import sys
import json
import logging
import re
import os
from unit_utils import UnitNormalizer
import requests
from typing import List, Dict, Any, Set

# === 引入自定义模块 (保持原有引用 + 新增 Agent 模块) ===
from data_model import Constraint
from wikidata_service import WikidataService
from optimizer import ConstraintOptimizer

# === [NEW] 引入 Agent 架构组件 ===
# 请确保这些文件已创建并在同一目录下
from graph_state import GraphState
from environment import GraphEnvironment
from critic import StatisticalCritic
from agent_brain import GoTAgent

from openai import OpenAI

os.makedirs("info_debug", exist_ok=True)
LOG_FILE = os.path.join("info_debug", "execution.log")
# ==============================================================================
# [配置日志]
# ==============================================================================
LOG_FILE = r"D:\GitHub\CCSP\info_debug\execution.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

if root_logger.hasHandlers():
    root_logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)


class NoisyLibFilter(logging.Filter):
    def filter(self, record):
        noisy_loggers = ["httpx", "httpcore", "openai", "urllib3", "connectionpool"]
        if any(ns in record.name for ns in noisy_loggers):
            return False
        return True


console_handler.addFilter(NoisyLibFilter())
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger("CCSP-AgentLauncher")



# ==============================================================================
# 2. LLM 服务 (保留，作为 Agent 的大脑接口)
# ==============================================================================
class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_text(self, prompt: str) -> str:
        """生成自然语言回复"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Text Gen Error: {e}")
            return "Error generating response."

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """增强版 JSON 生成：自动清洗"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}  # 显式要求 JSON
            )
            text = response.choices[0].message.content

            # 简单的清洗逻辑 (现在的模型通常能很好地遵循 json_object 模式)
            text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text)

            return json.loads(text)
        except Exception as e:
            logger.error(f"LLM JSON Error: {e}")
            return {}


# ==============================================================================
# 3. Parsing (保留，作为 Agent 的任务输入)
# ==============================================================================
def parse_query_to_constraints(user_query: str, llm: LLMService, wiki_service: WikidataService) -> List[Constraint]:
    logger.info("Phase 1: Parsing natural language to constraints...")

    # === 修改点 1: Prompt 明确要求 LLM 只提取语义标签，不要猜测 ID ===
    prompt = f"""
        Role: You are a Semantic Parser for Knowledge Graphs.
        Task: Extract constraints from the user query and map them to STANDARD Wikidata property labels.

        User Query: "{user_query}"

        Guidelines:
        1. **Property Labels**: Do NOT guess P-IDs (e.g., P123). Output the precise English label used in Wikidata.
           - "born" -> "date of birth"
           - "won" -> "award received"
           - "taller than" -> "height" (P2048)
           - "album by" -> "performer" (P175) or "artist"
           - "genre" -> "genre" (P136)

        2. **Values**: 
           - **Dates**: Convert ALL dates to ISO 8601 format (YYYY-MM-DD). E.g., "2013" -> "2013-01-01" or "2013-12-31" depending on operator.
           - **Entities**: Keep names as strings (e.g., "Taylor Swift", "Pop Rock").
           - **Numbers**: Extract pure numbers.
           - **Unit**: (CRITICAL) Extract the unit if present (e.g., "minutes", "km"). If no unit, return null.

        3. **Operator**: =, >, <, contains.

        Output Format (JSON list wrapped in "constraints"):
        {{
            "constraints": [
                {{ "property_label": "publication date", "operator": ">", "value": "2013-12-31", "unit": null }},
                {{ "property_label": "performer", "operator": "=", "value": "Taylor Swift", "unit": null }},
                {{ "property_label": "duration", "operator": "<", "value": "5471", "unit": "minutes" }}
            ]
        }}
    """
    try:
        data = llm.generate_json(prompt)

        # === 调试日志：打印 LLM 到底返回了什么 ===
        logger.info(f"DEBUG: Raw Parsed JSON: {json.dumps(data, ensure_ascii=False)}")

        constraints = []

        # === 修改点 2: 鲁棒的 JSON 解析逻辑 ===
        target_list = []

        if isinstance(data, list):
            target_list = data
        elif isinstance(data, dict):
            # 优先找 "constraints"
            if "constraints" in data and isinstance(data["constraints"], list):
                target_list = data["constraints"]
            else:
                # 如果没有 "constraints" 键，尝试找字典里第一个是 list 的值 (防守策略)
                for key, val in data.items():
                    if isinstance(val, list):
                        logger.warning(f"Warning: Found constraints under unexpected key '{key}'. Using it.")
                        target_list = val
                        break

        for item in target_list:
            # 1. 获取 LLM 提取的语义标签和值
            raw_label = item.get("property_label", "")
            final_value = item.get("value", "")

            # === 修改点 3 [CRITICAL FIX]: 提前定义 is_quantity ===
            # 逻辑：尝试判断值是否为数值或日期，如果是，则不需要进行实体链接
            is_quantity = False

            # 尝试判断是否为纯数字/浮点数
            try:
                float(str(final_value))
                is_quantity = True
            except ValueError:
                pass

            # 尝试判断是否为年份或日期 (YYYY 或 YYYY-MM-DD)
            # 如果是日期，也被视为 Quantity 类数据，不查 QID
            if not is_quantity and re.match(r'^\d{4}(-\d{2}-\d{2})?$', str(final_value)):
                is_quantity = True

            # 2. [关键] Relation Linking: 标签 -> P-ID
            # 我们不再信任 LLM 的 ID，即使它输出了 (通常是错的)
            # 强制调用 WikiService 进行搜索
            linked_pid = wiki_service.search_property(raw_label)

            if not linked_pid:
                logger.warning(
                    f"  [Linker Failed] Could not map label '{raw_label}' to a Property ID. Dropping this constraint.")
                continue  # 丢弃无法链接的属性，防止后续查询报错

            logger.info(f"  [Linker] '{raw_label}' -> {linked_pid}")

            # 3. Entity Linking: 值 -> Q-ID
            # 如果不是数值/日期，且不是 QID，尝试链接实体
            if not is_quantity and not re.match(r'^Q\d+$', str(final_value)):
                linked_qid = wiki_service.search_entity(final_value)
                if linked_qid:
                    logger.info(f"  [Entity Linker] '{final_value}' -> {linked_qid}")
                    final_value = linked_qid
                else:
                    # 如果搜不到实体，可能它本身就是字符串值（如名字），保留原值
                    logger.info(f"  [Entity Linker] Could not find QID for '{final_value}', keeping as string.")

            # 4. 构建约束对象
            c = Constraint(
                id=item.get("id", f"c{len(constraints) + 1}"),
                property_id=linked_pid,  # 使用 Linker 查到的真实 PID
                property_label=raw_label,
                operator=item.get("operator", "="),
                value=final_value,  # 使用 Linker 查到的真实 QID 或清洗后的值
                softness=0.0,

                # === [新增] 这里一定要加上 unit 字段的提取 ===
                unit=item.get("unit")
            )
            constraints.append(c)

        return constraints

    except Exception as e:
        logger.error(f"Parsing failed details: {e}", exc_info=True)  # 打印详细堆栈以便调试
        return []


# ==============================================================================
# 4. Final Response Generation (适配 Agent 结果)
# ==============================================================================
def generate_final_report(user_query: str, agent_history: List[str], final_candidates: Set[str], llm: LLMService,
                          wiki_service: WikidataService):
    """
    Phase 3: 让 LLM 基于 Agent 的思考过程生成最终报告。
    """
    logger.info("Phase 3: Generating Final Report...")

    # 1. 获取最终实体的 Label
    entity_labels = []
    if final_candidates:
        # 只取前 20 个避免溢出
        target_qids = list(final_candidates)[:20]
        values_str = " ".join([f"wd:{qid}" for qid in target_qids])
        sparql = f"""
        SELECT ?itemLabel WHERE {{
            VALUES ?item {{ {values_str} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        results = wiki_service.execute_sparql(sparql)
        for r in results:
            entity_labels.append(r.get('itemLabel', {}).get('value', 'Unknown'))

    # 2. 格式化上下文
    history_str = "\n".join([f"- {h}" for h in agent_history])
    results_str = ", ".join(entity_labels) if entity_labels else "None found."

    prompt = f"""
    Role: You are an AI Research Assistant summarizing a complex reasoning process.

    User Query: "{user_query}"

    Reasoning History (Graph of Thoughts Trace):
    {history_str}

    Final Results:
    {results_str}

    Task:Based on the above information, please give the answer you think is appropriate. No unnecessary explanations are needed.
    """

    report = llm.generate_text(prompt)
    print("\n" + "=" * 50)
    print("Final Agent Report:")
    print("=" * 50)
    print(report)
    print("=" * 50)


# ==============================================================================
# 5. 主流程 (The New Agentic Main)
# ==============================================================================
def main():
    print("=== CCSP Framework: Agentic Graph of Thoughts ===\n")

    # 1. 配置
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("model_name")  # 确保模型名正确

    # 2. 基础设施初始化
    try:
        llm_service = LLMService(api_key, base_url, model_name)
        wiki_service = WikidataService()

        # [CHANGE] 初始化优化器，传入 wiki_service，不再需要 metadata_path
        optimizer = ConstraintOptimizer(wiki_service)

        logger.info("Infrastructure initialized.")

    except Exception as e:
        logger.error(f"Init Failed: {e}")
        return

    # 3. 用户查询
    user_query = "Which comedy film starring Taylor Lautner was released after 2009 and has a runtime of less than 122.5 minutes?"
    print(f"Query: {user_query}\n")

    # 4. Phase 1: Parsing (将自然语言转为 Agent 的待办事项)
    constraints = parse_query_to_constraints(user_query, llm_service, wiki_service)

    print("\n--- Normalizing Units ---")
    normalizer = UnitNormalizer()
    constraints = normalizer.normalize(constraints)

    print("\n[System] Probing database for optimal execution path...")
    constraints = optimizer.optimize(constraints)

    if not constraints:
        logger.error("No constraints parsed. Exiting.")
        return

    print(f"Parsed {len(constraints)} constraints.")
    for c in constraints:
        print(f" - {c.property_label}: {c.value} (Op: {c.operator})")

    # 5. Phase 2: Agent Assembly & Execution (Agent 组装与执行)
    print("\n--- Handing over to GoT Agent ---")

    # 组装部件
    env = GraphEnvironment(wiki_service)  # 工具箱
    critic = StatisticalCritic(optimizer)
    agent = GoTAgent(llm_service, env, critic)
    # Agent 开始自主解题
    final_candidates = agent.solve(user_query, constraints)

    # 6. Phase 3: Reporting
    generate_final_report(user_query, agent.state.history, final_candidates, llm_service, wiki_service)


if __name__ == "__main__":
    main()