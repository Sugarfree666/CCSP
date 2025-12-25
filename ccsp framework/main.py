import sys
import json
import logging
import re
import os
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
# 1. 实体链接工具 (保留，因为 Parsing 阶段仍需要它)
# ==============================================================================
def search_wikidata(label: str) -> str:
    """使用 Wikidata API 搜索实体的真实 QID。"""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "CCSP-Bot/1.0 (Research Project - Educational Use)",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        if data.get("search"):
            return data["search"][0]["id"]
    except Exception as e:
        logger.warning(f"[Entity Search] Failed for '{label}': {e}")
    return None


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
def parse_query_to_constraints(user_query: str, llm: LLMService) -> List[Constraint]:
    logger.info("Phase 1: Parsing natural language to constraints...")

    # === 修改点 1: Prompt 明确要求包裹在 "constraints" 键中 ===
    prompt = f"""
        Role: You are a Knowledge Graph Query Parser.
        Task: Convert the user's question into structured constraints.

        User Query: "{user_query}"

        Requirements:
        1. Identify atomic constraints.
        2. Property ID: Predict P-ID if sure (e.g. P57), else empty.
        3. Value: 
           - **Entity**: Output the English Name (e.g. "Chester Bennington").
           - **Quantity/Number**: Output ONLY the number, REMOVE units. (e.g. "109.5 minutes" -> "109.5").
           - **Date**: Format as YYYY or YYYY-MM-DD.
        4. Operator: =, >, <, contains.

        IMPORTANT Output Format:
        Return a JSON Object with a single key "constraints" containing the list.
        Example:
        {{
            "constraints": [
                {{ "id": "c1", "property_id": "Pxx", "property_label": "...", "operator": "=", "value": "...", "softness": 0.0 }}
            ]
        }}
    """

    try:
        data = llm.generate_json(prompt)

        # === 调试日志：打印 LLM 到底返回了什么 ===
        logger.info(f"DEBUG: Raw Parsed JSON: {json.dumps(data, ensure_ascii=False)}")

        constraints = []

        # === 修改点 2: 更强的容错解析逻辑 ===
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

        # 开始处理提取到的列表
        for item in target_list:
            raw_value = str(item.get("value", ""))
            operator = item.get("operator", "=")

            # ... (保留你原来的数值清洗逻辑: is_quantity 判断等) ...
            final_value = raw_value
            is_quantity = False
            number_match = re.search(r'^(\d+(\.\d+)?)\s*[a-zA-Z]*$', raw_value)

            if operator in [">", "<"] or (number_match and " " in raw_value):
                if number_match:
                    final_value = number_match.group(1)
                    is_quantity = True
                else:
                    is_quantity = True

            # ... (保留你原来的实体链接逻辑: search_wikidata) ...
            if (not is_quantity
                    and final_value
                    and not re.match(r'^Q\d+$', final_value)
                    and not re.match(r'^[\d\.\-\:]+$', final_value)):

                # 只有还没转成 QID 的才查
                found_qid = search_wikidata(final_value)
                if found_qid:
                    final_value = found_qid
                    operator = "="

            c = Constraint(
                id=item.get("id", "unknown"),
                property_id=item.get("property_id", ""),
                property_label=item.get("property_label", "unknown"),
                operator=operator,
                value=final_value,
                softness=float(item.get("softness", 0.0))
            )
            constraints.append(c)

        return constraints

    except Exception as e:
        logger.error(f"Parsing failed details: {e}", exc_info=True)  # 打印详细堆栈
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

    Task: Write a concise and natural answer. Explain HOW the system found the answer based on the history.
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
    api_key = os.getenv("LLM_API_KEY", "sk-diwupeelrzsrpfibyrdxlbebzvwrawvhfvlktobvlirjsefm")
    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1/")
    model_name = "deepseek-ai/DeepSeek-V3.2"  # 确保模型名正确

    # 2. 基础设施初始化
    try:
        llm_service = LLMService(api_key, base_url, model_name)
        wiki_service = WikidataService()

        # Optimizer 加载元数据 (Critic 的核心)
        optimizer = ConstraintOptimizer("property_metadata_final.json", llm_service)
        logger.info("Infrastructure initialized.")

    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        return

    # 3. 用户查询
    user_query = "Which male poet who was born after 1683 and died before 1826 influenced Charles Dickens?"
    print(f"Query: {user_query}\n")

    # 4. Phase 1: Parsing (将自然语言转为 Agent 的待办事项)
    constraints = parse_query_to_constraints(user_query, llm_service)

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
    critic = StatisticalCritic(optimizer)  # 评价器 (注入了 Optimizer)
    agent = GoTAgent(llm_service, env, critic)  # 大脑

    # Agent 开始自主解题
    final_candidates = agent.solve(user_query, constraints)

    # 6. Phase 3: Reporting
    generate_final_report(user_query, agent.state.history, final_candidates, llm_service, wiki_service)


if __name__ == "__main__":
    main()