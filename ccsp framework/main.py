import sys
import json
import logging
import re
import os
import requests
from typing import List, Dict, Any, Set

# === 引入自定义模块 ===
# 确保 data_model.py, optimizer.py, wikidata_service.py 在同一目录下
from data_model import Constraint
from optimizer import ConstraintOptimizer
from wikidata_service import WikidataService
from openai import OpenAI, OpenAIError

# === 配置日志 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CCSP-GraphEngine")


# ==============================================================================
# 0. 工具函数：实体链接 (解决 LLM 幻觉的关键)
# ==============================================================================
def search_wikidata(label: str) -> str:
    """
    使用 Wikidata API 搜索实体的真实 QID。
    """
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
        # 添加 headers 参数
        response = requests.get(url, params=params, headers=headers, timeout=5)

        # 增加状态码检查
        if response.status_code != 200:
            logger.warning(f"[Entity Search] HTTP Error {response.status_code} for '{label}'")
            return None

        data = response.json()
        if data.get("search"):
            return data["search"][0]["id"]

    except json.JSONDecodeError:
        logger.warning(f"[Entity Search] JSON Decode Error for '{label}'. Response text: {response.text[:100]}...")
    except Exception as e:
        logger.warning(f"[Entity Search] Failed for '{label}': {e}")

    return None


# ==============================================================================
# 1. LLM 服务 (支持代理与清洗)
# ==============================================================================
class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_text(self, prompt: str) -> str:
        """生成自然语言回复 (非 JSON)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # 稍微提高温度，让回答更自然
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Text Gen Error: {e}")
            return "Sorry, I could not generate a final answer due to an error."

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """增强版 JSON 生成：自动清洗特殊字符"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = response.choices[0].message.content

            # 清洗：移除 Markdown 标记和不可见空格
            text = text.replace('\u00A0', ' ')
            json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)

            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(text)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}


# ==============================================================================
# 2. Parsing (解析阶段)
# ==============================================================================
def parse_query_to_constraints(user_query: str, llm: LLMService) -> List[Constraint]:
    logger.info("Phase 1: Parsing natural language to constraints...")

    # Prompt 明确要求不猜 ID，只输出英文原名
    prompt = f"""
    Role: You are a Knowledge Graph Query Parser.
    Task: Convert the user's question into structured constraints.

    User Query: "{user_query}"

    Requirements:
    1. Identify atomic constraints.
    2. Property ID: Predict P-ID if sure (e.g. P57), else empty.
    3. Value: 
       - **DO NOT GUESS QIDs**. 
       - Output the exact **English Name** of the entity (e.g. "Chester Bennington", "Horror film").
       - For numbers/dates, keep them as is.
    4. Operator: =, >, <, contains.

    Output JSON List:
    [{{ "id": "c1", "property_id": "Pxx", "property_label": "...", "operator": "=", "value": "English Label Here", "softness": 0.0 }}]
    """

    try:
        data = llm.generate_json(prompt)
        constraints = []
        if isinstance(data, list):
            for item in data:
                label_value = str(item.get("value", ""))
                # === 实体链接逻辑 ===
                # 如果不是 QID 且不是纯数字/日期，尝试搜索真实 QID
                real_value = label_value
                if label_value and not re.match(r'^Q\d+$', label_value) and not re.match(r'^[\d\.\-\:]+$', label_value):
                    logger.info(f"Linking entity: '{label_value}' ...")
                    found_qid = search_wikidata(label_value)
                    if found_qid:
                        logger.info(f"  -> Found: {found_qid}")
                        real_value = found_qid
                    else:
                        logger.warning(f"  -> Not found, using original string.")

                c = Constraint(
                    id=item.get("id", "unknown"),
                    property_id=item.get("property_id", ""),
                    property_label=item.get("property_label", "unknown"),
                    operator=item.get("operator", "="),
                    value=real_value,
                    softness=float(item.get("softness", 0.0))
                )
                constraints.append(c)
        return constraints
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return []


# ==============================================================================
# 3. 核心：图推理执行引擎 (Graph Reasoning Engine)
# ==============================================================================
class GraphReasoningExecutor:
    """
    实现“Anchor -> Step-by-Step Screening”的执行逻辑。
    """

    def __init__(self, wikidata_service: WikidataService):
        self.service = wikidata_service
        self.trace = []  # 用于记录推理轨迹 (Evidence)

    def execute(self, sorted_constraints: List[Constraint]) -> Dict[str, Any]:
        """
        执行推理并返回结果和证据。
        Return: {
            "final_entities": [{"id": "Qxx", "label": "Saw 3D"}],
            "trace": ["Selected Anchor...", "Filtered by...", "Remaining..."]
        }
        """
        self.trace = []  # 重置轨迹
        if not sorted_constraints:
            return {"final_entities": [], "trace": ["No constraints provided."]}

        # 1. Anchor 阶段
        anchor = sorted_constraints[0]
        anchor_log = f"Step 1 (Anchor): Started search with [{anchor.property_label} = {anchor.value}]."
        logger.info(anchor_log)
        self.trace.append(anchor_log)

        candidates = self._fetch_anchor_candidates(anchor)
        count_log = f"  -> Found {len(candidates)} initial candidates."
        logger.info(count_log)
        self.trace.append(count_log)

        if not candidates:
            return {"final_entities": [], "trace": self.trace}

        # 2. 逐步筛选 (Iterative Pruning)
        for i, constraint in enumerate(sorted_constraints[1:], 2):
            if not candidates:
                break

            step_log = f"Step {i} (Filter): Applying constraint [{constraint.property_label} {constraint.operator} {constraint.value}]."
            logger.info(step_log)
            self.trace.append(step_log)

            candidates = self._apply_filter(candidates, constraint)

            remain_log = f"  -> Candidates remaining: {len(candidates)}"
            logger.info(remain_log)
            self.trace.append(remain_log)

        # 3. 获取最终结果的详细信息 (Label)
        final_details = self._fetch_labels_for_qids(candidates)

        return {
            "final_entities": final_details,
            "trace": self.trace
        }

    def _fetch_anchor_candidates(self, c: Constraint) -> Set[str]:
        """
        针对 Anchor 节点生成初始 SPARQL 并执行。
        """
        val_str = str(c.value)

        # 情况 A: 已经是 QID (e.g., Q19198) - 最理想情况
        if re.match(r'^Q\d+$', val_str):
            where_clause = f"?item wdt:{c.property_id} wd:{val_str} ."

        # 情况 B: 仍然是字符串 (Entity Linking 失败)
        # 我们不能直接比较 ?item wdt:Pxx "String"，因为对象通常是 URI。
        # 我们需要查找该对象的 Label 是否匹配字符串。
        else:
            logger.info(f"Fallback: Searching by label match for {val_str} on property {c.property_id}")
            # 这是一个比较昂贵的操作，但比返回0结果要好
            # 逻辑：?item -> ?target_entity -> [Label == "Chester Bennington"]
            where_clause = f"""
                ?item wdt:{c.property_id} ?target .
                ?target rdfs:label ?targetLabel .
                FILTER(LCASE(STR(?targetLabel)) = LCASE("{val_str}")) .
                FILTER(LANG(?targetLabel) = "en") .
            """

        sparql = f"""
        SELECT DISTINCT ?item WHERE {{
            {where_clause}
        }}
        LIMIT 1000
        """

        # 调试用：打印生成的 SPARQL
        print(f"DEBUG SPARQL:\n{sparql}")

        results = self.service.execute_sparql(sparql)

        qids = set()
        for r in results:
            url = r['item']['value']
            if "entity/" in url:
                qids.add(url.split("/")[-1])
        return qids

    def _apply_filter(self, current_candidates: Set[str], c: Constraint) -> Set[str]:
        """
        构造 VALUES 子句，对现有 candidates 进行 SPARQL 过滤。
        """
        # 将当前候选集转换为 VALUES 字符串 (e.g., "wd:Q1 wd:Q2 ...")
        # 注意：如果候选集太大，可能需要分批处理。这里简化为一次处理。
        values_str = " ".join([f"wd:{qid}" for qid in current_candidates])

        val_str = str(c.value)
        is_qid = bool(re.match(r'^Q\d+$', val_str))
        is_date = bool(re.match(r'^\d{4}-\d{2}-\d{2}', val_str))
        is_number = val_str.replace('.', '', 1).isdigit()

        # 构造过滤逻辑
        filter_clause = ""
        target = f"wd:{val_str}" if is_qid else "?val"

        triple = f"?item wdt:{c.property_id} {target} ."

        if not is_qid:
            # 构造 FILTER 表达式
            if is_date:
                val_fmt = f"'{val_str}'^^xsd:dateTime"
            elif is_number:
                val_fmt = val_str
            else:
                val_fmt = f"'{val_str}'"

            if c.operator == ">":
                filter_clause = f"FILTER(?val > {val_fmt})"
            elif c.operator == "<":
                filter_clause = f"FILTER(?val < {val_fmt})"
            elif c.operator == "contains":
                filter_clause = f"FILTER(CONTAINS(LCASE(?val), LCASE({val_fmt})))"
            else:
                filter_clause = f"FILTER(?val = {val_fmt})"

        sparql = f"""
        SELECT DISTINCT ?item WHERE {{
            VALUES ?item {{ {values_str} }}
            {triple}
            {filter_clause}
        }}
        """

        results = self.service.execute_sparql(sparql)

        # 提取符合条件的 QID
        valid_qids = set()
        for r in results:
            url = r['item']['value']
            valid_qids.add(url.split("/")[-1])

        return valid_qids

    def _fetch_labels_for_qids(self, qids: Set[str]) -> List[Dict[str, str]]:
        """
        根据 QID 获取 Label，不再只是打印，而是返回数据结构
        """
        if not qids:
            return []

        # 限制数量，防止 Prompt 过长
        target_qids = list(qids)[:20]
        values_str = " ".join([f"wd:{qid}" for qid in target_qids])

        sparql = f"""
        SELECT ?item ?itemLabel WHERE {{
            VALUES ?item {{ {values_str} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        results = self.service.execute_sparql(sparql)

        entities = []
        for r in results:
            url = r['item']['value']
            qid = url.split("/")[-1]
            label = r.get('itemLabel', {}).get('value', 'Unknown')
            entities.append({"id": qid, "label": label})

        return entities


def generate_final_response(user_query: str, execution_result: Dict, llm: LLMService):
    """
    框架第 7 步：基于答案和证据生成最终回复。
    """
    logger.info("Phase 3: Generating Final Answer with LLM...")

    entities = execution_result["final_entities"]
    trace = execution_result["trace"]

    # 1. 格式化证据 (Evidence)
    trace_str = "\n".join(trace)

    # 2. 格式化答案 (Answers)
    if not entities:
        answer_str = "No specific entities were found matching all constraints."
    else:
        answer_str = ", ".join([f"{e['label']} ({e['id']})" for e in entities])

    # 3. 构建 Prompt
    prompt = f"""
    Role: You are an intelligent Knowledge Graph Question Answering Assistant.

    User Question: "{user_query}"

    System Execution Trace (Evidence of how the answer was found):
    {trace_str}

    Final Retrieved Entities from Knowledge Graph:
    {answer_str}

    Task: 
    Based ONLY on the provided evidence and retrieved entities, answer the user's question naturally. 
    1. Direct Answer: State the answer clearly.
    2. Explanation: Briefly explain the reasoning path (e.g., "We started by looking for... then filtered by...").
    3. If no results were found, explain which constraints might have been too strict based on the trace.
    """

    # 4. 调用 LLM
    final_response = llm.generate_text(prompt)

    print("\n" + "=" * 50)
    print("🤖 Final LLM Response:")
    print("=" * 50)
    print(final_response)
    print("=" * 50)

# ==============================================================================
# 4. 主流程
# ==============================================================================
def main():
    print("=== CCSP Framework: Graph of Thoughts Execution ===\n")

    # 配置 API (请从环境变量或直接填入)
    api_key = os.getenv("LLM_API_KEY", "sk-wZPm2CCFydnh7Nuh9vuaMBLYiJxBxP0MsIMwp6rGZ87JVzkF")
    base_url = os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1")
    model = "gpt-3.5-turbo"

    llm = LLMService(api_key, base_url,model)
    wiki_service = WikidataService()

    try:
        optimizer = ConstraintOptimizer("property_metadata_final.json", llm)
        logger.info("Optimizer loaded.")
    except Exception as e:
        logger.error(f"Init failed: {e}")
        return

    # 示例查询
    user_query = "Which film starring Chester Bennington and directed by Kevin Greutert was released after 1995, is a horror film, and has a runtime shorter than 109.5 minutes?"
    print(f"Query: {user_query}\n")

    # 1. Parsing
    constraints = parse_query_to_constraints(user_query, llm)
    if not constraints: return

    # 2. Optimization (Planning)
    sorted_constraints = optimizer.optimize(constraints)

    print("\n--- Execution Plan ---")
    for i, c in enumerate(sorted_constraints):
        print(f"Step {i + 1}: {c.property_label} = {c.value} (Score: {c.priority_score:.2f})")

    # 3. Execution (Graph Reasoning)
    engine = GraphReasoningExecutor(wiki_service)

    # === 修改点：获取返回结果，而不是只打印 ===
    execution_result = engine.execute(sorted_constraints)

    # 4. Final Generation (Step 7)
    # 把所有上下文送给 LLM 做总结
    generate_final_response(user_query, execution_result, llm)


if __name__ == "__main__":
    main()