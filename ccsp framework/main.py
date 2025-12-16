import json
import os
import uuid
from typing import List, Dict, Any, Optional, Set
from wikidata_service import WikidataKG
from openai import OpenAI


# ==========================================
# 1. 配置与 LLM 适配层 (LLM Adapter)
# ==========================================

class LLMService:
    """
    基于 OpenAI 格式的 LLM 服务封装 (支持 DeepSeek, GPT-4 等)
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # 保持低温度以获得稳定的结构化输出
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return "{}"

    def decompose_query(self, query: str) -> List[Dict]:
        """
        核心 Prompt：将复杂自然语言问题分解为 Anchor 和 Filter
        """
        system_prompt = """
        You are a query understanding engine for a Knowledge Graph QA system.
        Your task is to decompose a complex natural language question into a list of structured constraints (Thoughts).

        Analyze the question and break it down into:
        1. **Anchors**: Concrete entities that are the starting point of the search (e.g., people, specific movies, countries).
        2. **Filters**: Constraints on attributes, such as dates, numbers, genres, or specific relation values.

        Output Format (JSON):
        {
            "thoughts": [
                {
                    "content": "Description of the thought",
                    "key": "Attribute name or Relation ID (e.g., 'cast', 'publication date', 'P577')",
                    "value": "The value to look for",
                    "op": "Operator (==, >, <, >=, <=, contains)",
                    "role": "anchor" or "filter",
                    "type": "entity" or "literal"
                }
            ]
        }

        Example:
        Input: "Which movie starring Taylor Lautner was released after 2009 and has a runtime shorter than 122 minutes?"
        Output:
        {
            "thoughts": [
                {"content": "Find movies starring Taylor Lautner", "key": "starring", "value": "Taylor Lautner", "op": "contains", "role": "anchor", "type": "entity"},
                {"content": "Released date after 2009", "key": "release_date", "value": 2009, "op": ">", "role": "filter", "type": "literal"},
                {"content": "Runtime less than 122 minutes", "key": "runtime", "value": 122, "op": "<", "role": "filter", "type": "literal"}
            ]
        }
        """

        user_prompt = f"Analyze and decompose this question: '{query}'"

        result_str = self.chat(system_prompt, user_prompt)
        try:
            data = json.loads(result_str)
            return data.get("thoughts", [])
        except json.JSONDecodeError:
            print("Failed to decode JSON from LLM")
            return []

    def classify_constraint(self, constraint_info: Dict) -> str:
        """
        判断约束是硬约束(Hard)还是软约束(Soft)
        """
        # 简单逻辑：实体匹配通常是硬约束，数值范围通常是软约束
        if constraint_info.get("type") == "entity" or constraint_info.get("op") == "==":
            return "hard"
        return "soft"

    # 在 LLMService 类中添加
    def select_best_path(self, question: str, anchor_text: str, candidates: List[Dict]) -> Dict:
        """
        从 KG 返回的真实关系列表中，选择最符合问题的一条。
        """
        # 构造选项列表字符串
        # 格式: [P50] author (direction: reverse)
        options_str = ""
        for item in candidates:
            dir_str = "Answer -> Anchor" if item['direction'] == "reverse" else "Anchor -> Answer"
            options_str += f"- ID: {item['pid']} | Label: {item['label']} | Flow: {dir_str}\n"

        system_prompt = """
        You are a Path Selection Expert for Knowledge Graphs.
        Your task: Select the SINGLE best relation ID that connects the Anchor Entity to the Target Answer intended by the User Question.

        Example 1:
        Question: "Books by Beverly Cleary?" (Anchor: Beverly Cleary)
        Candidates include: "author (P50, Answer->Anchor)", "birth place (P19, Anchor->Answer)"
        Choice: {"pid": "P50", "direction": "reverse"} (Because books point TO the author)

        Example 2:
        Question: "Capital of France?" (Anchor: France)
        Candidates include: "capital (P36, Anchor->Answer)", "continent (P30, Anchor->Answer)"
        Choice: {"pid": "P36", "direction": "forward"}

        Return JSON: {"pid": "Pxxx", "direction": "forward/reverse"}
        """

        user_prompt = f"""
        User Question: "{question}"
        Anchor Entity: "{anchor_text}"

        Candidate Relations from KG:
        {options_str}

        Which relation leads to the answer?
        """

        try:
            res = self.chat(system_prompt, user_prompt)
            return json.loads(res)
        except:
            # 保底策略：如果 LLM 选不出来，根据经验返回一个常见的
            return {"pid": "P50", "direction": "reverse"}


    def query_anchor(self, key: str, value: Any, op: str = "==") -> Set[str]:
        """
        根据 Anchor 条件查找实体 ID
        """
        results = set()
        print(f"    [KG Query] Search {key} {op} {value}...")
        for item in self.mock_db:
            item_val = item.get(key)
            if not item_val: continue

            # 简单的包含/相等匹配
            if op == "contains" and isinstance(item_val, list):
                if value in item_val: results.add(item['id'])
            elif str(item_val) == str(value):
                results.add(item['id'])
        return results

    def get_entity_details(self, ids: Set[str]) -> List[Dict]:
        return [item for item in self.mock_db if item['id'] in ids]

    def check_filter(self, entity_id: str, constraints: List[Dict]) -> bool:
        """
        在 Python 端执行复杂的 Filter 逻辑 (>, <, etc.)
        """
        entity = next((e for e in self.mock_db if e['id'] == entity_id), None)
        if not entity: return False

        for c in constraints:
            key = c['key']
            target = c['value']
            op = c['op']
            actual = entity.get(key)

            if actual is None: return False

            try:
                if op == ">" and not (float(actual) > float(target)): return False
                if op == "<" and not (float(actual) < float(target)): return False
                if op == ">=" and not (float(actual) >= float(target)): return False
                if op == "<=" and not (float(actual) <= float(target)): return False
                if op == "==" and str(actual) != str(target): return False
                if op == "contains" and isinstance(actual, list) and target not in actual: return False
            except ValueError:
                continue  # 类型转换失败忽略

        return True

    def find_nearest_value(self, base_ids: Set[str], target_attr: str, threshold: float) -> Optional[float]:
        """
        Refine 辅助：在候选集中查找数值属性的分布
        """
        values = []
        for pid in base_ids:
            entity = next((e for e in self.mock_db if e['id'] == pid), None)
            if entity and entity.get(target_attr):
                values.append(entity[target_attr])

        values.sort()
        # 寻找最接近 threshold 的值
        # 简单逻辑：返回第一个大于 threshold 的值
        for v in values:
            if v > threshold: return v
        return None


# ==========================================
# 3. 思维图节点 (Thought Node)
# ==========================================
class ThoughtNode:
    def __init__(self, role: str, content: Dict, parents: List['ThoughtNode'] = None):
        self.role = role  # 'root', 'anchor', 'filter_raw', 'filter_aligned', 'sparql'
        self.content = content  # 具体的约束数据
        self.parents = parents if parents else []
        self.children = []

    def add_child(self, node):
        self.children.append(node)


# ==========================================
# 4. 核心引擎 (GoT Engine for Complex Constraints)
# ==========================================

class GoTEngine:
    def __init__(self, api_key: str, base_url: str, model:str):
        self.llm = LLMService(api_key, base_url, model)
        self.kg = WikidataKG()
        self.root = None

    # [在 main.py 的 GoTEngine 类中替换 run 方法]
    def run(self, complex_question_data: Dict):
        question = complex_question_data['complex_question']
        print(f"\n{'=' * 60}\nUser Query: {question}\n{'=' * 60}")

        self.root = ThoughtNode("root", {"text": question})

        # =================================================================
        # [Layer 1] Decomposition
        # =================================================================
        print("\n[Layer 1] Decomposition (Generating Sub-thoughts)...")
        sub_constraints = self.llm.decompose_query(question)

        # 提取数据
        anchor_data = next((item for item in sub_constraints if item['role'] == 'anchor'), None)
        raw_filters = [item for item in sub_constraints if item['role'] != 'anchor']

        if not anchor_data:
            print("Error: No valid anchors found.")
            return

        # 1. Anchor Entity Linking
        # 尝试将 Anchor 文本 (Beverly Cleary) 转为 QID
        anchor_qid = self.kg.search_entity_id(anchor_data['value'])
        if not anchor_qid:
            print("Error: Anchor linking failed.")
            return

        print(f"  -> Anchor Linked: {anchor_data['value']} ({anchor_qid})")

        # 对 Filter 的 Value 进行 Refinement (Entity Linking)
        # 这一步是为了解决 "novel series" -> Qxxxx 的问题
        for item in raw_filters:
            if isinstance(item['value'], str) and item['op'] in ['contains', '==', 'is']:
                # 排除纯数字字符串
                if not item['value'].isdigit():
                    print(f"  [Refinement] Linking filter value: '{item['value']}'...")
                    qid = self.kg.search_entity_id(item['value'])
                    if qid:
                        print(f"    -> Success: {qid}")
                        item['value_qid'] = qid
                        item['type'] = 'entity'

        # =================================================================
        # [Layer 1.5] Topology-Aware Path Finding (拓扑感知路径发现)
        # =================================================================
        print("\n[Layer 1.5] Path Finding (Probing KG Topology)...")

        # 1. Discover: 从 KG 获取 Anchor 周围的真实关系
        candidate_relations = self.kg.get_candidate_relations(anchor_qid)
        print(f"  -> Found {len(candidate_relations)} connections around {anchor_data['value']}.")

        if not candidate_relations:
            print("  [Error] Isolated node in KG.")
            return

        # 2. Select: 让 LLM 从真实候选项里挑一个
        selected_path = self.llm.select_best_path(question, anchor_data['value'], candidate_relations)
        rel_pid = selected_path.get('pid')
        rel_dir = selected_path.get('direction')
        print(f"  -> Path Selected: {rel_pid} ({rel_dir})")

        # 3. Sample: 根据选定的方向进行采样，获取一个 Target Instance
        if rel_dir == 'reverse':
            # Answer -> Anchor (e.g., Book -> Author)
            sample_query = f"SELECT ?item WHERE {{ ?item wdt:{rel_pid} wd:{anchor_qid} }} LIMIT 1"
        else:
            # Anchor -> Answer (e.g., Country -> Capital)
            sample_query = f"SELECT ?item WHERE {{ wd:{anchor_qid} wdt:{rel_pid} ?item }} LIMIT 1"

        sample_results = self.kg.execute_query(sample_query)
        sample_qid = None

        if sample_results:
            sample_uri = sample_results[0]['item']['value']
            # 确保是实体而不是数值
            if "entity" in sample_uri:
                sample_qid = sample_uri.split('/')[-1]
                print(f"  -> Sampling Success: Found sample instance {sample_qid}")
            else:
                print("  [Info] Target is a literal value (e.g., date/number), skipping schema learning.")
        else:
            print("  [Warning] Sampling returned no results. Schema learning might be limited.")

        # =================================================================
        # [Layer 2] Alignment (基于 Sample Schema 对齐 Filter)
        # =================================================================
        print("\n[Layer 2] Alignment (Mapping Filters to Sample Schema)...")

        # 1. 获取 Sample 的属性列表 (Context)
        schema_context = self._fetch_available_properties(sample_qid) if sample_qid else {}
        print(f"  -> Learned {len(schema_context)} properties from sample.")

        aligned_filter_nodes = []
        for raw_item in raw_filters:
            # 传入 Context 进行对齐 (例如 genre -> P136)
            aligned_content = self._generate_aligned_thought(raw_item, schema_context)

            # 创建节点
            node = ThoughtNode("filter_aligned", aligned_content, parents=[self.root])
            aligned_filter_nodes.append(node)

            val_disp = aligned_content.get('value_qid') or aligned_content.get('value')
            print(f"  -> Filter: '{raw_item['key']}' => '{aligned_content['pid']}' (Value: {val_disp})")

        # =================================================================
        # [Layer 3] Aggregation (构建并执行最终查询)
        # =================================================================
        print("\n[Layer 3] Aggregation (Constructing SPARQL)...")

        # 构造 Final Anchors
        # 这里的关键是把 Layer 1.5 选出来的路径 (PID 和 Direction) 放进去
        final_anchors = [{
            'qid': anchor_qid,
            'pid': rel_pid,
            'direction': rel_dir,
            'role': 'anchor'
        }]

        final_filters = [n.content for n in aligned_filter_nodes]

        # 调用 wikidata_service 中的构建函数
        sparql_query = self.kg.construct_sparql_from_got(final_anchors, final_filters)
        print(f"  [Aggregated Query]:\n{sparql_query}")

        # 执行查询
        results = self.kg.execute_query(sparql_query)
        final_entities = self._parse_results(results)
        print(f"  => Found {len(final_entities)} results.")

        # 生成最终答案
        ans = self.llm.chat(
            "You are a helpful assistant.",
            f"Question: {question}\nData: {json.dumps(final_entities)}\nAnswer:"
        )
        print(f"\n[Final Answer]: {ans}")

    # 在 GoTEngine 类中添加这个方法
    def _parse_results(self, raw_results):
        """
        将 SPARQL 返回的复杂 JSON 格式解析为简单的字典列表
        """
        parsed = []
        for row in raw_results:
            entity = {}
            # 提取 ?item (URI)
            if 'item' in row:
                entity['uri'] = row['item']['value']
                entity['id'] = entity['uri'].split('/')[-1]  # QID
            # 提取 ?itemLabel
            if 'itemLabel' in row:
                entity['name'] = row['itemLabel']['value']
            parsed.append(entity)
        return parsed

    # -------- 核心变换逻辑：生成思维 --------

    def _generate_aligned_thought(self, raw_constraint: Dict, schema_context: Dict) -> Dict:
        """
        思维变换函数：T(Raw_Thought, Context) -> Aligned_Thought
        这是你框架中 '根据前思维进行的生成思维变换' 的具体实现。
        """
        # 如果已经是 PID 格式（数据集中可能存在），直接返回
        if raw_constraint['key'].startswith('P') and raw_constraint['key'][1:].isdigit():
            raw_constraint['pid'] = raw_constraint['key']
            return raw_constraint

        # 构造 Prompt，要求 LLM 基于 Context 生成新的属性定义
        candidates_str = "\n".join([f"{pid}: {label}" for pid, label in schema_context.items()])

        prompt = f"""
        Current Thought: User is looking for attribute "{raw_constraint['key']}" (Context: {raw_constraint['op']} {raw_constraint['value']}).

        Available Knowledge (KG Schema):
        {candidates_str}

        Task: Generate a new thought that aligns the user's attribute to a specific Wikidata Property ID (PID).
        Return JSON: {{"key": "mapped_label", "pid": "Pxxx"}}
        """

        # 这是一个生成过程
        response = self.llm.chat("You are a knowledge alignment engine.", prompt)

        try:
            result = json.loads(response)
            # 继承原有约束的值和操作符，但更新 key 为 PID
            new_thought = raw_constraint.copy()
            new_thought['key'] = result.get('pid', raw_constraint['key'])  # 使用 PID 作为 Key
            new_thought['original_key'] = raw_constraint['key']
            new_thought['pid'] = result.get('pid')
            return new_thought
        except:
            print(f"  [Warning] Alignment generation failed for {raw_constraint['key']}")
            return raw_constraint

    # -------- 辅助方法 (Helpers) --------

    def _get_sample_instance(self, anchor):
        """
        helper: 找一个具体的例子来学习 Schema
        """
        # 这里需要知道 Anchor 的关系方向。如果是 "Starring Taylor", 关系是 P161
        # 但我们可能不知道 P161。
        # 策略：直接查 ?item ?p wd:QAnchor.
        query = f"SELECT ?item WHERE {{ ?item ?p wd:{anchor['qid']} }} LIMIT 1"
        res = self.kg.execute_query(query)
        if res:
            return res[0]['item']['value'].split('/')[-1]
        return None

    # [在 main.py 的 GoTEngine 类中替换此方法]
    def _fetch_available_properties(self, sample_qid):
        """
        获取样本实体的所有属性，以及该属性对应的值的Label（用于帮助理解语义）。
        """
        if not sample_qid: return {}

        # 这里的查询稍微复杂一点，为了获取 Example Value，帮助 LLM 理解属性含义
        query = f"""
        SELECT DISTINCT ?p ?pLabel ?valLabel WHERE {{
          wd:{sample_qid} ?p ?val .
          ?prop wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          # 获取属性的 Label
          ?prop rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")

          # 顺便获取一下值的 Label (只取 Entity 类型的 label)
          OPTIONAL {{ ?val rdfs:label ?valLabel . FILTER(LANG(?valLabel) = "en") }}
        }} LIMIT 50
        """
        res = self.kg.execute_query(query)
        schema = {}
        for r in res:
            pid = r['p']['value'].split('/')[-1]
            p_label = r.get('pLabel', {}).get('value', 'Unknown')
            val_example = r.get('valLabel', {}).get('value', '')

            # 存储格式：P136 -> "genre (e.g., Children's fiction)"
            if pid not in schema:
                desc = f"{p_label}"
                if val_example:
                    desc += f" (e.g., {val_example})"
                schema[pid] = desc
        return schema

    def _map_filters_with_schema(self, raw_filters, schema_map):
        """
        让 LLM 根据 KG 返回的 schema_map 做映射
        """
        schema_desc = "\n".join([f"{k}: {v}" for k, v in schema_map.items()])

        prompt_content = []
        for f in raw_filters:
            prompt_content.append(f"Attribute: '{f['key']}' (Context: {f['op']} {f['value']})")

        user_prompt = f"""
        I have a list of user constraints (Attributes). Map them to the most likely Wikidata Property ID based on the Candidate Properties list provided.

        Candidate Properties from KG:
        {schema_desc}

        User Constraints:
        {json.dumps(prompt_content)}

        Output JSON format:
        [
            {{"original_key": "runtime", "mapped_pid": "P2047"}},
            ...
        ]
        If no good match is found in candidates, try to predict the PID or output null.
        """

        response = self.llm.chat("You are a Schema Mapping Expert.", user_prompt)
        try:
            mapping = json.loads(response)
            # 将 PID 回填到 filters 中
            mapped_filters = []
            for f in raw_filters:
                new_f = f.copy()
                # 找对应的 PID
                match = next((m for m in mapping if m.get('original_key') == f['key']), None)
                if match and match.get('mapped_pid'):
                    new_f['key'] = match['mapped_pid']  # 替换为 P2047
                    mapped_filters.append(new_f)
                else:
                    print(f"Warning: Could not map attribute '{f['key']}'")
            return mapped_filters
        except:
            print("Mapping failed.")
            return raw_filters


# ==========================================
# 运行脚本
# ==========================================
# ==========================================
# 运行脚本
# ==========================================
if __name__ == "__main__":
    # 1. 配置 API 信息
    API_KEY = "sk-wZPm2CCFydnh7Nuh9vuaMBLYiJxBxP0MsIMwp6rGZ87JVzkF"  # 填入你的真实 Key
    BASE_URL = "https://api.chatanywhere.tech"  # 或者 OpenAI 的地址
    MODEL_NAME = "gpt-3.5-turbo"

    # 2. 指定数据集路径 (请根据你本地实际路径修改)
    # 假设 main.py 和 datasets_wiki 文件夹在同一级目录下
    DATASET_PATH = "D:\GitHub\CCSP\datasets\complex_constraint_dataset_rewrite_queries.json"

    # 检查 Key 是否填入
    if not API_KEY or API_KEY == "sk-...":
        print("错误：请先在代码中填入有效的 API Key 和 Base URL。")
        exit()

    # 3. 初始化引擎 (只初始化一次，复用连接)
    engine = GoTEngine(api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME)

    try:
        # 4. 读取数据集
        print(f"正在加载数据集: {DATASET_PATH} ...")
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"成功加载 {len(dataset)} 条数据。开始执行...")

        # 5. 循环处理
        # 你可以使用 dataset[:5] 来先测试前5条，跑通后再去掉切片跑全量
        for i, item in enumerate(dataset[:1]):
            print(f"\n{'#' * 60}")
            print(f"进度: {i + 1}/{len(dataset)}")
            print(f"{'#' * 60}")
            try:
                # 调用引擎处理单条数据
                engine.run(item)

            except Exception as e:
                print(f"[Error] 处理第 {i + 1} 条数据时发生错误: {e}")
                # 继续处理下一条，不要因为一条报错就停止整个程序
                continue

    except FileNotFoundError:
        print(f"错误：找不到文件 {DATASET_PATH}。请检查路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：文件 {DATASET_PATH} 不是有效的 JSON 格式。")