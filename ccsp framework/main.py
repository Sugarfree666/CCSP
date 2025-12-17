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

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
        try:
            # 构造基本参数
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            }

            # 只有在 json_mode=True 时才强制 JSON 格式
            if json_mode:
                params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Call Error: {e}")
            # 如果是 JSON 模式失败，返回空 JSON；否则返回空字符串
            return "{}" if json_mode else "Error generating answer."

    # [在 main.py 的 LLMService 类中修改此方法]
    def decompose_query(self, query: str) -> List[Dict]:
        """
        核心 Prompt：将复杂自然语言问题分解为 Anchors 和 Filters
        """
        system_prompt = """
        You are a query understanding engine for a Knowledge Graph QA system.
        Your task is to decompose a complex natural language question into a list of structured constraints (Thoughts).

        Analyze the question and break it down into:
        1. **Anchors**: Concrete entities that are the STARTING POINTS of the search. 
           - IMPORTANT: People (Actors, Directors, Authors), Locations, and Organizations are usually Anchors.
           - If a query mentions multiple entities (e.g., "Movies by Director X starring Actor Y"), BOTH X and Y are Anchors.
        2. **Filters**: Constraints on attributes, such as dates, numbers, genres, or simple adjectives.

        Output Format (JSON):
        {
            "thoughts": [
                {
                    "content": "Description",
                    "key": "Attribute name",
                    "value": "Entity Name or Literal Value",
                    "op": "Operator (==, >, <, contains)",
                    "role": "anchor" or "filter",
                    "type": "entity" or "literal"
                }
            ]
        }

        Example:
        Input: "Which film starring Chester Bennington and directed by Kevin Greutert was released after 1995?"
        Output:
        {
            "thoughts": [
                {"content": "Starring Chester Bennington", "key": "cast", "value": "Chester Bennington", "op": "contains", "role": "anchor", "type": "entity"},
                {"content": "Directed by Kevin Greutert", "key": "director", "value": "Kevin Greutert", "op": "contains", "role": "anchor", "type": "entity"},
                {"content": "Released after 1995", "key": "release_date", "value": 1995, "op": ">", "role": "filter", "type": "literal"}
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

    # [替换 main.py 中 GoTEngine 类的 run 方法]
    def run(self, complex_question_data: Dict):
        question = complex_question_data['complex_question']
        print(f"\n{'=' * 60}\nUser Query: {question}\n{'=' * 60}")

        self.root = ThoughtNode("root", {"text": question})

        # =================================================================
        # [Layer 1] Decomposition
        # =================================================================
        print("\n[Layer 1] Decomposition (Generating Sub-thoughts)...")
        sub_constraints = self.llm.decompose_query(question)

        # 1. 提取所有 Anchors 和 Filters
        anchor_items = [item for item in sub_constraints if item['role'] == 'anchor']
        raw_filters = [item for item in sub_constraints if item['role'] != 'anchor']

        if not anchor_items:
            print("Error: No valid anchors found.")
            return

        final_anchors_config = []  # 用于存储处理好的 Anchor 配置 (QID + Path)
        sample_qid = None  # 用于学习 Schema 的样本

        # =================================================================
        # [Layer 1.5] Multi-Anchor Path Finding (Parallel Processing)
        # =================================================================
        print(f"\n[Layer 1.5] Processing {len(anchor_items)} Anchors in Parallel...")

        for idx, anchor_data in enumerate(anchor_items):
            print(f"\n  --- Processing Anchor {idx + 1}: {anchor_data['value']} ---")

            # A. Entity Linking
            qid = self.kg.search_entity_id(anchor_data['value'])
            if not qid:
                print(f"    [Skip] Could not link entity '{anchor_data['value']}'")
                continue
            print(f"    -> Linked: {qid}")

            # B. Path Finding (Probing)
            # 探测这个实体和“答案”之间的关系
            candidate_relations = self.kg.get_candidate_relations(qid)
            if not candidate_relations:
                print("    [Error] Isolated node.")
                continue

            # C. Selection (让 LLM 选择最佳路径)
            # Prompt 会根据 Anchor 和问题上下文选择，比如对于导演选 P57，对于演员选 P161
            selected_path = self.llm.select_best_path(question, anchor_data['value'], candidate_relations)
            rel_pid = selected_path.get('pid')
            rel_dir = selected_path.get('direction')
            print(f"    -> Path Selected: {rel_pid} ({rel_dir})")

            # D. 保存配置
            final_anchors_config.append({
                'qid': qid,
                'pid': rel_pid,
                'direction': rel_dir,
                'role': 'anchor',
                'name': anchor_data['value']
            })

            # E. Sampling (只需要做一次，或者直到成功为止)
            # 我们只需要一个样本来学习“电影”这个类别的 Schema，不需要每个 Anchor 都采样一次
            if not sample_qid:
                if rel_dir == 'reverse':
                    query = f"SELECT ?item WHERE {{ ?item wdt:{rel_pid} wd:{qid} }} LIMIT 1"
                else:
                    query = f"SELECT ?item WHERE {{ wd:{qid} wdt:{rel_pid} ?item }} LIMIT 1"

                res = self.kg.execute_query(query)
                if res and "entity" in res[0]['item']['value']:
                    sample_qid = res[0]['item']['value'].split('/')[-1]
                    print(f"    -> Sampling Success: Found sample instance {sample_qid}")

        if not final_anchors_config:
            print("Error: All anchors failed to link or find paths.")
            return

        # =================================================================
        # [Layer 2] Alignment (Mapping Filters to Schema)
        # =================================================================
        print("\n[Layer 2] Alignment (Mapping Filters to Sample Schema)...")

        # 1. 获取 Sample 的属性列表
        # 如果采样失败，_fetch_available_properties 会返回空，后续逻辑会 fallback 到常用属性字典
        schema_context = self._fetch_available_properties(sample_qid)

        # 2. 对 Filter 中的 Value 做实体链接 (例如 'documentary' -> Q93204)
        for item in raw_filters:
            if isinstance(item['value'], str) and not item['value'].isdigit():
                qid = self.kg.search_entity_id(item['value'])
                if qid: item['value_qid'] = qid

        # 3. 对齐属性
        aligned_filters = []
        for raw_item in raw_filters:
            aligned_content = self._generate_aligned_thought(raw_item, schema_context)
            if aligned_content.get('pid'):
                aligned_filters.append(aligned_content)
                val_disp = aligned_content.get('value_qid') or aligned_content.get('value')
                print(f"  -> Filter: '{raw_item['key']}' => '{aligned_content.get('pid')}' (Value: {val_disp})")
            else:
                print(f"  [Warning] Dropping filter '{raw_item['key']}'")

        # =================================================================
        # [Layer 3] Aggregation (Constructing SPARQL)
        # =================================================================
        print("\n[Layer 3] Aggregation (Intersection of all constraints)...")

        # wikidata_service.py 里的 construct_sparql_from_got 已经支持传入 anchor 列表
        # 它会生成多个 ?item wdt:Px wd:Anchor 语句，天然构成了 AND 逻辑 (Intersection)

        sparql_query = self.kg.construct_sparql_from_got(final_anchors_config, aligned_filters)
        print(f"  [Aggregated Query]:\n{sparql_query}")

        # 执行查询
        results = self.kg.execute_query(sparql_query)
        final_entities = self._parse_results(results)
        print(f"  => Found {len(final_entities)} results.")

        # [新增] 增强数据：将 Anchors 信息注入到每个结果中
        # 因为 SPARQL 是 AND 逻辑，所以查出来的结果一定满足所有 Anchor 条件
        enriched_data = []
        for entity in final_entities:
            # 1. 基础信息
            context_str = f"Entity: {entity.get('name')} ({entity.get('id')})\n"
            context_str += f"Description: {entity.get('description', 'N/A')}\n"

            # 2. 属性证据 (Filters) - 这里只有日期等
            context_str += "Matched Attributes:\n"
            for k, v in entity.get('attributes', {}).items():
                context_str += f"  - {k}: {v}\n"

            # 3. [关键!] 关系证据 (Anchors) - 强行把 Diana Ross 写进去
            # 逻辑：既然这个实体是靠搜 Diana Ross 找到的，那它一定和 Diana Ross 有关系
            context_str += "Verified Relationships (Anchors):\n"
            for anchor in final_anchors_config:
                role = anchor.get('name')
                pid = anchor.get('pid', 'Unknown Relation')
                context_str += f"  - Connected to: {role} (via {pid})\n"

            enriched_data.append(context_str)

        # 4. 组装最终 Prompt
        final_context = "\n---\n".join(enriched_data[:5])

        # 生成最终答案
        # [修改] 这里的 Prompt 稍微改一下，强调使用提供的 Evidence
        system_prompt = "You are a Knowledge Graph QA assistant. Synthesize the answer based strictly on the provided Evidence."
        user_prompt = f"""
            Question: {question}

            Evidence Retrieved from Knowledge Graph:
            {final_context}

            Please answer the question. If the evidence contains the answer, state it clearly.
            """

        ans = self.llm.chat(system_prompt, user_prompt, json_mode=False)
        print(f"\n[Final Answer]: {ans}")

    def _parse_results(self, raw_results):
        """
        将 SPARQL 返回结果解析为字典，包含所有“证据变量”。
        """
        parsed = []
        for row in raw_results:
            entity = {}
            evidence = {}

            # 1. 提取核心 Item 信息
            if 'item' in row:
                entity['uri'] = row['item']['value']
                entity['id'] = entity['uri'].split('/')[-1]
            if 'itemLabel' in row:
                entity['name'] = row['itemLabel']['value']

            # [新增] 提取描述信息，非常有助于 LLM 理解这是个电影还是书，还是人
            if 'itemDescription' in row:
                entity['description'] = row['itemDescription']['value']

            # 2. 提取 Filter 产生的证据 (Evidence)
            for key, val in row.items():
                # [修改] 排除列表增加 itemDescription
                if key not in ['item', 'itemLabel', 'itemDescription']:
                    readable_key = key.rsplit('_', 1)[0] if '_' in key else key
                    # 如果是日期，截取前10位看起来更干净
                    val_str = val['value']
                    if "T00:00:00Z" in val_str:
                        val_str = val_str.split('T')[0]
                    evidence[readable_key] = val_str

            if evidence:
                entity['attributes'] = evidence  # 改个名字叫 attributes 更直观

            parsed.append(entity)
        print(parsed)
        return parsed


    # -------- 核心变换逻辑：生成思维 --------


    def _generate_aligned_thought(self, raw_constraint: Dict, schema_context: Dict) -> Dict:
        """
        思维变换函数：T(Raw_Thought, Context) -> Aligned_Thought
        优化版：强制 LLM 基于 Value 的语义来选择 Property，而不是盲信用户的 key。
        """
        # 如果已经是 PID 格式，直接返回
        if raw_constraint['key'].startswith('P') and raw_constraint['key'][1:].isdigit():
            raw_constraint['pid'] = raw_constraint['key']
            return raw_constraint

        # 构造 Context 描述
        candidates_str = "\n".join([f"- {pid}: {label}" for pid, label in schema_context.items()])

        # 获取 Value 的显示名称 (如果有 QID，说明已经链接了实体)
        val_display = raw_constraint.get('value')
        if raw_constraint.get('value_qid'):
            # 这里我们只把 value_qid 给 LLM 参考，虽然它可能不知道 QID 具体是啥，
            # 但我们主要依赖 raw_constraint['value'] 的文本 (如 "novel series") 来做判断
            pass

        prompt = f"""
        You are a Semantic Alignment Expert for Knowledge Graphs.

        Task: Map the User's Constraint to the correct Wikidata Property ID (PID) from the provided Schema.

        User Constraint:
        - Attribute Name (User Guess): "{raw_constraint['key']}"
        - Value: "{val_display}"
        - Operator: {raw_constraint['op']}

        Available KG Properties (Schema from a similar item):
        {candidates_str}

        CRITICAL RULES:
        1. Ignore the "Attribute Name" if it conflicts with how the "Value" is typically used in Wikidata.
        2. "Novel series", "Film", "Book" are usually values for P31 (instance of).
        3. "Horror", "Comedy", "Fiction" are usually values for P136 (genre).
        4. "USA", "France" are usually values for P17 (country) or P495 (country of origin).
        5. Dates (1990, 2020) are usually P577 (publication date).

        Decision Logic:
        - Does "{val_display}" look like a Genre (P136) or a Type/Category (P31)?
        - Does it look like a Date (P577)?

        Return JSON: {{"reasoning": "why you chose this PID", "pid": "Pxxx"}}
        """

        response = self.llm.chat("You are a smart ontology mapper.", prompt)

        try:
            result = json.loads(response)
            new_thought = raw_constraint.copy()

            # 更新 PID
            selected_pid = result.get('pid')

            # 如果 LLM 没选出来，或者是瞎编的 PID (不在 schema 里)，我们要小心
            # 但有时候 Schema 不全，允许 LLM 预测常见的 P31/P136
            if selected_pid:
                new_thought['key'] = selected_pid
                new_thought['pid'] = selected_pid
                print(
                    f"    [Align Logic] Mapped '{raw_constraint['key']}' ({val_display}) -> {selected_pid}. Reason: {result.get('reasoning', 'None')}")
            else:
                print(f"    [Align Warning] LLM did not return a PID for {raw_constraint['key']}")

            return new_thought
        except Exception as e:
            print(f"  [Error] Alignment failed: {e}")
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

    def _fetch_available_properties(self, sample_qid):
        """
        获取样本实体的所有属性（移除 LIMIT 限制，确保不漏掉关键属性）。
        """
        if not sample_qid: return {}

        # 移除 LIMIT 50，改为 LIMIT 500 或不设限
        query = f"""
        SELECT DISTINCT ?p ?pLabel ?valLabel WHERE {{
          wd:{sample_qid} ?p ?val .
          ?prop wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          ?prop rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")
          OPTIONAL {{ ?val rdfs:label ?valLabel . FILTER(LANG(?valLabel) = "en") }}
        }} LIMIT 500
        """
        res = self.kg.execute_query(query)
        schema = {}

        # 过滤无用的 ID 属性
        ignore_keywords = ["ID", "identifier", "code", "number"]

        for r in res:
            pid = r['p']['value'].split('/')[-1]
            p_label = r.get('pLabel', {}).get('value', 'Unknown')
            val_example = r.get('valLabel', {}).get('value', '')

            # 简单过滤：跳过包含 ID 的属性，除非是特定关键属性
            if any(k in p_label for k in ignore_keywords) and "tax" not in p_label:
                continue

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
if __name__ == "__main__":
    # 1. 配置 API 信息
    API_KEY = "sk-wZPm2CCFydnh7Nuh9vuaMBLYiJxBxP0MsIMwp6rGZ87JVzkF"  # 填入你的真实 Key
    BASE_URL = "https://api.chatanywhere.tech"  # 或者 OpenAI 的地址
    MODEL_NAME = "gpt-3.5-turbo"

    # 2. 指定数据集路径 (请根据你本地实际路径修改)
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