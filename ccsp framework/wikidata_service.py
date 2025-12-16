import time
import json
import hashlib
import os
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError


class WikidataKG:
    def __init__(self, cache_file="wikidata_cache.json"):
        self.endpoint = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        # 关键点1：必须设置独特的 User-Agent，包含你的联系方式，这是Wikidata要求的
        self.sparql.addCustomHttpHeader("User-Agent", "CCSP-Research-Bot/1.0 (uniqueyqlf@gmail.com)")

        # 关键点2：初始化本地缓存
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.save_counter = 0

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        # 每查询10次保存一次，防止程序崩溃丢失所有缓存
        self.save_counter += 1
        if self.save_counter % 10 == 0:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False)

    def execute_query(self, query):
        # 1. 检查缓存
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        if query_hash in self.cache:
            print("  [Cache Hit] 直接读取本地缓存")
            return self.cache[query_hash]

        # 2. 如果没缓存，执行网络请求（带重试机制）
        max_retries = 5
        wait_time = 1  # 初始等待1秒

        for attempt in range(max_retries):
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                bindings = results["results"]["bindings"]

                # 存入缓存
                self.cache[query_hash] = bindings
                self._save_cache()

                # 关键点3：每次成功后稍微睡一下，礼貌爬虫
                time.sleep(0.5)
                return bindings

            except Exception as e:
                # 检查是否是 429 (Too Many Requests)
                is_429 = "429" in str(e) or "Too Many Requests" in str(e)

                if is_429:
                    print(f"  [Rate Limit] 触发限流，等待 {wait_time} 秒后重试...")
                else:
                    print(f"  [Network Error] 尝试 {attempt + 1}/{max_retries}: {e}")

                # 关键点4：指数退避 (1s -> 2s -> 4s -> 8s ...)
                time.sleep(wait_time)
                wait_time *= 2

        print("  [Failure] 多次重试失败，放弃该查询")
        return []

    def search_entity_id(self, label):
        """
        Entity Linking: 将自然语言实体名 (如 'Taylor Lautner') 转为 QID (如 'Q143716')
        """
        # 1. 缓存检查
        cache_key = f"ENTITY:{label}"
        if cache_key in self.cache:
            print(f"    [Cache Hit] 实体 '{label}' -> {self.cache[cache_key]}")
            return self.cache[cache_key]

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label,
            "limit": 1
        }

        # --- [修改点] 定义 Headers，必须包含 User-Agent ---
        headers = {
            "User-Agent": "CCSP-Research-Bot/1.0 (uniqueyqlf@gmail.com)"  # 使用和你 __init__ 中一样的邮箱
        }

        try:
            # --- [修改点] 请求时带上 headers ---
            response = requests.get(url, params=params, headers=headers)

            # 调试：如果状态码不是 200，抛出异常
            response.raise_for_status()

            data = response.json()
            if data.get("search"):
                qid = data["search"][0]["id"]
                print(f"    [Link Success] '{label}' -> {qid}")

                # 存缓存
                self.cache[cache_key] = qid
                self._save_cache()
                return qid
            else:
                print(f"    [Link Failed] Wikidata 中找不到: {label}")

        except Exception as e:
            # 打印更详细的错误信息
            print(f"    [Link Error] Search failed for '{label}': {e}")
            # 如果是解析错误，打印一下返回的内容到底是啥（通常是 Access Denied 的 HTML）
            if "Expecting value" in str(e) and 'response' in locals():
                print(f"    [Debug] Server response: {response.text[:200]}...")  # 只打印前200个字符

        return None

        # 在 WikidataKG 类中添加此方法
    def get_candidate_relations(self, qid):
        """
        [通用方法] 获取与 Anchor 相连的所有属性（Top 50），包括正向和反向。
        用于让 LLM 从中选择正确的路径，而不是盲猜。
        """
        # 1. 反向关系 (Reverse): ?target ?p wd:Anchor (例如 ?book wdt:P50 wd:Author)
        # 这对于 "Books by...", "Movies starring..." 非常常见
        query_reverse = f"""
        SELECT DISTINCT ?p ?pLabel WHERE {{
          ?s ?p wd:{qid} .
          ?prop wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          ?prop rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")
        }} LIMIT 30
        """

        # 2. 正向关系 (Forward): wd:Anchor ?p ?target (例如 wd:Country wdt:P31 ?)
        # 这对于 "What is the capital of..." 非常常见
        query_forward = f"""
        SELECT DISTINCT ?p ?pLabel WHERE {{
          wd:{qid} ?p ?o .
          ?prop wikibase:directClaim ?p .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          ?prop rdfs:label ?pLabel .
          FILTER(LANG(?pLabel) = "en")
        }} LIMIT 30
        """

        relations = []

        # 执行查询并标记方向
        try:
            res_rev = self.execute_query(query_reverse)
            for r in res_rev:
                relations.append({
                    "pid": r['p']['value'].split('/')[-1],
                    "label": r.get('pLabel', {}).get('value', 'Unknown'),
                    "direction": "reverse"  # 意味着答案在 ?s 位置
                })

            res_fwd = self.execute_query(query_forward)
            for r in res_fwd:
                relations.append({
                    "pid": r['p']['value'].split('/')[-1],
                    "label": r.get('pLabel', {}).get('value', 'Unknown'),
                    "direction": "forward"  # 意味着答案在 ?o 位置
                })
        except Exception as e:
            print(f"Error fetching candidate relations: {e}")

        return relations


    def construct_sparql_from_got(self, anchors, filters):
        """
        根据 GoT 的节点构建 SPARQL。
        修改点：增加了对 filter 中 value_qid 的支持，实现了精确的实体匹配。
        """
        sparql = """
        SELECT DISTINCT ?item ?itemLabel WHERE {
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        """

        # ==========================================
        # 1. 处理 Anchors (起点实体)
        # ==========================================
        # 1. 处理 Anchors (支持方向)
        for anchor in anchors:
            if 'qid' in anchor:
                prop = anchor.get('pid') or anchor.get('key')
                if prop and prop.startswith('P'):
                    # 核心修改：判断方向
                    direction = anchor.get('direction', 'reverse')  # 默认为 reverse (常见于 book by author)

                    if direction == 'reverse':
                        # Answer(item) -> Anchor
                        sparql += f"  ?item wdt:{prop} wd:{anchor['qid']} .\n"
                    else:
                        # Anchor -> Answer(item)
                        # 注意：这种情况下，?item 是 Anchor 的属性值
                        # 但通常我们的 Filter 是加在 Answer 上的。
                        # 如果问题是 "Countries where Portuguese is spoken" (Anchor: Portuguese)
                        # 关系可能是: ?country wdt:P2936 wd:Portuguese. (这依然是 Reverse)

                        # 如果问题是 "Capital of France" (Anchor: France)
                        # 关系: wd:France wdt:P36 ?item. (这是 Forward)
                        sparql += f"  wd:{anchor['qid']} wdt:{prop} ?item .\n"

        # ==========================================
        # 2. 处理 Filters (约束条件)
        # ==========================================
        for i, flt in enumerate(filters):
            prop = flt.get('pid') or flt.get('key')
            val = flt.get('value')
            op = flt.get('op', '==')

            # [新增] 获取上游可能传入的 value_qid (例如 "children's fiction" -> "Q131539")
            val_qid = flt.get('value_qid')

            # 如果没有有效的属性 ID (Pxxx)，跳过该约束
            if not prop or not prop.startswith('P'):
                continue

            var_name = f"?v_{i}"

            # --- 分支 A: 精确实体匹配 (新增的核心逻辑) ---
            # 如果我们知道 Value 对应的 QID，直接使用对象属性匹配，不再依赖字符串
            if val_qid and val_qid.startswith('Q'):
                sparql += f"  ?item wdt:{prop} wd:{val_qid} .\n"
                continue  # 处理完这个 filter，直接进入下一次循环

            # --- 分支 B: 数值与日期范围查询 ---
            # 如果没有 QID，先声明变量
            sparql += f"  ?item wdt:{prop} {var_name} .\n"

            # 处理特殊字符，防止注入
            safe_val = str(val).replace("'", "\\'") if isinstance(val, str) else val

            # 判断是否为日期或数值比较
            is_numeric_op = op in ['>', '<', '>=', '<=']
            is_date_prop = "date" in str(prop).lower() or "time" in str(prop).lower() or "born" in str(
                flt.get('content', '')).lower()

            if is_numeric_op or is_date_prop:
                if isinstance(val, int) and val < 3000:  # 简单的年份判断
                    # 处理年份简写，如 2009 -> 2009-01-01
                    date_str = f"{val}-01-01T00:00:00Z"
                    sparql += f"  FILTER({var_name} {op} '{date_str}'^^xsd:dateTime)\n"
                elif isinstance(val, (int, float)):
                    # 普通数值
                    sparql += f"  FILTER({var_name} {op} {val})\n"
                else:
                    # 尝试处理字符串格式的日期
                    date_val = str(val) if "T" in str(val) else f"{val}T00:00:00Z"
                    sparql += f"  FILTER({var_name} {op} '{date_val}'^^xsd:dateTime)\n"

            # --- 分支 C: 字符串模糊匹配 (保底逻辑) ---
            else:
                # 只有当它是字符串时才进行 Label 匹配
                if isinstance(val, str):
                    var_label = f"?v_{i}Label"
                    # 获取 Label
                    sparql += f"  {var_name} rdfs:label {var_label} .\n"
                    # 限制为英文，提高查询速度
                    sparql += f"  FILTER(LANG({var_label}) = 'en')\n"
                    # 使用小写包含匹配
                    sparql += f'  FILTER(CONTAINS(LCASE({var_label}), LCASE("{safe_val}")))\n'

        sparql += "} LIMIT 20"
        return sparql
