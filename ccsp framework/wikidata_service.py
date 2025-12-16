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

        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["https_proxy"] = "http://127.0.0.1:7890"

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


    def get_candidate_relations(self, qid):
        """
        [通用方法] 获取与 Anchor 相连的所有属性（Top 50），包括正向和反向。
        优化版：增加了 PREFIX，修复了 Label 获取逻辑。
        """
        # 定义标准前缀
        prefixes = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            """

        # 1. 反向关系 (Reverse): 别人连我 (e.g. ?book wdt:P50 wd:Author)
        query_reverse = f"""
            {prefixes}
            SELECT DISTINCT ?p ?propLabel WHERE {{
              ?s ?p wd:{qid} .
              ?prop wikibase:directClaim ?p .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }} LIMIT 30
            """

        # 2. 正向关系 (Forward): 我连别人 (e.g. wd:Country wdt:P36 ?capital)
        query_forward = f"""
            {prefixes}
            SELECT DISTINCT ?p ?propLabel WHERE {{
              wd:{qid} ?p ?o .
              ?prop wikibase:directClaim ?p .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }} LIMIT 30
            """

        relations = []

        try:
            # 执行查询
            res_rev = self.execute_query(query_reverse)
            for r in res_rev:
                # 注意：Service 返回的 label 变量名通常是 ?propLabel
                label = r.get('propLabel', {}).get('value', 'Unknown')
                relations.append({
                    "pid": r['p']['value'].split('/')[-1],
                    "label": label,
                    "direction": "reverse"
                })

            res_fwd = self.execute_query(query_forward)
            for r in res_fwd:
                label = r.get('propLabel', {}).get('value', 'Unknown')
                relations.append({
                    "pid": r['p']['value'].split('/')[-1],
                    "label": label,
                    "direction": "forward"
                })

        except Exception as e:
            print(f"Error fetching candidate relations: {e}")

        return relations


    def construct_sparql_from_got(self, anchors, filters):
        """
        根据 GoT 的节点构建 SPARQL。
        修改点：
        1. 增加 ?itemDescription
        2. 变量名标准化为 ?v_0, ?v_1... 以便 main.py 回溯语义
        """
        # --- [修改点 1] 增加 itemDescription ---
        select_vars = ["?item", "?itemLabel", "?itemDescription"]
        where_clauses = []

        prefixes = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        """

        # 确保 Service 能拉取 description
        where_clauses.append('SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }')

        # 1. 处理 Anchors (保持不变)
        for anchor in anchors:
            if 'qid' in anchor:
                prop = anchor.get('pid') or anchor.get('key')
                if prop and prop.startswith('P'):
                    direction = anchor.get('direction', 'reverse')
                    if direction == 'reverse':
                        where_clauses.append(f"?item wdt:{prop} wd:{anchor['qid']} .")
                    else:
                        where_clauses.append(f"wd:{anchor['qid']} wdt:{prop} ?item .")

        # 2. 处理 Filters (核心修改)
        for i, flt in enumerate(filters):
            prop = flt.get('pid') or flt.get('key')
            val = flt.get('value')
            op = flt.get('op', '==')
            val_qid = flt.get('value_qid')

            if not prop or not prop.startswith('P'):
                continue

            # --- [修改点 2] 变量名强制使用索引格式 v_0, v_1 ---
            # 这样我们在解析结果时，就知道 v_0 对应 filters[0]
            var_name = f"?v_{i}"
            select_vars.append(var_name)

            # 分支 A: 精确 QID (例如 Genre == Horror)
            if val_qid and val_qid.startswith('Q'):
                where_clauses.append(f"?item wdt:{prop} wd:{val_qid} .")
                # 如果是实体匹配，我们通常不需要把 QID 选出来，但为了证据显示，我们可以选 Label
                # 这里我们稍微变通一下：把具体的实体值赋给 var_name 没意义（因为已经在WHERE里定死了），
                # 但我们可以提取这个属性的实际值用于展示（其实就是 val_qid）
                # 为了保持统一，这里不做额外操作，只依赖 filter 逻辑
                continue

            # 分支 B: 数值与字符串
            where_clauses.append(f"?item wdt:{prop} {var_name} .")

            # --- 日期与数值处理 (保持你的原有逻辑，只是把变量名换成了 var_name) ---
            is_numeric_op = op in ['>', '<', '>=', '<=']
            is_date_prop = any(k in str(prop).lower() or k in str(flt).lower() for k in
                               ["date", "time", "born", "died", "publication"])

            if is_numeric_op or is_date_prop:
                date_str = None
                if isinstance(val, (int, float)):
                    if val < 3000:
                        date_str = f"{int(val)}-01-01T00:00:00Z"
                    else:
                        where_clauses.append(f"FILTER({var_name} {op} {val})")
                elif isinstance(val, str):
                    val = val.strip()
                    if val.isdigit() and len(val) == 4:
                        date_str = f"{val}-01-01T00:00:00Z"
                    elif "T" in val:
                        date_str = val
                    else:
                        date_str = f"{val}T00:00:00Z"

                if date_str:
                    where_clauses.append(f"FILTER({var_name} {op} '{date_str}'^^xsd:dateTime)")

            # 分支 C: 字符串模糊匹配
            elif isinstance(val, str):
                safe_val = str(val).replace("'", "\\'")
                var_label = f"{var_name}_Label"
                where_clauses.append(f"{var_name} rdfs:label {var_label} .")
                where_clauses.append(f"FILTER(LANG({var_label}) = 'en')")
                where_clauses.append(f'FILTER(CONTAINS(LCASE({var_label}), LCASE("{safe_val}")))')
                select_vars.append(var_label)

        # 3. 组装
        sparql = f"{prefixes}\nSELECT DISTINCT {' '.join(select_vars)} WHERE {{\n"
        sparql += "\n".join(f"  {clause}" for clause in where_clauses)
        sparql += "\n} LIMIT 20"

        return sparql
