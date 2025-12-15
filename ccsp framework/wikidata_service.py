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
        # 1. 简单缓存检查 (你也可以把这个存进 cache_file 里，这里为了演示简单处理)
        # 注意：为了更稳健，你可以像 execute_query 一样实现一个基于文件的缓存

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label,
            "limit": 1
        }
        try:
            # 这里也可以加重试机制
            response = requests.get(url, params=params)
            data = response.json()
            if data.get("search"):
                qid = data["search"][0]["id"]
                print(f"    [Link Success] '{label}' -> {qid}")
                return qid
        except Exception as e:
            print(f"    [Link Error] Search failed for {label}: {e}")
        return None

    def construct_sparql_from_got(self, anchors, filters):
        """
        将聚合后的 Anchor 和 Filter 转换为 SPARQL 查询
        """
        sparql = """
        SELECT DISTINCT ?item ?itemLabel WHERE {
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        """

        # 1. 处理 Anchors (实体约束)
        for anchor in anchors:
            # anchor 结构预期: {'qid': 'Q143716', 'key': 'P161', ...}
            # 如果 main.py 里已经解析了 qid，直接用
            if 'qid' in anchor:
                # 这里的 key 应该是 PID (如 P161 starring)
                prop = anchor.get('key')
                # 如果 key 不是 PID (比如是 'starring')，需要 LLM 之前做过映射，或者在这里做容错
                # 假设已经是 PID
                if prop.startswith('P'):
                    sparql += f"  ?item wdt:{prop} wd:{anchor['qid']} .\n"
            else:
                # 备用：如果没有 qid，尝试现场搜 (不推荐，效率低)
                pass

        # 2. 处理 Filters (数值/属性约束)
        # filters 结构预期: [{'pid': 'P2047', 'op': '<', 'value': 122.5}, ...]
        for i, flt in enumerate(filters):
            # 使用 PID (Layer 2 对齐后的结果)
            prop = flt.get('pid') or flt.get('key')
            op = flt.get('op', '=')
            val = flt.get('value')

            if not prop or not prop.startswith('P'):
                continue

            var_name = f"?v_{i}"
            sparql += f"  ?item wdt:{prop} {var_name} .\n"

            # 数值/时间过滤
            # 简单的类型判断逻辑
            if isinstance(val, str) and ("-" in val or "date" in prop.lower()):
                sparql += f"  FILTER({var_name} {op} '{val}T00:00:00Z'^^xsd:dateTime)\n"
            else:
                sparql += f"  FILTER({var_name} {op} {val})\n"

        sparql += "} LIMIT 20"
        return sparql

    # search_entity_id 也可以加上类似的缓存逻辑
    # ...
