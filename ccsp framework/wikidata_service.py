import sys
import time
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError


class WikidataService:
    def __init__(self, user_agent="CCSP-Bot/1.0 (Research Project)"):
        """
        初始化 Wikidata SPARQL 服务
        """
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.user_agent = user_agent

    def search_entity(self, label: str) -> str:
        return self._search_wikidata(label, "item")

    def search_property(self, label: str) -> str:
        return self._search_wikidata(label, "property")

    def _search_wikidata(self, label: str, type_filter: str) -> str:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "type": type_filter,
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": self.user_agent}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)
            data = response.json()
            if data.get("search"):
                return data["search"][0]["id"]
        except Exception as e:
            print(f"[Wikidata Search] Error: {e}")
        return None

    def probe_query_count(self, query: str, timeout_sec=2.0) -> int:
        """
        [NEW] 基于 LIMIT 的探测
        返回查到的行数。如果超时或出错，返回 -1。
        """
        try:
            params = {"query": query, "format": "json"}
            headers = {"User-Agent": self.user_agent}

            # 执行请求
            response = requests.get(
                self.endpoint_url,
                params=params,
                headers=headers,
                timeout=timeout_sec
            )

            if response.status_code == 200:
                data = response.json()
                bindings = data["results"]["bindings"]
                return len(bindings)  # 直接返回 List 长度
            else:
                return -1  # HTTP Error

        except requests.exceptions.Timeout:
            # 超时意味着即便 LIMIT 1000 也没跑完（或者网络太差）
            # 这种情况下绝对不能做 Anchor
            return -1
        except Exception as e:
            # print(f"[Probe Error] {e}")
            return -1

    def execute_sparql(self, query: str, retries=3):
        """
        执行 SPARQL 查询并返回结果 (JSON 格式)。
        包含自动重试机制以应对 Wikidata 的网络波动。
        """
        sparql = SPARQLWrapper(self.endpoint_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        # Wikidata 强制要求设置 User-Agent，否则会返回 403 Forbidden
        sparql.addCustomHttpHeader("User-Agent", self.user_agent)

        sparql.setMethod("POST")
        sparql.setRequestMethod("postdirectly")

        for attempt in range(retries):
            try:
                results = sparql.query().convert()
                return results["results"]["bindings"]
            except HTTPError as e:
                if e.code == 429:  # Too Many Requests
                    wait_time = (attempt + 1) * 2
                    print(f"[Wikidata] Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Wikidata] HTTP Error: {e}")
                    raise e
            except Exception as e:
                print(f"[Wikidata] Error: {e}")
                # 如果是最后一次尝试，则抛出异常
                if attempt == retries - 1:
                    raise e
                time.sleep(1)

        return []

    def search_property(self, label: str) -> str:
        """
        [Relation Linker]
        严格的科研级实现：调用 Wikidata API 将自然语言 Label 映射为 Property ID (Pxx)。
        """
        if not label:
            return None

        # 1. 尝试完全匹配搜索
        pid = self._search_wikidata_api(label, "property")
        if pid:
            return pid

        # 2. 如果没搜到，尝试去除停用词后再次搜索 (简单的 Fallback)
        # 例如 "date of birth" -> "birth date" 有时能命中别名
        # 这里为了严谨，我们暂时只做直接搜索。在论文中可以提到这里可以使用更高级的 Dense Retrieval (如 BERT-based linker)。
        return None

    def _search_wikidata_api(self, query: str, type_filter: str) -> str:
        """底层 API 调用"""
        try:
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "type": type_filter,
                "format": "json",
                "limit": 1  # 科研 Baseline 通常取 Top-1，进阶版取 Top-5 配合 Re-ranking
            }
            headers = {"User-Agent": self.user_agent}
            resp = requests.get("https://www.wikidata.org/w/api.php", params=params, headers=headers, timeout=5)
            data = resp.json()

            if data.get("search"):
                # 返回第一个匹配项的 ID
                return data["search"][0]["id"]
        except Exception as e:
            print(f"[Linker Error] Search failed for '{query}': {e}")
        return None
    def print_results(self, bindings):
        """
        格式化打印 SPARQL 查询结果
        """
        if not bindings:
            print("No results found.")
            return

        print(f"Found {len(bindings)} results:")

        # 获取所有变量名 (表头)
        if len(bindings) > 0:
            vars = bindings[0].keys()

            # 简单打印
            for i, result in enumerate(bindings):
                print(f"[{i + 1}]")
                for var in vars:
                    value = result[var]['value']
                    # 如果是实体 URL，只显示 QID 简化显示
                    if "entity/Q" in value:
                        label = value.split("/")[-1]
                        print(f"  {var}: {label} ({value})")
                    else:
                        print(f"  {var}: {value}")
                print("-" * 20)
        else:
            print("No bindings in results.")

    def get_cardinality(self, query: str, timeout_sec=0.5) -> int:
        """
        [NEW] 探测专用：执行 COUNT 查询。
        关键点：timeout_sec 默认很短 (0.5s)。
        数据库原则：如果是高选择率索引(High Selectivity)，COUNT 会瞬间返回。
        如果卡住了，说明它需要全表扫描，直接视为 Bad Path。
        """
        try:
            params = {"query": query, "format": "json"}
            headers = {"User-Agent": self.user_agent}

            response = requests.get(
                self.endpoint_url,
                params=params,
                headers=headers,
                timeout=timeout_sec
            )

            if response.status_code == 200:
                data = response.json()
                return int(data["results"]["bindings"][0]["c"]["value"])
            else:
                return 999_999_999  # HTTP 错误视为高代价

        except requests.exceptions.Timeout:
            # print(f"[Probe] Timeout ({timeout_sec}s). Too expensive.")
            return 999_999_999  # 超时视为高代价
        except Exception as e:
            # print(f"[Probe] Error: {e}")
            return 999_999_999
