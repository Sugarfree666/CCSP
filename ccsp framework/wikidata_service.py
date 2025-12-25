import sys
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError


class WikidataService:
    def __init__(self, user_agent="CCSP-Bot/1.0 (Research Project)"):
        """
        初始化 Wikidata SPARQL 服务
        """
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.user_agent = user_agent

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


