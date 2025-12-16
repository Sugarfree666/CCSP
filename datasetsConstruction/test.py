# import json
#
# # 读取原始文件
# with open('../datasets/data_with_attributes.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)  # data 是一个列表
#
# # 取前n个元素
# n = 20  # 例如：取前3个元素
# first_n_elements = data[:n]
#
# # 保存到新文件
# with open('../datasets/datasets.json', 'w', encoding='utf-8') as f:
#     json.dump(first_n_elements, f, indent=2, ensure_ascii=False)
#
# print(f"已保存前 {len(first_n_elements)} 个元素到 datasets.json")

import os
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


# ================= 配置区 =================
# 如果你在国内无法直连 Wikidata，请取消下面两行的注释并修改端口
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# ==========================================

def get_name_via_sparql(qid):
    """
    方法 1: 使用 SPARQL 查询 (最准确，和你主程序逻辑一致)
    """
    endpoint = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    # 必须设置 User-Agent
    sparql.addCustomHttpHeader("User-Agent", "EntityVerifier/1.0 (test@gmail.com)")

    query = f"""
    SELECT ?label WHERE {{
      wd:{qid} rdfs:label ?label .
      FILTER(LANG(?label) = "en")
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]

        if bindings:
            return bindings[0]["label"]["value"]
        else:
            return "Label not found (Entity might not have an English label)"

    except Exception as e:
        return f"Error: {e}"


def get_name_via_api(qid):
    """
    方法 2: 使用 Wikidata API (轻量级，更快)
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels",
        "languages": "en",
        "format": "json"
    }
    headers = {
        "User-Agent": "EntityVerifier/1.0 (test@gmail.com)"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        if "entities" in data and qid in data["entities"]:
            entity = data["entities"][qid]
            if "labels" in entity and "en" in entity["labels"]:
                return entity["labels"]["en"]["value"]
            else:
                return "No English label found"
        return "Entity ID not found"

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 40)
        qid_input = input("请输入实体ID (例如 Q7289900) 或输入 'q' 退出: ").strip()

        if qid_input.lower() == 'q':
            break

        if not qid_input.startswith("Q") and not qid_input.startswith("P"):
            print("格式错误：ID 必须以 Q 或 P 开头")
            continue

        print(f"\n正在查询 {qid_input} ...")

        # 使用 API 方法查询 (速度快)
        name = get_name_via_api(qid_input)
        print(f"👉 实体名称: {name}")

        # 也可以取消注释下面这行来测试 SPARQL 方法
        # print(f"SPARQL 结果: {get_name_via_sparql(qid_input)}")