import json

# 读取原始文件
with open('datasets/data_with_attributes.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # data 是一个列表

# 取前n个元素
n = 20  # 例如：取前3个元素
first_n_elements = data[:n]

# 保存到新文件
with open('datasets/datasets.json', 'w', encoding='utf-8') as f:
    json.dump(first_n_elements, f, indent=2, ensure_ascii=False)

print(f"已保存前 {len(first_n_elements)} 个元素到 datasets.json")

#
# import requests
# import json
#
#
# def query_wikidata_by_qid(qid, lang='en'):
#     """
#     通过 Wikidata QID 查询实体信息
#
#     Args:
#         qid: Wikidata 实体ID，如 'Q1754478'
#         lang: 语言代码，默认为 'en'（英文）
#     """
#     # Wikidata API URL
#     url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
#
#     try:
#         # 发送请求
#         response = requests.get(url, headers={
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         })
#         response.raise_for_status()  # 检查请求是否成功
#
#         # 解析 JSON 数据
#         data = response.json()
#
#         if 'entities' in data and qid in data['entities']:
#             entity = data['entities'][qid]
#
#             # 提取基本信息
#             result = {
#                 'id': qid,
#                 'labels': {},
#                 'descriptions': {},
#                 'aliases': {},
#                 'claims': {}
#             }
#
#             # 获取标签
#             if 'labels' in entity:
#                 for lang_code, label_info in entity['labels'].items():
#                     result['labels'][lang_code] = label_info.get('value', '')
#
#             # 获取描述
#             if 'descriptions' in entity:
#                 for lang_code, desc_info in entity['descriptions'].items():
#                     result['descriptions'][lang_code] = desc_info.get('value', '')
#
#             # 获取别名
#             if 'aliases' in entity:
#                 for lang_code, aliases_list in entity['aliases'].items():
#                     result['aliases'][lang_code] = [alias.get('value', '') for alias in aliases_list]
#
#             # 获取属性声明
#             if 'claims' in entity:
#                 for prop_id, claims_list in entity['claims'].items():
#                     prop_values = []
#                     for claim in claims_list:
#                         mainsnak = claim.get('mainsnak', {})
#                         if mainsnak.get('datatype') == 'wikibase-item' and 'datavalue' in mainsnak:
#                             # 如果是 Wikidata 实体引用
#                             value = mainsnak['datavalue']['value']
#                             prop_values.append({
#                                 'id': value.get('id'),
#                                 'numeric-id': value.get('numeric-id')
#                             })
#                         elif 'datavalue' in mainsnak:
#                             # 其他类型值
#                             prop_values.append(mainsnak['datavalue'].get('value'))
#
#                     result['claims'][prop_id] = prop_values
#
#             return result
#         else:
#             return {"error": f"未找到实体 {qid}"}
#
#     except requests.exceptions.RequestException as e:
#         return {"error": f"网络请求失败: {e}"}
#     except json.JSONDecodeError as e:
#         return {"error": f"JSON 解析失败: {e}"}
#     except Exception as e:
#         return {"error": f"发生未知错误: {e}"}
#
#
# def print_wikidata_info(qid):
#     """打印 Wikidata 实体信息"""
#     info = query_wikidata_by_qid(qid)
#
#     if 'error' in info:
#         print(f"错误: {info['error']}")
#         return
#
#     print(f"=== Wikidata 实体: {qid} ===")
#
#     # 打印标签
#     if 'en' in info['labels']:
#         print(f"英文名称: {info['labels']['en']}")
#     if 'zh' in info['labels']:
#         print(f"中文名称: {info['labels']['zh']}")
#
#     # 打印描述
#     if 'en' in info['descriptions']:
#         print(f"英文描述: {info['descriptions']['en']}")
#     if 'zh' in info['descriptions']:
#         print(f"中文描述: {info['descriptions']['zh']}")
#
#     # 打印一些常用属性
#     print("\n=== 主要属性 ===")
#
#     # 定义一些常用属性的映射
#     property_names = {
#         'P31': '实例属于 (instance of)',
#         'P279': '子类 (subclass of)',
#         'P361': '部分属于 (part of)',
#         'P17': '国家 (country)',
#         'P131': '位于行政区 (located in)',
#         'P571': '成立时间 (inception)',
#         'P577': '出版日期 (publication date)',
#         'P50': '作者 (author)',
#         'P170': '创作者 (creator)',
#         'P106': '职业 (occupation)',
#         'P21': '性别 (gender)',
#         'P569': '出生日期 (date of birth)',
#         'P570': '死亡日期 (date of death)',
#         'P27': '国籍 (country of citizenship)'
#     }
#
#     for prop_id, prop_name in property_names.items():
#         if prop_id in info['claims']:
#             values = info['claims'][prop_id]
#             print(f"{prop_name} ({prop_id}): {values}")
#
#     # 如果还想查看更多属性
#     other_props = [pid for pid in info['claims'] if pid not in property_names]
#     if other_props:
#         print(f"\n=== 其他属性 ({len(other_props)}个) ===")
#         print(f"属性ID列表: {', '.join(other_props[:10])}" +
#               ("..." if len(other_props) > 10 else ""))
#
#
# # 查询 Q1754478
# print_wikidata_info('Q234576')


# from SPARQLWrapper import SPARQLWrapper, JSON
#
# # 1. 定义 Endpoint 和 User-Agent (Wikidata 要求必须设置 User-Agent)
# endpoint_url = "https://query.wikidata.org/sparql"
# user_agent = "Python-SPARQL-Client/1.0"
#
# # 2. 你的原始查询语句
# query = """
# PREFIX wd: <http://www.wikidata.org/entity/>
# PREFIX wdt: <http://www.wikidata.org/prop/direct/>
#
# SELECT DISTINCT ?x WHERE {
#   wd:Q794775 wdt:P1346 ?x.
# }
# """
#
#
# def get_answer_qids():
#     # 初始化包装器
#     sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#
#     try:
#         # 执行查询
#         results = sparql.query().convert()
#
#         qids = []
#         # 遍历结果
#         for result in results["results"]["bindings"]:
#             # 获取完整的 URI，例如: http://www.wikidata.org/entity/Q234576
#             entity_uri = result["x"]["value"]
#
#             # 提取 QID (即 URL 的最后一部分)
#             qid = entity_uri.split("/")[-1]
#             qids.append(qid)
#
#         return qids
#
#     except Exception as e:
#         print(f"查询出错: {e}")
#         return []
#
#
# # 运行并打印结果
# if __name__ == "__main__":
#     answer_qids = get_answer_qids()
#     print(f"找到 {len(answer_qids)} 个答案:")
#     print(answer_qids)