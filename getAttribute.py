import json
import time
import requests
from requests.exceptions import RequestException
import random
from urllib.error import HTTPError

# 配置
INPUT_FILE = 'datasets/seed_1_to_n_questions.json'  # 上一步的输出文件
OUTPUT_FILE = 'datasets/data_with_attributes.json'
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"


# --- 1. 将属性分类 ---
# 不需要单位的属性 (字符串, 日期, 实体)
SIMPLE_PROPS = [
    "P31", "P279", "P569", "P570", "P27", "P106", "P166", "P69", "P21",
    "P577", "P136", "P57", "P161", "P175", "P407", "P17", "P30", "P1376",
    "P571", "P159", "P118", "P112", "P186", "P61"
]

# 需要单位的数值属性 (身高, 票房, 面积, 员工数等)
# 注意：P1128(员工数) 虽然通常无单位，但也可以作为 Quantity 处理
QUANTITY_PROPS = [
    "P2047", "P2142", "P2130", "P1082", "P2046", "P2044", "P2048", "P2067"
]

ALL_PROPS = SIMPLE_PROPS + QUANTITY_PROPS


def execute_sparql_with_retry(query, max_retries=5):
    """
    带有重试机制的 SPARQL 请求执行器
    :param query: SPARQL 查询语句
    :param max_retries: 最大重试次数
    :return: JSON 数据 或 None (如果全部失败)
    """
    headers = {'User-Agent': 'ResearchBot/1.0 (Academic Research)'}
    base_delay = 3  # 基础等待 3 秒

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                WIKIDATA_ENDPOINT,
                params={'format': 'json', 'query': query},
                headers=headers,
                timeout=30  # 设置超时，防止无限卡死
            )
            resp.raise_for_status()  # 如果状态码是 4xx 或 5xx，直接抛出异常
            return resp.json()

        except RequestException as e:
            # 计算等待时间：指数退避 (5s, 10s, 20s...) + 随机抖动 (防止并发冲突)
            sleep_time = (base_delay * (2 ** attempt)) + random.uniform(1, 3)

            print(f"   [Warn] Request failed (Attempt {attempt + 1}/{max_retries}). Error: {e}")

            if attempt < max_retries - 1:
                print(f"   Waiting {sleep_time:.1f}s before retry...")
                time.sleep(sleep_time)
            else:
                print("   [Error] Max retries reached. Skipping this batch.")
                return None


def fetch_attributes_for_qids(qids):
    if not qids: return {}

    values_clause = " ".join([f"wd:{qid}" for qid in qids])
    entity_attrs = {qid: {} for qid in qids}

    # ==========================================
    # 1. 查询普通属性 (Simple Props) - 使用重试机制
    # ==========================================
    if SIMPLE_PROPS:
        query_simple = f"""
        SELECT ?item ?prop ?valueLabel WHERE {{
          VALUES ?item {{ {values_clause} }}
          VALUES ?prop {{ {" ".join(["wdt:" + p for p in SIMPLE_PROPS])} }}
          ?item ?prop ?value .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        # --- 核心修改：调用重试函数 ---
        data = execute_sparql_with_retry(query_simple)

        # 只有在 data 成功拿到时才解析
        if data and "results" in data and "bindings" in data["results"]:
            for item in data["results"]["bindings"]:
                try:
                    qid = item['item']['value'].split('/')[-1]
                    prop_url = item['prop']['value']
                    prop = prop_url.split('/')[-1]
                    val_label = item['valueLabel']['value']

                    if prop not in entity_attrs[qid]:
                        entity_attrs[qid][prop] = []
                    if val_label not in entity_attrs[qid][prop]:
                        entity_attrs[qid][prop].append(val_label)
                except Exception as parse_err:
                    print(f"Error parsing item: {parse_err}")

    # ==========================================
    # 2. 查询量化属性 (Quantity Props) - 使用重试机制
    # ==========================================
    if QUANTITY_PROPS:
        query_quantity = f"""
        SELECT ?item ?propStr ?amount ?unitLabel WHERE {{
          VALUES ?item {{ {values_clause} }}
          VALUES ?propStr {{ {" ".join([f'"{p}"' for p in QUANTITY_PROPS])} }} 
          BIND(IRI(CONCAT("http://www.wikidata.org/prop/", ?propStr)) AS ?p)
          BIND(IRI(CONCAT("http://www.wikidata.org/prop/statement/value/", ?propStr)) AS ?psv)
          ?item ?p ?statement .
          ?statement ?psv ?valNode .
          ?valNode wikibase:quantityAmount ?amount .
          OPTIONAL {{ 
            ?valNode wikibase:quantityUnit ?unit . 
            ?unit rdfs:label ?unitLabel .
            FILTER(LANG(?unitLabel) = "en")
          }}
        }}
        """

        # --- 核心修改：调用重试函数 ---
        data = execute_sparql_with_retry(query_quantity)

        if data and "results" in data and "bindings" in data["results"]:
            for item in data["results"]["bindings"]:
                try:
                    qid = item['item']['value'].split('/')[-1]
                    prop = item['propStr']['value']
                    amount = item['amount']['value']
                    unit = item.get('unitLabel', {}).get('value', '1')

                    if prop not in entity_attrs[qid]:
                        entity_attrs[qid][prop] = []

                    val_obj = {"amount": amount, "unit": unit}

                    if val_obj not in entity_attrs[qid][prop]:
                        entity_attrs[qid][prop].append(val_obj)
                except Exception as parse_err:
                    pass

    return entity_attrs


def process_attributes():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"找不到文件 {INPUT_FILE}，请先等待第一步脚本跑完。")
        return

    print(f"加载了 {len(data)} 个种子问题。开始提取属性...")

    enhanced_data = []

    for idx, entry in enumerate(data):
        print(f"[{idx + 1}/{len(data)}] Processing: {entry['question']} (Answers: {entry['answer_count']})")

        qids = entry['answers']

        # 核心步骤：去 Wikidata 查这些答案的属性
        # 注意：如果答案太多(>30)，可能导致 SPARQL URL 过长，可以切片查询，这里简化处理取前30个
        batch_qids = qids[:30]
        attrs_map = fetch_attributes_for_qids(batch_qids)

        # 将属性整合回 entry
        # 结构变成: entry['answers_detail'] = { "Q123": {"P577": ["2010-01-01"], ...}, ... }
        entry['answers_attributes'] = attrs_map

        # 简单的统计，看看有没有查到有用的东西
        valid_attr_count = sum(1 for props in attrs_map.values() if props)
        print(f"   -> Fetched attributes for {valid_attr_count}/{len(batch_qids)} entities.")

        enhanced_data.append(entry)
        time.sleep(1)  # 礼貌延时

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    print(f"完成！数据已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    process_attributes()