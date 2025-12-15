import json
import time
import requests
from urllib.error import HTTPError

# 1. 配置参数
INPUT_FILE = r'C:\Users\sugarfree\Downloads\wikidata-emnlp23-master\train.json'  # 你上传的文件名
OUTPUT_FILE = '../datasets/seed_1_to_n_questions.json'
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
MIN_ANSWERS = 3  # 只有答案数量 >= 3 的问题才会被保留作为“复杂约束”的种子
MAX_ANSWERS = 100  # 排除答案太多的问题（比如“所有人类”），避免噪音


def execute_sparql(query):
    """
    向 Wikidata 发送 SPARQL 请求并返回结果列表
    """
    try:
        # Wikidata 要求 User-Agent，否则会报 403 错误
        headers = {
            'User-Agent': 'MyResearchBot/1.0 (contact@example.com)'
        }
        params = {'format': 'json', 'query': query}

        response = requests.get(WIKIDATA_ENDPOINT, params=params, headers=headers, timeout=10)

        if response.status_code == 429:  # 触发了速率限制
            print("Too many requests, sleeping...")
            time.sleep(5)
            return execute_sparql(query)  # 重试

        data = response.json()
        results = []

        # 解析返回的 JSON，提取变量值（通常是 URL，我们需要提取 QID）
        if "results" in data and "bindings" in data["results"]:
            for binding in data["results"]["bindings"]:
                # 假设查询结果变量通常是 ?x 或 ?y，我们取第一个找到的变量
                for var in binding:
                    url = binding[var]['value']
                    # 提取 QID (例如 http://www.wikidata.org/entity/Q123 -> Q123)
                    if "entity/Q" in url:
                        qid = url.split("/")[-1]
                        results.append(qid)
        return results

    except Exception as e:
        print(f"Error executing query: {e}")
        return []


# 2. 主流程
def process_dataset():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    seed_dataset = []

    print(f"开始处理 {len(raw_data)} 条数据...")

    # 限制处理数量用于测试，正式跑时去掉 [:20]
    for idx, item in enumerate(raw_data):
        question = item['utterance']
        sparql = item['sparql']

        print(f"[{idx + 1}] Querying: {question}")

        # 执行查询
        answers = execute_sparql(sparql)

        num_answers = len(answers)
        print(f"   -> Found {num_answers} answers.")

        # 筛选逻辑：只保留 1-to-N 问题
        if MIN_ANSWERS <= num_answers <= MAX_ANSWERS:
            new_entry = {
                "original_id": item['id'],
                "question": question,
                "original_sparql": sparql,
                "answers": answers,  # 保存答案 QID 列表，供下一步查属性用
                "answer_count": num_answers
            }
            seed_dataset.append(new_entry)
            print(f"   -> [KEEP] Added to seed set.")
        else:
            print(f"   -> [SKIP] Answer count not in range.")

        # 礼貌性延时，避免被 Wikidata 封 IP
        time.sleep(0.5)

        # 3. 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(seed_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成！共筛选出 {len(seed_dataset)} 个适合构造约束的种子问题。")
    print(f"结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dataset()