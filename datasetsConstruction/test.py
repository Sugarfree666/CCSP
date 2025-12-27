import json
import requests
import time

# ================= 配置区域 =================
# 如果您开启了 VPN，请在此处填写代理地址。
# 常见的本地代理端口是 7890 (Clash) 或 10809 (v2ray)，请根据您的软件设置修改。
# 如果不确定，可以在 VPN 软件的“设置”中查看 "HTTP Proxy" 端口。
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# Wikidata 强制要求 User-Agent，否则可能会拦截请求
HEADERS = {
    "User-Agent": "MyDatasetLabelFetcher/1.0 (contact: your_email@example.com)"
}


# ===========================================

def fetch_wikidata_labels(qids):
    base_url = "https://www.wikidata.org/w/api.php"
    qid_to_label = {}
    unique_qids = list(set(qids))
    batch_size = 50
    total_batches = (len(unique_qids) + batch_size - 1) // batch_size

    print(f"正在获取 {len(unique_qids)} 个实体的标签，共 {total_batches} 批...")

    for i in range(0, len(unique_qids), batch_size):
        batch = unique_qids[i:i + batch_size]
        ids_str = "|".join(batch)

        params = {
            "action": "wbgetentities",
            "ids": ids_str,
            "format": "json",
            "props": "labels",
            "languages": "en"
        }

        try:
            # 加入 proxies 和 headers 参数
            response = requests.get(base_url, params=params, headers=HEADERS, proxies=PROXIES, timeout=15)

            # 尝试解析 JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                print(f"\n[错误] 第 {i // batch_size + 1} 批返回的不是 JSON 数据。")
                print(f"返回内容片段: {response.text[:200]}...")  # 打印前200个字符用于调试
                continue

            if "entities" in data:
                for qid, entity in data["entities"].items():
                    if "labels" in entity and "en" in entity["labels"]:
                        qid_to_label[qid] = entity["labels"]["en"]["value"]
                    else:
                        qid_to_label[qid] = qid

            print(f"第 {i // batch_size + 1}/{total_batches} 批获取成功")

        except requests.exceptions.ProxyError:
            print(f"\n[错误] 代理连接失败。请检查 PROXIES 设置中的端口是否正确。")
            break
        except requests.exceptions.ConnectionError:
            print(f"\n[错误] 网络连接失败。请确认 VPN 已开启且可以访问 Wikidata。")
            break
        except Exception as e:
            print(f"第 {i // batch_size + 1} 批请求发生未知异常: {e}")

        time.sleep(1)  # 增加延迟以保持稳定

    return qid_to_label


def add_answer_labels(input_file, output_file):
    try:
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"找不到文件: {input_file}，请确认路径是否正确。")
        return

    all_qids = []
    for entry in dataset:
        if "new_ground_truth" in entry:
            for item in entry["new_ground_truth"]:
                if isinstance(item, str) and item.startswith("Q"):
                    all_qids.append(item)

    if not all_qids:
        print("未找到任何以 Q 开头的 ID。")
        return

    label_map = fetch_wikidata_labels(all_qids)

    print("正在添加 answer_label 字段...")
    for entry in dataset:
        answer_labels = []
        if "new_ground_truth" in entry:
            for qid in entry["new_ground_truth"]:
                label = label_map.get(qid, qid)
                answer_labels.append(label)
        entry["answer_label"] = answer_labels

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"处理完成！新文件已保存为: {output_file}")


if __name__ == "__main__":
    # 请确保路径正确，Windows路径建议使用原始字符串 r"..." 或双反斜杠 \\
    # 输入文件路径 (您上传的文件名)
    input_filename = 'D:\GitHub\CCSP\datasets\complex_constraint_dataset_rewrite_queries.json'
    # 输出文件路径
    output_filename = 'complex_constraint_dataset_with_labels.json'
    add_answer_labels(input_filename, output_filename)