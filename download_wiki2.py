import duckdb
import json
import time
import os
import requests
import math
from huggingface_hub import list_repo_files, hf_hub_download
from typing import List, Dict

# ================= 配置区域 =================
# 1. 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"  # 建议正式运行时通过环境变量获取

REPO_ID = "CleverThis/wikidata-truthy"
# 注意：全量模式下这个变量虽然定义了但实际上被下面的逻辑忽略了，这是符合预期的
SAMPLE_FILES_COUNT = 1600

# [修复] 使用 raw string (r) 避免 Windows 路径转义错误
OUTPUT_FILE = r"/ccsp framework/property_metadata.json"


# ===========================================

def fetch_property_details(pids: List[str]) -> Dict[str, Dict]:
    """
    批量调用 Wikidata API 获取属性的 Label 和 Description。
    增加重试机制，提高全量跑的稳定性。
    """
    print(f"   [API] 正在获取 {len(pids)} 个属性的语义描述...")

    url = "https://www.wikidata.org/w/api.php"
    results = {}

    headers = {"User-Agent": "CCSP-Research/1.0 (PropertyStatsBuilder)"}

    batch_size = 50
    for i in range(0, len(pids), batch_size):
        batch = pids[i: i + batch_size]
        ids_str = "|".join(batch)

        params = {
            "action": "wbgetentities",
            "ids": ids_str,
            "languages": "en",
            "props": "labels|descriptions",
            "format": "json"
        }

        # [优化] 增加简单的重试机制
        retries = 3
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                if resp.status_code == 429:  # 限流
                    time.sleep(5)
                    continue

                data = resp.json()

                if "entities" in data:
                    for pid, content in data["entities"].items():
                        label = content.get("labels", {}).get("en", {}).get("value", "Unknown")
                        desc = content.get("descriptions", {}).get("en", {}).get("value", "No description available.")
                        results[pid] = {"label": label, "description": desc}

                break  # 成功则跳出重试循环

            except Exception as e:
                if attempt == retries - 1:
                    print(f"   [Error] API请求失败 (Batch {i}) 且重试耗尽: {e}")
                else:
                    time.sleep(2)  # 等待后重试

        # 礼貌性延时
        time.sleep(0.5)

        # 简单进度显示
        if (i + batch_size) % 1000 == 0:
            print(f"   [API 进度] 已处理 {i + batch_size}/{len(pids)}")

    return results


def run_pipeline():
    # --- 第一步：下载数据 ---
    print(f"1. [环境] 检查 HuggingFace 缓存 (全量模式: {REPO_ID})...")

    try:
        all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        target_files = [f for f in all_files if f.endswith(".parquet")]
        target_files.sort()

        print(f"   [全量准备] 共发现 {len(target_files)} 个 Parquet 文件，准备加载...")

        local_paths = []
        for idx, filename in enumerate(target_files):
            path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
            local_paths.append(path)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(target_files):
                print(f"   进度: {idx + 1}/{len(target_files)} 文件已就绪")

    except Exception as e:
        print(f"   [Fatal] 无法获取文件列表或下载失败: {e}")
        return

    # --- 第二步：DuckDB 统计 ---
    print("2. [计算] DuckDB 全量聚合 (这可能需要几分钟到几十分钟)...")
    start_time = time.time()

    # [优化] 使用磁盘支持的数据库，避免内存溢出 (OOM)
    # 处理完后可以手动删除 temp_stats.duckdb
    con = duckdb.connect("temp_stats.duckdb")

    # 增加内存限制 (根据你的机器调整，例如 '16GB')
    # con.execute("PRAGMA memory_limit='16GB'")
    # con.execute("PRAGMA threads=8") # 利用多核

    # 注意：local_paths 如果文件太多，SQL字符串可能过长。
    # DuckDB支持直接传列表，但在SQL中需要格式化好。
    # 这里保持你的逻辑，因为通常几千个文件的路径字符串还是在限制内的。

    query = fr"""
    SELECT 
        regexp_extract(predicate, 'P\d+', 0) as pid,
        COUNT(*) as cnt,
        APPROX_COUNT_DISTINCT("object") as unique_cnt
    FROM read_parquet({local_paths})
    WHERE regexp_matches(predicate, 'P\d+')
    GROUP BY pid
    HAVING cnt > 10
    ORDER BY cnt DESC
    """

    print("   [DuckDB] 开始执行 SQL (Aggregation)...")
    df = con.execute(query).df()

    total_sampled_rows = df['cnt'].sum()
    print(f"   统计完成! 耗时: {time.time() - start_time:.2f}s")
    print(f"   分析三元组总数: {total_sampled_rows:,}")
    print(f"   发现有效属性: {len(df)} 个")

    # 关闭连接，释放锁
    con.close()

    # 可选：删除临时数据库文件
    if os.path.exists("temp_stats.duckdb"):
        try:
            os.remove("temp_stats.duckdb")
            print("   [System] 临时数据库已清理")
        except:
            pass

    # --- 第三步：获取语义描述 (API) ---
    all_pids = df['pid'].tolist()
    semantic_data = fetch_property_details(all_pids)

    # --- 第四步：构建最终元数据 ---
    print("4. [合并] 生成最终 JSON 文件...")

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Wikidata-Truthy Full Dump",
        "total_triples_analyzed": int(total_sampled_rows),
        "file_count": len(target_files),
        "properties": {}
    }

    for _, row in df.iterrows():
        pid = row['pid']
        cnt = int(row['cnt'])
        unique_raw = int(row['unique_cnt'])
        unique = min(unique_raw, cnt)
        cr_val = unique / cnt if cnt > 0 else 0.0
        semantics = semantic_data.get(pid, {"label": "Unknown", "description": "No description."})

        metadata["properties"][pid] = {
            "label": semantics["label"],
            "description": semantics["description"],
            "cnt": cnt,
            "cr": round(cr_val, 6),
            "stats": {
                "total": cnt,
                "unique": unique
            }
        }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        json.dump(metadata, out, indent=2, ensure_ascii=False)

    print(f"🎉 全量统计成功! 文件已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()