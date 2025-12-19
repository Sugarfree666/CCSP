import duckdb
import json
import time
import os
from huggingface_hub import list_repo_files, hf_hub_download

# ================= 配置 =================
# 1. 强制设置国内镜像 (核心修复点)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"

REPO_ID = "CleverThis/wikidata-truthy"
SAMPLE_FILES_COUNT = 100  # 只下载前 5 个文件
OUTPUT_FILE = "property_metadata_sampled.json"


# =======================================

def run_duckdb_local_sampling():
    print(f"1. [网络] 正在连接镜像站获取文件列表: {os.environ['HF_ENDPOINT']} ...")

    try:
        # 获取所有 parquet 文件列表
        all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        parquet_files = [f for f in all_files if f.endswith(".parquet")]
        parquet_files.sort()

        # 截取前 N 个文件
        target_files = parquet_files[:SAMPLE_FILES_COUNT]
        print(f"   选中文件: {target_files}")

    except Exception as e:
        print("   获取文件列表失败，请检查网络或代理设置。")
        print(f"   错误信息: {e}")
        return

    print(f"2. [下载] 正在将 {SAMPLE_FILES_COUNT} 个文件缓存到本地 (利用 huggingface_hub)...")

    local_paths = []
    for idx, filename in enumerate(target_files):
        print(f"   正在下载 ({idx + 1}/{SAMPLE_FILES_COUNT}): {filename} ...")
        # hf_hub_download 会自动处理断点续传和缓存，下载过一次就不需要在下载了
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset"
        )
        local_paths.append(local_path)

    print(f"   下载完成！文件路径示例: {local_paths[0]}")

    # ================= DuckDB 本地分析 =================

    print("3. [计算] 开始 DuckDB 本地极速聚合...")
    start_time = time.time()

    con = duckdb.connect()

    # 注意：这里我们把 Windows 路径转义一下，或者直接传 list 给 read_parquet 也是支持的
    # read_parquet 接受文件列表

    query = f"""
    SELECT 
        regexp_extract(predicate, 'P\d+', 0) as pid,
        COUNT(*) as count
    FROM read_parquet({local_paths})
    GROUP BY pid
    ORDER BY count DESC
    """

    df = con.execute(query).df()

    total_sampled_rows = df['count'].sum()

    elapsed = time.time() - start_time
    print(f"4. 统计完成！耗时: {elapsed:.2f}s")
    print(f"   抽样总行数: {total_sampled_rows:,}")

    # ================= 保存逻辑 (保持不变) =================
    print("5. 保存元数据...")
    metadata = {
        "source": "DuckDB Sampling (Local Cache)",
        "total_sampled_rows": int(total_sampled_rows),
        "properties": {}
    }

    # 获取 P31 (instance of) 的计数作为基准分母
    # 如果抽样中没遇到 P31 (极小概率)，就用总行数代替
    p31_row = df[df['pid'] == 'P31']
    if not p31_row.empty:
        p31_count = int(p31_row['count'].iloc[0])
    else:
        p31_count = total_sampled_rows

    for _, row in df.iterrows():
        pid = row['pid']
        count = int(row['count'])

        if count < 10: continue

        # 计算相对于 P31 的密度
        r_normalized = count / p31_count
        r_normalized = min(r_normalized, 1.0)

        # 启发式 s_base
        s_base = 1.0 - (r_normalized * 0.8)

        metadata["properties"][pid] = {
            "r": round(r_normalized, 6),
            "s_base": round(s_base, 6)
        }

    with open(OUTPUT_FILE, 'w') as out:
        json.dump(metadata, out, indent=2)
    print(f"成功! 结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    run_duckdb_local_sampling()