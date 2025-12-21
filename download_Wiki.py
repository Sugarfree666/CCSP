import duckdb
import json
import time
import os
import math
from huggingface_hub import list_repo_files, hf_hub_download

# ================= 配置区域 =================
# 1. 设置国内镜像 (确保下载速度)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 请确保 Token 有效，或者如果在本地已登录可注释掉下行
os.environ["HF_TOKEN"] ="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"

REPO_ID = "CleverThis/wikidata-truthy"
SAMPLE_FILES_COUNT = 100  # 建议 50-100 个文件以覆盖长尾属性
OUTPUT_FILE = "ccsp framework/property_metadata_final.json"
# ===========================================

def run_pipeline():
    # --- 第一步：下载数据 ---
    print(f"1. [网络] 连接镜像站: {os.environ.get('HF_ENDPOINT')} ...")

    try:
        all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        parquet_files = [f for f in all_files if f.endswith(".parquet")]
        parquet_files.sort()
        target_files = parquet_files[:SAMPLE_FILES_COUNT]
        print(f"   选中文件数: {len(target_files)}")
    except Exception as e:
        print(f"   错误: 无法获取文件列表 ({e})")
        return

    print(f"2. [下载] 缓存 {SAMPLE_FILES_COUNT} 个 Parquet 文件...")
    local_paths = []
    for idx, filename in enumerate(target_files):
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
        local_paths.append(path)
        if (idx + 1) % 10 == 0: print(f"   进度: {idx + 1}/{SAMPLE_FILES_COUNT}")

    # --- 第二步：DuckDB 统计 ---
    print("3. [计算] DuckDB 聚合 (统计 Total 和 Unique)...")
    start_time = time.time()

    con = duckdb.connect()
    # SQL: 提取 Pxxx, 统计总数, 统计去重数
    query = f"""
    SELECT 
        regexp_extract(predicate, 'P\d+', 0) as pid,
        COUNT(*) as total_count,
        APPROX_COUNT_DISTINCT("object") as unique_count
    FROM read_parquet({local_paths})
    GROUP BY pid
    ORDER BY total_count DESC
    """
    df = con.execute(query).df()

    total_sampled_rows = df['total_count'].sum()
    print(f"   统计完成! 耗时: {time.time() - start_time:.2f}s")
    print(f"   总三元组行数: {total_sampled_rows:,}")

    # --- 第三步：计算指标并保存 ---
    print("4. [生成] 计算 s_base, lambda, CR 并保存...")

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "DuckDB Sampling with Log-Cardinality",
        "total_rows_analyzed": int(total_sampled_rows),
        "properties": {}
    }

    # 获取 P31 计数 (用于计算 r 的分母)
    p31_row = df[df['pid'] == 'P31']
    p31_count = int(p31_row['total_count'].iloc[0]) if not p31_row.empty else total_sampled_rows

    for _, row in df.iterrows():
        pid = row['pid']
        count = int(row['total_count'])
        unique = int(row['unique_count'])

        if count < 10: continue

        # 1. 计算 r(v) (Reliability)
        r_val = min(count / p31_count, 1.0)

        # 2. 计算 CR (Linear Cardinality Ratio)
        # CR = Unique / Total
        cr_val = unique / count

        # 3. 计算 s_base (New Log Formula)
        # 公式: 0.2 + 0.8 * (ln(U+1) / ln(T+1))
        if count <= 1:
            s_base = 0.2
        else:
            log_unique = math.log(unique + 1)
            log_total = math.log(count + 1)
            ratio = log_unique / log_total
            s_base = 0.2 + (0.8 * ratio)
        s_base = min(s_base, 1.0)

        # 4. 计算 lambda (LLM Weight)
        # 公式: 0.8 * (1 - CR)
        # CR 越高(ID类)，lambda 越低(不信LLM)
        lambda_val = 0.8 * (1.0 - cr_val)
        lambda_val = max(0.0, lambda_val)  # 保证非负

        metadata["properties"][pid] = {
            # "label": label_text,  # 已移除
            "r": round(r_val, 6),  # 密度
            "s_base": round(s_base, 6),  # 基础区分度 (对数版)
            "lambda": round(lambda_val, 6),  # LLM 权重 (线性版)
            "CR": round(cr_val, 6),  # 原始 CR 用于分析
            "stats": {  # 记录原始统计数据方便Debug
                "total": count,
                "unique": unique
            }
        }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        json.dump(metadata, out, indent=2, ensure_ascii=False)

    print(f"🎉 成功! 元数据表已保存至: {OUTPUT_FILE}")
    print(f"   共收录属性: {len(metadata['properties'])} 个")


if __name__ == "__main__":
    run_pipeline()