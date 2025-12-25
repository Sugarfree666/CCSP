import json
import re
import time
from SPARQLWrapper import SPARQLWrapper, JSON

# ================= 配置区域 =================
# === 配置 ===
INPUT_FILE = r"D:\GitHub\CCSP\datasets\complex_constraint_dataset_rewrite_queries.json"  # 你的问题集
METADATA_FILE = r"D:\GitHub\CCSP\ccsp framework\property_metadata.json"  # 你的统计表
OUTPUT_FILE = "train_data_pointwise.jsonl"

# 判定标准：结果数量在 [1, 1000] 之间为好锚点
MIN_ANCHOR_SIZE = 1
MAX_ANCHOR_SIZE = 1000

# SPARQL 端点
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


# ===========================================

class DatasetBuilderFinal:
    def __init__(self):
        self.metadata = self._load_json(METADATA_FILE).get("properties", {})
        self.sparql = SPARQLWrapper(SPARQL_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", "CCSP-DatasetBuilder/3.1 (Research)")

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            return {}

    def get_stats_text(self, pid):
        """生成统计特征文本 (Feature Injection)"""
        meta = self.metadata.get(pid, {})
        if not meta:
            return "Frequency: Unknown, Diversity: Unknown"

        cnt = meta.get("cnt", 0)
        cr = meta.get("cr", 0.0)

        if cnt > 1000000:
            freq = "Universal (>1M)"
        elif cnt > 10000:
            freq = "Common"
        elif cnt > 100:
            freq = "Moderate"
        else:
            freq = "Rare"

        if cr > 0.9:
            div = "Unique Identifier"
        elif cr > 0.1:
            div = "High Diversity"
        else:
            div = "Low Diversity"

        return f"Frequency: {freq}, Diversity: {div} (CR:{cr:.2f})"

    def get_real_count_limit(self, query_sparql):
        """[核心优化] 使用 LIMIT 检测法获取数量"""
        try:
            limit_val = MAX_ANCHOR_SIZE + 1
            if "LIMIT" not in query_sparql.upper():
                query_sparql += f" LIMIT {limit_val}"

            self.sparql.setQuery(query_sparql)
            # 建议稍微放宽一点超时时间，或者保持 10s
            self.sparql.setTimeout(15)

            results = self.sparql.query().convert()["results"]["bindings"]
            count = len(results)

            if count >= limit_val:
                return 999999  # 溢出，Bad Anchor
            return count

        except Exception as e:
            error_str = str(e).lower()  # 转小写，通杀所有大小写情况

            # 捕获各种超时情况
            if "timed out" in error_str or "timeout" in error_str or "504" in error_str:
                # 核心修改：超时 = 极慢 = Bad Anchor
                # 不要打印错误刷屏，直接返回大数
                return 999999

                # 其他错误才打印
            print(f"   [SPARQL Error]: {e}")
            return -1

    def recover_subject_anchor(self, simple_question_text, answer_qid):
        """
        [策略优化] 使用 原始简单问题 (simple_question_text) 进行实体匹配
        理由：原始问题中的实体通常未经变形，匹配成功率更高
        """
        query = f"""
        SELECT ?neighbor ?neighborLabel ?p ?dir WHERE {{
          {{ ?neighbor ?p wd:{answer_qid} . BIND("incoming" AS ?dir) }} 
          UNION 
          {{ wd:{answer_qid} ?p ?neighbor . BIND("outgoing" AS ?dir) }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 200
        """

        candidates = []
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()["results"]["bindings"]

            # 使用简单问题进行匹配
            q_lower = simple_question_text.lower()

            for row in results:
                lbl = row.get("neighborLabel", {}).get("value", "").strip()
                pid = row["p"]["value"].split("/")[-1]

                if pid in ["P31", "P17", "P279", "P131", "P_score"]: continue
                if not lbl: continue

                # === 核心匹配：在简单问题中寻找 ===
                if len(lbl) > 2 and lbl.lower() in q_lower:
                    neighbor_qid = row["neighbor"]["value"].split("/")[-1]
                    direction = row["dir"]["value"]

                    candidates.append({
                        "type": "recovered",
                        "pid": pid,
                        "subject_label": lbl,
                        "subject_qid": neighbor_qid,
                        "direction": direction
                    })
        except:
            pass

        return candidates

    def parse_filter_constraints(self, logic_str):
        if not logic_str: return []
        constraints = []
        parts = logic_str.split(" AND ")
        for part in parts:
            clean_part = part.strip("() ")
            match = re.match(r"(P\d+)\s+(is|[<>=]+)\s+(.+)", clean_part)
            if match:
                pid, op_str, val_raw = match.groups()
                op = "=" if op_str == "is" else op_str
                val = val_raw.strip("'\"")
                constraints.append({
                    "type": "filter",
                    "pid": pid,
                    "op": op,
                    "val": val
                })
        return constraints

    def process(self):
        questions = self._load_json(INPUT_FILE)
        print(f"🚀 Processing {len(questions)} questions...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(questions):
                # 关键修改：分离用途
                complex_q_text = item['complex_question']  # 用作训练特征 (Feature)
                simple_q_text = item['original_question']  # 用作挖掘匹配 (Mining)

                # === 1. 尝试找回正样本 (Label 1) ===
                if item.get("new_ground_truth"):
                    ans_qid = item["new_ground_truth"][0]

                    # 使用 简单问题 找回 Anchor
                    recovered_anchors = self.recover_subject_anchor(simple_q_text, ans_qid)

                    for anchor in recovered_anchors:
                        # 验证 Count
                        if anchor['direction'] == "incoming":
                            sparql = f"SELECT ?s WHERE {{ wd:{anchor['subject_qid']} wdt:{anchor['pid']} ?s }}"
                        else:
                            sparql = f"SELECT ?s WHERE {{ ?s wdt:{anchor['pid']} wd:{anchor['subject_qid']} }}"

                        count = self.get_real_count_limit(sparql)

                        # 生成文本
                        stats = self.get_stats_text(anchor['pid'])
                        pid_label = self.metadata.get(anchor['pid'], {}).get('label', anchor['pid'])

                        cand_text = f"Constraint: {pid_label} ({anchor['pid']}) = '{anchor['subject_label']}'. Stats: {stats}"

                        label = 1.0 if MIN_ANCHOR_SIZE <= count <= MAX_ANCHOR_SIZE else 0.0

                        # 写入训练数据时，Query 使用 复杂问题
                        record = {"query": complex_q_text, "text": cand_text, "label": label}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(f"   [Recovered] {cand_text[:60]}... -> Count:{count} (Label {label})")

                # === 2. 处理原有 Filters (通常是 Label 0) ===
                filters = self.parse_filter_constraints(item.get('constraint_logic', ''))
                for filt in filters:
                    if filt['op'] in ['>', '<', '>=', '<=']:
                        count = 999999
                    else:
                        safe_val = filt['val'].replace("'", "\\'")
                        sparql = f"""
                        SELECT ?s WHERE {{ 
                            ?s wdt:{filt['pid']} ?o .
                            ?o rdfs:label ?label .
                            FILTER(LCASE(STR(?label)) = LCASE("{safe_val}")) .
                            FILTER(LANG(?label) = "en")
                        }}
                        """
                        count = self.get_real_count_limit(sparql)

                    stats = self.get_stats_text(filt['pid'])
                    pid_label = self.metadata.get(filt['pid'], {}).get('label', filt['pid'])
                    cand_text = f"Constraint: {pid_label} ({filt['pid']}) {filt['op']} '{filt['val']}'. Stats: {stats}"

                    label = 1.0 if MIN_ANCHOR_SIZE <= count <= MAX_ANCHOR_SIZE else 0.0

                    # 写入训练数据时，Query 依然使用 复杂问题
                    record = {"query": complex_q_text, "text": cand_text, "label": label}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if label == 1.0:
                        print(f"   [Filter]    {cand_text[:60]}... -> Count:{count} (Label {label})")

                time.sleep(0.5)

        print(f"Done! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    builder = DatasetBuilderFinal()
    builder.process()