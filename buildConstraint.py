import json
import re
import statistics
import random
from collections import Counter

# ==========================================
# 1. 配置区域
# ==========================================
INPUT_FILE = 'datasets/datasets.json'
OUTPUT_FILE = 'datasets/complex_constraint_dataset.json'

TARGET_ANSWER_COUNT = 1  # 目标：唯一答案
MIN_CONSTRAINTS = 2  # 允许 2 个约束
MAX_TRIALS_PER_TARGET = 5  # 尝试次数

# ==========================================
# 属性配置字典
# ==========================================
PROP_CONFIG = {
    "P31": {"label": "is a", "format": "string"},
    "P279": {"label": "is a subclass of", "format": "string"},
    "P569": {"label": "born year", "format": "year"},
    "P570": {"label": "died in", "format": "year"},
    "P27": {"label": "citizenship", "format": "string"},
    "P106": {"label": "occupation", "format": "string"},
    "P166": {"label": "award received", "format": "string"},
    "P69": {"label": "educated at", "format": "string"},
    "P21": {"label": "gender", "format": "string"},
    "P577": {"label": "released date", "format": "year"},
    "P136": {"label": "genre", "format": "string"},
    "P57": {"label": "directed by", "format": "string"},
    "P161": {"label": "starring", "format": "string"},
    "P175": {"label": "performed by", "format": "string"},
    "P2047": {"label": "duration", "format": "number", "unit_suffix": " min"},
    "P2142": {"label": "box office", "format": "currency", "unit_prefix": "$"},
    "P2130": {"label": "budget", "format": "currency", "unit_prefix": "$"},
    "P407": {"label": "language", "format": "string"},
    "P17": {"label": "country", "format": "string"},
    "P1082": {"label": "population", "format": "large_number"},
    "P2046": {"label": "area", "format": "large_number", "unit_suffix": " sq km"},
    "P2044": {"label": "elevation", "format": "number", "unit_suffix": " m"},
    "P30": {"label": "continent", "format": "string"},
    "P1376": {"label": "capital of", "format": "string"},
    "P571": {"label": "founded in", "format": "year"},
    "P159": {"label": "headquartered in", "format": "string"},
    "P1128": {"label": "employees", "format": "large_number"},
    "P118": {"label": "league", "format": "string"},
    "P112": {"label": "founded by", "format": "string"},
    "P2048": {"label": "height", "format": "number", "unit_suffix": " m"},
    "P2067": {"label": "mass", "format": "number", "unit_suffix": " kg"},
    "P186": {"label": "made of", "format": "string"},
    "P61": {"label": "discovered by", "format": "string"},
}


# ==========================================
# 2. 辅助工具
# ==========================================
def normalize_value(prop_id, raw_data):
    """
    prop_id: 属性 ID
    raw_data: 可能是简单的字符串 (对于普通属性)，也可能是字典 (对于量化属性)
    """

    # 1. 兼容旧的字符串数据 (针对 SIMPLE_PROPS)
    if isinstance(raw_data, str):
        val_str = raw_data
        # ... (保留原有的年份处理逻辑) ...
        if prop_id in ["P577", "P569", "P570", "P571"]:
            match = re.search(r'^-?(\d{4})', val_str)
            if match: return int(match.group(1))
        return val_str

    # 2. 处理新的字典数据 (针对 QUANTITY_PROPS)
    if isinstance(raw_data, dict):
        amount_str = raw_data.get("amount", "0")
        unit_str = raw_data.get("unit", "").lower()

        try:
            val = float(amount_str)


            # Case A: 身高 (P2048)
            if prop_id == "P2048":
                if "centimetre" in unit_str or "cm" in unit_str:
                    return val / 100.0
                if "inch" in unit_str:
                    return val * 0.0254
                if "foot" in unit_str or "feet" in unit_str:
                    return val * 0.3048
                # 默认为米，无需转换
                return val

            # Case B: 面积 (P2046) - 经常有公顷、平方英里
            if prop_id == "P2046":
                if "hectare" in unit_str:
                    return val * 0.01  # 转为平方千米
                if "mile" in unit_str:  # square mile
                    return val * 2.5899
                # 默认平方千米
                return val

            # Case C: 质量 (P2067)
            if prop_id == "P2067":
                if "gram" in unit_str and "kilogram" not in unit_str:
                    return val / 1000.0
                if "pound" in unit_str:
                    return val * 0.453592
                return val

            return val

        except:
            return None

    return str(raw_data)


def format_human_readable(prop_id, value, operator=">"):
    config = PROP_CONFIG.get(prop_id, {})
    label = config.get("label", prop_id)
    fmt_type = config.get("format", "string")

    # 1. 年份处理
    if fmt_type == "year":
        # [Fix 1] 显式处理 None 或空值，防止生成 "released in around None"
        if value is None or value == "":
            return f"{label} is unknown"

        try:
            # [Fix 2] 尝试安全转换
            # 先转 float 是为了兼容 "2010.0" 这种字符串格式，再转 int 去掉小数点
            val_int = int(float(value))

            if operator == ">":
                return f"{label} after {val_int}"
            elif operator == "<":
                return f"{label} before {val_int}"
            else:
                # 对于等于的情况，用 "is" 比 "around" 更确切
                return f"{label} is {val_int}"

        except (ValueError, TypeError):
            # [Fix 3] 兜底逻辑
            # 如果转换失败（例如 value 是 "2020s" 或 "2010-05-01" 这种非纯数字格式）
            # 直接把原文本括起来显示，而不是猜测它是 "around"
            return f"{label} is '{value}'"

    # 2. 货币与大数值
    elif fmt_type in ["currency", "large_number"]:
        try:
            num = float(value)
            prefix = config.get("unit_prefix", "")
            suffix = config.get("unit_suffix", "")

            if num >= 1_000_000_000:
                val_str = f"{num / 1_000_000_000:.1f}B"
            elif num >= 1_000_000:
                val_str = f"{num / 1_000_000:.1f}M"
            elif num >= 1_000:
                val_str = f"{num / 1_000:.1f}K"
            else:
                val_str = str(int(num))

            # --- 修正：完善逻辑判断 ---
            if operator == ">":
                op_text = "more than"
            elif operator == "<":
                op_text = "less than"
            else:
                # 处理 "=" 的情况
                return f"{label} is {prefix}{val_str}{suffix}"

            return f"{label} is {op_text} {prefix}{val_str}{suffix}"
        except:
            return f"{label} is {value}"

    # 3. 普通数值
    elif fmt_type == "number":
        try:
            val_float = float(value)
            suffix = config.get("unit_suffix", "")

            if val_float.is_integer():
                display_val = str(int(val_float))
            else:
                display_val = f"{val_float:.2f}"

            # --- 修正：完善逻辑判断 ---
            if operator == ">":
                op_text = "more than"
            elif operator == "<":
                op_text = "less than"
            else:
                return f"{label} is {display_val}{suffix}"

            return f"{label} is {op_text} {display_val}{suffix}"
        except:
            return f"{label} {operator} {value}"

    # 4. 默认
    else:
        return f"{label} is '{value}'"

# ==========================================
# 3. 核心类：深度约束挖掘器
# ==========================================
class ComplexConstraintMiner:
    def __init__(self, entry):
        self.entry = entry
        self.all_answers = set(entry['answers'])
        self.attrs = entry.get('answers_attributes', {})
        self.atomic_constraints = []

    def mine(self):
        """主入口"""
        if not self.attrs or len(self.all_answers) < 2:
            return []

        # 1. 获取所有原子属性
        self._mine_atomic()

        unique_results = []
        seen_signatures = set()

        # 2. 遍历所有可能的答案作为 Target
        for target_id in self.all_answers:
            if target_id not in self.attrs: continue

            applicable_constraints = [
                c for c in self.atomic_constraints
                if target_id in c['subset']
            ]

            if len(applicable_constraints) < MIN_CONSTRAINTS:
                continue

            for _ in range(MAX_TRIALS_PER_TARGET):
                combo = self._greedy_stacking(target_id, applicable_constraints)

                if combo:
                    signature = tuple(sorted([c['logic_str'] for c in combo['constraints']]))
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    unique_results.append(combo)

                    if len(unique_results) >= 5:
                        return unique_results

        return unique_results

    def _mine_atomic(self):
        """挖掘原子约束 (修复：显式分离数值和字符串，防止混合类型排序报错)"""
        self.atomic_constraints = []
        prop_data = {}

        # 1. 收集数据
        for qid in self.all_answers:
            if qid not in self.attrs: continue
            for pid, vals in self.attrs[qid].items():
                if not vals: continue
                if pid not in prop_data: prop_data[pid] = []
                norm_val = normalize_value(pid, vals[0])
                if norm_val is not None:
                    prop_data[pid].append((qid, norm_val))

        # 2. 处理每个属性
        for pid, pairs in prop_data.items():
            if not pairs: continue

            # --- 关键修复：先将数据按类型分离 ---
            # 只有 int/float 才能计算分位点和大于小于
            numeric_pairs = [p for p in pairs if isinstance(p[1], (int, float))]
            # 字符串只能做相等匹配
            string_pairs = [p for p in pairs if isinstance(p[1], str)]

            # --- A. 处理数值型 (Continuous) ---
            if len(numeric_pairs) >= 2:
                nums = [p[1] for p in numeric_pairs]

                try:
                    # 计算分位点
                    quantiles = statistics.quantiles(nums, n=4) if len(nums) >= 4 else [statistics.median(nums)]
                    unique_thresholds = set(quantiles)

                    for thresh in unique_thresholds:
                        # 仅在 numeric_pairs 中筛选，防止比较字符串报错
                        subset_gt = {qid for qid, v in numeric_pairs if v > thresh}
                        subset_lt = {qid for qid, v in numeric_pairs if v < thresh}

                        # 保存 > 约束
                        if 0 < len(subset_gt) < len(self.all_answers):
                            self.atomic_constraints.append({
                                "type": "continuous", "prop": pid, "val": thresh,
                                "op": ">",
                                "logic_str": f"({pid} > {thresh})", "subset": subset_gt
                            })

                        # 保存 < 约束
                        if 0 < len(subset_lt) < len(self.all_answers):
                            self.atomic_constraints.append({
                                "type": "continuous", "prop": pid, "val": thresh,
                                "op": "<",
                                "logic_str": f"({pid} < {thresh})", "subset": subset_lt
                            })
                except Exception as e:
                    print(f"Warning: Numeric calculation failed for {pid}: {e}")

            # --- B. 处理离散型 (Discrete) ---
            # 字符串数据（或者被归类为字符串的混合数据）
            if string_pairs:
                values = [p[1] for p in string_pairs]
                counts = Counter(values)
                for val, count in counts.items():
                    if count < len(self.all_answers):
                        subset = {qid for qid, v in string_pairs if v == val}
                        self.atomic_constraints.append({
                            "type": "discrete", "prop": pid, "val": val,
                            "op": "=",
                            "logic_str": f"({pid} is '{val}')", "subset": subset
                        })

    def _greedy_stacking(self, target_id, candidates):
        """贪婪堆叠 (确保使用 op 生成正确描述)"""
        current_set = self.all_answers.copy()
        chosen = []
        used_props = set()

        random.shuffle(candidates)
        candidates.sort(key=lambda x: len(x['subset']), reverse=True)

        for constr in candidates:
            if constr['prop'] in used_props: continue

            intersection = current_set & constr['subset']

            if len(intersection) < len(current_set):
                chosen.append(constr)
                used_props.add(constr['prop'])
                current_set = intersection

            if len(current_set) == TARGET_ANSWER_COUNT:
                if target_id not in current_set: return None

                if len(chosen) >= MIN_CONSTRAINTS:
                    # --- 核心修复：传入 c['op'] ---
                    desc_list = []
                    for c in chosen:
                        # 确保调用的是含有 operator 参数的函数
                        desc = format_human_readable(c['prop'], c['val'], c['op'])
                        desc_list.append(desc)

                    return {
                        "constraints": chosen,
                        "constraint_logic": " AND ".join([c['logic_str'] for c in chosen]),
                        "constraint_description": " AND ".join(desc_list),
                        "final_answer_count": len(current_set),
                        "final_answers": list(current_set)
                    }
        return None


# ==========================================
# 4. 主执行流程
# ==========================================
def main():
    print(f"正在读取输入文件: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    final_dataset = []
    print(f"开始挖掘... (目标: 唯一答案, 约束 >= {MIN_CONSTRAINTS})")

    for i, entry in enumerate(data):
        if i % 50 == 0: print(f"Processing {i}/{len(data)}...")

        miner = ComplexConstraintMiner(entry)
        results = miner.mine()

        if results:
            for res in results:
                new_entry = {
                    "original_question": entry['question'],
                    "source_id": entry.get('original_id'),
                    "original_answer_count": len(entry['answers']),
                    "constraint_description": res['constraint_description'],
                    "constraint_logic": res['constraint_logic'],
                    "new_ground_truth": res['final_answers'],
                    "new_answer_count": res['final_answer_count']
                }
                final_dataset.append(new_entry)

    print(f"挖掘完成！共生成 {len(final_dataset)} 条数据。")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()