# environment.py
import re
import copy
from typing import Set, Dict, List
import logging
from wikidata_service import WikidataService
from data_model import Constraint
import math  # <--- 新增
import re
import copy
from graph_state import GraphState, ThoughtNode

logger = logging.getLogger(__name__)


class GraphEnvironment:
    """
    环境层：封装 WikidataService，提供原子操作工具 (Tools)。
    负责具体的 SPARQL 构造、执行和结果解析。
    """

    def __init__(self, wiki_service: WikidataService):
        self.service = wiki_service

    # --- Tool 1: Generate (生成思维) ---
    def tool_search_anchor(self, constraint: Constraint) -> Set[str]:
        """
        对应 GoT 的 Generate 操作：从无到有生成候选集。
        """
        logger.info(
            f"[Tool: Anchor] Searching {constraint.property_label} (ID: {constraint.property_id}) {constraint.operator} {constraint.value}")
        if constraint.operator == "IGNORE":
            logger.warning(f"[Tool: Anchor] Cannot search with IGNORE operator on {constraint.property_label}.")
            return set()
        try:
            val_str = str(constraint.value)
            pid = constraint.property_id

            # === [FIX] 1. 针对 QID 的查询 (Object Property) ===
            if re.match(r'^Q\d+$', val_str):
                # ?item wdt:Pxxx wd:Qxxx
                where_clause = f"?item wdt:{pid} wd:{val_str} ."

            # === [FIX] 2. 针对 日期/数值 的查询 (Datatype Property) ===
            # 如果是日期格式 YYYY-MM-DD 或 YYYY
            elif re.match(r'^\d{4}(-\d{2}-\d{2})?$', val_str):
                logger.info(f"  -> Detected Date Literal: {val_str}")

                # Wikidata 日期通常是 xsd:dateTime 格式 (e.g. "1974-12-31T00:00:00Z"^^xsd:dateTime)
                # 针对 Anchor，我们通常做精确匹配或基于 Operator 的匹配
                # 如果是 Anchor，我们暂时只支持 = 或 Operator 逻辑

                # 处理日期格式化
                if len(val_str) == 4:  # YYYY
                    # 如果只有年份，使用 YEAR() 函数
                    filter_logic = f"YEAR(?v) {constraint.operator} {val_str}"
                else:
                    # 完整日期，加上 ^^xsd:dateTime 类型转换
                    # 注意：Wikidata 存储通常带 T00:00:00Z，简单的字符串相等可能匹配不到
                    # 建议使用 >= <= 逻辑或者精确构造
                    if constraint.operator == "=":
                        # 尝试构建标准 Wikidata 日期格式
                        date_literal = f"'{val_str}T00:00:00Z'^^xsd:dateTime"
                        filter_logic = f"?v = {date_literal}"
                    else:
                        date_literal = f"'{val_str}T00:00:00Z'^^xsd:dateTime"
                        filter_logic = f"?v {constraint.operator} {date_literal}"

                where_clause = f"""
                    ?item wdt:{pid} ?v .
                    FILTER({filter_logic})
                """

            # === [FIX] 3. 针对 纯数值 的查询 ===
            elif re.match(r'^-?\d+(\.\d+)?$', val_str):
                logger.info(f"  -> Detected Number Literal: {val_str}")
                where_clause = f"""
                    ?item wdt:{pid} ?v .
                    FILTER(?v {constraint.operator} {val_str})
                """

            # === [FIX] 4. 针对 字符串标签 的查询 (Fallback) ===
            else:
                logger.info(f"Fallback: Searching by label match for '{val_str}' on property {pid}")
                # 只有当 Object 是 Entity 时才查 label
                # ?item -> ?target_entity -> [Label == "Value"]
                where_clause = f"""
                    ?item wdt:{pid} ?target .
                    ?target rdfs:label ?targetLabel .
                    FILTER(LCASE(STR(?targetLabel)) = LCASE("{val_str}")) .
                    FILTER(LANG(?targetLabel) = "en") .
                """

            sparql = f"""
            SELECT DISTINCT ?item WHERE {{
                {where_clause}
            }}
            LIMIT 1000
            """

            # [DEBUG] 打印生成的 SPARQL 以便调试
            # print(f"[SPARQL Debug] {sparql}")

            # 执行查询
            results = self.service.execute_sparql(sparql)

            # 解析结果
            qids = set()
            for r in results:
                url = r['item']['value']
                if "entity/" in url:
                    qids.add(url.split("/")[-1])

            logger.info(f"  -> Found {len(qids)} candidates.")
            return qids

        except Exception as e:
            logger.error(f"[Tool: Anchor] Execution failed: {e}")
            return set()

    # environment.py -> class GraphEnvironment

    def _align_magnitude(self, constraint: Constraint, parent_candidates: Set[str] = None,
                         sample_limit=10) -> Constraint:
        """
        [Robust] 稳健的动态数量级对齐逻辑。
        策略：
        1. 采样：获取 10 个样本的中位数 (Median)，避免被 AVG 的异常值误导。
        2. 比率分析：计算 Ratio = DB_Median / User_Value。
        3. 语义匹配：优先匹配时间因子 (60, 3600) 和数量级因子 (10^3, 10^6)。
        """
        # 1. 前置检查：只处理数值类型的 > 或 < 操作
        if constraint.operator not in [">", "<"]:
            return constraint

        try:
            user_val = float(constraint.value)
            if user_val == 0: return constraint
        except ValueError:
            return constraint

        try:
            pid = constraint.property_id

            # === [CORE FIX] 构造探测查询 ===
            # 策略：如果提供了候选集，只探测这些候选实体的属性值 (局部探测)
            # 这能避免"全局随机抽样"带来的巨大方差 (例如同时抽到 1分钟的短视频 和 120分钟的电影)

            if parent_candidates and len(parent_candidates) > 0:
                # 为了性能，如果候选集太大，只取前 20 个做样本
                sample_qids = list(parent_candidates)[:20]
                values_str = " ".join([f"wd:{qid}" for qid in sample_qids])

                sparql = f"""
                        SELECT ?v WHERE {{
                          VALUES ?item {{ {values_str} }}
                          ?item wdt:{pid} ?v .
                          FILTER(isNumeric(?v))
                        }} LIMIT {sample_limit}
                        """
            else:
                # Fallback: 如果没有候选集（极少情况，如Anchor阶段），才用全局随机探测
                sparql = f"""
                        SELECT ?v WHERE {{
                          ?item wdt:{pid} ?v .
                          FILTER(isNumeric(?v))
                        }} LIMIT {sample_limit}
                        """

            results = self.service.execute_sparql(sparql)
            if not results:
                return constraint

            # 提取数值并过滤
            values = []
            for r in results:
                try:
                    v = float(r['v']['value'])
                    if v > 0: values.append(v)
                except:
                    pass

            if not values:
                return constraint

            # 3. 计算中位数 (Median) - 比平均值更稳健
            values.sort()
            mid_idx = len(values) // 2
            db_median = values[mid_idx]

            # 计算比率
            ratio = db_median / user_val

            logger.info(
                f"[Auto-Align] Probing {constraint.property_label}: User={user_val}, DB_Median={db_median}, Ratio={ratio:.4f}")

            # 如果比率接近 1 (例如 0.5 ~ 2.0)，说明单位一致，无需调整
            if 0.5 <= ratio <= 2.0:
                return constraint

            factor = 1.0
            aligned_reason = ""

            # === 4. 语义因子匹配逻辑 (Semantic Factor Matching) ===

            # 辅助函数：判断 ratio 是否在 target 的容忍范围内 (+/- 50%)
            # 容忍度设大一点，因为"平均电影"和"特定电影"的时长本身就有差异
            def is_close_to(val, target, tolerance=0.5):
                return (target * (1 - tolerance)) < val < (target * (1 + tolerance))

            # --- A. 时间/单位换算检查 (60进制) ---

            # 场景: 库里是秒(大)，用户是分(小) -> Ratio ~ 60
            if is_close_to(ratio, 60.0):
                factor = 60.0
                aligned_reason = "Minute -> Second (x60)"

            # 场景: 库里是分(小)，用户是秒(大) -> Ratio ~ 1/60 (0.0166)
            elif is_close_to(ratio, 1 / 60.0):
                factor = 1 / 60.0
                aligned_reason = "Second -> Minute (x1/60)"

            # 场景: 库里是秒，用户是小时 -> Ratio ~ 3600
            elif is_close_to(ratio, 3600.0):
                factor = 3600.0
                aligned_reason = "Hour -> Second (x3600)"

            # --- B. 数量级检查 (10进制: k, M, B) ---

            # 如果没命中时间单位，再看是不是数量级搞错了 (Million/Billion)
            else:
                log_diff = math.log10(ratio)
                round_log = round(log_diff)

                # 只有当差异超过 2 个数量级 (100倍) 时才介入
                # 且 log_diff 必须接近整数 (误差 < 0.4)，避免误伤正常的数据波动
                if abs(round_log) >= 2 and abs(log_diff - round_log) < 0.4:
                    factor = 10 ** round_log
                    aligned_reason = f"Magnitude Correction (10^{round_log})"

            # 5. 应用修正
            if factor != 1.0:
                new_c = copy.deepcopy(constraint)
                new_c.value = str(user_val * factor)
                logger.warning(
                    f"[Auto-Align] FIXED {constraint.property_label}: {user_val} -> {new_c.value} ({aligned_reason})")
                return new_c

        except Exception as e:
            logger.warning(f"[Auto-Align] Logic error for {constraint.property_label}: {e}")

        return constraint

    # --- Tool 2: Filter (剪枝/过滤 - 增强版) ---
    def tool_filter(self, parent_candidates: Set[str], constraint: Constraint) -> Set[str]:
        """
        对应 GoT 的 Filter 操作：在现有集合上施加新约束。
        [Upgrade] 支持 Subclass (P279) 推理。
        [Upgrade] 支持 IGNORE 操作符。
        [Upgrade] 支持动态数量级对齐 (Dynamic Magnitude Alignment)。
        """
        # 1. IGNORE 检查
        if constraint.operator == "IGNORE":
            logger.info(f"[Tool: Filter] Constraint '{constraint.property_label}' is IGNORE. Skipping.")
            return parent_candidates

        if not parent_candidates:
            return set()
        # === [NEW] 动态对齐调用 ===
        # 在构造 SPARQL 之前，先检查并修正数值单位
        # 只有当包含数值比较时才触发，避免浪费时间
        # 这里的 align_constraint 是一个新的临时对象，不会污染原始 constraints 列表
        align_constraint = self._align_magnitude(constraint, parent_candidates)

        # 记录日志方便调试
        if align_constraint.value != constraint.value:
            logger.info(f"[Tool: Filter] aligned value {constraint.value} -> {align_constraint.value}")

        # 使用对齐后的约束对象进行后续操作
        constraint = align_constraint

        logger.info(
            f"[Tool: Filter] Filtering {len(parent_candidates)} items by {constraint.property_label} {constraint.operator} {constraint.value}")

        try:
            # 构造 VALUES 子句
            values_str = " ".join([f"wd:{qid}" for qid in parent_candidates])
            val_str = str(constraint.value)

            # === [Optimized] 类型判断逻辑优化 (互斥判断) ===
            is_qid = False
            is_year = False
            is_date_full = False
            is_number = False

            # 优先级：QID > 年份 > 完整日期 > 浮点数
            if re.match(r'^Q\d+$', val_str):
                is_qid = True
            elif re.match(r'^\d{4}$', val_str):
                is_year = True
            elif re.match(r'^\d{4}-\d{2}-\d{2}', val_str):
                is_date_full = True
            else:
                # 只有当前面都不是时，才尝试转浮点数
                try:
                    float(val_str)
                    is_number = True
                except ValueError:
                    pass

            # === 构造过滤逻辑 ===
            filter_clause = ""
            triple = ""

            if is_qid:
                # === 方案 A: 子类推理 (Subclass Inference) ===
                # 逻辑：?item 的属性值 ?actual_val，必须是 目标值(val_str) 本身，或者是它的子类
                triple = f"""
                    ?item wdt:{constraint.property_id} ?actual_val .
                    ?actual_val wdt:P279* wd:{val_str} .
                """
            else:
                # === 非 QID (数值/日期/字符串) ===
                triple = f"?item wdt:{constraint.property_id} ?val ."

                # 注意：这里根据上面计算的 flag 进行分支，不再重复正则
                if is_year and (
                        "date" in constraint.property_label.lower() or "publication" in constraint.property_label.lower()):
                    filter_clause = f"FILTER(YEAR(?val) {constraint.operator} {val_str})"

                elif is_date_full:
                    # 加上 ^^xsd:dateTime 类型
                    val_fmt = f"'{val_str}'^^xsd:dateTime"
                    filter_clause = f"FILTER(?val {constraint.operator} {val_fmt})"

                elif is_number:
                    # 纯数值直接拼接
                    filter_clause = f"FILTER(?val {constraint.operator} {val_str})"

                elif constraint.operator == "contains":
                    filter_clause = f"FILTER(CONTAINS(LCASE(?val), LCASE('{val_str}')))"
                else:
                    # 默认字符串精确匹配
                    filter_clause = f"FILTER(?val = '{val_str}')"

            sparql = f"""
                    SELECT DISTINCT ?item WHERE {{
                        VALUES ?item {{ {values_str} }}
                        {triple}
                        {filter_clause}
                    }}
                    """

            # 执行查询
            results = self.service.execute_sparql(sparql)

            # 解析结果
            valid_qids = set()
            for r in results:
                url = r['item']['value']
                valid_qids.add(url.split("/")[-1])

            logger.info(f"  -> {len(valid_qids)} items remain after filtering.")
            return valid_qids

        except Exception as e:
            logger.error(f"[Tool: Filter] Execution failed: {e}")
            return set()

    # --- Tool 3: Aggregate (聚合思维) ---
    def tool_intersect(self, set_a: Set[str], set_b: Set[str]) -> Set[str]:
        """
        对应 GoT 的 Aggregate 操作：多路思维合并 (求交集)
        """
        try:
            result = set_a.intersection(set_b)
            logger.info(f"[Tool: Intersect] Merging {len(set_a)} and {len(set_b)} sets -> {len(result)} remaining")
            return result
        except Exception as e:
            logger.error(f"[Tool: Intersect] Failed: {e}")
            return set()

    # --- Tool 4: Refine (精炼/修正思维) ---
    def tool_relax_constraint(self, constraint: Constraint) -> Constraint:
        """
        对应 GoT 的 Refine 操作：当结果为空时，根据 Softness 放宽条件。
        [Optimized] 逻辑去重，优先处理实体类型降级为 IGNORE。
        """
        logger.info(f"[Tool: Refine] Relaxing constraint {constraint.property_label}")

        # 深拷贝以防修改原始引用
        new_c = copy.deepcopy(constraint)

        try:
            val_str = str(new_c.value)

            # 提前判断数据类型，避免后续重复正则
            is_qid = bool(re.match(r'^Q\d+$', val_str))

            # 策略 A: 数值/日期范围放宽 (针对 > 或 <)
            # 注意：排除掉 QID，因为 QID 即使是数字开头也不应该进这里的逻辑
            if new_c.operator in [">", "<"] and not is_qid:
                # 尝试转 float，转换失败则跳过
                try:
                    val = float(new_c.value)
                    if new_c.operator == "<":
                        new_c.value = str(val * 1.5)  # 大幅放宽上限 (e.g. 1.1 -> 1.5)
                        logger.info(f"  -> Relaxed '<' value to {new_c.value}")
                    elif new_c.operator == ">":
                        new_c.value = str(val * 0.5)  # 大幅放宽下限 (e.g. 0.9 -> 0.5)
                        logger.info(f"  -> Relaxed '>' value to {new_c.value}")
                except ValueError:
                    # 如果转换失败（比如日期字符串），直接 Fallback 到 IGNORE
                    new_c.operator = "IGNORE"
                    logger.warning(f"  -> Numeric relaxation failed for {val_str}. Defaulting to IGNORE.")

            # 策略 B: 实体 (QID) -> 直接 IGNORE
            # 解释：如果已经用了子类推理还是 0 结果，说明数据大概率缺失。
            # 实体 ID 是离散的，不能把 Q123 变成 Q123.5，所以只能忽略。
            elif is_qid:
                new_c.operator = "IGNORE"
                logger.info(f"  -> Entity constraint (QID) too strict or data missing. Changed to 'IGNORE'.")

            # 策略 C: 字符串精确匹配 -> 尝试模糊匹配 (Contains)
            elif new_c.operator == "=":
                new_c.operator = "contains"
                logger.info(f"  -> Relaxed operator '=' to 'contains'")

            # 策略 D: 已经是 contains 或者其他情况 -> IGNORE
            else:
                new_c.operator = "IGNORE"
                logger.warning("  -> No other relaxation strategy. Defaulting to IGNORE.")

            return new_c

        except Exception as e:
            logger.error(f"[Tool: Refine] Failed: {e}")
            return constraint