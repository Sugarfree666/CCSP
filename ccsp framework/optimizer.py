import math
import logging
from typing import List
from data_model import Constraint

logger = logging.getLogger(__name__)


class ConstraintOptimizer:
    def __init__(self, wiki_service):
        self.wiki_service = wiki_service
        # [SETTING] 阈值：如果数量超过这个数，就认为不适合做 Anchor
        self.PROBE_LIMIT = 1000

    def optimize(self, constraints: List[Constraint]) -> List[Constraint]:
        logger.info("--- Starting Dynamic Probing (Limit-based) ---")

        for c in constraints:
            # 1. 构造 LIMIT 查询 (不再是 COUNT)
            sparql = self._build_probe_query(c, limit=self.PROBE_LIMIT + 1)

            # 2. 执行探测
            # 这里的 timeout 可以设短一点 (比如 2s)，因为 LIMIT 查询通常极快
            # 如果 2s 还没返回前 1000 个，那网络肯定有问题或者查询太复杂
            rows_found = self.wiki_service.probe_query_count(sparql, timeout_sec=2.0)

            # 3. 逻辑判定
            if rows_found > self.PROBE_LIMIT:
                # 超过阈值，说明是个大集合
                c.estimated_rows = 999_999_999  # 标记为极大，强迫排在后面
                c.priority_score = 0.0
                logger.info(f"Probe: {c.property_label} -> Hit Limit (> {self.PROBE_LIMIT})")
            elif rows_found == -1:
                # 超时或错误
                c.estimated_rows = 999_999_999
                c.priority_score = 0.0
                logger.info(f"Probe: {c.property_label} -> Timeout/Error")
            else:
                # 小于阈值，这是精确的具体数量，适合做 Anchor
                c.estimated_rows = rows_found
                # +2 防止 log(0) 或 log(1)
                c.priority_score = 1.0 / math.log10(rows_found + 2)
                logger.info(f"Probe: {c.property_label} -> {rows_found} rows (Anchor Candidate!)")

        # 4. 排序
        sorted_constraints = sorted(constraints, key=lambda x: x.priority_score, reverse=True)
        return sorted_constraints

    def _build_probe_query(self, c: Constraint, limit: int) -> str:
        """
        构造带 LIMIT 的 SELECT 查询
        """
        pid = c.property_id if c.property_id else "P0"

        # 基础三元组
        triple = f"?item wdt:{pid} ?v ."
        filter_clause = ""

        # === 构造 Filter 逻辑 (保持不变) ===
        if c.operator == "=":
            if str(c.value).startswith("Q") and c.value[1:].isdigit():
                triple = f"?item wdt:{pid} wd:{c.value} ."
            else:
                filter_clause = f"FILTER(?v = '{c.value}')"
        elif c.operator in [">", "<"]:
            val_str = str(c.value)

            # Case 1: 年份 (2020)
            if val_str.isdigit() and len(val_str) == 4:
                filter_clause = f"FILTER(YEAR(?v) {c.operator} {val_str})"

            # Case 2: [FIX] 完整日期 (YYYY-MM-DD) -> 必须加引号和类型
            elif "-" in val_str:
                # 补全 ISO 格式并加引号
                date_literal = f"'{val_str}T00:00:00Z'^^xsd:dateTime"
                filter_clause = f"FILTER(?v {c.operator} {date_literal})"

            # Case 3: 纯数值
            else:
                filter_clause = f"FILTER(?v {c.operator} {val_str})"

        elif c.operator == "contains":
            filter_clause = f"FILTER(CONTAINS(LCASE(STR(?v)), LCASE('{c.value}')))"

        # === [Change] 使用 LIMIT ===
        # 我们只查 ?item，不需要 ?v，且加上 DISTINCT
        query = f"""
        SELECT DISTINCT ?item WHERE {{
            {triple} 
            {filter_clause}
        }}
        LIMIT {limit}
        """
        return query