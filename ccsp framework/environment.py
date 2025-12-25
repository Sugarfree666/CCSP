# environment.py
import re
import copy
from typing import Set, Dict, List
import logging
from wikidata_service import WikidataService
from data_model import Constraint
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
        逻辑源自原 GraphReasoningExecutor._fetch_anchor_candidates
        """
        logger.info(
            f"[Tool: Anchor] Searching {constraint.property_label} (ID: {constraint.property_id}) = {constraint.value}")

        try:
            val_str = str(constraint.value)

            # 情况 A: 已经是 QID (e.g., Q19198) - 最理想情况
            if re.match(r'^Q\d+$', val_str):
                where_clause = f"?item wdt:{constraint.property_id} wd:{val_str} ."

            # 情况 B: 仍然是字符串 (Entity Linking 失败)
            else:
                logger.info(f"Fallback: Searching by label match for '{val_str}' on property {constraint.property_id}")
                # 逻辑：?item -> ?target_entity -> [Label == "Value"]
                where_clause = f"""
                    ?item wdt:{constraint.property_id} ?target .
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
            return set()  # 出错返回空集合，防止 NoneType 错误

    # --- Tool 2: Filter (剪枝/过滤) ---
    def tool_filter(self, parent_candidates: Set[str], constraint: Constraint) -> Set[str]:
        """
        对应 GoT 的 Filter 操作：在现有集合上施加新约束。
        逻辑源自原 GraphReasoningExecutor._apply_filter
        """
        logger.info(
            f"[Tool: Filter] Filtering {len(parent_candidates)} items by {constraint.property_label} {constraint.operator} {constraint.value}")

        if not parent_candidates:
            return set()

        try:
            # 构造 VALUES 子句 (限制搜索空间)
            values_str = " ".join([f"wd:{qid}" for qid in parent_candidates])

            val_str = str(constraint.value)
            is_qid = bool(re.match(r'^Q\d+$', val_str))

            # === 日期与数值检测逻辑 ===
            is_year = bool(re.match(r'^\d{4}$', val_str))
            is_date_full = bool(re.match(r'^\d{4}-\d{2}-\d{2}', val_str))
            # 简单的数值检测
            is_number = val_str.replace('.', '', 1).isdigit()

            # 构造过滤逻辑
            filter_clause = ""
            target = f"wd:{val_str}" if is_qid else "?val"
            triple = f"?item wdt:{constraint.property_id} {target} ."

            if not is_qid:
                val_fmt = val_str

                # 针对不同类型的 Filter 生成
                if is_year and (
                        "date" in constraint.property_label.lower() or "publication" in constraint.property_label.lower()):
                    # 情况 A: 只有年份 -> 使用 YEAR() 函数
                    filter_clause = f"FILTER(YEAR(?val) {constraint.operator} {val_str})"

                elif is_date_full:
                    # 情况 B: 完整日期 -> 强转类型比较
                    val_fmt = f"'{val_str}'^^xsd:dateTime"
                    filter_clause = f"FILTER(?val {constraint.operator} {val_fmt})"

                elif is_number:
                    # 情况 C: 普通数值 -> 直接比较
                    val_fmt = val_str
                    filter_clause = f"FILTER(?val {constraint.operator} {val_fmt})"

                elif constraint.operator == "contains":
                    # 情况 D: 字符串包含
                    filter_clause = f"FILTER(CONTAINS(LCASE(?val), LCASE('{val_str}')))"
                else:
                    # 情况 E: 字符串相等 (默认)
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
        此方法返回一个新的 Constraint 对象，而不是修改原来的。
        """
        logger.info(f"[Tool: Refine] Relaxing constraint {constraint.property_label}")

        # 深拷贝以防修改原始引用
        new_c = copy.deepcopy(constraint)

        try:
            # 简单的松弛逻辑示例
            # 1. 如果是数值比较，放宽 10%
            if new_c.operator in [">", "<"] and re.match(r'^\d+(\.\d+)?$', str(new_c.value)):
                val = float(new_c.value)
                if new_c.operator == "<":
                    new_c.value = str(val * 1.1)  # 放宽上限
                    logger.info(f"  -> Relaxed '<' value to {new_c.value}")
                elif new_c.operator == ">":
                    new_c.value = str(val * 0.9)  # 放宽下限
                    logger.info(f"  -> Relaxed '>' value to {new_c.value}")

            # 2. 如果是精确匹配，尝试改为模糊匹配 (Contains)
            elif new_c.operator == "=" and not re.match(r'^Q\d+$', str(new_c.value)):
                new_c.operator = "contains"
                logger.info(f"  -> Relaxed operator '=' to 'contains'")

            else:
                logger.warning("  -> No relaxation strategy available for this constraint type.")

            return new_c

        except Exception as e:
            logger.error(f"[Tool: Refine] Failed: {e}")
            return constraint  # 出错则返回原约束