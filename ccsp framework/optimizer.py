import json
import logging
from typing import List, Dict
from data_model import Constraint

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstraintOptimizer:
    def __init__(self, metadata_path: str, llm_service):
        """
        :param metadata_path: property_metadata_final.json 的路径
        :param llm_service: 提供 generate_json(prompt) 方法的 LLM 服务实例
        """
        self.llm = llm_service
        self.metadata = self._load_metadata(metadata_path)

        # === [重构] 移除冗余的 ALPHA/BETA 权重 ===
        # 我们将采用 "乘法漏斗" 模型，不再做加法权衡。
        self.K_SOFT = 0.5  # Softness 衰减系数保持不变

    def _load_metadata(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("properties", {})
        except FileNotFoundError:
            logger.error(f"Metadata file not found at {path}")
            return {}

    def optimize(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        主入口：对传入的约束列表进行 补全 -> 估算 -> 融合 -> 排序
        """
        logger.info("Starting constraint optimization phase (Refactored)...")

        for c in constraints:
            # 1. 注入统计元数据 (Lookup)
            self._inject_metadata(c)

            # 2. LLM 语义估算 (Zero-shot Estimation) & 融合
            self._estimate_and_fuse(c)

            # 3. 计算最终优先级分数 (Scoring) - [核心修改点]
            self._calculate_priority(c)

        # 4. 排序 (Sorting) - 降序
        sorted_constraints = sorted(constraints, key=lambda x: x.priority_score, reverse=True)

        logger.info(f"Optimization complete. Order: {[c.property_label for c in sorted_constraints]}")
        return sorted_constraints

    def _inject_metadata(self, c: Constraint):
        """步骤 1: 查表获取 r, s_base, lambda"""
        meta = self.metadata.get(c.property_id)
        if meta:
            c.r_score = meta.get('r', 0.5)
            c.s_base = meta.get('s_base', 0.1)  # 默认降低 s_base，假设未知属性都不具备高区分度
            c.lambda_val = meta.get('lambda', 0.8)
        else:
            logger.warning(f"Property {c.property_id} not found. Using defaults.")
            c.r_score = 0.5
            c.s_base = 0.1
            c.lambda_val = 0.8

    def _estimate_and_fuse(self, c: Constraint):
        """步骤 2: 保持原有的融合逻辑 (这是 Hybrid Reasoning 的精华，不冗余)"""

        # 极速通道：如果是 ID 类属性 (High Cardinality)，直接信任统计数据，跳过 LLM
        if c.lambda_val < 0.1:
            logger.debug(f"Skipping LLM for {c.property_label} (ID-like).")
            weight = 0.0
            c.s_raw = 0.0
        else:
            estimation = self._call_llm_estimator(c)
            if not estimation:
                c.s_raw = 0.5
                c.confidence = 0.0
            else:
                c.s_raw = estimation.get('filtering_power', 50) / 100.0
                c.confidence = estimation.get('confidence', 0) / 100.0

            weight = c.lambda_val * c.confidence

        # 融合公式: s(v) = W * s_raw + (1 - W) * s_base
        c.s_final = (weight * c.s_raw) + ((1 - weight) * c.s_base)

    def _call_llm_estimator(self, c: Constraint) -> Dict:
        """LLM 估算器 (保持不变)"""
        val_str = str(c.value)
        prompt = f"""
        Role: Database Optimizer.
        Task: Estimate Filtering Power (0-100) for Property: "{c.property_label}" Value: "{val_str}".

        Scale:
        0-20: Common (e.g. "Male", "Human") -> Removes almost nothing.
        21-60: Moderate (e.g. "New York") -> Removes some.
        61-90: Rare (e.g. "Svalbard") -> Very specific.
        91-100: Unique ID.

        Output JSON ONLY: {{ "filtering_power": <int>, "confidence": <int> }}
        """
        return self.llm.generate_json(prompt)

    def _calculate_priority(self, c: Constraint):
        """
        [重构核心] 步骤 4: 计算综合评分
        不再使用加法公式 (Alpha*S + Beta*R)，改用乘法惩罚模型。
        """

        # === 1. 起评分：区分度 (Selectivity) ===
        # 这是 Anchor 的灵魂。区分度越高，分数越高。
        score = c.s_final

        # === 2. 结构性惩罚 (Structural Penalties) ===
        # 即使区分度看起来不错，如果是以下情况，必须强制杀分：

        # A. 泛型属性惩罚 (Generic Property Penalty)
        # P31 (instance of) 和 P279 (subclass of) 除非 Value 极度罕见，否则通常是糟糕的 Anchor。
        if c.property_id in ["P31", "P279"]:
            logger.info(f"Penalty: Generic property '{c.property_label}' detected.")
            score *= 0.2  # 极其严厉的惩罚 (降为原分的 20%)

        # B. 操作符惩罚 (Operator Penalty)
        # 范围查询 (>, <) 和 模糊查询 (contains) 的搜索空间通常巨大。
        if c.operator in [">", "<", ">=", "<="]:
            logger.info(f"Penalty: Range operator '{c.operator}' on '{c.property_label}'.")
            score *= 0.1  # 范围查询几乎不配做 Anchor
        elif c.operator == "contains":
            score *= 0.5

        # C. LLM 恐慌惩罚 (Panic Penalty)
        # 如果 LLM 明确觉得这个值很常见 (s_raw < 0.2)，无论统计数据如何，都要压分。
        if hasattr(c, 's_raw') and c.s_raw < 0.2:
            score *= 0.5

        # === 3. 奖励加成 (Bonuses) ===

        # A. 实体 ID 奖励
        # 如果 Value 是明确的 QID (例如 Q12345)，且不是泛型属性，它通常是最好的切入点。
        is_qid = str(c.value).startswith("Q") and c.value[1:].isdigit()
        if is_qid and c.property_id not in ["P31", "P279"]:
            score *= 1.5  # 给予 50% 的加成

        # B. 可靠性微调 (Reliability Nudge)
        # 我们不再让 R 占主导，但如果两个选项区分度差不多，选那个覆盖率更高(R高)的。
        # 这是一个温和的 Tie-breaker。
        score *= (1.0 + 0.2 * c.r_score)

        # === 4. 用户意图衰减 ===
        soft_factor = 1.0 - (self.K_SOFT * c.softness)

        c.priority_score = score * soft_factor

        # 调试日志
        logger.debug(f"Scored {c.property_label}: {c.priority_score:.3f} (base_s={c.s_final:.2f})")