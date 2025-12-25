# critic.py
from typing import List, Dict
from data_model import Constraint
from optimizer import ConstraintOptimizer


class StatisticalCritic:
    """
    评价器：利用 Optimizer 中的数学指标 (s, r, CR) 指导 LLM。
    """

    def __init__(self, optimizer: ConstraintOptimizer):
        self.optimizer = optimizer

    def evaluate_constraints(self, constraints: List[Constraint]) -> str:
        """
        分析待处理的约束，返回建议文本。
        """
        # 1. 注入元数据 (复用 optimizer 逻辑)
        for c in constraints:
            self.optimizer._inject_metadata(c)
            # 注意：此处也可选择性调用 _estimate_and_fuse (涉及 LLM 开销)
            self.optimizer._calculate_priority(c)

        # 2. 生成基于数学的建议
        advice = "Statistical Critic Suggestions:\n"

        # 策略 A: 推荐 Anchor
        sorted_cons = sorted(constraints, key=lambda x: x.priority_score, reverse=True)
        best = sorted_cons[0]
        advice += f"  1. Recommended Starting Point (Anchor): '{best.property_label}' (Score: {best.priority_score:.2f}). High selectivity.\n"

        # 策略 B: 警告低效操作
        for c in sorted_cons:
            if c.s_base < 0.1 and c.lambda_val > 0.5:
                advice += f"  2. WARNING: '{c.property_label}' is very common (Low Selectivity). Avoid using it as a filter early on.\n"

        # 策略 C: 识别软约束
        soft_ones = [c for c in constraints if c.softness > 0.5]
        if soft_ones:
            labels = ", ".join([c.property_label for c in soft_ones])
            advice += f"  3. Refinement Hint: If results are empty, consider relaxing: {labels}.\n"

        return advice