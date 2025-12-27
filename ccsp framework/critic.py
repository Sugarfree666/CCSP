from typing import List
from data_model import Constraint

class StatisticalCritic:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def evaluate_constraints(self, constraints: List[Constraint]) -> str:
        """
        [Refactored] 基于探测到的真实行数生成建议
        """
        # 确保已经探测过
        if constraints and constraints[0].estimated_rows == -1:
             constraints = self.optimizer.optimize(constraints)

        advice = "Dynamic Probing Analysis:\n"

        # 1. 最佳切入点
        best = constraints[0]
        if best.estimated_rows < 1000:
            advice += f"  1. [STRONG ANCHOR] '{best.property_label}' is excellent. It yields only {best.estimated_rows} results.\n"
        elif best.estimated_rows < 10000:
            advice += f"  1. [ACCEPTABLE ANCHOR] '{best.property_label}' yields {best.estimated_rows} results. Use it if no better option.\n"
        else:
            advice += f"  1. [CAUTION] No highly selective anchor found. Best is '{best.property_label}' ({best.estimated_rows} rows).\n"

        # 2. 警告信息
        for c in constraints:
            if c.estimated_rows == 999_999_999:
                advice += f"  - WARNING: '{c.property_label}' is too expensive or timed out. Apply as late as possible.\n"
            elif c.estimated_rows > 100_000:
                 advice += f"  - NOTE: '{c.property_label}' has {c.estimated_rows} results. Inefficient as a filter.\n"

        return advice