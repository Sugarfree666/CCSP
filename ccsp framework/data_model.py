from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Constraint:
    # 基础信息
    id: str
    property_id: str
    property_label: str
    operator: str
    value: str
    unit: Optional[str] = None
    # 用户意图
    softness: float = 0.0

    # === [NEW] 动态探测结果 ===
    # -1 表示未探测，999999999 表示超时/代价无穷大
    estimated_rows: int = -1

    # 最终排序分 (基于 estimated_rows 计算)
    priority_score: float = 0.0

    def __repr__(self):
        return (f"<Constraint {self.property_label} {self.operator} {self.value} | "
                f"Rows={self.estimated_rows}, Score={self.priority_score:.3f}>")


@dataclass
class ExecutionPlan:
    """
    执行计划：包含排序后的约束列表和元数据
    """
    constraints: list[Constraint]
    reasoning_trace: str = ""