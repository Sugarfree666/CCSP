from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Constraint:
    """
    约束对象模型 (Constraint Object Model)
    代表一个原子约束条件：(Property, Operator, Value)
    """
    # 基础信息
    id: str  # 唯一标识符
    property_id: str  # e.g., "P19"
    property_label: str  # e.g., "place of birth"
    operator: str  # e.g., "=", ">", "contains"
    value: str  # e.g., "Svalbard"

    # 用户意图 (Scheme 1)
    softness: float = 0.0  # 0.0 (Hard) -> 1.0 (Soft)

    # 统计元数据 (Scheme 2 - From Metadata Table)
    r_score: float = 0.0  # Reliability (Density)
    s_base: float = 0.2  # Base Selectivity (Log-Normalized)
    lambda_val: float = 0.8  # Adaptive Weight (Based on Linear CR)

    # 语义估算 (Scheme 1 - From LLM)
    s_raw: float = 0.0  # LLM Estimated Selectivity (0.0-1.0)
    confidence: float = 0.0  # LLM Confidence (0.0-1.0)
    llm_reasoning: str = ""  # Chain of Thought trace

    # 融合结果
    s_final: float = 0.0  # 最终区分度 (Fused)
    priority_score: float = 0.0  # 执行优先级分数

    def __repr__(self):
        return (f"<Constraint {self.property_label}={self.value} | "
                f"Score={self.priority_score:.3f} (s={self.s_final:.2f}, r={self.r_score:.2f})>")


@dataclass
class ExecutionPlan:
    """
    执行计划：包含排序后的约束列表和元数据
    """
    constraints: list[Constraint]
    reasoning_trace: str = ""