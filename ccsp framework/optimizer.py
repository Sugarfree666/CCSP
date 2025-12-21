import json
import math
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

        # === 超参数配置 (Phase 1: Retrieval/Planning) ===
        self.ALPHA = 0.6  # 权重: Selectivity (s)
        self.BETA = 0.4  # 权重: Reliability (r)
        self.K_SOFT = 0.5  # 乘法衰减系数: Softness (soft)

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
        logger.info("Starting constraint optimization phase...")

        for c in constraints:
            # 1. 注入统计元数据 (Lookup)
            self._inject_metadata(c)

            # 2. LLM 语义估算 (Zero-shot Estimation) & 融合
            self._estimate_and_fuse(c)

            # 3. 计算最终优先级分数 (Scoring)
            self._calculate_priority(c)

        # 4. 排序 (Sorting)
        # 按分数降序排列
        sorted_constraints = sorted(constraints, key=lambda x: x.priority_score, reverse=True)

        logger.info(f"Optimization complete. Execution Order: {[c.property_label for c in sorted_constraints]}")
        return sorted_constraints

    def _inject_metadata(self, c: Constraint):
        """步骤 1: 查表获取 r, s_base, lambda"""
        meta = self.metadata.get(c.property_id)
        if meta:
            c.r_score = meta.get('r', 0.5)
            c.s_base = meta.get('s_base', 0.2)
            c.lambda_val = meta.get('lambda', 0.8)  # 这里的 lambda 基于 Linear CR
        else:
            # 冷启动：如果属性不在元数据表中，使用默认保守值
            logger.warning(f"Property {c.property_id} not found in metadata. Using defaults.")
            c.r_score = 0.5
            c.s_base = 0.2
            c.lambda_val = 0.8

    def _estimate_and_fuse(self, c: Constraint):
        """步骤 2 & 3: LLM 估算与贝叶斯融合"""

        # === Gating Mechanism (门控机制) ===
        # 如果 lambda 很小 (说明是 ID 类属性，CR高)，则跳过 LLM，直接信表
        # 这是一个科研创新点：Computation-Efficient Hybrid Reasoning
        if c.lambda_val < 0.1:
            logger.info(f"Skipping LLM for {c.property_label} (High-CR/ID attribute).")
            weight = 0.0
            c.s_raw = 0.0  # 占位，不影响计算
            c.confidence = 0.0
        else:
            # 调用 LLM 进行估算
            estimation = self._call_llm_estimator(c)

            # 防御性编程：处理 LLM 返回空字典的情况
            if not estimation:
                c.s_raw = 0.5
                c.confidence = 0.0
                c.llm_reasoning = "LLM failed to estimate."
            else:
                c.s_raw = estimation.get('filtering_power', 50) / 100.0
                c.confidence = estimation.get('confidence', 0) / 100.0
                c.llm_reasoning = estimation.get('reasoning', '')

            # 计算有效语义权重 W = lambda * Confidence
            weight = c.lambda_val * c.confidence

        # === Fusion Formula (融合公式) ===
        # s(v) = W * s_raw + (1 - W) * s_base
        c.s_final = (weight * c.s_raw) + ((1 - weight) * c.s_base)

        logger.debug(
            f"Fused {c.property_label}: Base={c.s_base:.2f}, LLM={c.s_raw:.2f} (Conf={c.confidence:.2f}), Final={c.s_final:.2f}")

    def _call_llm_estimator(self, c: Constraint) -> Dict:
        """构建通用 Prompt 并解析结果"""

        # 1. 强制类型转换：防止数字型 Value (如 109.5) 导致拼接错误
        val_str = str(c.value)

        prompt = f"""
        Role: You are a Database Query Optimizer expert in semantic selectivity estimation.
        Task: Analyze the "Filtering Power" of a specific VALUE for a given PROPERTY.

        Context:
        - Property: "{c.property_label}" (ID: {c.property_id})
        - Value: "{val_str}"

        Definitions:
        - Filtering Power (0-100): How effectively does this value narrow down the search space?
          * 0-20: Very Common (e.g., generic concepts, "Male"). Removes almost nothing.
          * 21-60: Moderate (e.g., common entities, "New York"). Removes some, but millions remain.
          * 61-90: Rare (e.g., specific entities, "Svalbard"). Very specific.
          * 91-100: Unique (e.g., Specific ID). Pinpoints a single entity.
        - Confidence (0-100): How sure are you about this real-world knowledge?

        Output JSON ONLY:
        {{
          "reasoning": "Brief explanation...",
          "filtering_power": <int>,
          "confidence": <int>
        }}
        """

        # 2. 真实调用 LLM (无任何 Mock)
        return self.llm.generate_json(prompt)

    def _calculate_priority(self, c: Constraint):
        """步骤 4: 计算综合评分"""
        # 基础分: 区分度 + 可靠性
        base_score = (self.ALPHA * c.s_final) + (self.BETA * c.r_score)

        # Softness 衰减 (Dampening)
        # 如果 soft=1.0 (软约束)，分数会打折 (e.g., * 0.5)
        # 这是为了体现 Constraint Object Model 中软硬约束的数学区分
        soft_factor = 1.0 - (self.K_SOFT * c.softness)

        c.priority_score = base_score * soft_factor