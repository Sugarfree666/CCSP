import json
import os
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Set

# === 1. 从 main.py 导入必要的类和函数 ===
from main import LLMService, parse_query_to_constraints

# === 2. 导入其他组件 ===
from wikidata_service import WikidataService
from optimizer import ConstraintOptimizer
from agent_brain import GoTAgent
from environment import GraphEnvironment
from critic import StatisticalCritic
from unit_utils import UnitNormalizer  # 务必导入这个

# === 配置日志 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evaluator")


class Evaluator:
    def __init__(self, dataset_path, limit):
        self.dataset_path = dataset_path
        self.limit = limit

        # === 初始化服务 (模拟 main.py 中的初始化逻辑) ===
        # 请确保环境变量已设置，或者在这里硬编码用于测试
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model_name = os.getenv("model_name")

        if not api_key:
            logger.warning("Environment variables for LLM not found. Please ensure LLM_API_KEY is set.")

        self.llm_service = LLMService(api_key, base_url, model_name)
        self.wiki_service = WikidataService()

        # 初始化核心组件
        self.optimizer = ConstraintOptimizer(self.wiki_service)
        self.env = GraphEnvironment(self.wiki_service)
        self.critic = StatisticalCritic(self.optimizer)
        self.normalizer = UnitNormalizer()  # 初始化单位标准化器

    def load_data(self):
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:self.limit]

    def compute_metrics(self, predicted: Set[str], gold: Set[str]):
        """计算 Precision, Recall, F1, EM"""
        tp = len(predicted.intersection(gold))  # True Positives
        fp = len(predicted - gold)  # False Positives
        fn = len(gold - predicted)  # False Negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        em = 1.0 if predicted == gold and len(gold) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "em": em,
            "predicted_count": len(predicted),
            "gold_count": len(gold),
            "tp": tp
        }

    def run_evaluation(self):
        data = self.load_data()
        results = []
        metrics_summary = {"precision": [], "recall": [], "f1": [], "em": []}

        logger.info(f"Starting evaluation on first {len(data)} samples...")

        for idx, entry in tqdm(enumerate(data), total=len(data)):
            query = entry['complex_question']

            # 兼容数据集格式：有的可能是 list，有的可能是 string
            gold_qids = set(entry['new_ground_truth']) if isinstance(entry['new_ground_truth'], list) else set(
                [entry['new_ground_truth']])

            start_time = time.time()
            error_msg = None
            pred_qids = set()

            try:
                # ==========================================================
                # 核心处理管线 (Pipeline) - 必须与 main.py 逻辑保持一致
                # ==========================================================

                # 1. Phase 1: Parsing
                constraints = parse_query_to_constraints(query, self.llm_service, self.wiki_service)

                # 2. Phase 1.5: Unit Normalization (关键步骤！)
                if constraints:
                    constraints = self.normalizer.normalize(constraints)

                # 3. Phase 2: Optimization
                # optimize 方法内部会进行探测(Probe)
                constraints = self.optimizer.optimize(constraints)

                if not constraints:
                    pred_qids = set()  # 解析失败或无有效约束
                else:
                    # 4. Phase 3: Agent Execution
                    # 每次重新实例化 Agent 以清除上一题的状态 (History)
                    agent = GoTAgent(self.llm_service, self.env, self.critic)
                    final_candidates = agent.solve(query, constraints)

                    if final_candidates:
                        pred_qids = set(final_candidates)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing query {idx}: {e}")
                pred_qids = set()

            duration = time.time() - start_time

            # === 计算指标 ===
            m = self.compute_metrics(pred_qids, gold_qids)

            # 记录详细结果
            record = {
                "id": idx,
                "source_id": entry.get("source_id", ""),
                "query": query,
                "gold": list(gold_qids),
                "predicted": list(pred_qids),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "em": m["em"],
                "duration": duration,
                "error": error_msg
            }
            results.append(record)

            metrics_summary["precision"].append(m["precision"])
            metrics_summary["recall"].append(m["recall"])
            metrics_summary["f1"].append(m["f1"])
            metrics_summary["em"].append(m["em"])

        # === 生成最终报告 ===
        avg_p = np.mean(metrics_summary["precision"])
        avg_r = np.mean(metrics_summary["recall"])
        avg_f1 = np.mean(metrics_summary["f1"])
        avg_em = np.mean(metrics_summary["em"])

        print("\n" + "=" * 30)
        print("EVALUATION REPORT")
        print("=" * 30)
        print(f"Total Samples: {len(data)}")
        print(f"Avg Precision: {avg_p:.4f}")
        print(f"Avg Recall:    {avg_r:.4f}")
        print(f"Avg F1 Score:  {avg_f1:.4f}")
        print(f"Exact Match:   {avg_em:.4f}")
        print("=" * 30)

        # 保存为 CSV
        df = pd.DataFrame(results)
        output_file = "evaluation_results.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    # 请修改为您本地的数据集路径
    DATASET_PATH = r"D:\GitHub\CCSP\datasets\complex_constraint_dataset_rewrite_queries.json"

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
    else:
        # 建议先跑 10 条测试一下
        evaluator = Evaluator(DATASET_PATH, limit=300)
        evaluator.run_evaluation()