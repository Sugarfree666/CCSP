import json
import logging
from typing import List, Dict, Any, Set
from data_model import Constraint
from graph_state import GraphState, ThoughtNode
from environment import GraphEnvironment
from critic import StatisticalCritic

logger = logging.getLogger(__name__)


class GoTAgent:
    def __init__(self, llm, tools: GraphEnvironment, critic: StatisticalCritic):
        self.llm = llm
        self.tools = tools
        self.critic = critic
        self.state = GraphState()
        self.max_steps = 15  # 稍微增加步数上限，以防复杂推理

    def solve(self, user_query: str, constraints: List[Constraint]):
        # 初始化节点：Root
        self.state.add_node(ThoughtNode("root", "Start", set()))
        constraint_map = {c.id: c for c in constraints}
        step = 0
        while step < self.max_steps:
            # 1. Observe: 获取当前状态
            graph_summary = self.state.get_summary()

            # 获取当前最新的节点信息，用于判断是否为空
            current_leaf_nodes = [node for node in self.state.nodes.values() if
                                  not any(edge[0] == node.node_id for edge in self.state.edges)]
            current_candidates_count = sum(len(n.candidates) for n in current_leaf_nodes) if current_leaf_nodes else 0

            # 2. Critic: 依然让 Critic 提供建议，但传入所有约束，让 Critic 评估整体优先级
            # 注意：Critic 还是基于数学计算优先级的，这对 LLM 决策很有帮助
            critic_advice = self.critic.evaluate_constraints(constraints)

            # 3. Think: 构建 Prompt
            # 关键修改：不再传入 partial list，而是传入所有 constraints，让 LLM 自己对照 History 判断
            prompt = self._build_prompt(user_query, graph_summary, critic_advice, constraints, step)

            # 4. Decide: LLM 决策
            action_json = self.llm.generate_json(prompt)

            # 5. Act: 执行工具
            # 注意：这里传入的是由 id 索引的完整约束字典

            result_node = self._execute_action(action_json, constraint_map)

            if result_node:
                self.state.add_node(result_node)
                self.state.history.append(f"Step {step}: {action_json.get('reasoning')}")

                # 终止条件：LLM 主动 FINISH
                if action_json.get("action") == "FINISH":
                    logger.info(f"Agent decided to FINISH at step {step}.")
                    return result_node.candidates
            else:
                # 如果执行失败（例如 Action 解析错误），记录日志但不 crash
                logger.warning(f"Step {step} action failed or returned None.")

            step += 1

            # 如果到了最后一步还没 finish，尝试返回最后的结果
            if step == self.max_steps:
                logger.warning("Max steps reached without FINISH.")
                if current_leaf_nodes:
                    return current_leaf_nodes[-1].candidates

        return set()

    def _build_prompt(self, query, graph, advice, constraints: List[Constraint], current_step: int) -> str:
        # 列出所有约束的定义，作为"工具书"供 LLM 参考
        definitions = "\n".join([f"- {c.id}: {c.property_label} {c.operator} {c.value}" for c in constraints])

        return f"""
        Role: You are an autonomous Graph of Thoughts Agent.
        Goal: Find the entity that satisfies ALL user constraints.

        User Query: "{query}"

        === Constraint Definitions (Reference) ===
        {definitions}

        === Current Graph State (History) ===
        {graph}

        === Statistical Critic Advice ===
        {advice}

        === Decision Instructions ===
        1. **ANALYZE HISTORY**: Look at the "Current Graph State". Which constraints have ALREADY been applied?
        2. **CHECK COMPLETION**: 
           - Do the current remaining candidates satisfy ALL "Constraint Definitions"? 
           - If you have 1-5 candidates left and you have applied all necessary filters, output "FINISH".
        3. **AVOID LOOPS**: Do NOT apply a constraint (FILTER/SEARCH) if it has already been applied in the current path.
        4. **NEXT STEP**: If constraints remain unfulfilled, choose the best one based on the Critic's advice.
        5.**HANDLE DEAD ENDS**: If a FILTER returns 0 entities, implies missing data or strict constraints. You MUST use 'RELAX_CONSTRAINT' on that constraint.

        Available Actions:
        1. SEARCH_ANCHOR(constraint_id): Start a new search path (Only if no good path exists).
        2. FILTER(parent_node_id, constraint_id): Apply a constraint to narrow down results.
        3. INTERSECT(node_id_1, node_id_2): Intersect two sets of candidates.
        4. FINISH(final_node_id): Return the final answer.
        5.RELAX_CONSTRAINT(constraint_id): **CRITICAL**. Use this if a FILTER yielded 0 results. It changes the constraint to 'IGNORE' so you can proceed.
        Output JSON:
        {{
            "reasoning": "Step-by-step reasoning: 1. I see constraints A, B, C are done. 2. Candidate count is X. 3. Therefore I will...",
            "action": "ACTION_NAME",
            "params": {{ ... }}
        }}
        """

    def _execute_action(self, action: dict, constraint_map: dict):
        act_type = action.get("action")
        params = action.get("params", {})

        try:
            if act_type == "SEARCH_ANCHOR":
                cid = params["constraint_id"]
                cons = constraint_map[cid]  # 注意这里变量名修正为 constraint_map 更好
                candidates = self.tools.tool_search_anchor(cons)
                return ThoughtNode(f"node_{cid}", f"Search {cons.property_label}", candidates, parent_ids=["root"])

            elif act_type == "FILTER":
                pid = params["parent_node_id"]
                cid = params["constraint_id"]
                parent = self.state.get_node(pid)
                cons = constraint_map[cid]
                if not parent:
                    logger.error(f"Parent node {pid} not found for FILTER.")
                    return None
                candidates = self.tools.tool_filter(parent.candidates, cons)
                return ThoughtNode(
                    f"node_{cid}",
                    f"Filter {cons.property_label}",
                    candidates,
                    parent_ids=[pid]
                )

            elif act_type == "RELAX_CONSTRAINT":
                cid = params.get("constraint_id")
                if cid not in constraint_map:
                    logger.error(f"Constraint {cid} not found.")
                    return None
                target_constraint = constraint_map[cid]
                relaxed_constraint = self.tools.tool_relax_constraint(target_constraint)
                # 这样下一次 loop 构建 Prompt 时，LLM 会看到这个约束变成了 IGNORE
                target_constraint.operator = relaxed_constraint.operator
                target_constraint.value = relaxed_constraint.value

                return ThoughtNode(
                    f"relax_{cid}",
                    f"Relaxed {cid} ({target_constraint.property_label}) -> {target_constraint.operator}",
                    set(),  # 这里不需要候选集，因为下一步通常是重试 Filter
                    parent_ids=[]
                )

            elif act_type == "INTERSECT":
                id1 = params["node_id_1"]
                id2 = params["node_id_2"]
                n1 = self.state.get_node(id1)
                n2 = self.state.get_node(id2)

                if not n1 or not n2:
                    return None

                candidates = self.tools.tool_intersect(n1.candidates, n2.candidates)
                return ThoughtNode(f"merge_{id1}_{id2}", "Intersection", candidates, parent_ids=[id1, id2])

            # === [修复] 新增 FINISH 处理逻辑 ===
            elif act_type == "FINISH":
                # 获取 LLM 指定的最终节点 ID
                final_node_id = params.get("final_node_id")
                target_node = self.state.get_node(final_node_id)

                # 如果 LLM 没传 ID 或 ID 错误，尝试使用最近的一个节点作为兜底
                if not target_node:
                    logger.warning(f"FINISH called with invalid node_id '{final_node_id}'. Using last node.")
                    if self.state.nodes:
                        last_id = list(self.state.nodes.keys())[-1]
                        target_node = self.state.nodes[last_id]

                return target_node
            # ===================================

            else:
                logger.warning(f"Unknown action type: {act_type}")
                return None

        except Exception as e:
            logger.error(f"Action Execution Failed: {e}", exc_info=True)
            return None