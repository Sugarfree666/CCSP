# graph_state.py
from typing import List, Set, Dict, Optional, Any
from data_model import Constraint


class ThoughtNode:
    """
    思维节点：代表推理过程中的一个中间状态。
    对应 GoT 中的顶点 (Vertex)。
    """

    def __init__(self, node_id: str, description: str, candidates: Set[str], parent_ids: List[str] = None):
        self.node_id = node_id
        self.description = description  # 语义描述，如 "Movies starring Chester"
        self.candidates = candidates  # 实体集合 (QIDs)
        self.parent_ids = parent_ids or []  # 依赖的前置节点 ID
        self.score = 0.0  # 节点的质量评分 (基于 Optimizer)
        self.is_terminal = False  # 是否是最终答案候选

    def __repr__(self):
        return f"<Node {self.node_id}: {len(self.candidates)} candidates | {self.description}>"


class GraphState:
    """
    图状态：维护当前的推理全景图。
    """

    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[tuple] = []  # (parent, child)
        self.history: List[str] = []  # 记录 Agent 的操作历史

    def add_node(self, node: ThoughtNode):
        self.nodes[node.node_id] = node
        for pid in node.parent_ids:
            self.edges.append((pid, node.node_id))

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        return self.nodes.get(node_id)

    def get_summary(self) -> str:
        """生成供 LLM 阅读的图状态摘要"""
        summary = "Current Graph State:\n"
        if not self.nodes:
            return summary + "  (Empty Graph)\n"

        for nid, node in self.nodes.items():
            parents = f" <- {node.parent_ids}" if node.parent_ids else " (Root)"
            summary += f"  - [{nid}] {node.description}: Found {len(node.candidates)} entities.{parents}\n"
        return summary