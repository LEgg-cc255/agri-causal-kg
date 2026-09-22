"""研究内容2：农业因果知识图谱 Neo4j 客户端模块（AKO-F v1.9）。

使用：
    from bate.src.kg import AgriCausalGraph, EntityNode, CausalEdge
    from bate.src.kg import QuestionParser, QuestionParse
    from bate.src.kg import ExplanationGenerator
    from bate.src.kg import CausalRAGPipeline, CausalAnswer
"""

from .neo4j_client import (
    AgriCausalGraph,
    EntityNode,
    CausalEdge,
    determine_edge_type,
    LABEL_ENTITY,
    REL_TRIGGERS,
    REL_PROPAGATES,
    REL_ALLEVIATES,
    REL_FORBIDDEN,
    ALL_CAUSAL_RELS,
)
from .question_parser import QuestionParser, QuestionParse
from .explanation_generator import ExplanationGenerator
from .pipeline import CausalRAGPipeline, CausalAnswer

# 向后兼容别名（旧代码可能仍引用 EventNode）
EventNode = EntityNode

__all__ = [
    "AgriCausalGraph", "EntityNode", "EventNode", "CausalEdge", "determine_edge_type",
    "LABEL_ENTITY", "REL_TRIGGERS", "REL_PROPAGATES", "REL_ALLEVIATES",
    "REL_FORBIDDEN", "ALL_CAUSAL_RELS",
    "QuestionParser", "QuestionParse",
    "ExplanationGenerator",
    "CausalRAGPipeline", "CausalAnswer",
]
