"""模块1.1 农业因果事件识别

包含：
- prompts: LLM prompt 模板（Event Causality Is Key 思路 + 农业few-shot）
- event_identifier: 事件+因果对联合抽取
- concept_aggregator: 事件概念聚合（纯规则同义词归一）
"""

from .event_identifier import EventIdentifier
from .concept_aggregator import ConceptAggregator

__all__ = ["EventIdentifier", "ConceptAggregator"]
