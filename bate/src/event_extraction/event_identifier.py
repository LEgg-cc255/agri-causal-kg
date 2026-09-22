"""模块1.1 农业因果事件识别（事件+因果对+因果链联合抽取）

设计依据（精简方案）：
- 参考 Event Causality Is Key (NAACL 2024) 的 prompt 框架
- 一次 LLM 调用同时完成事件识别 + 因果对抽取 + 因果链构建（避免多阶段串联）
- 用 agri_event_types.json 本体约束 LLM 输出概念标签
- 农业领域 few-shot 示例来自真实标注数据集

核心流程：
    农业文本 → LLM(单次prompt) → 原始事件+因果对+因果链 → 概念聚合校准 → 结构化输出
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils.llm_client import LLMClient
from .prompts import build_extraction_messages
from .concept_aggregator import ConceptAggregator

logger = logging.getLogger(__name__)


class ExtractedEvent(dict):
    """抽取事件的字典结构（便于类型提示）。"""

    __slots__ = ()


class CausalPair(dict):
    """因果关系对的字典结构。"""


class EventIdentifier:
    """农业因果事件识别器。

    Parameters
    ----------
    llm_client : LLMClient
        LLM 客户端实例（可为 Mock 模式）。
    aggregator : ConceptAggregator, optional
        概念聚合器，用于对 LLM 输出的概念做后校准。未提供时内部新建。
    num_fewshot : int
        few-shot 示例数量，默认 2。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        aggregator: ConceptAggregator | None = None,
        num_fewshot: int = 1,
    ) -> None:
        self.llm = llm_client
        self.aggregator = aggregator or ConceptAggregator()
        self.num_fewshot = num_fewshot

    # ------------------------------------------------------------------ #
    #  主接口
    # ------------------------------------------------------------------ #
    def extract(self, text: str) -> dict[str, Any]:
        """从农业文本中抽取事件、因果关系对与因果链。

        Parameters
        ----------
        text : str
            农业文本段落。

        Returns
        -------
        dict
            {
              "text": 原文,
              "events": [ {text, type, concept, sub_class_id, entity_class} ],
              "causal_pairs": [ {cause:{...}, effect:{...}, relation_type, confidence, evidence} ],
              "causal_chains": [ {chain:[concept,...], events:[text,...], confidence} ],
              "raw_llm_output": 原始LLM输出（调试用）
            }
        """
        if not text or not text.strip():
            return {"text": text, "events": [], "causal_pairs": [], "causal_chains": [], "raw_llm_output": ""}

        # 1. 构建prompt并调用LLM（单次调用完成全部抽取）
        messages = build_extraction_messages(text, num_fewshot=self.num_fewshot)
        try:
            result = self.llm.chat_json(messages)
        except Exception as e:  # noqa: BLE001
            logger.error("LLM 抽取失败: %s", e)
            return {
                "text": text, "events": [], "causal_pairs": [], "causal_chains": [],
                "raw_llm_output": "", "error": str(e),
            }

        # 2. 标准化结构
        raw_events = self._normalize_events(result.get("events", []))
        raw_pairs = self._normalize_pairs(result.get("causal_pairs", []))
        chains = self._normalize_chains(result.get("causal_chains", []))

        # 做副本——聚合不污染原始值
        import copy
        events = copy.deepcopy(raw_events)
        pairs = copy.deepcopy(raw_pairs)

        # 3. 概念聚合后校准（用同义词表对 LLM 输出的 concept 做归一化）
        for ev in events:
            calibrated = self.aggregator.aggregate(ev["text"], ev.get("concept", ""))
            ev["concept"] = calibrated["concept"]
            ev["sub_class_id"] = calibrated["sub_class_id"]
            ev["entity_class"] = calibrated["entity_class"]
            ev["sub_class_name"] = calibrated.get("sub_class_name", "")

        for pair in pairs:
            for role in ("cause", "effect"):
                ev = pair.get(role, {})
                calibrated = self.aggregator.aggregate(ev.get("text", ""), ev.get("concept", ""))
                ev["concept"] = calibrated["concept"]
                ev["sub_class_id"] = calibrated["sub_class_id"]
                ev["entity_class"] = calibrated["entity_class"]
                ev["sub_class_name"] = calibrated.get("sub_class_name", "")
                pair[role] = ev

        # 4. 对LLM输出的因果链做概念校准
        for chain in chains:
            calibrated_concepts: list[str] = []
            for c in chain.get("chain", []):
                cal = self.aggregator.aggregate(c)
                calibrated_concepts.append(cal["concept"])
            chain["chain"] = calibrated_concepts

        return {
            "text": text,
            "events": events,
            "causal_pairs": pairs,
            "causal_chains": chains,
            "raw_llm_output": result,
            "_debug_raw_events": raw_events,
            "_debug_raw_pairs": raw_pairs,
        }

    def extract_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """批量抽取。"""
        return [self.extract(t) for t in texts]

    # ------------------------------------------------------------------ #
    #  内部：结构标准化
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_events(raw_events: list) -> list[dict[str, Any]]:
        """标准化事件结构，过滤无效项。"""
        events: list[dict[str, Any]] = []
        for ev in raw_events:
            if not isinstance(ev, dict):
                continue
            text = (ev.get("text") or "").strip()
            if not text:
                continue
            etype = ev.get("type", "INTERMEDIATE")
            if etype not in ("CAUSE_EVENT", "EFFECT_EVENT", "INTERMEDIATE"):
                etype = "INTERMEDIATE"
            events.append({
                "text": text,
                "type": etype,
                "concept": (ev.get("concept") or "").strip(),
            })
        return events

    @staticmethod
    def _normalize_pairs(raw_pairs: list) -> list[dict[str, Any]]:
        """标准化因果对结构，过滤无效项。"""
        pairs: list[dict[str, Any]] = []
        for p in raw_pairs:
            if not isinstance(p, dict):
                continue
            cause = p.get("cause")
            effect = p.get("effect")
            if not isinstance(cause, dict) or not isinstance(effect, dict):
                continue
            cause_text = (cause.get("text") or "").strip()
            effect_text = (effect.get("text") or "").strip()
            if not cause_text or not effect_text:
                continue
            rtype = p.get("relation_type", "DIRECT")
            if rtype not in ("DIRECT", "INDIRECT", "CONDITIONAL"):
                rtype = "DIRECT"
            # confidence 容错
            try:
                conf = float(p.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            # crop 字段：LLM 可选标注该因果对涉及的作物主体（如 "番茄"），未标则空
            crop = (p.get("crop") or "").strip()
            pairs.append({
                "cause": {
                    "text": cause_text,
                    "type": cause.get("type", "CAUSE_EVENT"),
                    "concept": (cause.get("concept") or "").strip(),
                },
                "effect": {
                    "text": effect_text,
                    "type": effect.get("type", "EFFECT_EVENT"),
                    "concept": (effect.get("concept") or "").strip(),
                },
                "relation_type": rtype,
                "confidence": conf,
                "evidence": (p.get("evidence") or "").strip(),
                "crop": crop,
            })
        return pairs

    @staticmethod
    def _normalize_chains(raw_chains: list) -> list[dict[str, Any]]:
        """标准化因果链结构，过滤无效项。

        LLM 输出格式: {"concepts": [...], "events": [...], "confidence": float}
        标准化后: {"chain": [concept,...], "events": [text,...], "confidence": float}
        """
        chains: list[dict[str, Any]] = []
        for c in raw_chains:
            if not isinstance(c, dict):
                continue
            # 兼容 concepts / chain 两种字段名
            concepts = c.get("concepts") or c.get("chain") or []
            events = c.get("events") or []
            if not isinstance(concepts, list) or len(concepts) < 2:
                continue
            concepts = [str(x).strip() for x in concepts if str(x).strip()]
            if len(concepts) < 2:
                continue
            if not isinstance(events, list):
                events = []
            events = [str(x).strip() for x in events]
            try:
                conf = float(c.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            chains.append({
                "chain": concepts,
                "events": events,
                "confidence": conf,
                "is_valid": True,
                "reasoning": "LLM 在抽取阶段直接输出",
            })
        return chains
