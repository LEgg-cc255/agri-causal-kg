"""模块1.2 农业因果关系确认（纯规则后处理：本体约束 + 知识库增强 + 置信度评分）

设计依据（知识库增强方案 + Paths to Causality metapath 思路）：
- 在本体方向校验（entity_class级别）基础上，增加知识库校验（concept级别）
- 知识库提供 concept 级别的因果先验：
    已知因果 → 置信度提升（增强信号，取 max 避免拉低）
    已知禁止 → 置信度温和降权（不直接过滤，避免误杀）
    未知      → 不影响（避免伤害 Recall）
- 两层校验：本体校验（粗粒度） + 知识库校验（细粒度）

核心流程：
    模块1.1的因果对 → 本体方向校验 → 知识库校验 → 置信度评分 → 最终因果关系
"""

from __future__ import annotations

import logging
from typing import Any

from .confidence_scorer import ConfidenceScorer
from ..utils.knowledge_base import CausalKnowledgeBase

logger = logging.getLogger(__name__)


class CausalExtractor:
    """农业因果关系确认器（纯规则后处理 + 知识库增强）。

    Parameters
    ----------
    scorer : ConfidenceScorer, optional
        置信度评分器。未提供时内部新建。
    knowledge_base : CausalKnowledgeBase, optional
        农业因果规则知识库。未提供时内部新建。
    """

    def __init__(
        self,
        scorer: ConfidenceScorer | None = None,
        knowledge_base: CausalKnowledgeBase | None = None,
    ) -> None:
        self.scorer = scorer or ConfidenceScorer()
        self.kb = knowledge_base or CausalKnowledgeBase()
        if self.kb.is_loaded:
            logger.info("CausalExtractor 知识库增强已启用: %s", self.kb.stats)

    # ------------------------------------------------------------------ #
    #  主接口
    # ------------------------------------------------------------------ #
    def confirm_relation(
        self,
        cause: dict[str, str],
        effect: dict[str, str],
        context: str = "",
        relation_type_hint: str = "DIRECT",
        evidence: str = "",
        llm_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """确认单对因果关系（本体校验 + 知识库校验 + 置信度评分）。

        Parameters
        ----------
        cause, effect : dict
            因果事件结构（含 text/type/concept/sub_class_id/entity_class）。
        context : str
            原文上下文。
        relation_type_hint : str
            模块1.1给出的关系类型提示。
        evidence : str
            模块1.1给出的证据片段。
        llm_confidence : float
            模块1.1 LLM给出的置信度。

        Returns
        -------
        dict
            {
              "cause": {...}, "effect": {...},
              "is_causal": bool,
              "relation_type": str,
              "confidence": float,
              "reasoning": str,
              "evidence": str,
              "kb_match": {...},  # 知识库匹配结果
            }
        """
        # 1. 本体方向校验（entity_class 级别，粗粒度）
        ontology_result = self._ontology_check(cause, effect)

        # 2. 知识库校验（concept 级别，细粒度）
        kb_result = self._knowledge_check(cause, effect)

        # 3. 综合判定 is_causal
        # 知识库禁止方向不直接过滤，而是通过降低置信度让评分系统自然处理
        # （避免误杀正确因果对，FP主要来自"合理但未标注"的因果对，非反向因果）
        if not ontology_result["is_causal"]:
            is_causal = False
            reasoning = ontology_result["reason"]
        else:
            is_causal = True
            # 综合理由
            parts = [ontology_result["reason"]]
            if kb_result["known"]:
                parts.append(f"知识库已知因果(strength={kb_result['strength']})")
            elif kb_result["forbidden"]:
                parts.append(f"知识库禁止方向(penalty={kb_result['strength']})")
            reasoning = "；".join(parts)

        # 4. 关系类型校准
        relation_type = relation_type_hint if is_causal else "NONE"

        # 5. 置信度评分（3维：LLM + 本体 + 证据）
        confidence = self.scorer.score(
            is_causal=is_causal,
            evidence=evidence,
            llm_confidence=llm_confidence,
            context=context,
            cause_class=cause.get("entity_class", "") or cause.get("event_class", ""),
            effect_class=effect.get("entity_class", "") or effect.get("event_class", ""),
        )

        # 6. metapath 温和调整（Paths to Causality 思路，不改变 is_causal）
        # 已知因果：置信度取 max（避免加权拉低高置信度对）
        if kb_result["known"]:
            confidence = max(confidence, kb_result["strength"])
        # 禁止方向：温和降权（最多降30%，不归零，避免误杀正确因果对）
        elif kb_result["forbidden"]:
            confidence *= (1.0 - kb_result["strength"] * 0.3)
            confidence = max(0.0, min(1.0, confidence))

        return {
            "cause": cause,
            "effect": effect,
            "is_causal": is_causal,
            "relation_type": relation_type,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
            "evidence": evidence,
            "kb_match": {
                "known": kb_result["known"],
                "forbidden": kb_result["forbidden"],
                "strength": kb_result["strength"],
                "source": kb_result["source"],
            },
        }

    def confirm_batch(
        self, causal_pairs: list[dict[str, Any]], context: str = ""
    ) -> list[dict[str, Any]]:
        """批量确认因果关系对。

        Parameters
        ----------
        causal_pairs : list of dict
            模块1.1输出的因果对列表，每项含 cause/effect/relation_type/evidence/confidence。
        context : str
            原文上下文。
        """
        results: list[dict[str, Any]] = []
        for p in causal_pairs:
            res = self.confirm_relation(
                cause=p.get("cause", {}),
                effect=p.get("effect", {}),
                context=context,
                relation_type_hint=p.get("relation_type", "DIRECT"),
                evidence=p.get("evidence", ""),
                llm_confidence=p.get("confidence", 0.7),
            )
            # 透传 crop 字段（作物主体标注，用于建 INFLUENCE 边 + 前端展示）
            res["crop"] = p.get("crop", "")
            results.append(res)
        return results

    # ------------------------------------------------------------------ #
    #  内部：本体约束快速校验（entity_class 级别）
    # ------------------------------------------------------------------ #
    def _ontology_check(self, cause: dict[str, str], effect: dict[str, str]) -> dict[str, Any]:
        """基于本体(AKO-F v1.9)的因果方向校验。

        利用实体大类(entity_class)的因果方向约束做判定：
        - 环境因子/防治措施 → 功能症状/生物胁迫  : 合理方向
        - 功能症状 → 功能症状                    : 合理方向
        - 功能症状 → 环境因子                    : 不合理方向
        - 未知方向：默认保留，由置信度评分惩罚
        """
        cause_class = cause.get("entity_class", "") or cause.get("event_class", "")
        effect_class = effect.get("entity_class", "") or effect.get("event_class", "")

        # 合理因果方向（cause_class → effect_class）
        valid_directions = {
            ("环境因子", "功能症状"),
            ("环境因子", "生物胁迫"),
            ("环境因子", "防治措施"),
            ("防治措施", "功能症状"),
            ("防治措施", "环境因子"),
            ("防治措施", "生物胁迫"),
            ("生物胁迫", "功能症状"),
            ("功能症状", "功能症状"),
        }

        if (cause_class, effect_class) in valid_directions:
            return {
                "is_causal": True,
                "reason": f"本体方向合理: {cause_class} → {effect_class}",
            }

        # 明显不合理的方向
        invalid_directions = {
            ("功能症状", "环境因子"),
            ("功能症状", "防治措施"),
            ("功能症状", "生物胁迫"),
            ("生物胁迫", "环境因子"),
            ("生物胁迫", "防治措施"),
        }
        if (cause_class, effect_class) in invalid_directions:
            return {
                "is_causal": False,
                "reason": f"本体方向不合理: {cause_class} → {effect_class}",
            }

        # 未知方向（含 UNKNOWN）：默认保留，置信度评分会惩罚
        return {
            "is_causal": True,
            "reason": "本体无法判定，保留待置信度评分",
        }

    # ------------------------------------------------------------------ #
    #  内部：知识库校验（concept 级别）
    # ------------------------------------------------------------------ #
    def _knowledge_check(self, cause: dict[str, str], effect: dict[str, str]) -> dict[str, Any]:
        """基于农业因果规则知识库的 concept 级别校验。

        三种结果：
        - known=True:  concept 对在已知因果规则中 → 增强信号
        - forbidden=True: concept 对在禁止方向中 → 过滤信号
        - 两者皆 False: 知识库未覆盖 → 不影响判定
        """
        cause_concept = cause.get("concept", "")
        effect_concept = effect.get("concept", "")

        if not cause_concept or not effect_concept:
            return {"known": False, "forbidden": False, "strength": 0.5, "source": "概念缺失"}

        result = self.kb.lookup(cause_concept, effect_concept)
        return {
            "known": result["known"],
            "forbidden": result["forbidden"],
            "strength": result["strength"],
            "source": result["source"],
        }
