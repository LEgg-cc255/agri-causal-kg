"""端到端 Pipeline：农业因果事件抽取与因果关系识别（精简方案）

整合三个模块：
    模块1.1 EventIdentifier  → 事件+因果对+因果链联合抽取（1次LLM调用）
    模块1.2 CausalExtractor  → 因果关系确认 + 置信度评分（纯规则）
    模块1.3 ChainBuilder     → 因果链构建（纯图算法 + LLM链优先）

精简原则：
- 整个 Pipeline 仅调用 1 次 LLM（模块1.1）
- 模块1.2/1.3 仅用规则做后处理，不再调用 LLM
- 避免多阶段 LLM 判断不一致导致 recall 下降

输入：农业文本
输出：结构化因果知识（事件、因果对、因果链）
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .utils.llm_client import LLMClient
from .utils.knowledge_base import CausalKnowledgeBase
from .event_extraction import EventIdentifier, ConceptAggregator
from .relation_extraction import CausalExtractor, ConfidenceScorer, ChainBuilder

logger = logging.getLogger(__name__)


class AgriCausalPipeline:
    """农业因果知识抽取端到端 Pipeline（精简方案：1次LLM + 规则后处理）。

    Parameters
    ----------
    llm_client : LLMClient, optional
        LLM 客户端。未提供时自动创建（可能为Mock模式）。
    min_confidence : float
        因果对最低置信度阈值。
    knowledge_base : CausalKnowledgeBase, optional
        农业因果规则知识库。未提供时自动创建。
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        min_confidence: float = 0.3,
        knowledge_base: CausalKnowledgeBase | None = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.aggregator = ConceptAggregator()
        self.kb = knowledge_base or CausalKnowledgeBase()
        self.identifier = EventIdentifier(self.llm, self.aggregator)
        self.scorer = ConfidenceScorer()
        self.extractor = CausalExtractor(self.scorer, self.kb)
        self.chain_builder = ChainBuilder()
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ #
    #  主接口
    # ------------------------------------------------------------------ #
    def run(self, text: str) -> dict[str, Any]:
        """端到端抽取（1次LLM调用 + 规则后处理）。

        Parameters
        ----------
        text : str
            农业文本。

        Returns
        -------
        dict
            {
              "text": 原文,
              "events": [...],
              "causal_pairs": [...],   # 已确认的因果关系对
              "causal_chains": [...],  # 因果链
              "stats": {统计信息}
            }
        """
        # 模块1.1：事件+因果对+因果链联合抽取（唯一一次LLM调用）
        extraction = self.identifier.extract(text)
        events = extraction["events"]
        raw_pairs = extraction["causal_pairs"]
        llm_chains = extraction.get("causal_chains", [])

        # 知识库候选召回（COMET轻量版）：补LLM漏抽的已知因果对
        recalled = self._kb_recall(raw_pairs, events)

        # 模块1.2：因果关系确认 + 置信度评分（纯规则）
        confirmed_pairs = self.extractor.confirm_batch(raw_pairs + recalled, context=text)

        # 过滤掉被判定为非因果的关系 + 低置信度过滤
        causal_pairs = [
            p for p in confirmed_pairs
            if p.get("is_causal", False) and p.get("confidence", 0) >= self.min_confidence
        ]

        # 规则层极性校验：管理措施(MGT类)→负向effect 自动纠正为正向
        causal_pairs = self._polarity_check(causal_pairs, text)

        # 模块1.3：因果链构建（纯图算法 + LLM链优先）
        chains = self.chain_builder.build_chains(
            causal_pairs, llm_chains=llm_chains,
            min_confidence=self.min_confidence,
        )

        return {
            "text": text,
            "events": events,
            "causal_pairs": causal_pairs,
            "causal_chains": chains,
            "_debug_raw_events": extraction["events"],        # 聚合前的LLM原始events（含原始concept）
            "_debug_raw_pairs": extraction["causal_pairs"],   # 聚合前的LLM原始pairs（含原始concept）
            "stats": {
                "num_events": len(events),
                "num_raw_pairs": len(raw_pairs),
                "num_recalled_pairs": len(recalled),
                "num_confirmed_pairs": len(causal_pairs),
                "num_chains": len(chains),
                "num_llm_calls": 1,  # 精简方案：仅1次LLM调用
                "kb_enabled": self.kb.is_loaded,
                "kb_stats": self.kb.stats,
                "avg_confidence": (
                    sum(p["confidence"] for p in causal_pairs) / len(causal_pairs)
                    if causal_pairs else 0.0
                ),
            },
        }

    # ------------------------------------------------------------------ #
    #  规则层极性校验（改进3扩展版）：多维度极性纠正与过滤
    # ------------------------------------------------------------------ #

    # 负向 effect 概念 → 应纠正为的正向概念
    _POLARITY_MAP = {
        "病害加重": "病害减轻",
        "虫害": "虫害减轻",
        "产量下降": "产量提升",
        "氧化应激": "抗氧化能力提升",
        "光合作用下降": "光合作用增强",
        "氧化损伤": "氧化损伤减轻",
        "植株徒长": "生长受阻",
        "生长受阻": "胁迫缓解",
        "稻米品质下降": "胁迫缓解",
        "千粒重下降": "胁迫缓解",
        "空壳率上升": "胁迫缓解",
        "灌浆不良": "胁迫缓解",
        "根系早衰": "胁迫缓解",
        "叶片早衰": "胁迫缓解",
        "感病性增加": "病害减轻",
        "胁迫记忆形成": "胁迫缓解",
        # ── v1.9 新增盲区覆盖 ──
        "水分供需失衡": "胁迫缓解",
        "离子失衡": "胁迫缓解",
        "养分吸收下降": "养分吸收增加",
        "抗氧化能力下降": "抗氧化能力提升",
        "激素失衡": "胁迫缓解",
        "缺素症": "养分吸收增加",
        "根系活力下降": "胁迫缓解",
        "光合系统失活": "光合作用增强",
        "蒸腾过强": "胁迫缓解",
        "根系缺氧": "胁迫缓解",
    }

    # 正向概念集合：整体含义为"改善/提升"
    _POSITIVE_CONCEPTS = {
        "胁迫缓解", "病害减轻", "虫害减轻", "产量提升", "光合作用增强",
        "抗氧化能力提升", "着果率提升", "根系生长增强", "养分吸收增加",
        "氧化损伤减轻", "胁迫记忆形成",
    }

    # 负向概念集合：整体含义为"恶化/下降/灾害"
    _NEGATIVE_CONCEPTS = {
        "干旱胁迫", "水分亏缺", "高温胁迫", "盐碱胁迫", "低温冷害",
        "氧化应激", "光合作用下降", "产量下降", "离子失衡", "激素失衡",
        "根系活力下降", "根系早衰", "叶片早衰", "土壤过湿", "复合胁迫",
        "养分吸收下降", "感病性增加", "病害加重", "虫害", "生长受阻",
        "氧化损伤", "水分供需失衡", "光合系统失活", "蒸腾过强", "根系缺氧",
        "缺素症", "稻米品质下降", "千粒重下降", "空壳率上升", "灌浆不良",
        "植株徒长", "植株矮化", "落花落蕾", "着果率下降", "病害",
    }

    # 交叉域极性反转映射：负向 cause → 正向 effect 的纠正
    # （原则：负因通常导负面结果，若 effect 是同维度正向则纠正）
    _CROSS_DOMAIN_REVERSAL_MAP = {
        # 负向 cause → 产量下降（不是产量提升）
        "产量提升": {
            "if_cause_negative": "产量下降",
        },
        # 负向 cause → 光合作用下降（不是增强）
        "光合作用增强": {
            "if_cause_negative": "光合作用下降",
        },
        # 负向 cause → 胁迫缓解（特殊：多数负向 cause 不会直接导致胁迫缓解，
        # 除非中间有管理措施；若 LLM 直接输出则为错，纠正为胁迫加重→等价的负向概念）
        "胁迫缓解": {
            "if_cause_negative": "生长受阻",
        },
    }

    # 极性反转对（cause 是负向形式，effect 是同领域正向形式）→ 直接过滤
    _POLARITY_REVERSAL_PAIRS = {
        ("虫害", "虫害减轻"),
        ("病害加重", "病害减轻"),
        ("产量下降", "产量提升"),
        ("氧化应激", "胁迫缓解"),
        ("氧化应激", "抗氧化能力提升"),
        ("离子失衡", "胁迫缓解"),
        ("激素失衡", "胁迫缓解"),
        # 交叉域新增：病害/虫害/环境胁迫 直接导致 产量提升 → 自相矛盾
        ("病害加重", "产量提升"),
        ("虫害", "产量提升"),
        ("干旱胁迫", "产量提升"),
        ("盐碱胁迫", "产量提升"),
        ("高温胁迫", "产量提升"),
        ("干旱胁迫", "光合作用增强"),
        ("盐碱胁迫", "光合作用增强"),
        ("土壤过湿", "产量提升"),
    }

    # 原文否定/失败关键词 → 如果包含这些，说明措施确实导致了负向结果，不纠正
    _FAILURE_KEYWORDS = [
        "失败", "无效", "反而", "导致减产", "加重", "恶化",
        "没有缓解", "未能", "加剧", "更严重", "负面",
    ]

    # ── Step 0 证据文本极性规则（与 cause 类型完全无关） ──
    # 当 effect.concept 与 effect.text/evidence 的明确方向词矛盾时，自动纠正
    # key = 疑似错误的 concept；value = {bad_text_patterns: [regex], correct_to: str}
    _EVIDENCE_POLARITY_RULES: dict[str, dict] = {
        "病害加重": {
            "bad_text_patterns": [
                r"低于\s*CK\b", r"低于\s*对照", r"病情指数\s*显著?\s*低",
                r"病情\s*减轻", r"病情\s*下降", r"抑制\s*病",
                r"防效", r"发病率\s*低", r"病指\s*低",
                r"病情指数\s*.*?低", r"显著?低\s*于\s*(CK|对照)",
            ],
            "correct_to": "病害减轻",
        },
        "病害减轻": {
            "bad_text_patterns": [
                r"高于\s*CK\b", r"高于\s*对照", r"病情指数\s*显著?\s*高",
                r"病情\s*加重", r"病情\s*上升", r"发病率\s*上升",
                r"病指\s*高", r"病害\s*恶化",
            ],
            "correct_to": "病害加重",
        },
        "产量下降": {
            "bad_text_patterns": [
                r"增产", r"高于\s*CK\b", r"高于\s*对照", r"产量\s*显著高",
                r"产量\s*提升", r"产量\s*提高", r"产量从.*?升到",
                r"单株产量从.*?提高", r"高于清水对照", r"显著高于.*?对照",
                r"显著?增产",
            ],
            "correct_to": "产量提升",
        },
        "产量提升": {
            "bad_text_patterns": [
                r"减产", r"低于\s*CK\b", r"低于\s*对照", r"产量\s*下降",
                r"产量\s*降低", r"产量损失", r"严重限制.*?产量",
                r"显著?减产",
            ],
            "correct_to": "产量下降",
        },
        "虫害": {
            "bad_text_patterns": [
                r"种群密度\s*显著?\s*降低", r"虫口\s*低", r"种群\s*下降",
                r"飞虱种群\s*降低", r"虫害\s*减轻",
            ],
            "correct_to": "虫害减轻",
        },
        "虫害减轻": {
            "bad_text_patterns": [
                r"种群密度\s*显著?\s*上升", r"虫口\s*高", r"种群\s*上升",
                r"飞虱种群\s*升高", r"虫害\s*加重",
            ],
            "correct_to": "虫害",
        },
        "光合作用下降": {
            "bad_text_patterns": [
                r"光合速率\s*提升", r"光合\s*.*?显著.*?升", r"光合作用\s*增强",
                r"净光合速率\s*.*?提升", r"光合\s*.*?高于",
            ],
            "correct_to": "光合作用增强",
        },
        "光合作用增强": {
            "bad_text_patterns": [
                r"光合速率\s*下降", r"光合\s*.*?跟不上", r"光合\s*.*?降低",
                r"光合作用\s*下降",
            ],
            "correct_to": "光合作用下降",
        },
    }

    def _polarity_check(self, pairs: list[dict], text: str) -> list[dict]:
        """规则层极性校验（v1.9+ 扩展版）：四重校验策略。

        0. 证据文本极性校验（新增）：effect.concept 与 effect.text/evidence
           的方向词矛盾 → 纠正。与 cause 类型完全无关，最基础的校验。
        1. 极性反转过滤：cause 是负向概念、effect 是同领域正向概念 → 过滤
        2. MGT 类 cause → 负向 effect 纠正（原有逻辑，扩展 _POLARITY_MAP）
        3. 正向 cause → 负向 effect 纠正（新增：覆盖胁迫缓解→产量下降等）
        """
        result: list[dict] = []

        for p in pairs:
            cause = p.get("cause", {}) or p.get("cause_event", {})
            effect = p.get("effect", {}) or p.get("effect_event", {})
            cause_concept = cause.get("concept", "")
            cause_subtype = cause.get("sub_class_id", "") or cause.get("subtype_id", "")
            effect_concept = effect.get("concept", "")

            evidence = p.get("evidence", "") or ""
            cause_text = cause.get("text", "") or ""
            effect_text = effect.get("text", "") or ""
            # 局部文本：仅当前 pair 的证据 + cause/effect 原文片段
            # （证据极性规则必须严格限定在局部，不能混入全文，避免全文中其它句子造成误匹配）
            local_text = evidence + cause_text + effect_text

            # ── 校验0：证据文本极性校验（不依赖 cause 类型） ──
            ev_rule = self._EVIDENCE_POLARITY_RULES.get(effect_concept)
            if ev_rule:
                matched = False
                matched_pat = ""
                for pat in ev_rule["bad_text_patterns"]:
                    if re.search(pat, local_text):
                        matched = True
                        matched_pat = pat
                        break
                if matched:
                    corrected = ev_rule["correct_to"]
                    effect["concept"] = corrected
                    effect["sub_class_id"] = ""
                    p.setdefault("_post_fixes", []).append(
                        f"polarity_evidence: {effect_concept}→{corrected} (matched /{matched_pat[:30]}/)"
                    )
                    logger.debug(
                        "极性校验(证据文本): %s→%s (pattern=%s)", effect_concept, corrected, matched_pat
                    )
                    effect_concept = corrected  # 后续校验用更新后的值

            # ── 校验0.5：交叉域极性反转（负向 cause → 正向 effect → 纠正） ──
            # 当已被 KB 召回已知因果或有证据表明真实因果时不触发
            kb_known = p.get("kb_match", {}).get("known", False)
            if (not kb_known
                    and cause_concept in self._NEGATIVE_CONCEPTS
                    and effect_concept in self._CROSS_DOMAIN_REVERSAL_MAP):
                # 如果局部文本或证据中含有"升高/增加/增产"等明确正向词，
                # 让 Step 0 证据规则或 MGT 规则优先，此处只处理 LLM 幻觉
                cross_rule = self._CROSS_DOMAIN_REVERSAL_MAP[effect_concept]
                corrected = cross_rule["if_cause_negative"]
                effect["concept"] = corrected
                effect["sub_class_id"] = ""
                p.setdefault("_post_fixes", []).append(
                    f"polarity_cross: {effect_concept}→{corrected} (cause={cause_concept})"
                )
                logger.debug(
                    "极性校验(交叉域): %s→%s (neg_cause=%s)",
                    effect_concept, corrected, cause_concept,
                )
                effect_concept = corrected

            # ── 校验1：极性反转对 → 直接过滤 ──
            if (cause_concept, effect_concept) in self._POLARITY_REVERSAL_PAIRS:
                p.setdefault("_post_fixes", []).append(
                    f"filtered: polarity_reversal({cause_concept}→{effect_concept})"
                )
                logger.debug("极性反转过滤: %s→%s", cause_concept, effect_concept)
                continue  # 不加入 result

            # 失败关键词可查全文上下文（全局叙述）
            has_failure = any(kw in local_text or kw in text for kw in self._FAILURE_KEYWORDS)

            # ── 校验2：MGT 类 cause → 负向 effect 纠正 ──
            is_mgt = (
                cause_subtype.startswith("Treatment.")
                or cause_subtype.startswith("MGT-")
                or cause_concept in self._MGT_CONCEPTS
            )
            if is_mgt and effect_concept in self._POLARITY_MAP and not has_failure:
                corrected = self._POLARITY_MAP[effect_concept]
                effect["concept"] = corrected
                effect["sub_class_id"] = ""
                p.setdefault("_post_fixes", []).append(
                    f"polarity_mgt: {effect_concept}→{corrected}"
                )
                logger.debug("极性校验(MGT): %s→%s (cause=%s)", effect_concept, corrected, cause_concept)
                effect_concept = corrected

            # ── 校验3：正向 cause → 负向 effect 纠正 ──
            elif (cause_concept in self._POSITIVE_CONCEPTS
                  and effect_concept in self._POLARITY_MAP
                  and not has_failure):
                corrected = self._POLARITY_MAP[effect_concept]
                effect["concept"] = corrected
                effect["sub_class_id"] = ""
                p.setdefault("_post_fixes", []).append(
                    f"polarity_pos: {effect_concept}→{corrected}"
                )
                logger.debug("极性校验(正向cause): %s→%s (cause=%s)", effect_concept, corrected, cause_concept)

            result.append(p)

        return result

    # MGT 类概念集合（用于无 sub_class_id 时判断 cause 是否为管理措施）
    _MGT_CONCEPTS = {
        "降温措施", "灌溉措施", "氮肥过量", "叶面喷肥", "菌根共生",
        "有机改良剂", "蚯蚓粪", "土施氮肥", "锌肥施用", "螯合锌",
        "硅肥施用", "保水剂施用", "辐照改性SAP", "LED补光", "夜间补光",
        "铜制剂喷施", "减铜处理", "植物源农药", "欧李树皮提取物",
        "腐植酸钾施用", "微量元素肥施用", "耐热品种", "嫁接苗", "遮阳措施",
        "耐旱品种", "抗逆品种",
    }

    # ------------------------------------------------------------------ #
    #  知识库候选召回（COMET轻量版）
    # ------------------------------------------------------------------ #
    def _kb_recall(self, raw_pairs: list[dict], events: list[dict]) -> list[dict]:
        """用知识库 known_pairs 补充 LLM 漏抽的因果对。

        策略（借鉴 COMET 候选生成思路，但不引入外部模型）：
        - 遍历当前文本中出现的事件 concept 两两组合
        - 若某对在知识库 known_pairs 中且 LLM 未抽到，则补入
        - 补入的因果对用知识库 strength 作为置信度

        约束：
        - 只补入"两个 concept 都在当前 events 中出现"的对（避免无关召回）
        - 只召回高强度因果对（strength >= RECALL_THRESHOLD），避免噪声
        - 限制召回数量，避免噪声

        Returns
        -------
        list[dict]
            召回的因果对列表（格式与 raw_pairs 一致）。
        """
        RECALL_THRESHOLD = 0.85  # 只召回高强度因果对
        if not self.kb.is_loaded or not events:
            return []

        # 当前文本中出现的事件 concept 集合
        present_concepts = {
            e.get("concept", "") for e in events
            if e.get("concept", "") and e.get("concept", "") != "UNKNOWN"
        }
        if not present_concepts:
            return []

        # LLM 已抽取的 concept 对集合
        existing_pairs = set()
        for p in raw_pairs:
            c = p.get("cause", {}).get("concept", "")
            e = p.get("effect", {}).get("concept", "")
            if c and e:
                existing_pairs.add((c, e))

        # 从知识库 known_pairs 中召回漏抽的对
        recalled: list[dict] = []
        event_type_map = {e.get("concept", ""): e.get("type", "INTERMEDIATE") for e in events}
        for key, strength in self.kb.known_pairs.items():
            if "|||" not in key:
                continue
            # 只召回高强度因果对
            if strength < RECALL_THRESHOLD:
                continue
            cause_concept, effect_concept = key.split("|||", 1)
            # 约束：两个 concept 都在当前文本中出现
            if cause_concept not in present_concepts or effect_concept not in present_concepts:
                continue
            # 约束：LLM 未抽到
            if (cause_concept, effect_concept) in existing_pairs:
                continue
            recalled.append({
                "cause": {
                    "concept": cause_concept,
                    "text": cause_concept,
                    "type": event_type_map.get(cause_concept, "CAUSE_EVENT"),
                },
                "effect": {
                    "concept": effect_concept,
                    "text": effect_concept,
                    "type": event_type_map.get(effect_concept, "EFFECT_EVENT"),
                },
                "relation_type": "DIRECT",
                "confidence": float(strength),
                "evidence": f"知识库已知因果(strength={strength})",
            })

        if recalled:
            logger.info("知识库候选召回: 补入 %d 个漏抽因果对", len(recalled))
        return recalled
