"""研究内容2 Step 1: 问题解析模块（纯规则，0 LLM）。

将用户自然语言问题解析为结构化查询参数，供 CausalRAG 检索使用。

设计依据（精简方案：1篇核心论文 CausalRAG + 规则后处理）：
- 不调用 LLM，与研究内容1的「1 LLM + 规则后处理」架构约束一致
- 复用 ConceptAggregator 做目标事件 → 图谱 concept 归一化
- 规则化处理：疑问词识别 / 意图判定 / 方向判定 / 领域提取

输入：用户问题（自然语言，如「为什么番茄落花？」）
输出：QuestionParse
  - target_event     提取的目标事件文本（「落花」）
  - target_concept   归一化后的图谱 concept（「落花落蕾」）
  - target_sub_class_id 子类ID（「FunctionalSymptom.ReproductiveSymptom」）
  - intent           意图（cause_explanation / effect_prediction / mechanism）
  - direction        检索方向（backward 找原因 / forward 找结果）
  - domain           领域作物（「番茄」）
  - question_type    问题类型（why / what_effect / how / unknown）

使用：
    parser = QuestionParser()
    result = parser.parse("为什么番茄落花？")
    # result.target_concept == "落花落蕾"
    # result.direction == "backward"
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any

from ..event_extraction.concept_aggregator import ConceptAggregator

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
#  规则表
# -------------------------------------------------------------------- #
# 疑问词模式 → (question_type, intent, direction)
#   顺序敏感：先匹配「后果预测」（forward），再「原因解释」（backward），最后「机制」
#   避免「为什么...会导致什么」被 why 先吞掉
#   注意：不使用 .* 贪婪匹配，避免把目标事件一起吞掉
_QUESTION_PATTERNS: list[tuple[str, str, str, str]] = [
    # 后果预测类（forward：从目标事件往后找结果）
    (r"(会导致什么|会带来什么|造成什么|引起什么|的后果|影响是什么|影响$|会有什么影响|会怎样|会怎么样|会什么)",
     "what_effect", "effect_prediction", "forward"),
    # 原因解释类（backward：从目标事件往前找原因）
    (r"(为什么|为何|怎么会|何以|是什么原因|什么原因|的原因)",
     "why", "cause_explanation", "backward"),
    # 机制说明类（backward：找机制链）
    (r"(如何|怎样|的机制|的过程|的原理)",
     "how", "mechanism", "backward"),
]

# 作物词典（mention → 标准作物名）
_CROP_DICT: dict[str, str] = {
    "番茄": "番茄", "西红柿": "番茄",
    "茄子": "茄子", "辣椒": "辣椒", "黄瓜": "黄瓜",
    "水稻": "水稻", "稻": "水稻",
    "小麦": "小麦", "玉米": "玉米", "大麦": "大麦",
    "油菜": "油菜", "大豆": "大豆", "黄豆": "大豆",
    "棉花": "棉花", "蔬菜": "蔬菜",
    "茶树": "茶", "茶叶": "茶",
    "葡萄": "葡萄", "苹果": "苹果", "果树": "果树",
}

# 虚词/停用词（清理目标事件 mention 时去除）
_STOPWORDS = set("会了的是呢吗啊呀在都也又还把被让给对从向以")


@dataclass
class QuestionParse:
    """问题解析结果。"""
    raw_text: str
    question_type: str = "unknown"        # why / what_effect / how / unknown
    intent: str = "cause_explanation"     # cause_explanation / effect_prediction / mechanism
    direction: str = "backward"           # backward / forward
    domain: str = ""                      # 作物领域
    target_event: str = ""                # 提取的目标事件文本
    target_concept: str = "UNKNOWN"        # 归一化后的图谱 concept
    target_sub_class_id: str = "UNKNOWN"     # 子类ID (class_id.sub_class_id)
    is_valid: bool = False                 # 是否成功解析（target_concept 非 UNKNOWN）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuestionParser:
    """问题解析器（纯规则，0 LLM）。

    将用户自然语言问题解析为结构化查询参数，供 CausalRAG 检索使用。

    Parameters
    ----------
    aggregator : ConceptAggregator, optional
        概念聚合器。未提供时使用默认同义词典加载。
    crop_dict : dict, optional
        作物词典。未提供时使用默认 _CROP_DICT。
    """

    def __init__(
        self,
        aggregator: ConceptAggregator | None = None,
        crop_dict: dict[str, str] | None = None,
    ) -> None:
        self._re = re
        self.aggregator = aggregator or ConceptAggregator()
        self.crop_dict = crop_dict or dict(_CROP_DICT)
        # 按长度降序排列作物名，避免短词（如「稻」）先匹配导致「水稻」被拆
        self._sorted_crops = sorted(self.crop_dict.keys(), key=len, reverse=True)
        # 编译疑问词模式
        self._compiled_patterns: list[tuple[re.Pattern, str, str, str]] = [
            (self._re.compile(pat), qt, intent, direction)
            for pat, qt, intent, direction in _QUESTION_PATTERNS
        ]

    # ---------------------------------------------------------------- #
    #  公开接口
    # ---------------------------------------------------------------- #
    def parse(self, raw_text: str) -> QuestionParse:
        """解析用户问题。

        Parameters
        ----------
        raw_text : str
            用户自然语言问题，如「为什么番茄落花？」

        Returns
        -------
        QuestionParse
            解析结果，含 target_concept / direction / intent 等。
        """
        text = (raw_text or "").strip()
        if not text:
            return QuestionParse(raw_text=raw_text or "")

        # 1. 识别疑问词 → question_type / intent / direction
        question_type, intent, direction = self._detect_question_type(text)

        # how 类型特殊处理：「X如何影响Y」→ target = Y（被影响的结果事件）
        if question_type == "how":
            target_event = self._extract_how_target(text)
            if target_event:
                domain, _ = self._extract_domain(target_event)
                concept, sub_class_id = self._normalize_concept(target_event)
                result = QuestionParse(
                    raw_text=raw_text, question_type=question_type,
                    intent=intent, direction=direction, domain=domain,
                    target_event=target_event, target_concept=concept,
                    target_sub_class_id=sub_class_id,
                    is_valid=(concept != "UNKNOWN" and bool(target_event)),
                )
                if not result.is_valid:
                    logger.info("how 问题解析未锚定 concept: %r → mention=%r",
                                raw_text, target_event)
                return result

        # 常规流程：去除疑问词 → 提取作物 → 清理 → 归一化
        cleaned = self._remove_question_patterns(text)
        domain, cleaned = self._extract_domain(cleaned)
        target_event = self._clean_event_mention(cleaned)
        concept, sub_class_id = self._normalize_concept(target_event)

        result = QuestionParse(
            raw_text=raw_text,
            question_type=question_type,
            intent=intent,
            direction=direction,
            domain=domain,
            target_event=target_event,
            target_concept=concept,
            target_sub_class_id=sub_class_id,
            is_valid=(concept != "UNKNOWN" and bool(target_event)),
        )
        if not result.is_valid:
            logger.info("问题解析未能锚定 concept: %r → mention=%r", raw_text, target_event)
        return result

    def _extract_how_target(self, text: str) -> str:
        """从「X如何影响Y」提取 Y（被影响的目标事件）。

        支持：如何影响 / 怎样影响 / 如何导致
        """
        m = self._re.search(r"(?:如何|怎样)影响(.+?)$", text)
        if not m:
            m = self._re.search(r"(?:如何|怎样)导致(.+?)$", text)
        if not m:
            return ""
        target = m.group(1)
        target = self._re.sub(r"[，。？！？,.!?；;：:\s]+", "", target)
        target = "".join(ch for ch in target if ch not in _STOPWORDS)
        return target.strip()

    # ---------------------------------------------------------------- #
    #  内部方法
    # ---------------------------------------------------------------- #
    def _detect_question_type(self, text: str) -> tuple[str, str, str]:
        """识别疑问词模式，返回 (question_type, intent, direction)。"""
        for pat, qt, intent, direction in self._compiled_patterns:
            if pat.search(text):
                return qt, intent, direction
        # 默认：原因解释 + backward
        return "unknown", "cause_explanation", "backward"

    def _remove_question_patterns(self, text: str) -> str:
        """去除疑问词片段，保留事件主体。"""
        cleaned = text
        for pat, *_ in self._compiled_patterns:
            cleaned = pat.sub("", cleaned)
        return cleaned

    def _extract_domain(self, text: str) -> tuple[str, str]:
        """提取作物领域，返回 (domain, 去除作物后的文本)。"""
        for crop in self._sorted_crops:
            if crop in text:
                cleaned = text.replace(crop, "", 1)
                return self.crop_dict[crop], cleaned
        return "", text

    def _clean_event_mention(self, text: str) -> str:
        """清理虚词标点，提取目标事件 mention。"""
        # 去标点
        cleaned = self._re.sub(r"[，。？！？,.!?；;：:\s]+", "", text)
        # 去虚词
        cleaned = "".join(ch for ch in cleaned if ch not in _STOPWORDS)
        return cleaned.strip()

    def _normalize_concept(self, mention: str) -> tuple[str, str]:
        """用 ConceptAggregator 归一化 mention 到图谱 concept。

        Returns
        -------
        (concept, sub_class_id)
            归一化失败时返回 ("UNKNOWN", "UNKNOWN")。
        """
        if not mention:
            return "UNKNOWN", "UNKNOWN"
        # 同时传 mention 和 llm_concept=mention，让 aggregator 走策略2/3 命中本体收录的概念
        # （如"降水增加"在 mention_to_concept 未命中，但在 agri_event_types.json 的 PrecipitationParam.concepts 中）
        result = self.aggregator.aggregate(mention_text=mention, llm_concept=mention)
        concept = result.get("concept", "UNKNOWN")
        sub_class_id = result.get("sub_class_id", "UNKNOWN")
        return concept, sub_class_id
