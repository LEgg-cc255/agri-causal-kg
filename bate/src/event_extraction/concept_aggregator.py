"""事件概念聚合模块（纯规则，基于 AKO-F v1.9 本体）

设计依据：
- 参考 LKCER (COLING 2025) 的概念级事件聚合思想
- 简化实现：用自定义同义词表做精确+模糊匹配，不依赖 WordNet（中文农业覆盖差）
- 不引入 COMET 知识增强（农业领域噪声大）

本体格式（v2.0，AKO-F v1.9）：
- concept_to_subtype 的 value 格式为 "class_id.sub_class_id"
  例如 "高温胁迫" → "EnvFactor.WeatherParam"
- 返回字段：concept / sub_class_id / entity_class
  其中 sub_class_id = "EnvFactor.WeatherParam"，entity_class = "环境因子"

聚合策略（优先级递减）：
1. 精确匹配 mention_to_concept（优先于 LLM，避免坍缩）
2. LLM concept 在标准概念列表 → 采用（多数样本命中此步）
3. LLM concept 在 concept_to_subtype 中 → 采用（前移，覆盖正向 concept 如产量提升/光合作用增强）
4. mention 模糊包含匹配 mention_to_concept 键
5. 启发式关键词规则：对未知域文本，按关键词→concept 映射
6. 兜底：如果 LLM concept 非空，直接给 concept=llm_c（让下游评分/极性校验处理），
   仅在完全没信息时才标记 UNKNOWN
"""

from __future__ import annotations

import logging
from typing import Any

from ..utils.data_loader import load_synonym_dict, load_event_types, build_concept_index

logger = logging.getLogger(__name__)

# v2.1 新增：启发式关键词 → 标准概念映射
#   当 mention 中包含某 keyword（正则）时，锚定到指定标准概念。
#   仅作为策略 4 失败的兜底。按长度降序匹配。
_HEURISTIC_KEYWORDS: list[tuple[str, str]] = [
    # 养分吸收方向
    (r"\b镁吸收提升\b|\b锌吸收提升\b|\b铁吸收提升\b|\b钙吸收提升\b|\b养分吸收\b", "养分吸收增加"),
    (r"\b磷吸收\b.*提升|\b钾吸收\b.*提升", "养分吸收增加"),
    (r"\b吸收\b.*提升\b|提升.*\b吸收\b", "养分吸收增加"),
    # 离子平衡
    (r"Na⁺|Na\+.*升高|Na⁺.*上升|离子浓度.*上升", "离子失衡"),
    (r"离子平衡|离子稳态|离子.*降低", "离子失衡"),
    # 产量方向
    (r"产量.*提高|产量.*升到|单株产量.*提高|增产|显著增产", "产量提升"),
    (r"产量下降|减产|产量损失|低于.*产量|产量.*显著.*低", "产量下降"),
    # 光合方向
    (r"光合速率.*提升|光合.*增强|净光合.*提高|光合性能.*提高", "光合作用增强"),
    (r"光合速率.*下降|光合.*降低|光合.*受抑|叶绿素含量下降", "光合作用下降"),
    # 病害方向
    (r"病情指数.*低于|病情指数.*下降|病害减轻|防病.*效果|防效.*好|抑制.*发病", "病害减轻"),
    (r"病情指数.*高于|病情.*加重|发病率上升|病害蔓延", "病害加重"),
    # 坐果/结实方向
    (r"坐果率.*提升|坐果率.*上升|着果率.*提高|结实率.*提高", "着果率提升"),
    (r"坐果率.*下降|着果率.*减少|结实不良|落花落蕾", "着果率下降"),
    # 抗逆 / 品质改善
    (r"糖酸比.*上升|维生素C.*上升|品质改善|品质提升|品质优于|糖含量.*提高", "胁迫缓解"),
    (r"抗氧化.*提升|抗氧酶.*上升|POD.*提升|SOD.*提升|CAT.*提升|酶活.*提高", "抗氧化能力提升"),
    (r"叶绿素含量.*提升|叶绿素.*增加|叶绿素.*高于对照", "胁迫缓解"),
    (r"处理增产显著|增产效果.*显著|早发.*增产|增产", "产量提升"),
]


class ConceptAggregator:
    """事件概念聚合器。

    将文本中的事件 mention 归一化为标准概念标签，并映射到事件子类。

    Parameters
    ----------
    synonym_dict : dict, optional
        同义词典。未提供时从默认数据文件加载。
    event_types : dict, optional
        事件类型本体。未提供时从默认数据文件加载。
    """

    def __init__(
        self,
        synonym_dict: dict[str, Any] | None = None,
        event_types: dict[str, Any] | None = None,
    ) -> None:
        import re as _re
        self._re = _re
        self.synonym_dict = synonym_dict or load_synonym_dict()
        self.event_types = event_types or load_event_types()

        # mention -> concept 索引
        self.mention_to_concept: dict[str, str] = {
            k: v for k, v in self.synonym_dict.get("mention_to_concept", {}).items()
            if not k.startswith("_")
        }
        # 显式 concept -> "class_id.sub_class_id" 映射（用于覆盖本体未收录的正向概念等）
        self.concept_to_subtype: dict[str, str] = self.synonym_dict.get("concept_to_subtype", {})
        # 本体自动生成的 concept -> "class_id.sub_class_id" 索引（嵌套子类结构）
        self.concept_index: dict[str, str] = build_concept_index(self.event_types)
        # "class_id.sub_class_id" -> (class_id, class_name, sub_class_name)
        self.subtype_meta: dict[str, dict[str, str]] = self._build_subtype_meta()
        # 全部标准概念集合
        self.all_concepts: set[str] = set(self.concept_index.keys())

        # 按长度降序排列 mention keys，避免短词先匹配导致错误
        self._sorted_mentions = sorted(self.mention_to_concept.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------ #
    #  公开接口
    # ------------------------------------------------------------------ #
    def aggregate(self, mention_text: str, llm_concept: str = "") -> dict[str, str]:
        """将事件 mention 聚合为标准概念。"""
        mention = (mention_text or "").strip()
        llm_c = (llm_concept or "").strip()

        # 策略1：mention 精确命中 mention_to_concept
        if mention:
            hit = self.mention_to_concept.get(mention)
            if hit:
                return self._wrap(hit)

        # 策略2：LLM concept 在标准概念列表 → 采用
        if llm_c and llm_c in self.all_concepts:
            return self._wrap(llm_c)

        # 策略3（前移）：LLM concept 在 concept_to_subtype 中 → 采用
        #   → 覆盖正向 concept 如产量提升、光合作用增强等不在本体 concepts 但有 subtype 的
        if llm_c and llm_c in self.concept_to_subtype:
            return self._wrap(llm_c)

        # 策略4：mention 模糊包含（子串匹配）
        if mention:
            for key in self._sorted_mentions:
                if not key:
                    continue
                if key in mention or mention in key:
                    return self._wrap(self.mention_to_concept[key])

        # 策略5（v2.1 新增）：启发式关键词正则匹配
        if mention:
            for pat, concept in _HEURISTIC_KEYWORDS:
                if self._re.search(pat, mention):
                    return self._wrap(concept)

        # 策略6（v2.1 新增）：若 LLM concept 非空，即使未命中映射，也直接包装
        #   → 下游极性校验等规则层仍有机会纠正，避免过度 UNKNOWN 导致 Recall 坍塌
        if llm_c:
            wrapped = self._wrap(llm_c)
            if wrapped["sub_class_id"] == "UNKNOWN":
                # 尝试启发式推断 sub_class_id：根据 concept 关键词粗略归类
                subtype_guess = self._heuristic_subtype_for_llm_concept(llm_c)
                if subtype_guess:
                    wrapped["sub_class_id"] = subtype_guess
                    wrapped["entity_class"] = self.subtype_meta.get(subtype_guess, {}).get(
                        "class_name", wrapped["entity_class"]
                    )
                    wrapped["sub_class_name"] = self.subtype_meta.get(subtype_guess, {}).get(
                        "sub_class_name", wrapped.get("sub_class_name", "")
                    )
            return wrapped

        # 最终兜底：完全无信息时 UNKNOWN
        logger.debug("概念聚合(兜底)标记为 UNKNOWN: mention=%r, llm_concept=%r", mention, llm_concept)
        return {"concept": "UNKNOWN", "sub_class_id": "UNKNOWN", "entity_class": "UNKNOWN", "sub_class_name": ""}

    # ------------------------------------------------------------------ #
    #  v2.1 新增：对 LLM 未锚定 concept 的启发式 subtype 推断
    # ------------------------------------------------------------------ #
    def _heuristic_subtype_for_llm_concept(self, llm_c: str) -> str:
        """当 LLM concept 不在任何映射中时，根据关键词粗分类。

        返回 "class_id.sub_class_id" 格式（如 "EnvFactor.WeatherParam"）。
        """
        s = llm_c
        # 产量类
        if "产量" in s:
            return "FunctionalSymptom.YieldSymptom"
        # 光合类
        if "光合" in s or "叶绿素" in s:
            return "FunctionalSymptom.PhysiologicalSymptom"
        # 病害类
        if "病害" in s or "病" in s and ("加重" in s or "减轻" in s or "指数" in s):
            return "BioticStress.Disease"
        # 虫害类
        if "虫" in s and ("种群" in s or "虫口" in s or "减轻" in s or "加重" in s):
            return "BioticStress.Pest"
        # 养分吸收
        if "吸收" in s and ("养分" in s or "磷" in s or "钾" in s or "锌" in s or "镁" in s or "铁" in s):
            return "FunctionalSymptom.PhysiologicalSymptom"
        # 离子平衡
        if "离子" in s or "Na" in s or "Na⁺" in s:
            return "FunctionalSymptom.PhysiologicalSymptom"
        # 着果率
        if "着果" in s or "坐果" in s or "结实" in s:
            return "FunctionalSymptom.ReproductiveSymptom"
        # 胁迫缓解（正向发育结果）
        if "缓解" in s or "提升" in s or "改善" in s or "提高" in s:
            return "FunctionalSymptom.StressResponseSymptom"
        # 胁迫/环境类
        if "胁迫" in s or "干旱" in s or "盐碱" in s or "高温" in s or "低温" in s or "过湿" in s or "干热风" in s:
            # 先在本体中查找精确匹配
            for cls in self.event_types.get("entity_classes", []):
                for sub in cls.get("sub_classes", []):
                    if any(c in llm_c for c in sub.get("concepts", [])):
                        return f"{cls['class_id']}.{sub['sub_class_id']}"
            # 粗略归类
            if "盐碱" in s:
                return "EnvFactor.SoilParam"
            if "干旱" in s or "水分亏缺" in s:
                return "EnvFactor.WeatherParam"
            if "高温" in s:
                return "EnvFactor.WeatherParam"
            if "低温" in s or "冷害" in s:
                return "EnvFactor.WeatherParam"
            if "过湿" in s:
                return "EnvFactor.SoilParam"
        return ""

    def aggregate_batch(self, mentions: list[str]) -> list[dict[str, str]]:
        """批量聚合。"""
        return [self.aggregate(m) for m in mentions]

    # ------------------------------------------------------------------ #
    #  内部工具
    # ------------------------------------------------------------------ #
    def _wrap(self, concept: str) -> dict[str, str]:
        """将概念包装为完整结构（含 sub_class_id / entity_class / sub_class_name）。

        优先级：
        1. 显式映射 concept_to_subtype（用于本体未收录的正向概念等）
        2. 本体自动索引 concept_index（嵌套子类结构自动生成）

        返回字段：
        - concept: 标准概念名
        - sub_class_id: "class_id.sub_class_id" 格式（如 "EnvFactor.PrecipitationParam"）
        - entity_class: 实体大类中文名（如 "环境因子"）
        - sub_class_name: 子类中文名（如 "降水"）
        """
        sub_class_id = self.concept_to_subtype.get(concept) or self.concept_index.get(concept, "UNKNOWN")
        meta = self.subtype_meta.get(sub_class_id, {})
        return {
            "concept": concept,
            "sub_class_id": sub_class_id,
            "entity_class": meta.get("class_name", "UNKNOWN"),
            "sub_class_name": meta.get("sub_class_name", ""),
        }

    def _build_subtype_meta(self) -> dict[str, dict[str, str]]:
        """构建 "class_id.sub_class_id" -> 元信息 的索引，支持嵌套子类结构。

        AKO-F v1.9 结构：
        - 实体大类（class_id，如 EnvFactor）→ 子类（如 WeatherParam）→ 细粒度子类（如 AirTemperatureParam）
        - 复合胁迫/模糊节点存父类顶层（如 concept_to_subtype 值 = "EnvFactor.WeatherParam"）
        - 跨大类复合节点存 EnvFactor 顶层父类
        """
        meta: dict[str, dict[str, str]] = {}
        for cls in self.event_types.get("entity_classes", []):
            class_id = cls.get("class_id", "")
            class_name = cls.get("class_name", "")
            # 大类顶层键（用于跨大类复合节点，如"干旱与病害复合胁迫"→"EnvFactor"）
            meta[class_id] = {
                "class_id": class_id,
                "class_name": class_name,
                "sub_class_name": "",
            }
            # 无子类的实体类（如 Treatment）—— 用 class_id.class_id 键
            if "concepts" in cls and not cls.get("sub_classes"):
                key = f"{class_id}.{class_id}"
                meta[key] = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "sub_class_name": class_name,
                }

            # 递归处理子类（支持多层嵌套）
            self._collect_subclass_meta(cls.get("sub_classes", []), class_id, class_name, meta)
        return meta

    def _collect_subclass_meta(
        self,
        sub_classes: list[dict[str, Any]],
        class_id: str,
        class_name: str,
        meta: dict[str, dict[str, str]],
    ) -> None:
        """递归收集子类元信息到 meta。

        对每个 sub_class，同时添加：
        - "class_id.sub_class_id" 键（用于父类顶层存储的复合/模糊节点）
        - 如有嵌套 sub_classes，递归处理细粒度子类
        """
        for sub in sub_classes:
            sub_id = sub.get("sub_class_id", "")
            sub_name = sub.get("sub_class_name", "")
            key = f"{class_id}.{sub_id}"
            meta[key] = {
                "class_id": class_id,
                "class_name": class_name,
                "sub_class_name": sub_name,
            }
            # 递归处理细粒度子类
            nested = sub.get("sub_classes", [])
            if nested:
                self._collect_subclass_meta(nested, class_id, class_name, meta)
