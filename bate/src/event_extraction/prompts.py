"""LLM Prompt 模板（精简方案：一次LLM调用完成全部抽取）

设计依据：
- 参考 Event Causality Is Key (NAACL 2024) 的 prompt 框架：事件与因果对联合抽取
- 扩展：让 LLM 在同一次调用中直接输出因果链（causal_chains）
- 用 agri_event_types.json 的本体类型作为约束，避免 LLM 自由发挥
- 农业领域 few-shot 示例来自真实标注数据集

精简原则：
- 一次 LLM 调用同时完成「事件识别 + 因果对抽取 + 因果链构建」
- 后续模块（1.2/1.3）仅用规则做后处理，不再调用 LLM
- 避免多阶段 LLM 判断不一致导致 recall 下降
"""

from __future__ import annotations

# ================================================================== #
#  系统 Prompt：定义角色、任务、输出格式、本体约束
# ================================================================== #

SYSTEM_PROMPT_EXTRACTION = """你是农业因果事件抽取专家。任务：从农业文本中识别因果事件，抽取因果关系对，并构建因果链。

## 事件类型（对应 AKO-F v1.9 本体4大类）
- CAUSE_EVENT: 原因事件（环境因子/生物胁迫/防治措施等触发事件）
- EFFECT_EVENT: 结果事件（功能症状中的产量/品质/发育等终端变化）
- INTERMEDIATE: 中间事件（因果链中的功能症状-生理/生殖变化）

## 关系类型
- DIRECT: 直接因果 | INDIRECT: 间接因果 | CONDITIONAL: 条件性因果（特定阈值/生育期下成立）

## 原则（优先级最高，务必遵守）
1. 事件text必须是原文片段，不得改写
2. concept从下方概念表选最贴近的
3. **只抽取有明确因果机制的关系**（原文有"导致/使得/造成/引起"等因果动词），不抽取相关关系或并列关系
4. 因果对数量约束：短文本（<100字）最多3-5个因果对，长文本最多5-8个，避免过度抽取
5. causal_chains是从因果对合并的多跳链（≥2节点），概念序列需与因果对一致
6. confidence: 有明确机制≥0.85，间接推断0.6-0.85，弱推断<0.6
7. **未知概念不硬套**：原文中的防治措施（如某种肥料、药剂、补光、保水剂等）不属于下方概念表时，concept输出"UNKNOWN"；但功能症状类事件（如"养分吸收""光合""产量""着果率"等已知域事件）尽量从概念表中选方向一致的，不要轻易输出UNKNOWN
8. **极性一致性（原因侧）**：防治措施类原因→效果极性必须与原文一致。原文说"增产/提升/缓解"则效果概念为正向（产量提升、胁迫缓解等），不能输出反向概念
9. **极性一致性（文本侧）**：每个event的concept极性必须与对应text片段的语义方向一致。示例：① text写"病情指数低于CK/病情下降/抑制发病" → concept必须选病害减轻（不得写病害加重）；② text写"增产/产量高于CK/从X升到Y" → concept必须选产量提升（不得写产量下降）；③ text写"叶绿素含量下降" → concept不得写光合作用增强。选择方向匹配的概念（如"养分吸收增加"对应"养分吸收提高/上升"，"养分吸收下降"对应"降低/流失"），不应因谨慎输出UNKNOWN
10. **作物一致性**：效果concept必须与文本讨论的作物一致。番茄文本不出现"稻米品质"，水稻文本不出现"着果率"
11. **作物主体标注（crop 字段）**：每条 causal_pair 可选带 crop 字段，标该因果对涉及的作物主体（如"番茄""水稻"）。
    - 原文明确提到具体作物（如"高温胁迫使得番茄花粉量减少"）→ crop="番茄"
    - 原文未提具体作物或为通用机制（如"高温导致蒸腾增强"）→ crop=""（空字符串）
    - 一段文本里同时涉及多个作物时，每条因果对标该条直接涉及的作物

## 输出格式（严格JSON，不要解释文字）
{"events":[{"text":"原文片段","type":"CAUSE_EVENT|EFFECT_EVENT|INTERMEDIATE","concept":"标准概念"}],"causal_pairs":[{"cause":{"text":"原文","type":"CAUSE_EVENT","concept":"概念"},"effect":{"text":"原文","type":"EFFECT_EVENT|INTERMEDIATE","concept":"概念"},"relation_type":"DIRECT|INDIRECT|CONDITIONAL","confidence":0.0-1.0,"evidence":"原文片段","crop":"番茄或空"}],"causal_chains":[{"concepts":["原因概念","中间概念","结果概念"],"events":["原文1","原文2","原文3"],"confidence":0.0-1.0}]}

## 农业事件概念表（AKO-F v1.9 本体，动态注入，选择concept时参考）
{CONCEPT_TABLE_PLACEHOLDER}
"""


# ================================================================== #
#  Few-shot 示例（来自真实农业文献，含因果链输出）
# ================================================================== #

FEWSHOT_EXAMPLES = [
    {
        "input": "高温胁迫使得番茄花粉量减少，花药开裂受制，花粉管伸长受制，花粉活力降低，造成落花现象。",
        "output": {
            "events": [
                {"text": "高温胁迫", "type": "CAUSE_EVENT", "concept": "高温胁迫"},
                {"text": "番茄花粉量减少", "type": "INTERMEDIATE", "concept": "花粉量减少"},
                {"text": "花药开裂受制", "type": "INTERMEDIATE", "concept": "花药开裂受阻"},
                {"text": "花粉管伸长受制", "type": "INTERMEDIATE", "concept": "花粉管伸长受阻"},
                {"text": "花粉活力降低", "type": "INTERMEDIATE", "concept": "花粉活力下降"},
                {"text": "落花", "type": "EFFECT_EVENT", "concept": "落花落蕾"}
            ],
            "causal_pairs": [
                {"cause": {"text": "高温胁迫", "type": "CAUSE_EVENT", "concept": "高温胁迫"},
                 "effect": {"text": "番茄花粉量减少", "type": "INTERMEDIATE", "concept": "花粉量减少"},
                 "relation_type": "DIRECT", "confidence": 0.92, "evidence": "高温胁迫使得番茄花粉量减少", "crop": "番茄"},
                {"cause": {"text": "高温胁迫", "type": "CAUSE_EVENT", "concept": "高温胁迫"},
                 "effect": {"text": "花药开裂受制", "type": "INTERMEDIATE", "concept": "花药开裂受阻"},
                 "relation_type": "DIRECT", "confidence": 0.90, "evidence": "花药开裂受制", "crop": "番茄"},
                {"cause": {"text": "高温胁迫", "type": "CAUSE_EVENT", "concept": "高温胁迫"},
                 "effect": {"text": "花粉管伸长受制", "type": "INTERMEDIATE", "concept": "花粉管伸长受阻"},
                 "relation_type": "DIRECT", "confidence": 0.90, "evidence": "花粉管伸长受制", "crop": "番茄"},
                {"cause": {"text": "高温胁迫", "type": "CAUSE_EVENT", "concept": "高温胁迫"},
                 "effect": {"text": "花粉活力降低", "type": "INTERMEDIATE", "concept": "花粉活力下降"},
                 "relation_type": "DIRECT", "confidence": 0.92, "evidence": "花粉活力降低", "crop": "番茄"},
                {"cause": {"text": "花粉活力降低", "type": "INTERMEDIATE", "concept": "花粉活力下降"},
                 "effect": {"text": "落花", "type": "EFFECT_EVENT", "concept": "落花落蕾"},
                 "relation_type": "DIRECT", "confidence": 0.88, "evidence": "造成落花现象", "crop": "番茄"}
            ],
            "causal_chains": [
                {"concepts": ["高温胁迫", "花粉活力下降", "落花落蕾"],
                 "events": ["高温胁迫", "花粉活力降低", "落花"],
                 "confidence": 0.90}
            ]
        }
    },
    {
        "input": "小麦灌浆期遇上干热风，叶片水分蒸发太快，光合作用跟不上，籽粒灌浆不足，千粒重下降，严重的能减产10%-20%。",
        "output": {
            "events": [
                {"text": "干热风", "type": "CAUSE_EVENT", "concept": "干热风"},
                {"text": "叶片水分蒸发太快", "type": "INTERMEDIATE", "concept": "蒸腾过强"},
                {"text": "光合作用跟不上", "type": "INTERMEDIATE", "concept": "光合作用下降"},
                {"text": "籽粒灌浆不足", "type": "INTERMEDIATE", "concept": "灌浆不良"},
                {"text": "千粒重下降", "type": "EFFECT_EVENT", "concept": "千粒重下降"},
                {"text": "减产10%-20%", "type": "EFFECT_EVENT", "concept": "产量下降"}
            ],
            "causal_pairs": [
                {"cause": {"text": "干热风", "type": "CAUSE_EVENT", "concept": "干热风"},
                 "effect": {"text": "叶片水分蒸发太快", "type": "INTERMEDIATE", "concept": "蒸腾过强"},
                 "relation_type": "DIRECT", "confidence": 0.90, "evidence": "叶片水分蒸发太快", "crop": "小麦"},
                {"cause": {"text": "叶片水分蒸发太快", "type": "INTERMEDIATE", "concept": "蒸腾过强"},
                 "effect": {"text": "光合作用跟不上", "type": "INTERMEDIATE", "concept": "光合作用下降"},
                 "relation_type": "DIRECT", "confidence": 0.88, "evidence": "光合作用跟不上", "crop": "小麦"},
                {"cause": {"text": "光合作用跟不上", "type": "INTERMEDIATE", "concept": "光合作用下降"},
                 "effect": {"text": "籽粒灌浆不足", "type": "INTERMEDIATE", "concept": "灌浆不良"},
                 "relation_type": "DIRECT", "confidence": 0.88, "evidence": "籽粒灌浆不足", "crop": "小麦"},
                {"cause": {"text": "籽粒灌浆不足", "type": "INTERMEDIATE", "concept": "灌浆不良"},
                 "effect": {"text": "千粒重下降", "type": "EFFECT_EVENT", "concept": "千粒重下降"},
                 "relation_type": "DIRECT", "confidence": 0.90, "evidence": "千粒重下降", "crop": "小麦"},
                {"cause": {"text": "千粒重下降", "type": "EFFECT_EVENT", "concept": "千粒重下降"},
                 "effect": {"text": "减产10%-20%", "type": "EFFECT_EVENT", "concept": "产量下降"},
                 "relation_type": "CONDITIONAL", "confidence": 0.85, "evidence": "严重的能减产10%-20%", "crop": "小麦"}
            ],
            "causal_chains": [
                {"concepts": ["干热风", "蒸腾过强", "光合作用下降", "灌浆不良", "千粒重下降", "产量下降"],
                 "events": ["干热风", "叶片水分蒸发太快", "光合作用跟不上", "籽粒灌浆不足", "千粒重下降", "减产10%-20%"],
                 "confidence": 0.88}
            ]
        }
    },
    {
        "input": "抽穗扬花期日最高温度持续高于35℃时，高温使花药壁失水过快，导致花药无法正常开裂散粉，花粉粒活性急剧下降，出现空壳率大幅上升的“花而不实”现象。",
        "output": {
            "events": [
                {"text": "抽穗扬花期日最高温度持续高于35℃", "type": "CAUSE_EVENT", "concept": "花期高温胁迫"},
                {"text": "花药壁失水过快", "type": "INTERMEDIATE", "concept": "花药壁失水"},
                {"text": "花药无法正常开裂散粉", "type": "INTERMEDIATE", "concept": "花药开裂受阻"},
                {"text": "花粉粒活性急剧下降", "type": "INTERMEDIATE", "concept": "花粉活力下降"},
                {"text": "空壳率大幅上升", "type": "EFFECT_EVENT", "concept": "空壳率上升"}
            ],
            "causal_pairs": [
                {"cause": {"text": "抽穗扬花期日最高温度持续高于35℃", "type": "CAUSE_EVENT", "concept": "花期高温胁迫"},
                 "effect": {"text": "花药壁失水过快", "type": "INTERMEDIATE", "concept": "花药壁失水"},
                 "relation_type": "CONDITIONAL", "confidence": 0.92, "evidence": "高温使花药壁失水过快"},
                {"cause": {"text": "花药壁失水过快", "type": "INTERMEDIATE", "concept": "花药壁失水"},
                 "effect": {"text": "花药无法正常开裂散粉", "type": "INTERMEDIATE", "concept": "花药开裂受阻"},
                 "relation_type": "DIRECT", "confidence": 0.90, "evidence": "导致花药无法正常开裂散粉"},
                {"cause": {"text": "高温", "type": "CAUSE_EVENT", "concept": "高温胁迫"},
                 "effect": {"text": "花粉粒活性急剧下降", "type": "INTERMEDIATE", "concept": "花粉活力下降"},
                 "relation_type": "DIRECT", "confidence": 0.90, "evidence": "花粉粒活性急剧下降"},
                {"cause": {"text": "花药无法正常开裂散粉", "type": "INTERMEDIATE", "concept": "花药开裂受阻"},
                 "effect": {"text": "空壳率大幅上升", "type": "EFFECT_EVENT", "concept": "空壳率上升"},
                 "relation_type": "DIRECT", "confidence": 0.88, "evidence": "出现空壳率大幅上升"}
            ],
            "causal_chains": [
                {"concepts": ["花期高温胁迫", "花药壁失水", "花药开裂受阻", "空壳率上升"],
                 "events": ["抽穗扬花期日最高温度持续高于35℃", "花药壁失水过快", "花药无法正常开裂散粉", "空壳率大幅上升"],
                 "confidence": 0.90}
            ]
        }
    },
    {
        "input": "田间试验表明，喷施植物源杀菌剂（蒽醌类提取物）后，晚疫病病情指数显著低于对照，防效与全铜处理相当；减铜处理与植物源杀菌剂交替使用，在降低铜投入的同时维持了防效，最终产量显著高于清水对照。",
        "output": {
            "events": [
                {"text": "喷施植物源杀菌剂", "type": "CAUSE_EVENT", "concept": "植物源农药"},
                {"text": "晚疫病病情指数显著低于对照", "type": "EFFECT_EVENT", "concept": "病害减轻"},
                {"text": "减铜处理与植物源杀菌剂交替使用", "type": "CAUSE_EVENT", "concept": "减铜处理"},
                {"text": "降低了铜投入", "type": "EFFECT_EVENT", "concept": "胁迫缓解"},
                {"text": "维持了防效", "type": "EFFECT_EVENT", "concept": "病害减轻"},
                {"text": "产量显著高于清水对照", "type": "EFFECT_EVENT", "concept": "产量提升"}
            ],
            "causal_pairs": [
                {"cause": {"text": "喷施植物源杀菌剂", "type": "CAUSE_EVENT", "concept": "植物源农药"},
                 "effect": {"text": "晚疫病病情指数显著低于对照", "type": "EFFECT_EVENT", "concept": "病害减轻"},
                 "relation_type": "DIRECT", "confidence": 0.92, "evidence": "晚疫病病情指数显著低于对照"},
                {"cause": {"text": "减铜处理与植物源杀菌剂交替使用", "type": "CAUSE_EVENT", "concept": "减铜处理"},
                 "effect": {"text": "降低了铜投入", "type": "EFFECT_EVENT", "concept": "胁迫缓解"},
                 "relation_type": "DIRECT", "confidence": 0.88, "evidence": "降低了铜投入"},
                {"cause": {"text": "减铜处理与植物源杀菌剂交替使用", "type": "CAUSE_EVENT", "concept": "减铜处理"},
                 "effect": {"text": "维持了防效", "type": "EFFECT_EVENT", "concept": "病害减轻"},
                 "relation_type": "DIRECT", "confidence": 0.85, "evidence": "维持了防效"},
                {"cause": {"text": "晚疫病病情指数显著低于对照", "type": "EFFECT_EVENT", "concept": "病害减轻"},
                 "effect": {"text": "产量显著高于清水对照", "type": "EFFECT_EVENT", "concept": "产量提升"},
                 "relation_type": "INDIRECT", "confidence": 0.82, "evidence": "产量显著高于清水对照"}
            ],
            "causal_chains": [
                {"concepts": ["植物源农药", "病害减轻", "产量提升"],
                 "events": ["喷施植物源杀菌剂", "晚疫病病情指数显著低于对照", "产量显著高于清水对照"],
                 "confidence": 0.88}
            ]
        }
    }
]


# ================================================================== #
#  Prompt 构造函数
# ================================================================== #

def build_concept_table(event_types: dict | None = None) -> str:
    """从 agri_event_types.json 动态生成概念表字符串（按 v1.9 子类结构组织）。

    格式示例：
      环境因子(天气/空气温度): 低温冷害,高温胁迫,...
      环境因子(天气/降水): 降水增加,降水减少,...
      环境因子(土壤/土壤湿度): 土壤过湿,...
      生物胁迫(病害): 晚疫病,...
      ...
    """
    from ..utils.data_loader import load_event_types
    et = event_types or load_event_types()
    lines: list[str] = []

    def _collect_concepts(node: dict, prefix: str) -> list[str]:
        """递归收集节点及其子节点的所有 concepts。"""
        cs = list(node.get("concepts", []))
        for sub in node.get("sub_classes", []):
            cs.extend(_collect_concepts(sub, prefix))
        return cs

    for cls in et.get("entity_classes", []):
        cls_name = cls.get("class_name", "")
        # 无子类的实体类（如 Treatment）直接列概念
        if not cls.get("sub_classes"):
            cs = cls.get("concepts", [])
            if cs:
                lines.append(f"{cls_name}: {', '.join(cs)}")
            continue
        # 有子类的实体类，按子类分组
        for sub in cls.get("sub_classes", []):
            sub_name = sub.get("sub_class_name", "")
            # 收集该子类及其嵌套子类的所有概念
            cs = _collect_concepts(sub, sub_name)
            if cs:
                lines.append(f"{cls_name}({sub_name}): {', '.join(cs)}")
            else:
                # 即使无概念也列出（提示 LLM 该子类存在）
                nested = sub.get("sub_classes", [])
                if nested:
                    nested_names = "/".join(n.get("sub_class_name", "") for n in nested)
                    lines.append(f"{cls_name}({sub_name}/{nested_names}): （暂无标准概念）")
    return "\n".join(lines)


# 模块级缓存（首次调用时生成，避免重复加载本体）
_CONCEPT_TABLE_CACHE: str | None = None


def get_concept_table() -> str:
    """获取动态概念表（带缓存）。"""
    global _CONCEPT_TABLE_CACHE
    if _CONCEPT_TABLE_CACHE is None:
        _CONCEPT_TABLE_CACHE = build_concept_table()
    return _CONCEPT_TABLE_CACHE


def build_extraction_messages(text: str, num_fewshot: int = 2) -> list[dict[str, str]]:
    """构建事件+因果对+因果链联合抽取的 chat messages。

    Parameters
    ----------
    text : str
        待抽取的农业文本。
    num_fewshot : int
        few-shot 示例数量（默认2个，平衡效果与成本）。
    """
    # 动态注入概念表
    system_prompt = SYSTEM_PROMPT_EXTRACTION.replace(
        "{CONCEPT_TABLE_PLACEHOLDER}", get_concept_table()
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # 注入 few-shot 示例
    for ex in FEWSHOT_EXAMPLES[:num_fewshot]:
        messages.append({"role": "user", "content": f"请从以下农业文本中抽取事件、因果关系对与因果链：\n\n{ex['input']}"})
        messages.append({
            "role": "assistant",
            "content": "```json\n" + _to_json_str(ex["output"]) + "\n```",
        })

    # 实际待抽取文本
    messages.append({
        "role": "user",
        "content": f"请从以下农业文本中抽取事件、因果关系对与因果链：\n\n{text}",
    })
    return messages


def _to_json_str(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)
