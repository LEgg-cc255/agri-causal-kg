"""新增 subtype/concept 扩展完整性自检

每次在本体加了新 subtype/concept 后运行：
    python -m bate.scripts.check_new_subtype_integrity

检查 4 类缺口（对应扩展标准流程 Step 1~4）：

  A. 本体 concept 未在 concept_to_subtype 注册
       → 缺 A 会让 _wrap() 解析出 UNKNOWN sub_class_id/entity_class，
          导致本体方向过滤静默失效（本次最初的 21 个 concept 就是这个）

  B. 同义词典 concept_to_subtype 有、但本体缺失
       → 词典比本体多了幽灵 concept，以后 LLM 输出时策略3会命中，
          但 all_concepts 里找不到，影响一致性

  C. 本体 concept 在 mention_to_concept 中作为"归一目标准"出现，
     但 mention_to_concept 找不到该 concept 的任何 mention 键
       → 加了 concept 但从没给它配同义词，只能靠 LLM 输出正确才会命中，
          回归概率高（这次的"光照充足/遮阳措施"最初就是这个状态）

  D. （仅 entity_class 级）causal_extractor 的方向矩阵缺新加的大类对
       → 如果新增了 CLASS-XXX，需要补 valid/invalid_directions
          （subtype 级不用，因为方向矩阵是 entity_class 级的）

输出：按严重程度 A→B→C→D 列，附修复指引。
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.utils.data_loader import (
    load_event_types,
    load_synonym_dict,
    get_all_concepts,
)


def _skip_comment_keys(d: dict) -> dict:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def load_entity_classes() -> set[str]:
    """从本体提取所有 entity_class 名称（class_name）。"""
    et = load_event_types()
    classes: set[str] = set()
    for cls in et.get("entity_classes", []):
        cn = cls.get("class_name")
        if cn:
            classes.add(cn)
    return classes


def load_direction_classes() -> tuple[set[str], set[str]]:
    """从 causal_extractor._ontology_check 提取 valid_directions 涉及的
    cause_class 集合与 effect_class 集合（用于判断 D 类缺项）。"""
    # 保持与 causal_extractor.py valid_directions 同步
    valid = {
        ("环境胁迫事件", "生理生化事件"),
        ("环境胁迫事件", "发育与产量事件"),
        ("环境胁迫事件", "生物胁迫事件"),
        ("管理措施事件", "生理生化事件"),
        ("管理措施事件", "发育与产量事件"),
        ("管理措施事件", "环境胁迫事件"),
        ("生物胁迫事件", "生理生化事件"),
        ("生物胁迫事件", "发育与产量事件"),
        ("生理生化事件", "生理生化事件"),
        ("生理生化事件", "发育与产量事件"),
        ("发育与产量事件", "发育与产量事件"),
    }
    cause_classes = {a for a, _ in valid}
    effect_classes = {b for _, b in valid}
    return cause_classes, effect_classes


def extract_ontology(
    event_types: dict,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """提取本体三类关键信息：

    Returns
    -------
    (canonical_concepts, synonym_mentions, all_mentions)
      canonical_concepts: 标准 concept -> sub_class_id（每个 subtype concepts[0]）
      synonym_mentions:   本体同义 mention -> 对应标准 concept（concepts[1:]）
      all_mentions:       本体里出现过的所有 mention（含标准 concept 自己）
    """
    canonical: dict[str, str] = {}  # concept -> sub_class_id
    synonyms: dict[str, str] = {}   # mention -> canonical
    all_mentions: set[str] = set()
    for cls in event_types.get("entity_classes", []):
        for sub in cls.get("sub_classes", []):
            concepts = sub.get("concepts") or []
            if not concepts:
                continue
            canon = concepts[0]
            sid = sub["sub_class_id"]
            canonical[canon] = sid
            all_mentions.add(canon)
            for mention in concepts[1:]:
                synonyms[mention] = canon
                all_mentions.add(mention)
    return canonical, synonyms, all_mentions


def main() -> None:
    event_types = load_event_types()
    syn = load_synonym_dict()

    canonical_concepts, ontology_synonyms, all_ontology_mentions = extract_ontology(event_types)
    concept_to_subtype: dict[str, str] = _skip_comment_keys(syn.get("concept_to_subtype", {}))
    mention_to_concept: dict[str, str] = _skip_comment_keys(syn.get("mention_to_concept", {}))

    # A. 本体【标准 concept】未在 concept_to_subtype 注册
    a_missing = sorted(c for c in canonical_concepts if c not in concept_to_subtype)

    # B. concept_to_subtype 有键不在【标准 concept】集合（幽灵 concept）
    #    说明：本体同义 mention 本来就不该在 concept_to_subtype 里（它们是 mention），
    #    所以 B 只拿 canonical 当参照
    b_ghost = sorted(c for c in concept_to_subtype if c not in canonical_concepts)

    # C1. 【标准 concept】在 mention_to_concept 里没有锚定（自映射或作为归一目标准）
    concept_targets: set[str] = set(mention_to_concept.values())
    canon_self_anchored = {m for m, t in mention_to_concept.items() if m == t and m in canonical_concepts}
    canon_anchored = concept_targets | canon_self_anchored
    c1_missing = sorted(c for c in canonical_concepts if c not in canon_anchored)

    # C2. 【本体同义 mention】在 mention_to_concept 里缺精确键
    #     （模糊匹配可能会被更长/更短 mention 截断，精确键更稳妥）
    c2_missing = sorted(m for m in ontology_synonyms if m not in mention_to_concept)

    # D. 新 entity_class 未进方向矩阵
    ontology_classes = load_entity_classes()
    dir_cause, dir_effect = load_direction_classes()
    d_orphan = sorted(
        c for c in ontology_classes
        if c not in dir_cause and c not in dir_effect
    )
    # 更精细：任何本体大类，作为 cause 时不在 dir_cause 里 → valid_directions 缺它作为原因
    d_missing_as_cause = sorted(c for c in ontology_classes if c not in dir_cause)

    # ---------- 输出 ----------
    W, H = 72, "="
    print(H * W)
    print("New Subtype/Concept 扩展完整性自检")
    print(H * W)
    print(f"\n[概览]")
    print(f"  本体标准概念(canonical): {len(canonical_concepts)}")
    print(f"  本体同义 mention: {len(ontology_synonyms)}")
    print(f"  concept_to_subtype 条目: {len(concept_to_subtype)}")
    print(f"  mention_to_concept 条目: {len(mention_to_concept)}")
    print(f"  本体 entity_class 数: {len(ontology_classes)}  ({', '.join(sorted(ontology_classes))})")

    # A
    print(f"\n{H * W}")
    print(f"【A】本体【标准概念】缺 concept_to_subtype 映射（共 {len(a_missing)}）")
    print(H * W)
    if a_missing:
        print("  影响: _wrap() sub_class_id=UNKNOWN → entity_class=UNKNOWN → 方向过滤失效")
        print("  修复: 在 agri_synonym_dict.json 的 concept_to_subtype 里加：")
        for c in a_missing:
            # 尝试从本体里找对应的 sub_class_id 提示
            sid = canonical_concepts.get(c, "<sub_class_id>")
            # 找 class_name 提示
            class_name = ""
            for cls in event_types.get("entity_classes", []):
                for sub in cls.get("sub_classes", []):
                    if sub.get("sub_class_id") == sid:
                        class_name = cls.get("class_name", "")
                        break
            hint = f"  # 建议映射到: {sid} ({class_name})" if class_name else ""
            print(f'    "{c}": "<sub_class_id>",{hint}')
    else:
        print("  ✓ 全部对齐")

    # B
    print(f"\n{H * W}")
    print(f"【B】concept_to_subtype 有、但本体缺失（幽灵概念，共 {len(b_ghost)}）")
    print(H * W)
    if b_ghost:
        print("  影响: 策略3会命中，但 all_concepts 查不到，LLM输出这些概念时策略2/1也不覆盖")
        print("  修复二选一: (a) 本体对应 subtype 加 concept；(b) 从 concept_to_subtype 删除")
        for c in b_ghost:
            sid = concept_to_subtype[c]
            print(f'    "{c}" (当前映射到 {sid})')
    else:
        print("  ✓ 无幽灵")

    # C1
    print(f"\n{H * W}")
    print(f"【C1】标准 concept 在 mention_to_concept 无锚定（共 {len(c1_missing)}）")
    print(H * W)
    if c1_missing:
        print("  影响: 只能靠 LLM 策略2/3 正确输出 concept 才能命中；LLM输出错时词典无法兜底")
        print("  修复: 加一条自映射（或给它配同义词 mention）：")
        for c in c1_missing:
            print(f'    "{c}": "{c}",')
    else:
        print("  ✓ 全部标准 concept 有锚定")

    # C2
    print(f"\n{H * W}")
    print(f"【C2】本体同义 mention 缺 mention_to_concept 精确键（共 {len(c2_missing)}）")
    print(H * W)
    if c2_missing:
        print("  影响: 模糊匹配(包含子串)可能误匹配到更短 mention, 归一错概念(这次的DOC-001就是)")
        print("  修复: 本体里有这些同义 mention, 建议同步到 mention_to_concept：")
        top = c2_missing[:50]
        for m in top:
            canon = ontology_synonyms[m]
            print(f'    "{m}": "{canon}",')
        if len(c2_missing) > 50:
            print(f"    ... 剩余 {len(c2_missing) - 50} 项（格式同上, mention→对应标准概念）")
    else:
        print("  ✓ 本体所有同义 mention 在 mention_to_concept 均有精确键")

    # D
    print(f"\n{H * W}")
    print(f"【D】本体 entity_class 缺方向矩阵覆盖（共 {len(d_orphan)}）")
    print(H * W)
    if d_missing_as_cause:
        print("  说明: causal_extractor 的 valid_directions 是 entity_class 级（大类间合法方向对）。")
        print("        下面这些大类从没作为合法原因出现，若它们实际上也能触发其它事件，需补对。")
        print("        作为「结果」方向缺的类不单独列，因为多数类都可当结果。")
        for c in d_missing_as_cause:
            print(f"    - {c}  → 在 valid_directions 里没有以它为 cause 的合法对")
    else:
        print("  ✓ 所有大类均在方向矩阵中作为原因出现过")

    # 汇总
    total_issues = len(a_missing) + len(b_ghost) + len(c1_missing) + len(c2_missing) + len(d_orphan)
    print(f"\n{H * W}")
    if total_issues == 0:
        print("✓ 完美通过。可以放心地用这些新 subtype/concept 去跑数据。")
    else:
        print(f"共发现 {total_issues} 项缺口:"
              f" A(标准概念缺映射)={len(a_missing)}"
              f"  B(幽灵概念)={len(b_ghost)}"
              f"  C1(标准概念缺锚定)={len(c1_missing)}"
              f"  C2(本体同义mention缺精确键)={len(c2_missing)}"
              f"  D(大类方向缺)={len(d_orphan)}")
        print("严重度排序：A > B > C1 > C2 > D  "
              "（A方向过滤静默失效最致命；D除非加新CLASS，否则一般都0）")
    print(H * W)


if __name__ == "__main__":
    main()
