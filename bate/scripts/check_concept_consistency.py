"""数据一致性校验：标注数据集中的 concept 是否已在本体/同义词典注册

扫描 agri_causal_events.json 中所有因果对与因果链的 concept 字段，对比：
  1. agri_event_types.json 的本体概念列表（entity_classes 下所有 sub_classes 的 concepts）
  2. agri_synonym_dict.json 的 concept_to_subtype 键（标准概念 → 子类映射）

输出三类问题：
  A. 完全未注册 —— concept 既不在本体也不在同义词典
       后果：概念聚合时聚合器查不到，返回 UNKNOWN，事件归一与方向过滤全失效
       修复：把该 concept 补进本体 + 同义词典；或把标注的 concept 归一到已有标准概念
  B. 本体有但缺 subtype 映射 —— concept 在本体，但 concept_to_subtype 没有
       后果：聚合能识别概念，但 _wrap 里 sub_class_id/entity_class 解析为 UNKNOWN，
             本体方向过滤（entity_class 级别）失效
       修复：在 agri_synonym_dict.json 的 concept_to_subtype 里补一条映射
  C. 数据层交叉不一致（信息性）—— 词典与本体互相缺失，影响维护

运行：python -m bate.scripts.check_concept_consistency
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.utils.data_loader import (
    load_causal_events,
    load_event_types,
    load_synonym_dict,
    get_all_concepts,
)


def _skip_comment_keys(d: dict) -> dict:
    """过滤以 _ 开头的注释/元信息键（如 _comment）。"""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def collect_annotation_concepts(dataset: dict) -> dict[str, list[dict]]:
    """收集标注数据集中所有 concept 及其出现位置。

    Returns
    -------
    dict[str, list[dict]]
        concept -> [{doc_id, where, text}]，text 为该事件的原文表述（链节点无 text）。
    """
    occ: dict[str, list[dict]] = defaultdict(list)
    for doc in dataset.get("documents", []):
        doc_id = doc.get("doc_id", "?")
        for pair in doc.get("causal_pairs", []):
            for role in ("cause_event", "effect_event"):
                ev = pair.get(role, {})
                c = (ev.get("concept") or "").strip()
                if c:
                    occ[c].append({
                        "doc_id": doc_id,
                        "where": f"{pair.get('pair_id', '?')}.{role}",
                        "text": (ev.get("text") or "").strip(),
                    })
        for chain in doc.get("causal_chains", []):
            for c in chain.get("concepts") or []:
                c = (c or "").strip()
                if c:
                    occ[c].append({
                        "doc_id": doc_id,
                        "where": f"{chain.get('chain_id', '?')}.concepts",
                        "text": "",
                    })
    return occ


def _print_occurrences(concepts: list[str], occ: dict[str, list[dict]]) -> None:
    """按 concept 打印其所有出现位置。"""
    for c in sorted(concepts):
        locs = occ[c]
        print(f"  - 「{c}」 ({len(locs)} 处)")
        for loc in locs:
            text = f"  原文: {loc['text']}" if loc["text"] else ""
            print(f"      {loc['doc_id']} / {loc['where']}{text}")
        print()


def main() -> None:
    dataset = load_causal_events()
    event_types = load_event_types()
    syn = load_synonym_dict()

    # 参考集合
    ontology_concepts: set[str] = set(get_all_concepts(event_types))
    concept_to_subtype: dict[str, str] = _skip_comment_keys(syn.get("concept_to_subtype", {}))
    mention_to_concept: dict[str, str] = _skip_comment_keys(syn.get("mention_to_concept", {}))
    subtype_concepts: set[str] = set(concept_to_subtype.keys())
    mention_targets: set[str] = set(mention_to_concept.values())

    ann = collect_annotation_concepts(dataset)

    # 分类
    unregistered = {c for c in ann if c not in ontology_concepts and c not in subtype_concepts}
    missing_subtype = {c for c in ann if c in ontology_concepts and c not in subtype_concepts}
    registered = {c for c in ann if c in ontology_concepts or c in subtype_concepts}

    # 数据层交叉不一致（信息性）
    dict_not_in_onto = sorted(subtype_concepts - ontology_concepts)
    mention_not_registered = sorted(
        {t for t in mention_targets if t not in subtype_concepts and t not in ontology_concepts}
    )

    # ---------- 输出 ----------
    print("=" * 64)
    print("标注数据集 concept 一致性校验报告")
    print("=" * 64)

    total = len(ann)
    print(f"\n[统计] 标注中出现的不重复 concept 数: {total}")
    print(f"  - 已注册（本体或同义词典命中）: {len(registered)}")
    print(f"  - 完全未注册（A）: {len(unregistered)}")
    print(f"  - 本体有但缺 subtype 映射（B）: {len(missing_subtype)}")

    # A
    print(f"\n{'=' * 64}")
    print(f"【A】完全未注册 concept（{len(unregistered)} 个）—— 会导致聚合变 UNKNOWN")
    print("=" * 64)
    if unregistered:
        _print_occurrences(sorted(unregistered), ann)
    else:
        print("  （无）")

    # B
    print(f"\n{'=' * 64}")
    print(f"【B】本体有但缺 concept_to_subtype 映射（{len(missing_subtype)} 个）—— 方向过滤失效")
    print("=" * 64)
    if missing_subtype:
        _print_occurrences(sorted(missing_subtype), ann)
    else:
        print("  （无）")

    # C
    print(f"\n{'=' * 64}")
    print("【C】数据层交叉不一致（信息性，非标注直接问题）")
    print("=" * 64)
    if dict_not_in_onto:
        print(f"\n  同义词典 concept_to_subtype 有、但本体缺失（{len(dict_not_in_onto)} 个）:")
        print("    " + "、".join(dict_not_in_onto))
    else:
        print("\n  同义词典 concept_to_subtype 与本体完全对齐。")
    if mention_not_registered:
        print(f"\n  mention_to_concept 的归一目标准、但未注册（{len(mention_not_registered)} 个）:")
        print("    " + "、".join(mention_not_registered))
    else:
        print("\n  mention_to_concept 的归一目标均已注册。")

    # D. mention_to_concept 重复键检测（JSON 后者覆盖前者）
    import re as _re
    from collections import Counter as _Counter
    from bate.src.utils.data_loader import get_data_dir as _get_data_dir
    syn_raw = (_get_data_dir() / "agri_synonym_dict.json").read_text(encoding="utf-8")
    _m = _re.search(r'"mention_to_concept":\s*\{(.*?)\n  \}', syn_raw, _re.S)
    dups: list[str] = []
    if _m:
        keys = _re.findall(r'"([^"]+)":', _m.group(1))
        dups = [k for k, c in _Counter(keys).items() if c > 1]

    print(f"\n{'=' * 64}")
    print(f"【D】mention_to_concept 重复键（共 {len(dups)}）")
    print("=" * 64)
    if dups:
        print("  影响: JSON 重复键后者覆盖前者，先加的修复静默失效")
        print("  修复: 删掉旧的重复键，保留正确的")
        for k in dups:
            print(f'    "{k}" 出现 {_Counter(keys)[k]} 次')
    else:
        print("  ✓ 无重复键")

    print(f"\n{'=' * 64}")
    print("修复建议：A → 补本体+词典；B → 补 concept_to_subtype 映射；D → 删旧重复键。")
    print("=" * 64)


if __name__ == "__main__":
    main()
