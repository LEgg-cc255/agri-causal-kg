"""重建因果规则库 v3.2：用 v3.2 ConceptAggregator 重新归一数据集 concept。

输入：agri_causal_events.json (gold v1.5, 292对, 20文档)
处理：
  1. 遍历每对 cause_event/effect_event，用 v3.2 ConceptAggregator 重新归一 concept
  2. key = "cause_concept|||effect_concept"，value = confidence
  3. 自循环(cause==effect)→forbidden；低置信度(<0.3)过滤
  4. 去重：同key取max confidence
输出：agri_causal_rules.json (v3.2)

设计：备份原rules，记录concept变化，只增不删。
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from bate.src.event_extraction.concept_aggregator import ConceptAggregator

DATA = Path(r"c:\Users\lenovo\Desktop\开题\bate\data")
EVENTS_PATH = DATA / "agri_causal_events.json"
RULES_PATH = DATA / "agri_causal_rules.json"
BACKUP_DIR = Path(r"c:\Users\lenovo\Desktop\开题\kb备份") / "2026-08-18_v3.2规则库重建前"


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKUP_DIR / RULES_PATH.name
    if not dst.exists():
        shutil.copy2(RULES_PATH, dst)
        print(f"[备份] {RULES_PATH.name} -> {dst}")

    events = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    agg = ConceptAggregator()

    known: dict[str, float] = {}
    forbidden: dict[str, float] = {}
    concept_changes: list[tuple[str, str, str, str]] = []  # (doc, side, old, new)
    pair_count = 0
    self_loop = 0
    low_conf = 0

    for doc in events.get("documents", []):
        did = doc.get("doc_id", "?")
        for pair in doc.get("causal_pairs", []):
            pair_count += 1
            conf = float(pair.get("confidence", 0.85))
            if conf < 0.3:
                low_conf += 1
                continue
            for side_key in ("cause_event", "effect_event"):
                ev = pair.get(side_key, {})
                old_c = ev.get("concept", "") or ""
                text = ev.get("text", "") or ""
                # v3.2 重新归一（旧concept作为LLM hint传入）
                r = agg.aggregate(text, old_c)
                new_c = r["concept"]
                if new_c == "UNKNOWN":
                    # 聚合失败则保留旧concept
                    new_c = old_c if old_c else "UNKNOWN"
                if new_c != old_c and old_c:
                    concept_changes.append((did, side_key, old_c, new_c))
                ev["concept"] = new_c  # 更新数据集（内存中）

            cause_c = pair["cause_event"]["concept"]
            effect_c = pair["effect_event"]["concept"]
            if not cause_c or not effect_c or cause_c == "UNKNOWN":
                continue
            key = f"{cause_c}|||{effect_c}"
            if cause_c == effect_c:
                # 自循环→forbidden
                forbidden[key] = max(forbidden.get(key, 0.1), 0.1)
                self_loop += 1
                continue
            # 已知因果：取max confidence
            known[key] = max(known.get(key, 0.0), conf)

    # 读取原rules保留元信息
    old_rules = {}
    try:
        old_rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass

    new_rules = {
        "_comment": (
            f"Auto-rebuilt v3.2 from gold v1.5 (292 pairs, 20 docs) using v3.2 "
            f"ConceptAggregator (ontology 104 subtypes). Self-loop pairs forbidden. "
            f"Concept re-normalized: {len(concept_changes)} changes. "
            f"Low-conf(<0.3) filtered: {low_conf}."
        ),
        "known_causal_pairs": dict(sorted(known.items(), key=lambda x: -x[1])),
        "forbidden_pairs": dict(sorted(forbidden.items())),
        "causal_paths": old_rules.get("causal_paths", {}),
    }
    RULES_PATH.write_text(
        json.dumps(new_rules, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("【规则库重建完成 v3.2】")
    print(f"  数据集因果对总数:    {pair_count}")
    print(f"  低置信度过滤(<0.3):  {low_conf}")
    print(f"  自循环→forbidden:    {self_loop}")
    print(f"  known_causal_pairs:  {len(known)}")
    print(f"  forbidden_pairs:     {len(forbidden)}")
    print(f"  causal_paths(保留):  {len(new_rules['causal_paths'])}")
    print(f"  concept重新归一变化: {len(concept_changes)} 处")
    if concept_changes:
        print("  变化示例(前10):")
        for did, side, old, new in concept_changes[:10]:
            print(f"    {did}/{side}: {old!r} → {new!r}")
    # 统计新concept覆盖
    all_concepts = set()
    for k in known:
        c, e = k.split("|||")
        all_concepts.add(c); all_concepts.add(e)
    print(f"  规则库涉及concept数: {len(all_concepts)}")
    print(f"  备份: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
