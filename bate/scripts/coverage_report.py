"""扫描当前金标集(264对)实际覆盖了本体多少 subtype，列出零覆盖缺口。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.utils.data_loader import load_causal_events, load_event_types, load_synonym_dict


def build_concept_subtype_map(et, syn):
    """concept -> (sub_class_id, sub_class_name, class_name)"""
    m: dict[str, tuple[str, str, str]] = {}
    for cls in et.get("entity_classes", []):
        cn = cls["class_name"]
        for sub in cls.get("sub_classes", []):
            sid = sub["sub_class_id"]
            sn = sub["sub_class_name"]
            for concept in sub.get("concepts", []) or []:
                m[concept] = (sid, sn, cn)
    # synonym concept_to_subtype 覆盖一层
    for concept, sid in syn.get("concept_to_subtype", {}).items():
        if concept in m:
            continue
        # 再从 et 查 sid -> (sn, cn)
        for cls in et.get("entity_classes", []):
            for sub in cls.get("sub_classes", []):
                if sub["sub_class_id"] == sid:
                    m[concept] = (sid, sub["sub_class_name"], cls["class_name"])
                    break
    return m


def main():
    et = load_event_types()
    syn = load_synonym_dict()
    ds = load_causal_events()
    c2s = build_concept_subtype_map(et, syn)

    # 统计 264 对里 cause/effect 每个 concept 出现次数 -> 统计每个 subtype 命中次数
    subtype_hits: dict[str, int] = Counter()
    class_hits: dict[str, int] = Counter()
    concept_hits: Counter = Counter()

    for doc in ds["documents"]:
        for p in doc["causal_pairs"]:
            for role in ("cause_event", "effect_event"):
                cc = p[role]["concept"]
                concept_hits[cc] += 1
                if cc in c2s:
                    sid, sn, cn = c2s[cc]
                    subtype_hits[sid] += 1
                    class_hits[cn] += 1

    # 列出每个大类下每个 subtype 命中数（0命中的标红）
    print("=" * 96)
    print(f"本体 subtype 覆盖报告（金标集 {sum(len(d['causal_pairs']) for d in ds['documents'])} 对 × 2 端 = {sum(concept_hits.values())} concept 位置）")
    print("=" * 96)

    total_subtypes = 0
    covered_subtypes = 0
    for cls in et.get("entity_classes", []):
        cn = cls["class_name"]
        subs = cls.get("sub_classes", [])
        print(f"\n【{cn}】({len(subs)} subtypes, 命中 {sum(1 for s in subs if subtype_hits.get(s['sub_class_id'], 0) > 0)} 个)")
        print(f"  总命中数: {class_hits.get(cn, 0)}")
        for sub in subs:
            total_subtypes += 1
            sid = sub["sub_class_id"]
            h = subtype_hits.get(sid, 0)
            if h == 0:
                covered_subtypes += 0
                marker = "✗ ZERO"
            else:
                total_subtypes  # keep counter above right
                covered_subtypes += 0 if 0 else 1
                marker = f"✓ hits={h}"
            # 修正 covered counter
            if h > 0:
                pass  # 下面重算
            concepts = sub.get("concepts", []) or []
            show = concepts[:3]
            extra = f" ...+{len(concepts)-3}" if len(concepts) > 3 else ""
            print(f"  {marker:<12s} {sid:24s} {sub['sub_class_name']:16s}  concepts=[{', '.join(show)}{extra}]")
    # 重算上面的覆盖
    _covered = sum(1 for cls in et["entity_classes"]
                   for sub in cls["sub_classes"] if subtype_hits.get(sub["sub_class_id"], 0) > 0)
    _total = sum(len(cls["sub_classes"]) for cls in et["entity_classes"])
    print(f"\n{'='*96}")
    print(f"总 subtype: {_total}  已覆盖: {_covered}  未覆盖(零命中): {_total - _covered}  覆盖率: {_covered/_total*100:.1f}%")
    print("=" * 96)

    # 每个 CLASS 的建议
    print("\n【建议扩展方向（按优先级）】")
    priority = []
    for cls in et.get("entity_classes", []):
        cn = cls["class_name"]
        zeros = [s for s in cls["sub_classes"] if subtype_hits.get(s["sub_class_id"], 0) == 0]
        class_total = sum(len(doc["causal_pairs"]) for doc in ds["documents"])
        # 更细：计算该类概念在 cause/effect 两端出现次数
        hits = class_hits.get(cn, 0)
        if len(zeros) > 0:
            priority.append((len(zeros), hits, cn, zeros))
    priority.sort(key=lambda x: (-x[0], x[1]))  # 未覆盖数最多、已有命中数最少的先排
    for nz, hits, cn, zeros in priority:
        names = ", ".join(s["sub_class_name"] for s in zeros[:5])
        extra = f"等{len(zeros)}个" if len(zeros) > 5 else ""
        if hits == 0:
            state = "【大类零命中，最严重】"
        elif nz >= 4:
            state = "【缺口大】"
        else:
            state = "【有个别空】"
        print(f"  {state} {cn}: 已有命中{hits}次，缺 {nz} 个 subtype → {names}{extra}")

    # 作物覆盖情况
    print(f"\n{'='*96}")
    print("作物/领域覆盖情况")
    print("=" * 96)
    domain_counter = Counter()
    for doc in ds["documents"]:
        domain_counter[doc.get("domain", "?")] += 1
    for d, c in domain_counter.most_common():
        dids = [dd["doc_id"] for dd in ds["documents"] if dd.get("domain") == d]
        pairs = sum(len(dd["causal_pairs"]) for dd in ds["documents"] if dd.get("domain") == d)
        print(f"  {d:18s}  {c} docs ({', '.join(dids)})  {pairs} pairs")


if __name__ == "__main__":
    main()
