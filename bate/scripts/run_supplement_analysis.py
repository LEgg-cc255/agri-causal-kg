"""补充实验：三分档幻觉口径 + 分组Recall分析。

三分档：
- verified: 因果对命中图谱（TRIGGERS|PROPAGATES|ALLEVIATES|CAUSES）
- contradiction: 因果对命中 FORBIDDEN_CAUSES 禁止边（真幻觉）
- out_of_kg: 图谱覆盖外（概念不在图谱 / 概念在但无边）——不一定是错的

输出: bate/data/supplement_analysis.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from bate.src.kg import AgriCausalGraph
from bate.src.event_extraction.concept_aggregator import ConceptAggregator
from bate.scripts.run_hallucination_analysis import _extract_causal_pairs


def main() -> None:
    os.environ.setdefault("NEO4J_PASSWORD", "123456")

    exp_path = _ROOT / "bate" / "data" / "experiment_results.json"
    with open(exp_path, encoding="utf-8") as f:
        results = json.load(f)["results"]

    graph = AgriCausalGraph.from_env()
    aggregator = ConceptAggregator()

    # 加载因果边（4种）与禁止边
    with graph._open_session() as s:
        pairs = {(r["cause"], r["effect"]) for r in s.run(
            "MATCH (a)-[r:TRIGGERS|PROPAGATES|ALLEVIATES|CAUSES]->(b) "
            "RETURN a.concept AS cause, b.concept AS effect").data() if r["cause"] and r["effect"]}
        forbidden = {(r["cause"], r["effect"]) for r in s.run(
            "MATCH (a)-[r:FORBIDDEN_CAUSES]->(b) "
            "RETURN a.concept AS cause, b.concept AS effect").data() if r["cause"] and r["effect"]}
        # 反向边（图里存在 Y->X）用于检测方向颠倒型矛盾
        reverse = {(e, c) for (c, e) in pairs}
    print(f"[图谱] 因果对 {len(pairs)} | 禁止边 {len(forbidden)}")

    def analyze(text: str) -> dict:
        raw = _extract_causal_pairs(text)
        n = {"verified": 0, "contradiction": 0, "rev_dir": 0,
             "okg_concept": 0, "okg_relation": 0}
        for cause_t, effect_t in raw:
            cc = aggregator.aggregate(cause_t, "").get("concept", "UNKNOWN")
            ec = aggregator.aggregate(effect_t, "").get("concept", "UNKNOWN")
            if cc == "UNKNOWN" or ec == "UNKNOWN":
                n["okg_concept"] += 1
            elif (cc, ec) in pairs:
                n["verified"] += 1
            elif (cc, ec) in forbidden:
                n["contradiction"] += 1          # 真幻觉：命中禁止边
            elif (ec, cc) in pairs:
                n["rev_dir"] += 1                # 真幻觉：因果方向颠倒
            else:
                n["okg_relation"] += 1           # 图谱未覆盖（不必然错）
        total = sum(n.values())
        return {**n, "total": total}

    agg = {}
    for method, key in [("baseline", "baseline_answer"), ("causalrag", "causalrag_answer")]:
        s = {"verified": 0, "contradiction": 0, "rev_dir": 0, "okg_concept": 0,
             "okg_relation": 0, "total": 0}
        for r in results:
            a = analyze(r[key])
            for k in s:
                s[k] += a[k]
        t = s["total"]
        agg[method] = {
            **s,
            "verification_rate": round(s["verified"] / t, 3) if t else 0,
            "true_hallucination_rate": round((s["contradiction"] + s["rev_dir"]) / t, 3) if t else 0,
            "out_of_kg_rate": round((s["okg_concept"] + s["okg_relation"]) / t, 3) if t else 0,
        }
        print(f"\n[{method}] total={t}")
        for k, v in agg[method].items():
            if k != "total":
                print(f"  {k}: {v}")

    # 分组 Recall（方向/难度）
    groups: dict[str, dict] = {}
    for r in results:
        for dim in ("direction", "difficulty"):
            g = r.get(dim) or "unknown"
            groups.setdefault(dim, {}).setdefault(g, {"b": [], "c": []})
            groups[dim][g]["b"].append(r.get("baseline_recall", 0))
            groups[dim][g]["c"].append(r.get("causalrag_recall", 0))
    grouped = {}
    for dim, sub in groups.items():
        grouped[dim] = {}
        for g, v in sub.items():
            grouped[dim][g] = {
                "n": len(v["b"]),
                "baseline_recall": round(float(np.mean(v["b"])), 3),
                "causalrag_recall": round(float(np.mean(v["c"])), 3),
            }

    out = {"three_tier_hallucination": agg, "grouped_recall": grouped}
    out_path = _ROOT / "bate" / "data" / "supplement_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")
    graph.close()


if __name__ == "__main__":
    main()
