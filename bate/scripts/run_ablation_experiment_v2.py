"""消融实验 v2（修正版，与生产检索逻辑严格对齐）。

相对 v1 脚本的修正（v1 的"完整方法"0.699 与主实验 path recall 0.847 不一致的根因）：
1. 候选路径截断与生产一致：内层 Cypher LIMIT = top_k*3*5 = 75（v1 错误地用 25，
   高温/产量下降等高连通节点的金标链常在截断之外）。
2. 排序使用生产环境完整 4 维公式（含 evidence 维，权重和=1.0）：
   score = avg_conf*0.4 + coverage*0.3 + length_penalty*0.2 + evidence*0.1
   v1 脚本漏配 evidence 维，权重和仅 0.9。
3. 随机排序取 10 个随机种子报告均值±标准差（v1 单种子 seed=42）。
4. 同时报告两个队列：38 题有金标题 / 全部 61 题（无题按惯例记 1.0）。
   与主实验对比时统一使用 38 题队列。

5 组消融：
  A. 完整方法（生产 search_paths：4维排序, max_hops=4, 全部3子关系）
  B. w/o 4维排序（候选相同，按 avg_conf 单维排序）
  C. w/o 多跳（max_hops=1）
  D. w/o 3子关系（仅 TRIGGERS）
  E. w/o 路径排序（10 种子随机）
跳数实验：max_hops ∈ {1..5}

输出: bate/data/ablation_experiment_v2.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.kg import AgriCausalGraph
from bate.src.kg.question_parser import QuestionParser
from bate.src.kg.neo4j_client import ALL_CAUSAL_RELS

TOP_K = 5
RAW_LIMIT = TOP_K * 3 * 5  # 75，与生产 _find_paths_with_evidence 一致


def _build_rows(graph: AgriCausalGraph, target: str, direction: str,
                max_hops: int, rels: str) -> list[dict[str, Any]]:
    if not target or target == "UNKNOWN":
        return []
    if direction == "forward":
        start_c, end_c = target, None
    else:
        start_c, end_c = None, target
    where, params = [], {"limit": RAW_LIMIT}
    if start_c:
        where.append("s.concept = $s_c"); params["s_c"] = start_c
    if end_c:
        where.append("e.concept = $e_c"); params["e_c"] = end_c
    q = (
        f"MATCH path = (s)-[r:{rels}*1..{int(max_hops)}]->(e) "
        "WHERE " + (" AND ".join(where) if where else "true") + " "
        "RETURN [node IN nodes(path) | node.concept] AS concepts, "
        "       [rel IN relationships(path) | rel.confidence] AS confs, "
        "       [rel IN relationships(path) | rel.evidence] AS evidences, "
        "       length(path) AS len "
        "LIMIT $limit"
    )
    rows = []
    with graph._open_session() as sess:
        for rec in sess.run(q, params):
            cs = rec["concepts"]
            confs = rec["confs"] or []
            evidences = rec["evidences"] or []
            l = int(rec["len"])
            if not confs:
                continue
            conf_vals = [float(x) if x is not None else 0.0 for x in confs]
            avg_conf = sum(conf_vals) / len(conf_vals)
            coverage = len(set(cs)) / max(len(cs), 1)
            evi_count = sum(1 for ev in evidences if ev and str(ev).strip())
            evidence_score = min(evi_count, 5) / 5.0
            length_penalty = 1.0 / (l + 1)
            rows.append({
                "concepts": cs,
                "_avg_conf": avg_conf,
                "_coverage": coverage,
                "_length_penalty": length_penalty,
                "_evidence_score": evidence_score,
            })
    return rows


def recall_at_k(paths: list[dict], expected: list[str], k: int = TOP_K) -> float:
    if not expected:
        return 1.0
    retrieved = set()
    for p in paths[:k]:
        retrieved.update(p["concepts"])
    return sum(1 for c in expected if c in retrieved) / len(expected)


def _summarize(recalls: list[float], n_paths: list[float], has_gold: list[bool]):
    g = [r for r, h in zip(recalls, has_gold) if h]
    return {
        "recall_38": round(float(np.mean(g)), 3),
        "recall_61": round(float(np.mean(recalls)), 3),
        "avg_paths": round(float(np.mean(n_paths)), 1),
    }


def main() -> None:
    os.environ.setdefault("NEO4J_PASSWORD", "123456")
    data_path = _ROOT / "bate" / "data" / "agri_causal_qa_dataset.json"
    with open(data_path, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    print(f"[消融v2] 题数 {len(questions)}")

    graph = AgriCausalGraph.from_env()
    parser = QuestionParser()
    parses = [(q, parser.parse(q["question"])) for q in questions]
    has_gold = [bool(q.get("expected_concepts")) for q in questions]

    def retrieve(q_parse, expected, max_hops, sort_mode, rels, seed=42):
        rows = _build_rows(graph, q_parse.target_concept, q_parse.direction,
                           max_hops, rels)
        if sort_mode == "4dim":
            rows.sort(key=graph._score_path, reverse=True)
        elif sort_mode == "conf_only":
            rows.sort(key=lambda x: x["_avg_conf"], reverse=True)
        elif sort_mode == "random":
            random.Random(seed).shuffle(rows)
        top = rows[:TOP_K]
        return recall_at_k(top, expected), len(top)

    configs = [
        ("完整方法",          dict(max_hops=4, sort_mode="4dim",     rels=ALL_CAUSAL_RELS)),
        ("w/o 4维排序",       dict(max_hops=4, sort_mode="conf_only", rels=ALL_CAUSAL_RELS)),
        ("w/o 多跳(=1跳)",    dict(max_hops=1, sort_mode="4dim",     rels=ALL_CAUSAL_RELS)),
        ("w/o 3子关系",       dict(max_hops=4, sort_mode="4dim",     rels="TRIGGERS")),
        ("w/o 路径排序(随机)", dict(max_hops=4, sort_mode="random",   rels=ALL_CAUSAL_RELS)),
    ]

    results = {}
    for name, cfg in configs:
        if name == "w/o 路径排序(随机)":
            seed_means_38, seed_means_61 = [], []
            n_paths_last = None
            for seed in range(10):
                recalls, nps = [], []
                for (q, pp), h in zip(parses, has_gold):
                    r, npth = retrieve(pp, q.get("expected_concepts", []),
                                       cfg["max_hops"], cfg["sort_mode"], cfg["rels"], seed)
                    recalls.append(r); nps.append(npth)
                n_paths_last = nps
                seed_means_38.append(float(np.mean([x for x, hh in zip(recalls, has_gold) if hh])))
                seed_means_61.append(float(np.mean(recalls)))
            s = _summarize(recalls, n_paths_last, has_gold)
            s["recall_38"] = round(float(np.mean(seed_means_38)), 3)
            s["recall_61"] = round(float(np.mean(seed_means_61)), 3)
            s["std_38"] = round(float(np.std(seed_means_38)), 3)
            results[name] = s
        else:
            recalls, nps = [], []
            for (q, pp), h in zip(parses, has_gold):
                r, npth = retrieve(pp, q.get("expected_concepts", []),
                                   cfg["max_hops"], cfg["sort_mode"], cfg["rels"])
                recalls.append(r); nps.append(npth)
            results[name] = _summarize(recalls, nps, has_gold)
        print(f"  {name:20s} {results[name]}")

    hops = {}
    for h in range(1, 6):
        recalls, nps = [], []
        for (q, pp), hg in zip(parses, has_gold):
            r, npth = retrieve(pp, q.get("expected_concepts", []), h, "4dim", ALL_CAUSAL_RELS)
            recalls.append(r); nps.append(npth)
        g = [r for r, hh in zip(recalls, has_gold) if hh]
        hops[str(h)] = {
            "recall_38": round(float(np.mean(g)), 3),
            "recall_61": round(float(np.mean(recalls)), 3),
        }
        print(f"  hops={h} {hops[str(h)]}")

    out = {"note": "38=有金标题队列(与主实验同口径); 61=全部题(无金标题记1.0)",
           "ablation": results, "hops": hops}
    out_path = _ROOT / "bate" / "data" / "ablation_experiment_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")
    graph.close()


if __name__ == "__main__":
    main()
