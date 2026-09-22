"""消融实验 + 路径跳数实验（图谱端消融，不改源码）。

5 组消融：
  A. 完整方法（CausalRAG, max_hops=4, top_k=5, 4维排序）
  B. w/o 4维排序（路径按 avg_conf 单维排序）
  C. w/o 多跳路径（max_hops=1，仅直接邻居）
  D. w/o 3子关系（仅匹配 CAUSES 旧关系，若存在）
  E. w/o 路径排序（随机打乱，取前 top_k）

跳数实验：max_hops ∈ {1, 2, 3, 4, 5}

评估口径：Recall@k（前 top_k 路径覆盖期望概念的比例）。
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
from bate.src.kg.neo4j_client import ALL_CAUSAL_RELS, LABEL_ENTITY


def recall_at_k(paths: list[dict], expected: list[str], k: int = 5) -> float:
    """前 k 条路径覆盖期望概念的比例。"""
    if not expected:
        return 1.0
    retrieved = set()
    for p in paths[:k]:
        retrieved.update(p.get("concepts", []))
    hits = sum(1 for c in expected if c in retrieved)
    return hits / len(expected)


def run_ablation():
    os.environ.setdefault("NEO4J_PASSWORD", "123456")
    rng = random.Random(42)

    data_path = _ROOT / "bate" / "data" / "agri_causal_qa_dataset.json"
    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]
    print(f"[消融] 题数 {len(questions)}")

    graph = AgriCausalGraph.from_env()
    parser = QuestionParser()

    # 提前解析所有问题
    parses = []
    for q in questions:
        p = parser.parse(q["question"])
        parses.append((q, p))

    def search(target: str, direction: str, max_hops: int, top_k: int,
               sort_mode: str = "4dim", rel_filter: str = "all") -> list[dict]:
        """统一检索接口，支持消融选项。

        sort_mode: 4dim / conf_only / random
        rel_filter: all (3子关系) / triggers_only (仅TRIGGERS)
        """
        if not target or target == "UNKNOWN":
            return []
        start_c = target if direction == "forward" else None
        end_c = target if direction != "forward" else None
        rels = "TRIGGERS" if rel_filter == "triggers_only" else ALL_CAUSAL_RELS

        where = []
        params: dict[str, Any] = {"limit": top_k * 5}
        if start_c:
            where.append("s.concept = $s_c"); params["s_c"] = start_c
        if end_c:
            where.append("e.concept = $e_c"); params["e_c"] = end_c
        q = (
            f"MATCH path = (s)-[r:{rels}*1..{int(max_hops)}]->(e) "
            "WHERE " + (" AND ".join(where) if where else "true") + " "
            "RETURN [node IN nodes(path) | node.concept] AS concepts, "
            "       [rel IN relationships(path) | rel.confidence] AS confs, "
            "       length(path) AS len "
            "LIMIT $limit"
        )
        rows = []
        with graph._open_session() as sess:
            for rec in sess.run(q, params):
                cs = rec["concepts"]; confs = rec["confs"] or []
                l = int(rec["len"])
                if not confs:
                    continue
                conf_vals = [float(x) if x is not None else 0.0 for x in confs]
                avg_conf = sum(conf_vals) / len(conf_vals)
                coverage = len(set(cs)) / max(len(cs), 1)
                length_penalty = 1.0 / (l + 1)
                score_4d = avg_conf * 0.4 + coverage * 0.3 + length_penalty * 0.2
                rows.append({
                    "concepts": cs, "length": l, "avg_confidence": round(avg_conf, 3),
                    "_avg_conf": avg_conf, "_coverage": coverage,
                    "_length_penalty": length_penalty, "_score_4d": score_4d,
                })
        # 排序
        if sort_mode == "4dim":
            rows.sort(key=lambda x: x["_score_4d"], reverse=True)
        elif sort_mode == "conf_only":
            rows.sort(key=lambda x: x["_avg_conf"], reverse=True)
        elif sort_mode == "random":
            rng.shuffle(rows)
        return rows[:top_k]

    configs = [
        ("完整方法",        {"max_hops": 4, "sort_mode": "4dim",      "rel_filter": "all"}),
        ("w/o 4维排序",     {"max_hops": 4, "sort_mode": "conf_only",  "rel_filter": "all"}),
        ("w/o 多跳(=1跳)",  {"max_hops": 1, "sort_mode": "4dim",      "rel_filter": "all"}),
        ("w/o 3子关系",     {"max_hops": 4, "sort_mode": "4dim",      "rel_filter": "triggers_only"}),
        ("w/o 路径排序",    {"max_hops": 4, "sort_mode": "random",    "rel_filter": "all"}),
    ]

    results: dict[str, dict] = {}
    for name, cfg in configs:
        recalls = []
        n_paths_list = []
        for q, p in parses:
            expected = q.get("expected_concepts", [])
            paths = search(p.target_concept, p.direction,
                           cfg["max_hops"], 5,
                           cfg["sort_mode"], cfg["rel_filter"])
            r = recall_at_k(paths, expected, 5)
            recalls.append(r)
            n_paths_list.append(len(paths))
        r_mean = float(np.mean(recalls)) if recalls else 0
        n_mean = float(np.mean(n_paths_list)) if n_paths_list else 0
        results[name] = {
            "recall": round(r_mean, 3),
            "avg_paths": round(n_mean, 1),
            "recalls": [round(x, 3) for x in recalls],
        }
        print(f"  {name:18s} Recall={r_mean:.3f}  avg_paths={n_mean:.1f}")

    # 跳数实验
    hops_result = {}
    for h in [1, 2, 3, 4, 5]:
        recalls = []
        for q, p in parses:
            expected = q.get("expected_concepts", [])
            paths = search(p.target_concept, p.direction, h, 5, "4dim", "all")
            recalls.append(recall_at_k(paths, expected, 5))
        r_mean = float(np.mean(recalls)) if recalls else 0
        hops_result[h] = round(r_mean, 3)
        print(f"  hops={h} Recall={r_mean:.3f}")

    out = {"ablation": results, "hops": hops_result}
    out_path = _ROOT / "bate" / "data" / "ablation_experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")
    graph.close()


if __name__ == "__main__":
    run_ablation()
