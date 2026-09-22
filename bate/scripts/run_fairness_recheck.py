"""公平性复核：精确串 Recall vs 同义词归一化 Recall。

背景：
主实验 concept_recall 用精确子串匹配（expected concept 是否原样出现在解释文本中）。
Baseline 用通用说法（"热害""高温热害"）回答，即使语义正确也无法命中标准概念串
（"高温胁迫"），导致 Baseline Recall 被系统性低估。

本脚本在同一批已保存答案上重算两种口径：
- raw：精确子串（原口径）
- normalized：expected 标准概念 + 同义词表中的全部别名，任一命中即算对
仅在有 expected_concepts 的 38 题上评估（与 experiment_results.json 的 metrics 同队列）。

输出: bate/data/fairness_recheck.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.event_extraction.concept_aggregator import ConceptAggregator


def main() -> None:
    exp_path = _ROOT / "bate" / "data" / "experiment_results.json"
    with open(exp_path, encoding="utf-8") as f:
        results = json.load(f)["results"]

    agg = ConceptAggregator()

    # 标准概念 -> 别名集合（mention_to_concept 的反向索引，并入标准概念自身）
    aliases: dict[str, set[str]] = {}
    for mention, concept in agg.mention_to_concept.items():
        aliases.setdefault(concept, {concept}).add(mention)

    def hit(concept: str, text: str) -> int:
        names = aliases.get(concept, {concept})
        return int(any(name and name in text for name in names))

    rows = []
    for r in results:
        expected = r.get("expected_concepts") or []
        if not expected:
            continue
        for method_key in ("baseline_answer", "causalrag_answer"):
            text = r[method_key]
            raw = sum(1 for c in expected if c in text) / len(expected)
            norm = sum(hit(c, text) for c in expected) / len(expected)
            rows.append({"id": r["id"], "method": method_key, "raw": raw, "norm": norm})

    out = {"num_questions": len({x["id"] for x in rows})}
    for method, key in [("baseline", "baseline_answer"), ("causalrag", "causalrag_answer")]:
        sub = [x for x in rows if x["method"] == key]
        out[method] = {
            "raw_recall": round(float(np.mean([x["raw"] for x in sub])), 3),
            "normalized_recall": round(float(np.mean([x["norm"] for x in sub])), 3),
        }

    # 题级差异分布：归一化后比精确串多命中几题
    for method, key in [("baseline", "baseline_answer"), ("causalrag", "causalrag_answer")]:
        sub = [x for x in rows if x["method"] == key]
        out[method]["questions_improved"] = sum(1 for x in sub if x["norm"] > x["raw"])

    out_path = _ROOT / "bate" / "data" / "fairness_recheck.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
