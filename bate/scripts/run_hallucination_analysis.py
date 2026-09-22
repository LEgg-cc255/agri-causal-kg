"""Phase 7.3 因果幻觉分析。

评估 Baseline（纯LLM）vs CausalRAG（LLM+图谱）的因果幻觉率。

幻觉类型：
1. Concept Hallucination：因果对中的概念不在图谱中（无法归一化）
2. Relation Hallucination：概念在图谱中但因果关系不在（无 CAUSES 边）

评估流程：
1. 从解释文本中用正则提取因果对（X导致Y / X引起Y 等）
2. 用 ConceptAggregator 归一化 X 和 Y 到图谱 concept
3. 检查因果对是否在图谱中（236条 CAUSES 边）
4. 计算 verification_rate 和 hallucination_rate

用法：
    python bate/scripts/run_hallucination_analysis.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.kg import AgriCausalGraph
from bate.src.event_extraction.concept_aggregator import ConceptAggregator


# -------------------------------------------------------------------- #
#  中文因果模式（正则）
# -------------------------------------------------------------------- #
# 每个模式匹配 "X因果词Y" 结构，提取 X（原因）和 Y（结果）
# X/Y 长度限制 2-20 字符，避免匹配过长/过短文本
_CAUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(.{2,20}?)(?:导致|引起|造成|引发|促使|诱发|致使|使得)(.{2,20}?)"),
    re.compile(r"(.{2,20}?)(?:进而|进一步|从而|最终)(.{2,20}?)"),
    re.compile(r"(?:由于|因为)(.{2,20}?)[，,。；;](.{2,20}?)"),
    re.compile(r"(.{2,20}?)(?:会|可以|能够)(?:导致|引起|造成|引发)(.{2,20}?)"),
]

# 清理用：句末标点截断
_SENTENCE_END = re.compile(r"[。.！!？?；;]|$")


def _extract_causal_pairs(text: str) -> list[tuple[str, str]]:
    """从文本中提取因果对 (cause_text, effect_text)。"""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pat in _CAUSAL_PATTERNS:
        for m in pat.finditer(text):
            cause = m.group(1).strip()
            effect = m.group(2).strip()
            # 在 Y 处截断到句末
            eff_end = _SENTENCE_END.search(effect)
            if eff_end:
                effect = effect[:eff_end.start()].strip()
            # 清理标点
            cause = re.sub(r"[，,。.！!？?；;：:、\s]+$", "", cause)
            cause = re.sub(r"^[，,。.！!？?；;：:、\s]+", "", cause)
            effect = re.sub(r"[，,。.！!？?；;：:、\s]+$", "", effect)
            effect = re.sub(r"^[，,。.！!？?；;：:、\s]+", "", effect)
            # 过滤过短/过长
            if 2 <= len(cause) <= 20 and 2 <= len(effect) <= 20:
                key = (cause, effect)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


# -------------------------------------------------------------------- #
#  幻觉分析器
# -------------------------------------------------------------------- #
class CausalHallucinationAnalyzer:
    """因果幻觉分析器。

    从解释文本中提取因果对，归一化到图谱 concept，
    检查是否在图谱中，计算幻觉率。
    """

    def __init__(self, graph: AgriCausalGraph, aggregator: ConceptAggregator) -> None:
        self.graph = graph
        self.aggregator = aggregator
        self.causal_pairs: set[tuple[str, str]] = set()
        self._load_causal_pairs()

    def _load_causal_pairs(self) -> None:
        """从 Neo4j 加载所有因果边（3种子关系，兼容旧版CAUSES残留）。"""
        with self.graph._open_session() as s:
            rows = s.run(
                "MATCH (a)-[r:TRIGGERS|PROPAGATES|ALLEVIATES|CAUSES]->(b) "
                "RETURN a.concept AS cause, b.concept AS effect"
            ).data()
        for r in rows:
            if r["cause"] and r["effect"]:
                self.causal_pairs.add((r["cause"], r["effect"]))
        print(f"[图谱] 加载 {len(self.causal_pairs)} 条已知因果对")

    def _normalize(self, mention: str) -> str:
        """归一化 mention 到图谱 concept。"""
        if not mention:
            return "UNKNOWN"
        result = self.aggregator.aggregate(mention_text=mention, llm_concept="")
        return result.get("concept", "UNKNOWN")

    def analyze(self, explanation: str) -> dict[str, Any]:
        """分析解释文本的因果幻觉率。

        Returns
        -------
        dict
            含 total_claims / verified / hallucinated_concept /
            hallucinated_relation / hallucination_rate / verification_rate
        """
        raw_pairs = _extract_causal_pairs(explanation)

        verified = 0
        halluc_concept = 0  # 概念不在图谱（无法归一化）
        halluc_relation = 0  # 关系不在图谱（无 CAUSES 边）
        detail: list[dict] = []

        for cause_text, effect_text in raw_pairs:
            cause_c = self._normalize(cause_text)
            effect_c = self._normalize(effect_text)

            if cause_c == "UNKNOWN" or effect_c == "UNKNOWN":
                halluc_concept += 1
                detail.append({
                    "cause_text": cause_text, "effect_text": effect_text,
                    "cause_concept": cause_c, "effect_concept": effect_c,
                    "verdict": "concept_hallucination",
                })
            elif (cause_c, effect_c) in self.causal_pairs:
                verified += 1
                detail.append({
                    "cause_text": cause_text, "effect_text": effect_text,
                    "cause_concept": cause_c, "effect_concept": effect_c,
                    "verdict": "verified",
                })
            else:
                halluc_relation += 1
                detail.append({
                    "cause_text": cause_text, "effect_text": effect_text,
                    "cause_concept": cause_c, "effect_concept": effect_c,
                    "verdict": "relation_hallucination",
                })

        total = verified + halluc_concept + halluc_relation
        return {
            "total_claims": total,
            "verified": verified,
            "hallucinated_concept": halluc_concept,
            "hallucinated_relation": halluc_relation,
            "hallucination_rate": round(
                (halluc_concept + halluc_relation) / total, 3
            ) if total else 0.0,
            "verification_rate": round(verified / total, 3) if total else 0.0,
            "detail": detail,
        }


# -------------------------------------------------------------------- #
#  批量分析实验结果
# -------------------------------------------------------------------- #
def main() -> None:
    if not os.environ.get("NEO4J_PASSWORD"):
        os.environ["NEO4J_PASSWORD"] = "123456"

    # 加载实验结果
    exp_path = _ROOT / "bate" / "data" / "experiment_results.json"
    with open(exp_path, encoding="utf-8") as f:
        exp = json.load(f)
    results = exp["results"]

    # 初始化分析器
    graph = AgriCausalGraph.from_env()
    aggregator = ConceptAggregator()
    analyzer = CausalHallucinationAnalyzer(graph, aggregator)

    print("=" * 70)
    print("Phase 7.3 因果幻觉分析")
    print(f"分析 {len(results)} 题的 Baseline vs CausalRAG 解释")
    print("=" * 70)

    # 逐题分析
    b_stats = {"total": 0, "verified": 0, "h_concept": 0, "h_relation": 0}
    c_stats = {"total": 0, "verified": 0, "h_concept": 0, "h_relation": 0}
    analyzed: list[dict] = []

    with graph:
        for r in results:
            qid = r["id"]
            question = r["question"]

            b_analysis = analyzer.analyze(r["baseline_answer"])
            c_analysis = analyzer.analyze(r["causalrag_answer"])

            b_stats["total"] += b_analysis["total_claims"]
            b_stats["verified"] += b_analysis["verified"]
            b_stats["h_concept"] += b_analysis["hallucinated_concept"]
            b_stats["h_relation"] += b_analysis["hallucinated_relation"]

            c_stats["total"] += c_analysis["total_claims"]
            c_stats["verified"] += c_analysis["verified"]
            c_stats["h_concept"] += c_analysis["hallucinated_concept"]
            c_stats["h_relation"] += c_analysis["hallucinated_relation"]

            analyzed.append({
                "id": qid,
                "question": question,
                "baseline": {
                    "total_claims": b_analysis["total_claims"],
                    "verified": b_analysis["verified"],
                    "hallucinated_concept": b_analysis["hallucinated_concept"],
                    "hallucinated_relation": b_analysis["hallucinated_relation"],
                    "hallucination_rate": b_analysis["hallucination_rate"],
                    "verification_rate": b_analysis["verification_rate"],
                },
                "causalrag": {
                    "total_claims": c_analysis["total_claims"],
                    "verified": c_analysis["verified"],
                    "hallucinated_concept": c_analysis["hallucinated_concept"],
                    "hallucinated_relation": c_analysis["hallucinated_relation"],
                    "hallucination_rate": c_analysis["hallucination_rate"],
                    "verification_rate": c_analysis["verification_rate"],
                },
            })

    # 汇总
    b_total = b_stats["total"]
    c_total = c_stats["total"]
    b_halluc = b_stats["h_concept"] + b_stats["h_relation"]
    c_halluc = c_stats["h_concept"] + c_stats["h_relation"]

    print(f"\n{'='*70}")
    print(f"因果幻觉分析汇总（{len(results)} 题）")
    print(f"{'='*70}")
    print(f"\n{'指标':<32} {'Baseline':>12} {'CausalRAG':>12} {'差异':>10}")
    print("-" * 68)
    print(f"{'Total Causal Claims':<32} {b_total:>12} {c_total:>12} "
          f"{c_total-b_total:>+10}")
    print(f"{'Verified Claims':<32} {b_stats['verified']:>12} {c_stats['verified']:>12} "
          f"{c_stats['verified']-b_stats['verified']:>+10}")
    print(f"{'Hallucinated (Concept)':<32} {b_stats['h_concept']:>12} "
          f"{c_stats['h_concept']:>12} {c_stats['h_concept']-b_stats['h_concept']:>+10}")
    print(f"{'Hallucinated (Relation)':<32} {b_stats['h_relation']:>12} "
          f"{c_stats['h_relation']:>12} {c_stats['h_relation']-b_stats['h_relation']:>+10}")
    print(f"{'Total Hallucinated':<32} {b_halluc:>12} {c_halluc:>12} "
          f"{c_halluc-b_halluc:>+10}")
    b_rate = round(b_halluc / b_total, 3) if b_total else 0
    c_rate = round(c_halluc / c_total, 3) if c_total else 0
    b_vrate = round(b_stats["verified"] / b_total, 3) if b_total else 0
    c_vrate = round(c_stats["verified"] / c_total, 3) if c_total else 0
    print(f"{'Hallucination Rate':<32} {b_rate:>12.3f} {c_rate:>12.3f} "
          f"{c_rate-b_rate:>+10.3f}")
    print(f"{'Verification Rate':<32} {b_vrate:>12.3f} {c_vrate:>12.3f} "
          f"{c_vrate-b_vrate:>+10.3f}")
    print("-" * 68)

    # 保存详细结果
    out_path = _ROOT / "bate" / "data" / "hallucination_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "baseline": {
                    "total_claims": b_total,
                    "verified": b_stats["verified"],
                    "hallucinated_concept": b_stats["h_concept"],
                    "hallucinated_relation": b_stats["h_relation"],
                    "hallucination_rate": b_rate,
                    "verification_rate": b_vrate,
                },
                "causalrag": {
                    "total_claims": c_total,
                    "verified": c_stats["verified"],
                    "hallucinated_concept": c_stats["h_concept"],
                    "hallucinated_relation": c_stats["h_relation"],
                    "hallucination_rate": c_rate,
                    "verification_rate": c_vrate,
                },
            },
            "per_question": analyzed,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
