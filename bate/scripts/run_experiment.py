"""Phase 7.2 对比实验：Baseline（纯 LLM）vs CausalRAG（LLM + 因果图）。

实验设计：
- Baseline：纯 LLM 直接回答用户问题（无因果图注入）
- CausalRAG：Step1→2→3 pipeline（问题解析→双向检索→LLM 生成）

评估指标：
1. Concept Recall：解释中 expected_concepts 的命中率
2. Explanation Length：平均解释字符数
3. LLM Call Count：LLM 调用次数（CausalRAG 无路径时 0 调用）
4. Avg Latency：平均延迟（ms）

用法：
    python bate/scripts/run_experiment.py --limit 10        # 小规模验证
    python bate/scripts/run_experiment.py                    # 全量 61 题
    python bate/scripts/run_experiment.py --resume          # 断点续跑
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.kg import CausalRAGPipeline
from bate.src.utils.llm_client import LLMClient


# -------------------------------------------------------------------- #
#  Baseline：纯 LLM（无因果图）
# -------------------------------------------------------------------- #
BASELINE_SYSTEM_PROMPT = """你是一个农业专家。请直接回答用户的农业相关问题。

要求：
1. 用专业的农业知识回答
2. 语言通俗易懂
3. 解释要准确、完整
4. 直接输出自然语言解释，不要输出 JSON 或代码块"""


class BaselineGenerator:
    """纯 LLM 基线（无因果图增强）。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def answer(self, question: str) -> tuple[str, int]:
        """返回 (解释文本, 耗时ms)。"""
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        t0 = time.time()
        text = self.llm.chat(messages).strip()
        elapsed = int((time.time() - t0) * 1000)
        return text, elapsed


# -------------------------------------------------------------------- #
#  评估指标
# -------------------------------------------------------------------- #
def concept_recall(explanation: str, expected_concepts: list[str]) -> float:
    """解释中 expected_concepts 的命中率。"""
    if not expected_concepts:
        return 1.0  # 边界用例无期望概念
    hits = sum(1 for c in expected_concepts if c in explanation)
    return hits / len(expected_concepts)


def evaluate(results: list[dict]) -> dict[str, Any]:
    """计算汇总指标。"""
    n = len(results)
    if n == 0:
        return {}
    # 只对有 expected_concepts 的问题计算 recall
    with_expected = [r for r in results if r.get("expected_concepts")]
    recall_baseline = [concept_recall(r["baseline_answer"], r["expected_concepts"])
                        for r in with_expected]
    recall_causalrag = [concept_recall(r["causalrag_answer"], r["expected_concepts"])
                         for r in with_expected]
    # 路径覆盖率（仅 CausalRAG）
    path_recalls = [r.get("path_recall", 0.0) for r in with_expected]

    return {
        "num_questions": n,
        "num_with_expected": len(with_expected),
        "baseline": {
            "avg_concept_recall": round(sum(recall_baseline) / len(recall_baseline), 3)
                                    if recall_baseline else 0,
            "avg_length": round(sum(len(r["baseline_answer"]) for r in results) / n),
            "avg_latency_ms": round(sum(r["baseline_ms"] for r in results) / n),
            "llm_calls": sum(1 for r in results if r["baseline_answer"]),
        },
        "causalrag": {
            "avg_concept_recall": round(sum(recall_causalrag) / len(recall_causalrag), 3)
                                   if recall_causalrag else 0,
            "avg_path_recall": round(sum(path_recalls) / len(path_recalls), 3)
                               if path_recalls else 0,
            "avg_length": round(sum(len(r["causalrag_answer"]) for r in results) / n),
            "avg_latency_ms": round(sum(r["causalrag_ms"] for r in results) / n),
            "llm_calls": sum(r.get("causalrag_llm_calls", 0) for r in results),
        },
    }


# -------------------------------------------------------------------- #
#  实验主流程
# -------------------------------------------------------------------- #
def run_experiment(
    dataset_path: Path,
    output_path: Path,
    limit: int = 0,
    resume: bool = False,
) -> None:
    if not os.environ.get("NEO4J_PASSWORD"):
        os.environ["NEO4J_PASSWORD"] = "123456"

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]
    if limit > 0:
        questions = questions[:limit]

    # 断点续跑
    existing: list[dict] = []
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f).get("results", [])
        done_ids = {r["id"] for r in existing}
        questions = [q for q in questions if q["id"] not in done_ids]
        print(f"[续跑] 已完成 {len(done_ids)} 题，剩余 {len(questions)} 题")

    baseline_gen = BaselineGenerator()
    pipe = CausalRAGPipeline.from_env()

    print("=" * 70)
    print(f"Phase 7.2 对比实验")
    print(f"  问题数: {len(questions) + len(existing)} (已跑 {len(existing)}, 待跑 {len(questions)})")
    print(f"  LLM 模式: mock={baseline_gen.llm.mock} model={baseline_gen.llm.model}")
    print("=" * 70)

    results: list[dict] = list(existing)
    with pipe:
        for i, q in enumerate(questions, 1):
            qid = q["id"]
            question = q["question"]
            expected = q.get("expected_concepts", [])

            print(f"\n[{i}/{len(questions)}] {qid} {question}")

            # Baseline：纯 LLM
            b_answer, b_ms = baseline_gen.answer(question)
            b_recall = concept_recall(b_answer, expected)
            print(f"  [Baseline] recall={b_recall:.2f} len={len(b_answer)} {b_ms}ms")

            # CausalRAG：pipeline
            t0 = time.time()
            ans = pipe.answer(question, max_hops=4, top_k=5)
            c_ms = int((time.time() - t0) * 1000)
            c_answer = ans.explanation
            c_recall = concept_recall(c_answer, expected)
            # 路径覆盖率
            retrieved_concepts = set()
            for p in ans.paths:
                retrieved_concepts.update(p["concepts"])
            path_hits = sum(1 for ec in expected if ec in retrieved_concepts)
            path_recall = path_hits / len(expected) if expected else 1.0
            # LLM 调用次数
            c_llm_calls = 1 if (ans.parse.is_valid and ans.paths) else 0
            print(f"  [CausalRAG] recall={c_recall:.2f} path_recall={path_recall:.2f} "
                  f"len={len(c_answer)} {c_ms}ms llm={c_llm_calls}")

            results.append({
                "id": qid,
                "question": question,
                "expected_concepts": expected,
                "direction": q.get("direction"),
                "difficulty": q.get("difficulty"),
                "baseline_answer": b_answer,
                "baseline_ms": b_ms,
                "baseline_recall": round(b_recall, 3),
                "causalrag_answer": c_answer,
                "causalrag_ms": c_ms,
                "causalrag_recall": round(c_recall, 3),
                "path_recall": round(path_recall, 3),
                "causalrag_llm_calls": c_llm_calls,
                "causalrag_num_paths": len(ans.paths),
                "causalrag_target_concept": ans.parse.target_concept,
            })

            # 每 10 题保存一次（断点保护）
            if i % 10 == 0:
                _save_results(output_path, results, dataset.get("dataset_info", {}))
                print(f"  [保存] 已保存 {len(results)} 题结果")

    # 最终保存
    _save_results(output_path, results, dataset.get("dataset_info", {}))

    # 汇总报告
    metrics = evaluate(results)
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    print(f"问题数: {metrics.get('num_questions', 0)} "
          f"(有 expected_concepts: {metrics.get('num_with_expected', 0)})")
    b = metrics.get("baseline", {})
    c = metrics.get("causalrag", {})
    print(f"\n{'指标':<24} {'Baseline':>12} {'CausalRAG':>12} {'差异':>10}")
    print("-" * 60)
    print(f"{'Concept Recall':<24} {b.get('avg_concept_recall',0):>12.3f} "
          f"{c.get('avg_concept_recall',0):>12.3f} "
          f"{c.get('avg_concept_recall',0)-b.get('avg_concept_recall',0):>+10.3f}")
    print(f"{'Path Recall':<24} {'N/A':>12} {c.get('avg_path_recall',0):>12.3f} {'':>10}")
    print(f"{'Avg Length (chars)':<24} {b.get('avg_length',0):>12} "
          f"{c.get('avg_length',0):>12} {c.get('avg_length',0)-b.get('avg_length',0):>+10}")
    print(f"{'Avg Latency (ms)':<24} {b.get('avg_latency_ms',0):>12} "
          f"{c.get('avg_latency_ms',0):>12} {c.get('avg_latency_ms',0)-b.get('avg_latency_ms',0):>+10}")
    print(f"{'LLM Calls':<24} {b.get('llm_calls',0):>12} {c.get('llm_calls',0):>12} "
          f"{c.get('llm_calls',0)-b.get('llm_calls',0):>+10}")
    print("-" * 60)

    # 保存指标
    final = {"results": results, "metrics": metrics}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


def _save_results(path: Path, results: list[dict], info: dict) -> None:
    out = {"dataset_info": info, "results": results, "metrics": evaluate(results)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="限制问题数（0=全部）")
    ap.add_argument("--resume", action="store_true", help="断点续跑")
    args = ap.parse_args()

    dataset_path = _ROOT / "bate" / "data" / "agri_causal_qa_dataset.json"
    output_path = _ROOT / "bate" / "data" / "experiment_results.json"
    run_experiment(dataset_path, output_path, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
