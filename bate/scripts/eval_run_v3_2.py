"""BATE v3.2 重新评估：用 v3.2 本体+规则库对 5 个 EVAL 文档重新跑预测 + 自动判定。

输入：
  - eval_texts.v1.5.json (kb备份/2026-08-18_v2.2发布/)：5 个 EVAL 文档原文
  - eval_predictions.v2.2.json + eval_judgements.v2.2.json：v2.2 基线
  - v3.2 本体 (agri_event_types.json v3.2.1) + 规则库 (agri_causal_rules.json v3.2, known=227)

处理：
  1. 用 v3.2 AgriCausalPipeline 对 5 个 EVAL 文档重新跑 pipeline.run()（REAL LLM）
  2. 自动判定：v3.2 对 vs v2.2 对（按 cause/effect text 相似度匹配）
     - 匹配：继承 v2.2 verdict，记录 concept 归一变化
     - 新对：自动判定（concept 非 UNKNOWN + 极性正确 + 非 self-loop → y）
  3. 统计 Precision/Recall/UNKNOWN/极性/KB 召回，对比 v2.2

输出：
  - data/eval_predictions.v3.2.json
  - data/eval_judgements.v3.2.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.pipeline import AgriCausalPipeline
from bate.src.utils.knowledge_base import CausalKnowledgeBase
from bate.src.event_extraction.concept_aggregator import ConceptAggregator

DATA = Path(__file__).resolve().parent.parent / "data"
TEXTS_PATH = Path(r"c:\Users\lenovo\Desktop\开题\kb备份\2026-08-18_v2.2发布\eval_texts.v1.5.json")
V22_PRED = DATA / "eval_predictions.v2.2.json"
V22_JUDGE = DATA / "eval_judgements.v2.2.json"
OUT_PRED = DATA / "eval_predictions.v3.2.json"
OUT_JUDGE = DATA / "eval_judgements.v3.2.json"


def _norm(s: str) -> str:
    return (s or "").strip().replace(" ", "").lower()[:40]


def _sim_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na in nb or nb in na or na == nb


def auto_judge_pair(pair: dict, agg: ConceptAggregator) -> tuple[str, str, bool]:
    """自动判定单对。返回 (verdict, error_type, concept_correct)。"""
    cause = pair.get("cause") or pair.get("cause_event") or {}
    effect = pair.get("effect") or pair.get("effect_event") or {}
    cc = cause.get("concept", "")
    ec = effect.get("concept", "")
    cs = cause.get("sub_class_id", "")
    es = effect.get("sub_class_id", "")

    # UNKNOWN
    if cc == "UNKNOWN" or ec == "UNKNOWN":
        return "s", "UNKNOWN_CONCEPT", False
    # self-loop
    if cc == ec and cc:
        return "n", "SELF_LOOP", False
    # 极性：MGT cause 不应有负向 effect（除非文本显式失败）
    is_mgt = bool(cs and cs.startswith("MGT-"))
    neg_concepts = {"产量下降", "千粒重下降", "病害加重", "虫害加重", "感病性增加",
                    "光合作用下降", "灌浆不良", "落花落蕾", "蕾铃脱落", "叶片损伤",
                    "稻米品质下降", "空壳率上升", "植株矮化", "生长受阻"}
    if is_mgt and ec in neg_concepts:
        kb = pair.get("kb_match", {}) or {}
        if not kb.get("known"):
            return "s", "POLARITY_SUSPECT", True
    # 正常
    return "y", "", True


def main() -> None:
    eval_set = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    v22_pred = json.loads(V22_PRED.read_text(encoding="utf-8"))
    v22_judge = json.loads(V22_JUDGE.read_text(encoding="utf-8"))

    kb = CausalKnowledgeBase(DATA / "agri_causal_rules.json")
    pipe = AgriCausalPipeline(knowledge_base=kb)
    agg = ConceptAggregator()
    print(f"[LLM Mode] {'MOCK' if getattr(pipe.llm,'mock',True) else 'REAL'} model={pipe.llm.model}")
    print(f"[KB] known={len(kb.known_pairs)} forbid={len(kb.forbidden_pairs)}")
    print(f"[Ontology] subtypes=104 concepts=815 mention_to_concept=1105")

    # v2.2 pairs 索引（按 doc_id 分组，text 归一作 key）
    v22_pairs_by_doc: dict[str, list[dict]] = {}
    for d in v22_pred.get("per_doc", []):
        v22_pairs_by_doc[d["doc_id"]] = d.get("pairs", [])
    v22_judge_pw = v22_judge.get("pairwise_judgements", {})

    results = {"meta": {"llm_mode": "REAL" if not getattr(pipe.llm, "mock", True) else "MOCK",
                         "kb_version": "v3.2", "ontology_version": "v3.2.1"}, "per_doc": []}

    for d in eval_set["documents"]:
        did = d["doc_id"]
        text = d["text"]
        print(f"\n===== [{did}] {d.get('domain','')} =====")
        try:
            r = pipe.run(text)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["per_doc"].append({"doc_id": did, "ERROR": str(e)})
            continue
        pairs = r.get("causal_pairs") or r.get("confirmed_pairs") or []
        chains = r.get("causal_chains", [])
        events = r.get("events", [])
        stats = r.get("stats", {}) or {}

        v22_pairs = v22_pairs_by_doc.get(did, [])
        v22_jlist = v22_judge_pw.get(did, [])

        # 匹配 + 判定
        judged = []
        concept_changes = []
        new_pairs = 0
        for i, p in enumerate(pairs):
            cause = p.get("cause") or p.get("cause_event") or {}
            effect = p.get("effect") or p.get("effect_event") or {}
            ctxt = cause.get("text", "")
            etxt = effect.get("text", "")
            cc = cause.get("concept", "")
            ec = effect.get("concept", "")

            # 找 v2.2 对应
            matched_v22 = None
            matched_judge = None
            for j, vp in enumerate(v22_pairs):
                vc = vp.get("cause") or vp.get("cause_event") or {}
                ve = vp.get("effect") or vp.get("effect_event") or {}
                if _sim_match(ctxt, vc.get("text", "")) and _sim_match(etxt, ve.get("text", "")):
                    matched_v22 = vp
                    matched_judge = v22_jlist[j] if j < len(v22_jlist) else None
                    # 记录 concept 变化
                    vcc = vc.get("concept", "")
                    vec = ve.get("concept", "")
                    if vcc != cc or vec != ec:
                        concept_changes.append({
                            "v22_cause": vcc, "v32_cause": cc,
                            "v22_effect": vec, "v32_effect": ec,
                        })
                    break

            if matched_v22 and matched_judge:
                verdict = matched_judge.get("verdict", "y")
                err = matched_judge.get("error_type", "")
                cc_ok = matched_judge.get("concept_correct", True)
                source = "v2.2继承"
            else:
                verdict, err, cc_ok = auto_judge_pair(p, agg)
                source = "v3.2新对自动判定"
                if not matched_v22:
                    new_pairs += 1

            judged.append({
                "idx": i, "verdict": verdict, "error_type": err,
                "concept_correct": cc_ok, "source": source,
                "cause_concept": cc, "effect_concept": ec,
                "cause_subtype": cause.get("sub_class_id", ""),
                "effect_subtype": effect.get("sub_class_id", ""),
                "cause_text": ctxt[:50], "effect_text": etxt[:50],
                "confidence": p.get("confidence", 0),
                "kb_known": (p.get("kb_match", {}) or {}).get("known", False),
            })

        results["per_doc"].append({
            "doc_id": did, "domain": d.get("domain", ""),
            "gold_expected_pairs": d.get("human_gold_pairs_expected", 0),
            "num_events": len(events), "num_pairs": len(pairs),
            "num_chains": len(chains), "num_unknown_concept": sum(1 for j in judged if j["error_type"] == "UNKNOWN_CONCEPT"),
            "concept_changes_vs_v22": concept_changes, "new_pairs_vs_v22": new_pairs,
            "pairs": pairs, "chains": chains, "events": events, "stats": stats,
            "judged": judged,
        })
        y_count = sum(1 for j in judged if j["verdict"] == "y")
        print(f"  events={len(events)} pairs={len(pairs)} chains={len(chains)}")
        print(f"  y={y_count} new_vs_v22={new_pairs} concept_changes={len(concept_changes)} unknown={sum(1 for j in judged if j['error_type']=='UNKNOWN_CONCEPT')}")

    # 全局统计
    total_pairs = sum(len(d.get("judged", [])) for d in results["per_doc"])
    verdict_counter = Counter()
    error_counter = Counter()
    for d in results["per_doc"]:
        for j in d.get("judged", []):
            verdict_counter[j["verdict"]] += 1
            if j.get("error_type"):
                error_counter[j["error_type"]] += 1
    y_count = verdict_counter.get("y", 0)
    s_count = verdict_counter.get("s", 0)
    n_count = verdict_counter.get("n", 0)
    gold_total = sum(d.get("gold_expected_pairs", 0) for d in results["per_doc"])
    results["meta"].update({
        "num_eval_docs": len(results["per_doc"]),
        "total_pairs_extracted": total_pairs,
        "total_chains_extracted": sum(len(d.get("chains", [])) for d in results["per_doc"]),
        "verdict_counts": dict(verdict_counter),
        "error_distribution": dict(error_counter),
        "precision_y_over_all": round(y_count / total_pairs, 4) if total_pairs else 0,
        "precision_y_over_effective": round(y_count / (y_count + n_count), 4) if (y_count + n_count) else 0,
        "estimated_recall_vs_gold": round(y_count / gold_total, 4) if gold_total else 0,
        "gold_expected_total": gold_total,
        "concept_changes_total": sum(len(d.get("concept_changes_vs_v22", [])) for d in results["per_doc"]),
        "new_pairs_total": sum(d.get("new_pairs_vs_v22", 0) for d in results["per_doc"]),
    })

    OUT_PRED.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # v2.2 对比
    v22_metrics = v22_judge.get("metrics_overall", {})
    print("\n" + "=" * 70)
    print("【v3.2 评估完成】对比 v2.2")
    print("=" * 70)
    print(f"{'指标':<28}{'v2.2':<12}{'v3.2':<12}{'变化'}")
    print("-" * 70)
    for label, v22k, v32k in [
        ("total_pairs_predicted", "total_pairs_predicted", total_pairs),
        ("precision(y/全部)", "precision_y_over_all_pairs", results["meta"]["precision_y_over_all"]),
        ("precision(y/有效)", "precision_y_over_effective", results["meta"]["precision_y_over_effective"]),
        ("recall(vs gold)", "estimated_recall_vs_gold_expected", results["meta"]["estimated_recall_vs_gold"]),
        ("concept_changes_vs_v22", "-", results["meta"]["concept_changes_total"]),
        ("new_pairs_vs_v22", "-", results["meta"]["new_pairs_total"]),
    ]:
        v22v = v22_metrics.get(v22k, "-")
        if isinstance(v22v, float):
            v22v = round(v22v, 4)
        if isinstance(v32k, float):
            v32v = round(v32k, 4)
        else:
            v32v = v32k
        delta = ""
        if isinstance(v22v, (int, float)) and isinstance(v32v, (int, float)):
            d = round(v32v - v22v, 4)
            delta = f"+{d}" if d > 0 else (str(d) if d < 0 else "0")
        print(f"{label:<28}{str(v22v):<12}{str(v32v):<12}{delta}")
    print("-" * 70)
    print(f"verdict: {dict(verdict_counter)}")
    print(f"errors: {dict(error_counter)}")
    print(f"-> {OUT_PRED}")
    print(f"-> {OUT_JUDGE}")

    # judgements 摘要文件
    judge_summary = {
        "generated_at": "2026-08-18 (v3.2 本体+规则库重建后重新评估)",
        "metrics_overall": results["meta"],
        "per_doc": {d["doc_id"]: {
            "total": len(d.get("judged", [])),
            "y": sum(1 for j in d.get("judged", []) if j["verdict"] == "y"),
            "n": sum(1 for j in d.get("judged", []) if j["verdict"] == "n"),
            "s": sum(1 for j in d.get("judged", []) if j["verdict"] == "s"),
            "r": sum(1 for j in d.get("judged", []) if j["verdict"] == "r"),
            "concept_correct": sum(1 for j in d.get("judged", []) if j.get("concept_correct")),
            "gold_expected_pairs": d.get("gold_expected_pairs", 0),
            "num_events": d.get("num_events", 0),
            "num_pairs": d.get("num_pairs", 0),
            "num_chains": d.get("num_chains", 0),
            "concept_changes": len(d.get("concept_changes_vs_v22", [])),
            "new_pairs": d.get("new_pairs_vs_v22", 0),
            "errors": dict(Counter(j.get("error_type", "") for j in d.get("judged", []) if j.get("error_type"))),
        } for d in results["per_doc"]},
        "pairwise_judgements": {d["doc_id"]: d.get("judged", []) for d in results["per_doc"]},
    }
    OUT_JUDGE.write_text(json.dumps(judge_summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
