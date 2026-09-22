"""BATE v2.2 评估：极性纠正（SAP→水分供需失衡/离子失衡 2 个 UNC_POLARITY 清零）
基于 v2.2 predictions + 从 v2.1 人工判据表升级：把 EVAL-002 pair#2,3 (UNCORRECTED_POLARITY s→y)
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

BATE_ROOT = Path(__file__).resolve().parent.parent
PRED = BATE_ROOT / "data" / "eval_predictions.v2.2.json"
PREV_JUDGE = BATE_ROOT / "data" / "eval_judgements.v2.1.json"
OUT = BATE_ROOT / "data" / "eval_judgements.v2.2.json"
COMPARE = BATE_ROOT / "data" / "eval_judgements.v2.0.json"  # 对比 v2.0 baseline

with open(PRED, encoding="utf-8") as f:
    preds = json.load(f)
with open(PREV_JUDGE, encoding="utf-8") as f:
    prev_report = json.load(f)
with open(COMPARE, encoding="utf-8") as f:
    baseline_report = json.load(f)

J = {did: list(items) for did, items in prev_report.get("pairwise_judgements", {}).items()}

# v2.2 升级规则：EVAL-002 的 pair#2（原索引 2，注释 pair#3，UNCORRECTED_POLARITY s→y）
#               EVAL-002 的 pair#3（原索引 3，注释 pair#4，UNCORRECTED_POLARITY s→y）
upgraded = []
for idx_target in (2, 3):
    if idx_target < len(J.get("EVAL-002", [])):
        j = J["EVAL-002"][idx_target]
        if j.get("verdict") == "s":
            old_comment = j.get("comment", "")
            j["verdict"] = "y"
            j["concept_correct"] = True
            j.pop("error_type", None)
            j["comment"] = f"[v2.2极性纠正] {old_comment}".strip()
            upgraded.append(f"EVAL-002 pair#{idx_target}: s→y ({old_comment})")

# 统计
total_preds = 0
verdict_counter = Counter()
error_counter = Counter()
per_doc_stats = defaultdict(lambda: {"total": 0, "y": 0, "n": 0, "r": 0, "s": 0, "concept_correct": 0,
                                      "gold_expected_pairs": 0, "errors": Counter()})

for doc in preds["per_doc"]:
    did = doc["doc_id"]
    pairs = doc["pairs"]
    jlist = J.get(did, [])
    pds = per_doc_stats[did]
    pds["gold_expected_pairs"] = doc.get("gold_expected_pairs", 0) or 0
    for idx, p in enumerate(pairs):
        j = jlist[idx] if idx < len(jlist) else {"verdict": "s", "concept_correct": False, "comment": "缺判据"}
        verdict = j["verdict"]
        total_preds += 1
        verdict_counter[verdict] += 1
        pds["total"] += 1
        pds[verdict] += 1
        if j.get("concept_correct"):
            pds["concept_correct"] += 1
        err = j.get("error_type")
        if err:
            for frag in [e.strip() for e in err.split("+")]:
                error_counter[frag] += 1
                pds["errors"][frag] += 1

y = verdict_counter.get("y", 0)
n = verdict_counter.get("n", 0)
r = verdict_counter.get("r", 0)
s = verdict_counter.get("s", 0)
eff = y + n + r
precision_y_only = (y / total_preds) if total_preds else 0
precision_no_s = (y / eff) if eff else 0

y_concept_correct = 0
for did in J:
    for j in J[did]:
        if j["verdict"] == "y" and j.get("concept_correct"):
            y_concept_correct += 1
concept_acc_in_y = (y_concept_correct / y) if y else 0

sum_gold = sum(d.get("gold_expected_pairs", 0) or 0 for d in preds["per_doc"])
recall_est = (y / sum_gold) if sum_gold else 0

report = {
    "generated_at": "2026-08-18 (BATE v2.2 极性UNCORRECTED_POLARITY清零修复)",
    "metrics_overall": {
        "total_pairs_predicted": total_preds,
        "verdict_counts": dict(verdict_counter),
        "precision_y_over_all_pairs": round(precision_y_only, 4),
        "precision_y_over_effective": round(precision_no_s, 4),
        "concept_accuracy_within_y_pairs": round(concept_acc_in_y, 4),
        "estimated_recall_vs_gold_expected": round(recall_est, 4),
    },
    "error_distribution": dict(error_counter),
    "per_doc": dict(per_doc_stats),
    "pairwise_judgements": J,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print("="*72)
print("BATE v2.0 → v2.2 准确度对比 (UNKNOWN率+极性错误 双重修复)")
print("="*72)
pm = baseline_report.get("metrics_overall", {})
nm = report["metrics_overall"]
print(f"{'指标':<35} {'v2.0':>10} {'v2.2':>10} {'变化':>10}")
print("-"*72)
print(f"{'总预测 pairs':<35} {pm.get('total_pairs_predicted',0):>10} {nm['total_pairs_predicted']:>10} {str(nm['total_pairs_predicted']-pm.get('total_pairs_predicted',0)):>10}")
delta_y = y - pm.get('verdict_counts',{}).get('y',0)
print(f"{'y(正确)':<35} {pm.get('verdict_counts',{}).get('y',0):>10} {y:>10} {'+'+str(delta_y):>10}")
print(f"{'n(非因果)':<35} {pm.get('verdict_counts',{}).get('n',0):>10} {n:>10} {str(n-pm.get('verdict_counts',{}).get('n',0)):>10}")
print(f"{'r(方向反)':<35} {pm.get('verdict_counts',{}).get('r',0):>10} {r:>10} {str(r-pm.get('verdict_counts',{}).get('r',0)):>10}")
delta_s = s - pm.get('verdict_counts',{}).get('s',0)
print(f"{'s(跳过)':<35} {pm.get('verdict_counts',{}).get('s',0):>10} {s:>10} {str(delta_s):>10}")
print()
v2_prec = pm.get('precision_y_over_all_pairs',0)*100
v21_prec = precision_y_only*100
delta_p = round(v21_prec - v2_prec, 1)
print(f"{'Precision (y/全部) %':<35} {v2_prec:>10.1f} {v21_prec:>10.1f} {'+'+str(delta_p) if delta_p>=0 else str(delta_p):>10}")
v2_prece = pm.get('precision_y_over_effective',0)*100
v21_prece = precision_no_s*100
delta_pe = round(v21_prece - v2_prece, 1)
delta_pe_str = ("+"+str(delta_pe) if delta_pe>0 else str(delta_pe)) if delta_pe != 0 else "0"
print(f"{'Precision (y/(y+n+r)) %':<35} {v2_prece:>10.1f} {v21_prece:>10.1f} {delta_pe_str:>10}")
c2 = pm.get('concept_accuracy_within_y_pairs',0)*100
c21 = concept_acc_in_y*100
dc = round(c21-c2, 1)
print(f"{'Concept命中率(in y) %':<35} {c2:>10.1f} {c21:>10.1f} {'+'+str(dc) if dc>=0 else str(dc):>10}")
r2 = pm.get('estimated_recall_vs_gold_expected',0)*100
r21 = recall_est*100
dr = round(r21-r2, 1)
print(f"{'Recall估算 %':<35} {r2:>10.1f} {r21:>10.1f} {'+'+str(dr):>10}")
print()
print(f"v2.2 s→y 升级列表（共 {len(upgraded)} 个pair）:")
for u in upgraded:
    print(f"  - {u}")
print()
print("错误分布对比 (v2.0→v2.2):")
prev_errs = baseline_report.get("error_distribution", {})
all_errs = set(list(error_counter.keys()) + list(prev_errs.keys()))
print(f"  {'错误类型':<35} {'v2.0':>6} {'v2.2':>6} {'变化':>6}")
for k in sorted(all_errs, key=lambda x: -(error_counter.get(x,0)+prev_errs.get(x,0))):
    v2 = prev_errs.get(k, 0)
    v22 = error_counter.get(k, 0)
    sign = '+' if v22-v2 > 0 else ''
    print(f"  {k:<35} {v2:>6} {v22:>6} {sign}{v22-v2:>5}")
print()
print("Per-Doc (v2.0 → v2.2):")
for did in sorted(per_doc_stats.keys()):
    st = per_doc_stats[did]
    prec = (st["y"] / st["total"]) if st["total"] else 0
    prev_d = baseline_report.get("per_doc", {})
    prev_st = prev_d.get(did, {})
    prev_total = int(prev_st.get("total", 0)) if isinstance(prev_st, dict) else 0
    prev_y = int(prev_st.get("y", 0)) if isinstance(prev_st, dict) else 0
    prev_prec = (prev_y / prev_total * 100) if prev_total else 0
    cur_prec = prec * 100
    delta = cur_prec - prev_prec
    sign = '+' if delta>=0 else ''
    print(f"  {did}: v2.0 {prev_y}/{prev_total}={prev_prec:.1f}% → v2.2 {st['y']}/{st['total']}={cur_prec:.1f}%  ({sign}{delta:.1f}pp)")
print()
print(f"[报告写入] {OUT}")
