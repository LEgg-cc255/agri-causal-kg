"""概念聚合质量分析：QA问题解析与答案中的概念图谱命中率。"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

from bate.src.kg import AgriCausalGraph
from bate.src.kg.question_parser import QuestionParser
from bate.src.event_extraction.concept_aggregator import ConceptAggregator

os.environ.setdefault("NEO4J_PASSWORD", "123456")

with open(_ROOT / "bate" / "data" / "agri_causal_qa_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)
with open(_ROOT / "bate" / "data" / "experiment_results.json", encoding="utf-8") as f:
    exp = json.load(f)["results"]

graph = AgriCausalGraph.from_env()
parser = QuestionParser()
aggregator = ConceptAggregator()

# 图谱所有概念
with graph._open_session() as s:
    concepts_in_kg = {r["c"] for r in s.run("MATCH (n:Entity) RETURN n.concept AS c").data() if r["c"]}
print(f"[图谱] {len(concepts_in_kg)} 概念")

# 1. 问题解析端：target_concept 在图谱命中率
q_total = 0; q_hit = 0
for q in dataset["questions"]:
    p = parser.parse(q["question"])
    if p.target_concept and p.target_concept != "UNKNOWN":
        q_total += 1
        if p.target_concept in concepts_in_kg:
            q_hit += 1
q_rate = q_hit / max(q_total, 1)
print(f"[问题解析] target 在图谱: {q_hit}/{q_total} = {q_rate:.1%}")

# 2. 答案端：期望概念在图谱命中率（反映聚合召回）
exp_total = 0; exp_hit = 0
for r in exp:
    for c in r.get("expected_concepts", []):
        exp_total += 1
        if c in concepts_in_kg:
            exp_hit += 1
exp_rate = exp_hit / max(exp_total, 1)
print(f"[期望概念] 在图谱: {exp_hit}/{exp_total} = {exp_rate:.1%}")

# 3. 期望概念聚合：用聚合器统计 UNKNOWN 占比
unknown = 0; mapped = 0; sub_known = 0
for r in exp:
    for c in r.get("expected_concepts", []):
        agg = aggregator.aggregate(c, "")
        if agg["concept"] == "UNKNOWN":
            unknown += 1
        else:
            mapped += 1
            if agg.get("sub_class_id") and agg["sub_class_id"] != "UNKNOWN":
                sub_known += 1
print(f"[聚合] 总 {mapped+unknown} | 命中标准概念 {mapped} ({mapped/(mapped+unknown):.1%}) | 含子类 {sub_known} ({sub_known/(mapped+unknown):.1%}) | UNKNOWN {unknown}")

# 4. 聚合6级策略分布：用一个样本mention集测试
test_mentions = ["高温", "热害", "持续35℃以上高温", "防效80%", "病情指数", "产量提高", "减产", "光合作用增强", "土壤过湿", "干热风"]
strategy_dist = Counter()
for m in test_mentions:
    agg = aggregator.aggregate(m, "")
    sc = agg.get("sub_class_id", "")
    if sc and sc != "UNKNOWN":
        strategy_dist["命中本体"] += 1
    else:
        strategy_dist["未命中(UNKNOWN)"] += 1
print(f"[6级策略测试] {test_mentions}")
print(f"  命中本体: {strategy_dist['命中本体']}/{len(test_mentions)}")

out = {
    "question_parse_hit_rate": round(q_rate, 3),
    "question_parse_hit_count": q_hit,
    "question_parse_total": q_total,
    "expected_concept_in_kg_rate": round(exp_rate, 3),
    "expected_concept_in_kg_count": exp_hit,
    "expected_concept_total": exp_total,
    "aggregator": {
        "total": mapped + unknown,
        "mapped": mapped,
        "with_subclass": sub_known,
        "unknown": unknown,
        "mapped_rate": round(mapped / (mapped + unknown), 3),
        "with_subclass_rate": round(sub_known / (mapped + unknown), 3),
    },
    "kg_concept_count": len(concepts_in_kg),
}
out_path = _ROOT / "bate" / "data" / "aggregation_quality.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {out_path}")
graph.close()
