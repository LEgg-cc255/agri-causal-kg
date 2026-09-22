"""概念归一坍缩全量扫描：找出 16 篇文档中所有聚合异常

扫描两类问题：
  A. 坍缩（自环）：同一因果对里 cause.concept == effect.concept 且 mention 文本不同
     → 最危险，会让本体方向过滤无效，还会形成自环边破坏因果链
  B. 错配（高可疑）：mention 与聚合出的 concept 语义不符，基于硬规则标记：
     - mention 含"降低/减少/下降/抑制/损失"等负向词 → 但 concept 是正向干预（如降温措施/品种选择）
     - mention 含"具体数值单位℃/%" → 但 concept 不是量化类标签
     - 多 mention（>2 条不同 text）都归一到同一 concept（过度坍缩）

输出：建议补入 mention_to_concept 的映射列表，供人工审核后加进词典。
"""

from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.pipeline import AgriCausalPipeline
from bate.src.utils.data_loader import load_causal_events

NEG_WORDS = ["降低", "减少", "下降", "抑制", "损失", "低下", "不良", "减少", "缩短", "受损", "伤害", "缺水"]
QUANT_MARKERS = ["℃", "%", "mg/", "g/kg", "kg/ha", "d-", "天以上", "小时", "年一遇"]
# 明显的"措施类"concept（如果 mention 是描述结果却归到这些 concept，就是错配）
INTERVENTION_CONCEPTS = {"降温措施", "品种选择", "施肥措施", "灌溉措施", "遮阳措施", "种植调整", "覆盖/保墒", "播期调整"}


def scan_doc(doc, pipeline):
    text = doc["text"]
    result = pipeline.run(text)
    pairs = result["causal_pairs"]

    self_loops = []           # A
    mismatches = []           # B1 负向mention被归为正向干预
    quant_mismatches = []     # B2 量化mention被归为非量化concept
    # 统计 concept -> 不同 mentions（B3 过度坍缩）
    concept_mentions: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        for role in ("cause", "effect"):
            ev = p[role]
            concept_mentions[ev["concept"]].add(ev["text"])

    # A. 自环坍缩
    for i, p in enumerate(pairs, 1):
        if p["cause"]["concept"] == p["effect"]["concept"]:
            # mention 不同才算真坍缩（原文同词不算问题）
            if p["cause"]["text"].strip() != p["effect"]["text"].strip():
                self_loops.append({
                    "i": i,
                    "concept": p["cause"]["concept"],
                    "cause_text": p["cause"]["text"],
                    "effect_text": p["effect"]["text"],
                    "conf": p.get("confidence", 0),
                    "evidence": p.get("evidence", ""),
                })

    # B1/B2. 错配
    for i, p in enumerate(pairs, 1):
        for role in ("cause", "effect"):
            ev = p[role]
            mention = ev["text"]
            concept = ev["concept"]
            # B1 负向 mention 被归为正向干预 concept
            if concept in INTERVENTION_CONCEPTS and any(w in mention for w in NEG_WORDS):
                mismatches.append({
                    "i": i, "role": role, "mention": mention, "concept": concept,
                    "evidence": p.get("evidence", ""),
                })
            # B2 数值单位 mention 但 concept 不是温度/水分量化类
            if any(q in mention for q in QUANT_MARKERS):
                if concept not in {"高温胁迫", "低温冷害", "温度下降", "相对干旱",
                                   "渍害", "盐胁迫", "空气湿度", "光合有效辐射",
                                   "相对湿度下降"}:
                    quant_mismatches.append({
                        "i": i, "role": role, "mention": mention, "concept": concept,
                        "evidence": p.get("evidence", ""),
                    })

    # B3. 过度坍缩（同 concept 对应 3+ 差异明显 mention）
    over_collapsed = []
    for concept, mentions in concept_mentions.items():
        if len(mentions) >= 3 and concept != "UNKNOWN":
            # 差异判断：mention 互相不包含子串
            ms = list(mentions)
            diverse = True
            for a in ms:
                for b in ms:
                    if a != b and (a in b or b in a):
                        diverse = False
                        break
                if not diverse:
                    break
            if diverse:
                over_collapsed.append({"concept": concept, "mentions": sorted(ms, key=len)})

    return {
        "doc_id": doc["doc_id"],
        "pairs": pairs,
        "self_loops": self_loops,
        "mismatches": mismatches,
        "quant_mismatches": quant_mismatches,
        "over_collapsed": over_collapsed,
    }


def main():
    dataset = load_causal_events()
    docs = dataset["documents"]
    pipeline = AgriCausalPipeline()

    total_pairs = 0
    all_self_loops = []
    all_mismatches = []
    all_quant_mm = []
    all_over = []

    # 收集建议：mention → 更合适的 concept（基于本体常见候选）
    suggestions = {}  # mention_text -> set of likely_concepts

    print("=" * 72)
    print("全量概念归一坍缩扫描（16 篇）")
    print("=" * 72)

    for doc in docs:
        r = scan_doc(doc, pipeline)
        total_pairs += len(r["pairs"])
        flags = []
        if r["self_loops"]:
            flags.append(f"A自环×{len(r['self_loops'])}")
        if r["mismatches"]:
            flags.append(f"B1错配×{len(r['mismatches'])}")
        if r["quant_mismatches"]:
            flags.append(f"B2量化×{len(r['quant_mismatches'])}")
        if r["over_collapsed"]:
            flags.append(f"B3坍缩×{len(r['over_collapsed'])}")

        mark = "  " + " ".join(flags) if flags else ""
        print(f"\n[{r['doc_id']}] {doc['title'][:38]}  ({len(r['pairs'])}对){mark}")

        if r["self_loops"]:
            print("  └─ [A] 自环坍缩 (cause/effect同concept、text不同):")
            for s in r["self_loops"]:
                print(f"      * #{s['i']:02d} 「{s['concept']}」 ← cause='{s['cause_text']}'  effect='{s['effect_text']}'")
                print(f"          证据: {s['evidence'][:60]}")
        if r["mismatches"]:
            print("  └─ [B1] 负向mention被归为正向干预:")
            for m in r["mismatches"]:
                print(f"      * #{m['i']:02d} {m['role']}: '{m['mention']}' → 「{m['concept']}」")
        if r["quant_mismatches"]:
            print("  └─ [B2] 含数值单位但concept非量化类:")
            for q in r["quant_mismatches"]:
                print(f"      * #{q['i']:02d} {q['role']}: '{q['mention']}' → 「{q['concept']}」")
        if r["over_collapsed"]:
            print("  └─ [B3] 单一concept对多个差异mention（过度坍缩）:")
            for o in r["over_collapsed"]:
                print(f"      * 「{o['concept']}」 ← " + " / ".join(o["mentions"]))

        all_self_loops.extend((r["doc_id"], s) for s in r["self_loops"])
        all_mismatches.extend((r["doc_id"], m) for m in r["mismatches"])
        all_quant_mm.extend((r["doc_id"], q) for q in r["quant_mismatches"])
        all_over.extend((r["doc_id"], o) for o in r["over_collapsed"])

    # 汇总
    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    print(f"Pipeline 总抽对数: {total_pairs}")
    print(f"  A 自环坍缩: {len(all_self_loops)} 条 (影响方向过滤 + 因果链)")
    print(f"  B1 负向→干预错配: {len(all_mismatches)} 条")
    print(f"  B2 量化mention错归: {len(all_quant_mm)} 条")
    print(f"  B3 过度坍缩概念: {len(all_over)} 个")

    # 输出建议的 mention_to_concept 增补（基于自环和量化错配）
    # 自环：effect 侧经常是温度下降、遮光减少等结果 → 建议按 mention 关键词映射
    print("\n" + "=" * 72)
    print("【建议补入 mention_to_concept】（按出现频率排序）")
    print("=" * 72)

    # 收集所有 A/B2 中的问题 mention → 频次
    problem_mentions: dict[str, list[tuple[str, str]]] = defaultdict(list)  # mention -> [(doc_id, actual_concept_LLM_output)]
    for doc_id, s in all_self_loops:
        problem_mentions[s["cause_text"]].append((doc_id, s["concept"]))
        problem_mentions[s["effect_text"]].append((doc_id, s["concept"]))
    for doc_id, q in all_quant_mm:
        problem_mentions[q["mention"]].append((doc_id, q["concept"]))

    ranked = sorted(problem_mentions.items(), key=lambda kv: -len(kv[1]))
    print(f"共 {len(ranked)} 个问题 mention")
    for mention, occs in ranked[:50]:
        current = occs[0][1]  # 现在被归到的 concept
        docs = ",".join(sorted(set(d for d, _ in occs)))
        # 基于关键词的推荐映射
        rec = recommend(mention)
        print(f"\n  mention: '{mention}'")
        print(f"    当前被归为: 「{current}」  出现于: {docs}")
        print(f"    建议: {rec}")

    # --export 模式：输出结构化 JSON
    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        out = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "collapse_problems.json"
        problems = export_problems(all_self_loops, all_mismatches, out)
        print(f"\n[导出] {len(problems)} 条问题 → {out}")


def recommend_concept(mention: str) -> str | None:
    """基于 mention 关键词给出推荐的标准 concept（纯名称），无法判断返回 None。"""
    m = mention
    # 温度
    if any(k in m for k in ["气温降低", "温度降低", "降低2℃", "降低温度", "气温下降"]):
        return "温度下降"
    if any(k in m for k in ["升温抑制", "温升抑制", "抑制升温"]):
        return "温度下降"
    if any(k in m for k in ["℃高温", "℃的高温", "超过25℃", "℃ 高温", "平均气温超过"]):
        return "高温胁迫"
    # 光照
    if any(k in m for k in ["遮光减少", "遮光カーテン使用减少", "遮光少用"]):
        return "光照充足"
    if any(k in m for k in ["日射量充足", "日射量更充足", "日射量增加"]):
        return "日射量充足"
    if any(k in m for k in ["光照不足", "日照不足"]) or (any(k in m for k in ["遮光", "日射量", "光照"]) and any(k in m for k in ["减少", "降低", "不足"])):
        return "光照不足"
    # 设备
    if any(k in m for k in ["气化热", "パッドアンドファン", "细雾冷房", "降温设备"]):
        return "降温措施"
    if any(k in m for k in ["カーテン", "遮阳", "遮阴", "遮光布", "遮光カーテン"]) and not any(k in m for k in ["减少", "少用", "不足"]):
        return "遮阳措施"
    # 生理
    if any(k in m for k in ["肥大不良", "果实肥大", "膨大不"]):
        return "果实发育不良"
    if any(k in m for k in ["稔性", "花粉活力", "花粉量"]) and any(k in m for k in ["低下", "减少", "低"]):
        return "花粉活力下降"
    # 水分
    if any(k in m for k in ["氧气扩散", "充水", "孔隙充水"]):
        return "氧气扩散受阻"
    if any(k in m for k in ["供需矛盾", "供需失衡", "失水"]):
        return "水分供需失衡"
    # 氧化/复合
    if any(k in m for k in ["ROS", "活性氧", "氧化"]):
        return "氧化应激"
    if any(k in m for k in ["复合损害", "复合胁迫"]):
        return "复合胁迫"
    # 记忆/抗逆
    if any(k in m for k in ["转录记忆", "形成记忆"]):
        return "胁迫记忆形成"
    if any(k in m for k in ["存活与响应", "增强.*存活"]):
        return "抗逆性增强"
    return None


def recommend(mention: str) -> str:
    """兼容旧调用：返回带注释的推荐字符串。"""
    c = recommend_concept(mention)
    return f"{c}" if c else "(人工判断)"


def export_problems(all_self_loops, all_mismatches, output_path):
    """输出结构化问题列表到 JSON，供交互式修复使用。"""
    import json

    problems: list[dict] = []
    seen: set[str] = set()  # mention 去重

    for doc_id, s in all_self_loops:
        for role, key in [("cause", "cause_text"), ("effect", "effect_text")]:
            mention = s[key]
            if mention in seen:
                continue
            seen.add(mention)
            rec = recommend_concept(mention)
            if rec and rec != s["concept"]:  # 只导出"推荐与当前不同"的
                problems.append({
                    "doc_id": doc_id,
                    "mention": mention,
                    "current_concept": s["concept"],
                    "recommended_concept": rec,
                    "evidence": s.get("evidence", "")[:80],
                    "problem_type": "self_loop",
                    "role": role,
                })

    for doc_id, m in all_mismatches:
        mention = m["mention"]
        if mention in seen:
            continue
        seen.add(mention)
        rec = recommend_concept(mention)
        problems.append({
            "doc_id": doc_id,
            "mention": mention,
            "current_concept": m["concept"],
            "recommended_concept": rec or "",
            "evidence": m.get("evidence", "")[:80],
            "problem_type": "mismatch",
            "role": m["role"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
    return problems


def apply_fixes(fixes: dict[str, str], dict_path) -> int:
    """把修复写入 agri_synonym_dict.json 的 mention_to_concept。

    fixes: {mention_text: correct_concept}
    返回实际写入条数（已存在且相同的不算）。
    """
    import json

    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m2c = data.get("mention_to_concept", {})
    applied = 0
    for mention, concept in fixes.items():
        if m2c.get(mention) != concept:
            m2c[mention] = concept
            applied += 1

    data["mention_to_concept"] = m2c
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return applied


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--export" in args:
        idx = args.index("--export")
        out = args[idx + 1] if idx + 1 < len(args) else "collapse_problems.json"
        # main() 已经跑完扫描，需要把 all_self_loops 等传给 export_problems
        # 这里通过重新组织 main 来实现
        main()  # 先跑扫描打印
    else:
        main()
