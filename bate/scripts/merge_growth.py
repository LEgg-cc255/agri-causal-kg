"""合并 agri_causal_events_growth.json → agri_causal_events.json，输出最终金标集。

策略：
  1. 强去重：(doc_id, cause_concept, effect_concept) 在金标已有 → 删除增长集该条
     （因为知识库已知对中包含因果传递，只要 concept 对重复就视为同样的抽取单元，
      评估时 Pair-Level Evaluation 要求 cause.concept==gold.cause.concept，重复无意义）
  2. 弱去重：concept 对不同但 mention_text 完全相同 → 删除（防御性，实际很少）
  3. 合并：增长集幸存的 pair 追加到对应文档的 causal_pairs[]，pair_id 重排为 Pxx（从金标最大值续数）
     - 保留 source、verified 额外字段（兼容原金标 schema，只增不减）
  4. chain：增长集不生成新 causal_chains（增长集对是 pair 级的，链构建由 Pipeline 运行时再做），
     原金标 causal_chains 完全保留
  5. 最终 statistics 更新：版本号+0.1、pairs、chains、按领域覆盖的 doc 数、domains 列表
  6. 输出文件：data/agri_causal_events_merged.json（不覆盖原金标，确认正确后可改名）
"""

from __future__ import annotations
import json
import sys
import re
from collections import defaultdict, Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "bate" / "data"
GOLD_PATH = DATA_DIR / "agri_causal_events.json"
GROWTH_PATH = DATA_DIR / "agri_causal_events_growth.json"
OUT_PATH = DATA_DIR / "agri_causal_events_merged.json"


def parse_doc_suffix(pid: str) -> int:
    """从 pair_id/chain_id 中提取数字部分（P01→1，G12→12），找不到返回 0。"""
    m = re.search(r"([A-Za-z])(\d+)$", pid or "")
    return int(m.group(2)) if m else 0


def existing_max_pair_n(doc_causal_pairs: list[dict]) -> int:
    """文档内现有 pair 的最大序号。"""
    return max((parse_doc_suffix(p.get("pair_id", "")) for p in doc_causal_pairs), default=0)


def main():
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = json.load(f)
    with open(GROWTH_PATH, encoding="utf-8") as f:
        growth = json.load(f)

    # 把 gold 建成 (doc_id, concept_tuple) -> pair 的快速索引
    gold_index: dict[tuple[str, str, str], dict] = {}
    gold_mention_index: set[tuple[str, str, str]] = set()
    gold_docs: dict[str, dict] = {}
    for doc in gold["documents"]:
        did = doc["doc_id"]
        gold_docs[did] = doc
        for p in doc["causal_pairs"]:
            cc = p["cause_event"]["concept"]
            ee = p["effect_event"]["concept"]
            gold_index[(did, cc, ee)] = p
            ct = p["cause_event"].get("text", "").strip()
            et = p["effect_event"].get("text", "").strip()
            gold_mention_index.add((did, ct, et))

    stats = defaultdict(int)
    stats["growth_total_pairs"] = sum(
        len(d.get("causal_pairs", [])) for d in growth["documents"]
    )
    stats["gold_total_pairs"] = sum(
        len(d.get("causal_pairs", [])) for d in gold["documents"]
    )

    merged_docs: list[dict] = []
    merge_log: list[dict] = []

    # 遍历金标 16 篇文档的顺序，保证输出顺序与金标一致
    growth_docs_by_id = {d["doc_id"]: d for d in growth["documents"]}
    for doc in gold["documents"]:
        did = doc["doc_id"]
        new_pairs: list[dict] = []  # 这次要新增的
        doc_log = {"doc_id": did, "gold": len(doc["causal_pairs"]),
                   "growth_in": 0, "dup_concept": 0, "dup_mention": 0,
                   "added": 0, "skipped_unverified": 0}

        g_doc = growth_docs_by_id.get(did)
        if g_doc:
            g_pairs = g_doc.get("causal_pairs", [])
            doc_log["growth_in"] = len(g_pairs)
            max_n = existing_max_pair_n(doc["causal_pairs"])
            for gp in g_pairs:
                cc = gp["cause_event"]["concept"]
                ee = gp["effect_event"]["concept"]
                # 强去重：concept 对
                if (did, cc, ee) in gold_index:
                    doc_log["dup_concept"] += 1
                    continue
                ct = gp["cause_event"].get("text", "").strip()
                et = gp["effect_event"].get("text", "").strip()
                # 弱去重：mention text
                if (did, ct, et) in gold_mention_index:
                    doc_log["dup_mention"] += 1
                    continue
                # 只接 verified=true 的（增长集里全是 true，但留校验）
                if not gp.get("verified", False):
                    doc_log["skipped_unverified"] += 1
                    continue

                # 重排 pair_id：DOC-001-P05、DOC-001-P06 ...
                max_n += 1
                new_pair = dict(gp)
                new_pair["pair_id"] = f"{did}-P{max_n:02d}"
                # 保证 relation_type、confidence 有值
                if "relation_type" not in new_pair:
                    new_pair["relation_type"] = "DIRECT"
                if "confidence" not in new_pair:
                    new_pair["confidence"] = 0.70
                new_pairs.append(new_pair)
                doc_log["added"] += 1
                # 同步加入去重索引，防止增长集内部自身重复
                gold_index[(did, cc, ee)] = new_pair
                gold_mention_index.add((did, ct, et))

        # 组装新 document：深拷贝老的 pairs，加新 pairs，chain 完全保留
        merged_doc = dict(doc)
        merged_doc["causal_pairs"] = list(doc["causal_pairs"]) + new_pairs
        # chain 保持原金标不动
        merged_docs.append(merged_doc)
        merge_log.append(doc_log)

    # ----- statistics 汇总 -----
    total_pairs = sum(len(d["causal_pairs"]) for d in merged_docs)
    total_chains = sum(len(d.get("causal_chains", [])) for d in merged_docs)

    # domains 去重（用 domain 字段的 "-" 前缀取大类，兼容 "设施蔬菜-番茄"）
    domains = []
    seen_domain = set()
    for d in merged_docs:
        dm = d.get("domain", "")
        # 取大类："设施蔬菜-番茄" → "设施蔬菜"；只有大类的如 "水稻" 直接保留
        big = dm.split("-")[0] if dm else dm
        if big and big not in seen_domain:
            seen_domain.add(big)
            domains.append(big)
    # 与原 domains 合并，保持原有的排序习惯（水稻/小麦/设施蔬菜优先）
    original_domains = gold["dataset_info"]["statistics"].get("domains", [])
    final_domains = [d for d in original_domains if d in domains]
    final_domains += [d for d in domains if d not in original_domains]

    info = dict(gold["dataset_info"])
    # bump version：1.0 → 1.1，如果已经是 1.1+ 则加小数
    v = info.get("version", "1.0")
    try:
        parts = v.split(".")
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
            info["version"] = f"{major}.{minor + 1}"
        else:
            info["version"] = v + "+"
    except ValueError:
        info["version"] = v + "+"
    if "growth_note" not in info.get("description", ""):
        info["description"] = (
            info.get("description", "")
            + " [v1.1 2026-08] 并入主动学习增长集，去重概念对后新增对经人工复核确认。"
        )
    stats = info.get("statistics", {})
    stats["num_documents"] = len(merged_docs)
    stats["num_causal_pairs"] = total_pairs
    stats["num_causal_chains"] = total_chains
    stats["domains"] = final_domains
    stats["growth_pairs_merged"] = total_pairs - sum(
        len(d.get("causal_pairs", [])) for d in gold["documents"]
    )  # 净增
    stats["gold_original_pairs"] = stats.get("num_causal_pairs", 42) - stats.get("growth_pairs_merged", 0)

    info["statistics"] = stats
    info["construction_method"] = (
        gold["dataset_info"].get("construction_method", "")
        + " [2026-08] 半自动扩展：运行主动学习回填 Pipeline（LLM 联合抽取 + 启发式自动标注），"
        "人工复核后并入。增长集来源全部带 source=active_learning 标记。"
    )
    info["label_schema"] = gold["dataset_info"].get("label_schema", {})

    merged = {
        "dataset_info": info,
        "documents": merged_docs,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # ----- 打印 merge 报告 -----
    _gold_pairs = sum(len(d["causal_pairs"]) for d in gold["documents"])
    _gold_chains = sum(len(d.get("causal_chains", [])) for d in gold["documents"])
    _gp = sum(len(d.get("causal_pairs", [])) for d in growth["documents"])
    print("=" * 92)
    print("合并报告：增长集 → 金标集")
    print("=" * 92)
    print(f"  输入金标:   {GOLD_PATH}  ({len(gold_docs)} docs / {_gold_pairs} pairs / {_gold_chains} chains)")
    print(f"  输入增长集: {GROWTH_PATH}  ({len(growth['documents'])} docs / {_gp} pairs)")
    print(f"  输出:       {OUT_PATH}")
    print()

    total_gold = 0
    total_growth_in = 0
    total_dup_c = 0
    total_dup_m = 0
    total_added = 0
    print(f"  {'doc_id':8s} {'金标pairs':>10s} {'增长集入':>9s} {'dup_conc':>8s} {'dup_men':>8s} {'新增':>6s} {'最终pairs':>10s}")
    print(f"  {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
    for l in merge_log:
        total_gold += l["gold"]
        total_growth_in += l["growth_in"]
        total_dup_c += l["dup_concept"]
        total_dup_m += l["dup_mention"]
        total_added += l["added"]
        final_n = l["gold"] + l["added"]
        print(f"  {l['doc_id']:8s} {l['gold']:>10d} {l['growth_in']:>9d} {l['dup_concept']:>8d} {l['dup_mention']:>8d} {l['added']:>6d} {final_n:>10d}")
    print(f"  {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
    print(f"  {'合计':8s} {total_gold:>10d} {total_growth_in:>9d} {total_dup_c:>8d} {total_dup_m:>8d} {total_added:>6d} {total_gold + total_added:>10d}")

    print(f"\n{'='*92}")
    print(f"dataset_info.statistics 最终值:")
    print(f"  version          = {info['version']}")
    print(f"  num_documents    = {stats['num_documents']}")
    print(f"  num_causal_pairs = {stats['num_causal_pairs']}")
    print(f"  num_causal_chains= {stats['num_causal_chains']}")
    print(f"  domains          = {stats['domains']}")
    print(f"  净增 pairs       = {total_added}")
    print(f"  去重(概念级重复) = {total_dup_c}")
    print(f"  去重(文本级重复) = {total_dup_m}")
    print(f"\n  输出文件: {OUT_PATH}")
    print(f"  ❗ 未覆盖原金标: {GOLD_PATH}（请人工检查后再改名覆盖）")
    print("=" * 92)


if __name__ == "__main__":
    main()
