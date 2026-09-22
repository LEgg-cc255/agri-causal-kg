"""测试脚本：评估事件抽取与因果关系识别质量

评估指标：
- 事件识别：Precision / Recall / F1（基于概念标签匹配）
- 因果关系抽取：Precision / Recall / F1（基于因果对概念匹配）
- 因果链构建：覆盖率（构建出的链与标注链的重叠度）

运行方式：
    cd c:\\Users\\lenovo\\Desktop\\开题
    python -m bate.tests.test_event_extraction

或：
    cd c:\\Users\\lenovo\\Desktop\\开题\\bate
    python tests\\test_event_extraction.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.utils.llm_client import LLMClient
try:
    from bate.src.pipeline import AgriCausalPipeline
except ImportError:
    # 兼容直接运行
    sys.path.insert(0, str(_ROOT / "bate"))
    from src.pipeline import AgriCausalPipeline

import os as _os
from bate.src.utils.data_loader import load_causal_events, flatten_causal_pairs, flatten_causal_chains
from bate.src.utils.knowledge_base import CausalKnowledgeBase
from bate.src.utils.data_loader import load_synonym_dict


def evaluate_events(predicted_events: list[dict], gold_events: list[dict]) -> dict:
    """评估事件识别（基于概念标签匹配）。"""
    pred_concepts = {e["concept"] for e in predicted_events if e.get("concept", "UNKNOWN") != "UNKNOWN"}
    gold_concepts = {e["concept"] for e in gold_events if e.get("concept")}

    if not pred_concepts and not gold_concepts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(pred_concepts & gold_concepts)
    fp = len(pred_concepts - gold_concepts)
    fn = len(gold_concepts - pred_concepts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _build_subtype_map() -> dict[str, str]:
    """构建 concept -> sub_class_id 映射（LKCER 概念聚合思路，用于 subtype 级软匹配）。"""
    try:
        syn = load_synonym_dict()
        return dict(syn.get("concept_to_subtype", {}))
    except Exception:
        return {}


def _to_subtype_pair(c: str, e: str, subtype_map: dict[str, str]) -> tuple[str, str]:
    """将 concept 对归一化为 subtype 对（同 subtype 视为等价）。"""
    return (subtype_map.get(c, c), subtype_map.get(e, e))


def evaluate_pairs(
    predicted_pairs: list[dict],
    gold_pairs: list[dict],
    kb: CausalKnowledgeBase | None = None,
    subtype_map: dict[str, str] | None = None,
) -> dict:
    """评估因果关系抽取。

    匹配策略（三层，由严到宽）：
    1. 精确 concept 对匹配：(c, e) 完全一致
    2. subtype 级软匹配（LKCER思路）：同 subtype 的 concept 视为等价
    3. KB-supported FP 识别：FP 中命中知识库 known_pairs 的视为"合理但未标注"

    报告两个 F1：
    - strict_f1: 严格 F1（KB-supported FP 仍算 FP）
    - kb_adjusted_f1: 调整 F1（KB-supported FP 不算 FP，反映真实质量）

    注意：不把知识库 known_pairs 并入 gold_set（那会让 gold 暴涨、Recall 虚低）。
    """
    pred_set = set()
    for p in predicted_pairs:
        c = p.get("cause", {}).get("concept", "")
        e = p.get("effect", {}).get("concept", "")
        if c and e and c != "UNKNOWN" and e != "UNKNOWN":
            pred_set.add((c, e))

    gold_set = set()
    for p in gold_pairs:
        c = p.get("cause_event", {}).get("concept", "")
        e = p.get("effect_event", {}).get("concept", "")
        if c and e:
            gold_set.add((c, e))

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0,
                "exact_tp": 0, "subtype_tp": 0, "kb_supported_fp": 0,
                "kb_adjusted_precision": 1.0, "kb_adjusted_f1": 1.0}

    # 精确匹配
    exact_tp = pred_set & gold_set

    # subtype 级软匹配（LKCER思路：同 subtype 视为等价）
    subtype_tp = set()
    if subtype_map:
        gold_subtype = {_to_subtype_pair(c, e, subtype_map) for c, e in gold_set}
        for c, e in pred_set:
            if (c, e) in exact_tp:
                continue
            sp = _to_subtype_pair(c, e, subtype_map)
            if sp in gold_subtype:
                subtype_tp.add((c, e))

    tp = len(exact_tp) + len(subtype_tp)
    fp_pairs = pred_set - exact_tp - subtype_tp
    fp = len(fp_pairs)
    fn = len(gold_set) - len(exact_tp)  # FN 只按精确匹配算（软匹配不增加 FN）

    # KB-supported FP：FP 中命中知识库 known_pairs 的（合理但未标注）
    kb_supported_fp = 0
    if kb is not None and kb.is_loaded:
        for c, e in fp_pairs:
            key = f"{c}|||{e}"
            if key in kb.known_pairs:
                kb_supported_fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # KB-adjusted 指标：KB-supported FP 不算 FP（反映真实抽取质量）
    adjusted_fp = fp - kb_supported_fp
    kb_adjusted_precision = tp / (tp + adjusted_fp) if (tp + adjusted_fp) > 0 else 0.0
    kb_adjusted_f1 = (
        2 * kb_adjusted_precision * recall / (kb_adjusted_precision + recall)
        if (kb_adjusted_precision + recall) > 0 else 0.0
    )

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "exact_tp": len(exact_tp), "subtype_tp": len(subtype_tp),
        "kb_supported_fp": kb_supported_fp,
        "kb_adjusted_precision": kb_adjusted_precision,
        "kb_adjusted_f1": kb_adjusted_f1,
    }


def evaluate_chains(predicted_chains: list[dict], gold_chains: list[dict]) -> dict:
    """评估因果链构建（基于链的概念序列覆盖率）。"""
    pred_chain_sets = [set(c["chain"]) for c in predicted_chains if c.get("is_valid", True)]
    gold_chain_sets = [set(c.get("concepts", c.get("events", []))) for c in gold_chains]

    if not gold_chain_sets:
        return {"coverage": 0.0, "num_predicted": len(pred_chain_sets), "num_gold": 0, "covered": 0}

    if not pred_chain_sets:
        return {"coverage": 0.0, "num_predicted": 0, "num_gold": len(gold_chain_sets), "covered": 0}

    # 对每条gold链，检查是否有pred链与之重叠率>=0.6
    covered = 0
    for gset in gold_chain_sets:
        for pset in pred_chain_sets:
            if not gset:
                continue
            overlap = len(gset & pset) / len(gset)
            if overlap >= 0.6:
                covered += 1
                break

    coverage = covered / len(gold_chain_sets)
    return {
        "coverage": coverage,
        "num_predicted": len(pred_chain_sets),
        "num_gold": len(gold_chain_sets),
        "covered": covered,
    }


def main() -> None:
    print("=" * 70)
    print("农业因果事件抽取 - 评估测试")
    print("=" * 70)

    # 加载数据集（支持 ACED_DATASET 切换合并集）
    import pathlib as _pl
    _ds_env = _os.environ.get("ACED_DATASET", "")
    _ds_path: str | None = None
    if _ds_env:
        _candidates = [
            _pl.Path(_ds_env),
            _pl.Path(__file__).resolve().parent.parent / "data" / _ds_env,
        ]
        for _c in _candidates:
            if _c.exists():
                _ds_path = str(_c)
                break
        if not _ds_path:
            raise FileNotFoundError(f"ACED_DATASET={_ds_env} 找不到文件，试过 {_candidates}")
    dataset = load_causal_events(_ds_path)
    docs = dataset["documents"]
    info = dataset.get("dataset_info", {})
    name = _ds_path or "agri_causal_events.json"
    print(f"\n加载数据集: {name}")
    print(f"  version={info.get('version', '?')}  docs={len(docs)}  "
          f"pairs={info.get('statistics', {}).get('num_causal_pairs', '?')}  "
          f"chains={info.get('statistics', {}).get('num_causal_chains', '?')}")

    # 加载知识库 + 同义词典（评估层增强）
    kb = CausalKnowledgeBase()
    subtype_map = _build_subtype_map()
    print(f"知识库: {kb.stats['num_known_pairs']} 已知因果, {kb.stats['num_forbidden_pairs']} 禁止方向")
    print(f"同义词典: {len(subtype_map)} concept -> subtype 映射")

    # 创建 Pipeline
    llm = LLMClient()
    mode = "Mock（规则模拟）" if llm.mock else f"真实LLM（{llm.model}）"
    print(f"运行模式: {mode}\n")
    pipeline = AgriCausalPipeline(llm_client=llm, knowledge_base=kb)

    # 累计统计
    event_stats = Counter()
    pair_stats = Counter()
    chain_results = {"num_predicted": 0, "num_gold": 0, "covered": 0}
    total_gold_chains = 0

    for doc in docs:
        text = doc["text"]
        gold_events = []
        for p in doc.get("causal_pairs", []):
            gold_events.append(p.get("cause_event", {}))
            gold_events.append(p.get("effect_event", {}))
        gold_pairs = doc.get("causal_pairs", [])
        gold_chains = doc.get("causal_chains", [])
        total_gold_chains += len(gold_chains)

        # 运行 pipeline
        result = pipeline.run(text)

        # 评估事件
        ev_metrics = evaluate_events(result["events"], gold_events)
        event_stats["tp"] += ev_metrics["tp"]
        event_stats["fp"] += ev_metrics["fp"]
        event_stats["fn"] += ev_metrics["fn"]

        # 评估因果对（知识库扩充gold + subtype软匹配）
        pr_metrics = evaluate_pairs(result["causal_pairs"], gold_pairs, kb=kb, subtype_map=subtype_map)
        pair_stats["tp"] += pr_metrics["tp"]
        pair_stats["fp"] += pr_metrics["fp"]
        pair_stats["fn"] += pr_metrics["fn"]
        pair_stats["exact_tp"] += pr_metrics.get("exact_tp", 0)
        pair_stats["subtype_tp"] += pr_metrics.get("subtype_tp", 0)
        pair_stats["kb_supported_fp"] += pr_metrics.get("kb_supported_fp", 0)

        # 评估因果链
        ch_metrics = evaluate_chains(result["causal_chains"], gold_chains)
        chain_results["num_predicted"] += ch_metrics["num_predicted"]
        chain_results["num_gold"] += ch_metrics["num_gold"]
        chain_results["covered"] += ch_metrics["covered"]

    # 汇总指标
    def prf(stats):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    ev_p, ev_r, ev_f = prf(event_stats)
    pr_p, pr_r, pr_f = prf(pair_stats)
    chain_cov = chain_results["covered"] / chain_results["num_gold"] if chain_results["num_gold"] > 0 else 0.0

    print("=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"\n[事件识别]")
    print(f"  Precision: {ev_p:.3f}  Recall: {ev_r:.3f}  F1: {ev_f:.3f}")
    print(f"  TP={event_stats['tp']}  FP={event_stats['fp']}  FN={event_stats['fn']}")

    print(f"\n[因果关系抽取]")
    print(f"  严格指标  Precision: {pr_p:.3f}  Recall: {pr_r:.3f}  F1: {pr_f:.3f}")
    print(f"  TP={pair_stats['tp']}  FP={pair_stats['fp']}  FN={pair_stats['fn']}")
    kb_fp = pair_stats.get('kb_supported_fp', 0)
    print(f"  匹配明细: 精确TP={pair_stats.get('exact_tp',0)}  subtype软匹配TP={pair_stats.get('subtype_tp',0)}")
    print(f"  FP分析: KB-supported合理FP={kb_fp}  真正FP={pair_stats['fp']-kb_fp}")

    # KB-adjusted 指标（把 KB-supported 合理FP 从 FP 中剔除）
    tp_total = pair_stats['tp']
    fp_total = pair_stats['fp']
    fn_total = pair_stats['fn']
    adjusted_fp = fp_total - kb_fp
    kb_adj_p = tp_total / (tp_total + adjusted_fp) if (tp_total + adjusted_fp) > 0 else 0.0
    kb_adj_f1 = 2 * kb_adj_p * pr_r / (kb_adj_p + pr_r) if (kb_adj_p + pr_r) > 0 else 0.0
    print(f"  KB调整指标 Precision: {kb_adj_p:.3f}  F1: {kb_adj_f1:.3f}  (合理FP不计入FP)")
    print(f"  >>> F1 提升: {pr_f:.3f} → {kb_adj_f1:.3f} (+{kb_adj_f1-pr_f:.3f})")

    print(f"\n[因果链构建]")
    print(f"  覆盖率: {chain_cov:.3f}  "
          f"(覆盖 {chain_results['covered']}/{chain_results['num_gold']} 条标注链, "
          f"构建 {chain_results['num_predicted']} 条)")

    print(f"\n[说明]")
    if llm.mock:
        print("  当前为 Mock 模式，指标仅供参考（基于规则模拟，非真实LLM输出）。")
        print("  配置 LLM_API_KEY 环境变量后可获得真实评估结果。")
    else:
        print(f"  使用真实LLM ({llm.model}) 评估。")
    print("=" * 70)


if __name__ == "__main__":
    main()
