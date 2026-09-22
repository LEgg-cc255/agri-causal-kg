"""回填标注：遍历金标集文档，用主动学习补全"合理但未标注"的因果对

背景：当前金标集 42 对 vs Pipeline 抽 175 对，133 对"合理但未标注"被误判为 FP，
导致严格 F1 偏低。本脚本对每个金标文档重跑 Pipeline，把高价值候选通过
主动学习(y/n/r/s)确认后写入增长集（agri_causal_events_growth.json），
人工复核后再并入金标集。

与 run_active_learning 的区别：
- run_active_learning：面向自由输入文本，doc_id=AL-FREETEXT
- 本脚本：面向金标集已有文档，doc_id=DOC-xxx，doc_meta 从金标集取，便于追溯

运行：
    python -m bate.scripts.backfill_annotations               # 遍历全部文档
    python -m bate.scripts.backfill_annotations --doc DOC-001  # 只跑指定文档
    python -m bate.scripts.backfill_annotations --top-k 30     # 每轮标30个
    python -m bate.scripts.backfill_annotations --from DOC-005 # 从某文档开始
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.pipeline import AgriCausalPipeline
from bate.src.utils.data_loader import load_causal_events
from bate.src.utils.active_learning import (
    active_learning_round,
    select_candidates,
    apply_annotations,
    apply_to_growth_set,
)


def _auto_annotate(selected: list[dict]) -> list[dict]:
    """非交互自动打标（基于 pair.confidence 和 kb_match 的正确性判断）

    注意：value_score 的 score["total"] 是"人工标注价值"（越不确定越值得标=越高），
    不是因果正确性的置信度。正确性判断看 pair 自带的字段。
    """
    out: list[dict] = []
    for cand in selected:
        pair = cand["pair"]
        # 取 pair 的正确性相关字段
        llm_conf = pair.get("confidence", 0.5)
        kb_match = pair.get("kb_match", {}) or {}
        kb_known = bool(kb_match.get("known"))
        kb_forbid = bool(kb_match.get("forbidden"))
        llm_tag = pair.get("is_causal", True)

        p = pair
        cause_c = ""
        effect_c = ""
        # Pipeline 输出 pair 用 cause/effect；数据集格式用 cause_event/effect_event
        _cause = p.get("cause_event") or p.get("cause") or {}
        _effect = p.get("effect_event") or p.get("effect") or {}
        cause_c = _cause.get("concept", "") if isinstance(_cause, dict) else ""
        effect_c = _effect.get("concept", "") if isinstance(_effect, dict) else ""

        # --- 先判错误 ---
        # 1) 知识库禁止 → 直接 n
        if kb_forbid:
            out.append({"pair": pair, "label": "n",
                        "reason": f"auto-n kb_forbid conf={llm_conf:.2f}"})
            continue
        # 2) 自环（cause==effect） → 留人工，不自动入增长集
        if cause_c and effect_c and cause_c == effect_c:
            out.append({"pair": pair, "label": "s",
                        "reason": "auto-s 自环，需人工确认是否同义"})
            continue
        # 3) LLM 标成非因果 or 置信度极低 → n
        if not llm_tag or llm_conf < 0.25:
            out.append({"pair": pair, "label": "n",
                        "reason": f"auto-n 低置信 conf={llm_conf:.2f} tag={llm_tag}"})
            continue

        # --- 再判正确 ---
        # 4) 高置信 (≥0.85) 且 kb 不冲突 → y
        if llm_conf >= 0.85:
            out.append({"pair": pair, "label": "y",
                        "reason": f"auto-y 高置信 conf={llm_conf:.2f}"})
            continue
        # 5) 知识库已确认 known=True 且 conf≥0.70 → y
        if kb_known and llm_conf >= 0.70:
            out.append({"pair": pair, "label": "y",
                        "reason": f"auto-y kb+known conf={llm_conf:.2f}"})
            continue
        # 6) 中等置信 (0.50~0.85) → 留人工 s
        if llm_conf >= 0.50:
            out.append({"pair": pair, "label": "s",
                        "reason": f"auto-s 边界 conf={llm_conf:.2f} 留人工"})
            continue
        # 7) 0.25~0.50 低置信 → n
        out.append({"pair": pair, "label": "n",
                    "reason": f"auto-n 低置信 conf={llm_conf:.2f} 未达阈值"})
    return out


def backfill(
    pipeline: AgriCausalPipeline,
    dataset: dict,
    *,
    only_doc: str | None = None,
    from_doc: str | None = None,
    top_k: int = 20,
    batch: bool = False,
) -> None:
    """遍历金标集文档，逐个跑主动学习回填。"""
    docs = dataset.get("documents", [])
    started = from_doc is None  # 未指定 --from 则从头开始

    total_stats = {"candidates": 0, "annotated": 0, "y": 0, "n": 0, "r": 0, "s": 0,
                   "kb_known": 0, "kb_forbid": 0, "growth": 0}

    for doc in docs:
        doc_id = doc.get("doc_id", "?")

        if only_doc and doc_id != only_doc:
            continue
        if from_doc and not started:
            if doc_id == from_doc:
                started = True
            else:
                continue

        text = doc.get("text", "").strip()
        if not text:
            print(f"[{doc_id}] 无正文，跳过。")
            continue

        print(f"\n{'#'*64}")
        print(f"# {doc_id}  {doc.get('domain', '')}  {doc.get('title', '')[:40]}")
        print(f"# 正文: {text[:80]}...")
        print(f"{'#'*64}")

        # 文档元信息（写增长集时填充文档条目）
        doc_meta = {
            "doc_id": doc_id,
            "domain": doc.get("domain", "未分类"),
            "title": doc.get("title", ""),
            "source": doc.get("source", {}),
            "text": text,
        }

        # Pipeline 抽取
        result = pipeline.run(text)
        pairs = result["causal_pairs"]
        print(f"Pipeline 抽取: {len(result['events'])} 事件, {len(pairs)} 因果对")

        if not pairs:
            print("无因果对，跳过。")
            continue

        doc_s = {"candidates": 0, "annotated": 0, "y": 0, "n": 0, "r": 0, "s": 0}

        if batch:
            # 非交互模式：选候选 → 自动标注 → 直接写 KB+增长集
            selected = select_candidates(pairs, pipeline.kb, top_k=top_k)
            annotations = _auto_annotate(selected)
            print(f"自动标注 {len(annotations)} 个候选：")
            for ann in annotations:
                p0 = ann["pair"]
                ce = p0.get("cause_event") or p0.get("cause") or {}
                ee = p0.get("effect_event") or p0.get("effect") or {}
                cc = ce.get("concept", "")
                ec = ee.get("concept", "")
                print(f"  [{ann['label'].upper()}] {cc} → {ec}   ({ann['reason']})")
            # 转为两个函数期望的格式：
            #   - label: y/n/r → yes/no/reverse  (s 跳过)
            #   - 字段: cause_concept / effect_concept / label  (apply_to_growth_set 还保留 pair 原始引用)
            prepared: list[dict] = []
            for ann in annotations:
                if ann["label"] == "s":
                    continue
                mapping = {"y": "yes", "n": "no", "r": "reverse"}
                p0 = ann["pair"]
                ce = p0.get("cause_event") or p0.get("cause") or {}
                ee = p0.get("effect_event") or p0.get("effect") or {}
                cc = ce.get("concept", "")
                ec = ee.get("concept", "")
                lab = mapping[ann["label"]]
                prepared.append({
                    "cause_concept": cc,
                    "effect_concept": ec,
                    "label": lab,
                    "pair": p0,
                    "doc_id": doc_id,
                })
                char_label = ann["label"]
                doc_s[char_label] = doc_s.get(char_label, 0) + 1
                doc_s["annotated"] += 1
            doc_s["candidates"] = len(selected)
            # 写 KB
            kb_changes = apply_annotations(prepared, pipeline.kb)
            doc_s["kb_known"] = kb_changes.get("added_known", 0)
            doc_s["kb_forbid"] = kb_changes.get("added_forbidden", 0)
            # 写增长集
            growth_result = apply_to_growth_set(prepared, doc_meta=doc_meta)
            doc_s["growth"] = growth_result.get("added_pairs", 0) if isinstance(growth_result, dict) else 0
        else:
            # 交互模式（原有路径）
            stats = active_learning_round(
                pairs=pairs,
                kb=pipeline.kb,
                text=text,
                doc_id=doc_id,
                top_k=top_k,
                interactive=True,
                doc_meta=doc_meta,
            )
            s = stats["stats"]
            doc_s["candidates"] = s["num_candidates"]
            doc_s["annotated"] = s["num_annotated"]
            doc_s["kb_known"] = s.get("added_known", 0)
            doc_s["kb_forbid"] = s.get("added_forbidden", 0)
            doc_s["growth"] = s.get("growth_added", 0)
            anns = stats.get("annotations", [])
            for a in anns:
                lab = a.get("annotation", "s")
                doc_s[lab] = doc_s.get(lab, 0) + 1

        for k in total_stats:
            total_stats[k] = total_stats.get(k, 0) + doc_s.get(k, 0)
        print(f"\n[{doc_id} 完成] 候选={doc_s['candidates']} 标注={doc_s['annotated']} "
              f"y={doc_s.get('y',0)} n={doc_s.get('n',0)} r={doc_s.get('r',0)} s={doc_s.get('s',0)} "
              f"KB+{doc_s.get('kb_known',0)}已知/+{doc_s.get('kb_forbid',0)}禁止 "
              f"增长集+{doc_s.get('growth',0)}对")

    print(f"\n{'='*64}")
    print(f"批量汇总：候选={total_stats['candidates']} 标注={total_stats['annotated']} "
          f"y={total_stats['y']} n={total_stats['n']} r={total_stats['r']} s={total_stats['s']} "
          f"KB+{total_stats['kb_known']}已知/+{total_stats['kb_forbid']}禁止 "
          f"增长集+{total_stats['growth']}对")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="回填标注：遍历金标集文档，主动学习补全因果对写增长集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--doc", default=None, help="只跑指定文档（如 DOC-001）")
    parser.add_argument("--from", dest="from_doc", default=None, help="从指定文档开始")
    parser.add_argument("--top-k", type=int, default=20, help="每轮候选数量（默认20）")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="非交互批量模式：score≥0.70 且 不确定<0.30 → 自动 y；其余按阈值 n/s/r",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("回填标注工具 - 遍历金标集，主动学习写增长集")
    print("=" * 64)
    print(f"[配置] 候选数/轮: {args.top_k}  模式: {'批量自动' if args.batch else '人工交互(y/n/r/s)'}")
    print(f"[配置] 产出: data/agri_causal_events_growth.json")

    dataset = load_causal_events()
    num_docs = len(dataset.get("documents", []))
    print(f"[数据] 金标集: {num_docs} 篇文档")

    pipeline = AgriCausalPipeline()
    print(f"[模式] 知识库: {pipeline.kb.stats}")

    backfill(
        pipeline,
        dataset,
        only_doc=args.doc,
        from_doc=args.from_doc,
        top_k=args.top_k,
        batch=args.batch,
    )

    print(f"\n{'='*64}")
    print("全部完成。增长集待人工复核后并入金标集（可运行 merge_growth.py 合并）")
    print("=" * 64)


if __name__ == "__main__":
    main()
