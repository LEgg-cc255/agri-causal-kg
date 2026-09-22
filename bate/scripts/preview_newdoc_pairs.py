"""面向新文档的 Pipeline 抽取预览：跑 DOC-xxx 输出所有 events/pairs 供人工决策。

运行：
    python -m bate.scripts.preview_newdoc_pairs DOC-017          # 单篇
    python -m bate.scripts.preview_newdoc_pairs DOC-017 DOC-018  # 多篇
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.pipeline import AgriCausalPipeline
from bate.src.utils.data_loader import load_causal_events


def main():
    show_raw = "--raw" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python -m bate.scripts.preview_newdoc_pairs [--raw] DOC-017 [DOC-018 ...]")
        print("   --raw : 显示 LLM 抽取结果（概念聚合前），用于排查 mention 精确键是否漏掉")
        sys.exit(2)
    targets = set(args)

    ds = load_causal_events()
    p = AgriCausalPipeline()

    for doc in ds["documents"]:
        did = doc["doc_id"]
        if did not in targets:
            continue
        print("\n" + "=" * 96)
        print(f"# {did}  {doc.get('domain')}  {doc.get('title')[:60]}")
        print(f"# text 字数: {len(doc.get('text', ''))}")
        print("=" * 96)
        print("正文：")
        for line in doc.get("text", "").split("。"):
            line = line.strip().strip("。")
            if not line:
                continue
            print("   ", line, "。")
        result = p.run(doc.get("text", ""))
        evts = result.get("events", [])
        pairs = result.get("causal_pairs", [])
        if show_raw:
            raw_ev = result.get("_debug_raw_events") or []
            raw_pairs = result.get("_debug_raw_pairs") or []
            if raw_ev:
                print(f"\n[RAW EVENTS] LLM 抽取（聚合前）共 {len(raw_ev)} 个事件")
                for i, ev in enumerate(raw_ev, 1):
                    t = ev.get("text", "")
                    c = ev.get("concept", "")
                    cls = ev.get("entity_class", "?")
                    typ = ev.get("type", "?")
                    print(f"  RE{i:02d} [{cls:5s}/{typ:14s}]  mention={t!r}   → raw_concept={c!r}")
            if raw_pairs:
                print(f"\n[RAW PAIRS] LLM 抽取（聚合前）共 {len(raw_pairs)} 因果对")
                for i, pair in enumerate(raw_pairs, 1):
                    c = pair.get("cause_event") or pair.get("cause") or {}
                    e = pair.get("effect_event") or pair.get("effect") or {}
                    print(f"  RP{i:02d} conf={pair.get('confidence', 0):.2f}")
                    print(f"     原因: {c.get('text', '')!r}  → raw_concept={c.get('concept', '')!r}")
                    print(f"     结果: {e.get('text', '')!r}  → raw_concept={e.get('concept', '')!r}")
            if not raw_ev and not raw_pairs:
                print("\n[RAW] 未启用调试，请先在 pipeline 中保存 _debug_raw_events/_debug_raw_pairs")
        print(f"\n[EVENTS] 抽 {len(evts)} 个事件")
        for i, ev in enumerate(evts, 1):
            t = ev.get("text", "")
            c = ev.get("concept", "")
            cls = ev.get("entity_class", "?")
            typ = ev.get("type", "?")
            print(f"  E{i:02d} [{cls:5s}/{typ:14s}]  mention={t!r}   → concept={c!r}")
        print(f"\n[PAIRS] 抽 {len(pairs)} 因果对")
        for i, pair in enumerate(pairs, 1):
            c = pair.get("cause") or pair.get("cause_event") or {}
            e = pair.get("effect") or pair.get("effect_event") or {}
            ct = c.get("text", "")
            cc = c.get("concept", "")
            et = e.get("text", "")
            ec = e.get("concept", "")
            conf = pair.get("confidence", 0)
            ev = pair.get("evidence", "")
            kb = pair.get("kb_match", {}) or {}
            kb_flag = (
                "KNOWN" if kb.get("known") else
                "FORBID" if kb.get("forbidden") else
                "NEW"
            )
            print(f"  P{i:02d} 【{kb_flag}】 conf={conf:.2f}")
            print(f"     原因: {ct!r}  → concept={cc!r}")
            print(f"     结果: {et!r}  → concept={ec!r}")
            print(f"     证据: {ev!r}")


if __name__ == "__main__":
    main()
