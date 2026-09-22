"""主动学习脚本：交互式标注因果对，扩充农业知识库

运行方式：
    python -m bate.examples.run_active_learning
    python -m bate.examples.run_active_learning -i          # 交互式输入文本
    python -m bate.examples.run_active_learning "你的文本"   # 直接传入文本
    python -m bate.examples.run_active_learning --top-k 30   # 每轮标30个

核心流程：
    1. 输入农业文本
    2. Pipeline 抽取因果对
    3. 价值评分选 Top-K 高价值样本
    4. 人工逐个校验（y/n/r/s）
    5. 自动更新知识库 JSON 文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.pipeline import AgriCausalPipeline
from bate.src.utils.knowledge_base import CausalKnowledgeBase
from bate.src.utils.active_learning import active_learning_round


def collect_existing_concepts() -> set[str]:
    """收集已有标注集和知识库中的 concept 集合。"""
    existing = set()
    kb = CausalKnowledgeBase()
    for key in kb.known_pairs:
        if "|||" in key:
            c, e = key.split("|||", 1)
            existing.add(c)
            existing.add(e)
    for key in kb.forbidden_pairs:
        if "|||" in key:
            c, e = key.split("|||", 1)
            existing.add(c)
            existing.add(e)
    return existing


def run_single(pipeline: AgriCausalPipeline, text: str, top_k: int) -> None:
    """对单条文本运行主动学习。"""
    print("=" * 60)
    print(f"文本: {text[:100]}{'...' if len(text) > 100 else ''}")
    print("=" * 60)

    # 1. Pipeline 抽取
    result = pipeline.run(text)
    pairs = result["causal_pairs"]
    kb = pipeline.kb

    print(f"\nPipeline 抽取: {len(result['events'])} 事件, {len(pairs)} 因果对")
    print(f"知识库状态: {kb.stats}")

    if not pairs:
        print("没有抽取到因果对，跳过。")
        return

    # 2. 显示所有因果对的覆盖状态
    print(f"\n[因果对覆盖状态]")
    known_cnt = 0
    forbidden_cnt = 0
    uncovered_cnt = 0
    for i, p in enumerate(pairs):
        c = p["cause"]["concept"]
        e = p["effect"]["concept"]
        kb_match = p.get("kb_match", {})
        conf = p.get("confidence", 0)
        if kb_match.get("known"):
            status = "[已在知识库-已知因果]"
            known_cnt += 1
        elif kb_match.get("forbidden"):
            status = "[已在知识库-禁止方向]"
            forbidden_cnt += 1
        else:
            status = "[未覆盖-需标注]"
            uncovered_cnt += 1
        print(f"  {i+1}. {c} → {e}  conf={conf:.2f}  {status}")

    print(f"\n覆盖统计: 已知={known_cnt}  禁止={forbidden_cnt}  未覆盖={uncovered_cnt}")

    if uncovered_cnt == 0:
        print(f"\n[提示] 所有因果对都已在知识库中，无需标注。")
        print(f"[提示] 这些因果对价值低（模型已确定+知识库已覆盖），跳过主动学习。")
        print(f"[提示] 建议输入包含新现象/新机制的文本以扩充知识库。")
        return

    # 3. 收集已有 concept
    existing_concepts = collect_existing_concepts()

    # 4. 运行主动学习（交互式）
    stats = active_learning_round(
        pairs=pairs,
        kb=kb,
        existing_concepts=existing_concepts,
        text=text,
        doc_id="AL-FREETEXT",
        top_k=top_k,
        interactive=True,
        doc_meta={
            "doc_id": "AL-FREETEXT",
            "domain": "自由输入",
            "title": "自由输入文本",
            "text": text,
            "source": {"type": "主动学习-自由输入"},
        },
    )

    # 5. 显示本轮统计
    print(f"\n{'='*60}")
    print(f"本轮主动学习完成:")
    print(f"  候选数: {stats['stats']['num_candidates']}")
    print(f"  已标注: {stats['stats']['num_annotated']}")
    print(f"  新增已知因果: {stats['stats'].get('added_known', 0)}")
    print(f"  新增禁止方向: {stats['stats'].get('added_forbidden', 0)}")
    print(f"  写回增长集因果对: {stats['stats'].get('growth_added', 0)}")
    print(f"  知识库更新后: {kb.stats}")
    print(f"  增长集: data/agri_causal_events_growth.json（人工复核后并入金标集）")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="主动学习：交互式标注因果对，扩充农业知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="?", default=None,
                        help="农业文本（不传则进入交互模式）")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互式输入模式")
    parser.add_argument("--top-k", type=int, default=20,
                        help="每轮标注的候选数量（默认20）")
    args = parser.parse_args()

    print("=" * 60)
    print("农业因果知识 - 主动学习标注工具")
    print("=" * 60)
    print(f"\n[配置] 每轮标注候选数: {args.top_k}")
    print(f"[配置] 校验操作: y=确认 | n=否定 | r=反向 | s=跳过")

    # 初始化 Pipeline
    pipeline = AgriCausalPipeline()
    print(f"[模式] 已加载知识库: {pipeline.kb.stats}")

    # 模式1：交互式输入
    if args.interactive or (not args.text):
        print(f"\n[交互模式] 输入农业文本进行主动学习（输入 q 退出）")
        while True:
            try:
                text = input("\n请输入农业文本> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("q", "quit", "exit"):
                break
            run_single(pipeline, text, args.top_k)
        return

    # 模式2：直接传入文本
    run_single(pipeline, args.text, args.top_k)


if __name__ == "__main__":
    main()
