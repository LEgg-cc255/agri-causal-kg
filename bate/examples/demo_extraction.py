"""演示脚本：农业因果事件抽取与因果关系识别

运行方式（项目根目录 = bate/ 的上一级）：

1. 自定义文本抽取（直接传入文本）：
    python -m bate.examples.demo_extraction "高温胁迫使得番茄花粉量减少，花药开裂受制，造成落花"

2. 交互式输入：
    python -m bate.examples.demo_extraction -i

3. 运行预设示例：
    python -m bate.examples.demo_extraction

环境变量配置（启用真实LLM）：
    set LLM_API_KEY=sk-xxxx
    set LLM_BASE_URL=https://api.deepseek.com/v1
    set LLM_MODEL=deepseek-chat

未配置 API key 时自动使用 Mock 模式（基于规则模拟，用于功能验证）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 确保能导入 bate 包
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.utils.llm_client import LLMClient
from bate.src.pipeline import AgriCausalPipeline


# 演示用农业文本（来自数据集中的真实文献段落）
DEMO_TEXTS = [
    "高温胁迫使得番茄花粉量减少，花药开裂受制，花粉管伸长受制，"
    "花粉活力降低，造成落花现象。夜间温度高会使番茄植株营养生长加速，"
    "造成植株徒长，更多营养分配到茎秆和叶片上，对花的养分供应不足，造成落花。",

    "抽穗扬花期日最高温度持续高于35℃时，高温使花药壁失水过快，"
    "导致花药无法正常开裂散粉，花粉粒活性急剧下降，甚至失去授精能力，"
    "出现空壳率大幅上升的现象，即'花而不实'。",

    "小麦灌浆期遇上干热风，叶片水分蒸发太快，光合作用跟不上，"
    "籽粒灌浆不足，千粒重下降，严重的能减产10%-20%。",
]


def print_result(result: dict) -> None:
    """格式化打印抽取结果。"""
    stats = result["stats"]
    print(f"\n[统计] 事件数={stats['num_events']}  "
          f"因果对数={stats['num_confirmed_pairs']}  "
          f"因果链数={stats['num_chains']}  "
          f"知识库召回={stats.get('num_recalled_pairs', 0)}  "
          f"平均置信度={stats['avg_confidence']:.3f}")

    # 打印事件
    print("\n[识别到的事件]")
    for ev in result["events"]:
        print(f"  - {ev['text']}  →  概念: {ev['concept']}  "
              f"类型: {ev['type']}  大类: {ev.get('entity_class', '?')}")

    # 打印因果对
    print("\n[因果关系对]")
    for p in result["causal_pairs"]:
        c, e = p["cause"], p["effect"]
        kb = p.get("kb_match", {})
        kb_tag = ""
        if kb.get("known"):
            kb_tag = " [KB已知]"
        elif kb.get("forbidden"):
            kb_tag = " [KB禁止]"
        print(f"  - {c['concept']} → {e['concept']}  "
              f"[{p['relation_type']}]  conf={p['confidence']:.2f}{kb_tag}")

    # 打印因果链
    print("\n[因果链]")
    if result["causal_chains"]:
        for chain in result["causal_chains"]:
            status = "✓有效" if chain["is_valid"] else "✗无效"
            print(f"  {status} [{' → '.join(chain['chain'])}]  "
                  f"conf={chain['confidence']:.2f}")
    else:
        print("  （未构建出因果链）")
    print()


def run_single(pipeline: AgriCausalPipeline, text: str) -> None:
    """对单条文本运行抽取并打印结果。"""
    print("-" * 70)
    print(f"文本: {text}")
    result = pipeline.run(text)
    print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="农业因果事件抽取与因果关系识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="?", default=None,
                        help="待抽取的农业文本（不传则运行预设示例）")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互式输入模式")
    args = parser.parse_args()

    print("=" * 70)
    print("农业因果事件抽取与因果关系识别")
    print("=" * 70)

    # 创建 Pipeline（自动检测 API key，无 key 则 Mock 模式）
    llm = LLMClient()
    mode = "Mock（规则模拟）" if llm.mock else f"真实LLM（{llm.model}）"
    print(f"\n[运行模式] {mode}")

    pipeline = AgriCausalPipeline(llm_client=llm)

    # 模式1：交互式输入
    if args.interactive:
        print("\n[交互模式] 输入农业文本进行因果抽取（输入 q 退出）")
        while True:
            try:
                text = input("\n请输入农业文本> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("q", "quit", "exit"):
                break
            run_single(pipeline, text)
        return

    # 模式2：命令行直接传入文本
    if args.text:
        run_single(pipeline, args.text)
        # 保存结果
        output_path = Path(__file__).parent / "demo_output.json"
        result = pipeline.run(args.text)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_path}")
        return

    # 模式3：运行预设示例
    print("\n[预设示例模式]\n")
    all_results = []
    for i, text in enumerate(DEMO_TEXTS, 1):
        print(f"【示例 {i}】")
        run_single(pipeline, text)
        all_results.append(pipeline.run(text))

    output_path = Path(__file__).parent / "demo_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
