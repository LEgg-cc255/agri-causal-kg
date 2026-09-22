"""研究内容2 Step 3: 可解释生成模块（CausalRAG 核心，1 次 LLM 调用）。

设计依据（精简方案：1篇核心论文 CausalRAG + 规则后处理）：
- 将检索到的 Top-K 因果路径注入 LLM prompt，生成自然语言机制解释
- 整个推理 pipeline 仅此 1 次 LLM 调用（与研究内容1约束一致）
- 证据溯源：从路径边的 evidence 字段提取，标注到解释文本

输入：
  - 用户问题（str）
  - Top-K 因果路径（list[dict]，由 Step 2 search_paths 产出）
  - 问题解析结果（QuestionParse，可选，用于丰富上下文）

输出：
  - 自然语言机制解释（str，含证据溯源标注）

使用：
    gen = ExplanationGenerator()
    explanation = gen.generate(question="为什么番茄落花？", paths=top_k_paths)
"""
from __future__ import annotations

import logging
from typing import Any

from ..utils.llm_client import LLMClient
from .question_parser import QuestionParse

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
#  CausalRAG 系统 Prompt
# -------------------------------------------------------------------- #
SYSTEM_PROMPT_EXPLANATION = """你是农业因果推理专家。任务：基于因果知识图谱检索到的因果路径，生成自然语言的机制解释。

## 解释原则
1. 按因果链条逐步解释，说明每一步的生理生化或生态机制
2. 使用农业专业知识，但语言通俗易懂，避免堆砌术语
3. 在关键机制处标注证据来源（路径提供的 evidence 字段）；evidence 为空时省略标注
4. 优先解释 Top1 路径，其他路径作为补充（说明机制差异或并列原因）
5. 多条路径共享中间节点时合并解释，避免重复
6. 不编造图谱中未提供的因果关系；路径之外的内容仅作背景常识简述
7. 解释长度控制在 200-400 字，结构清晰

## 输出格式
直接输出自然语言解释段落，不要输出 JSON 或代码块。可使用「首先...其次...最终...」等顺序词。
证据溯源用括号标注，如「（证据：高温胁迫导致花粉活力下降）」。"""


# -------------------------------------------------------------------- #
#  可解释生成器
# -------------------------------------------------------------------- #
class ExplanationGenerator:
    """CausalRAG 核心：将 Top-K 因果路径注入 LLM prompt 生成机制解释。

    整个推理 pipeline 仅此 1 次 LLM 调用。

    Parameters
    ----------
    llm : LLMClient, optional
        LLM 客户端。未提供时使用默认配置（无 API key 时自动 Mock）。
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    # ---------------------------------------------------------------- #
    #  公开接口
    # ---------------------------------------------------------------- #
    def generate(
        self,
        question: str,
        paths: list[dict[str, Any]],
        question_parse: QuestionParse | None = None,
    ) -> str:
        """生成自然语言机制解释。

        Parameters
        ----------
        question : str
            用户原始问题。
        paths : list[dict]
            Top-K 因果路径（由 Step 2 ``search_paths`` 产出），每条含
            ``concepts`` / ``avg_confidence`` / ``score`` / ``evidences`` 等字段。
        question_parse : QuestionParse, optional
            Step 1 问题解析结果，用于在 prompt 中指明方向与领域。

        Returns
        -------
        str
            自然语言解释文本。无路径时返回 fallback 解释。
        """
        if not paths:
            return self._fallback_no_paths(question, question_parse)

        # Mock 模式：规则生成（避免调用真实 LLM）
        if self.llm.mock:
            return self._mock_explanation(question, paths, question_parse)

        # 真实 LLM：构建 CausalRAG prompt
        user_prompt = self._build_prompt(question, paths, question_parse)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EXPLANATION},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.chat(messages).strip()

    # ---------------------------------------------------------------- #
    #  CausalRAG Prompt 构建
    # ---------------------------------------------------------------- #
    def _build_prompt(
        self,
        question: str,
        paths: list[dict[str, Any]],
        question_parse: QuestionParse | None,
    ) -> str:
        """构建 CausalRAG prompt：注入 Top-K 因果路径 + 证据。"""
        lines: list[str] = []
        lines.append(f"用户问题: {question}")
        if question_parse is not None:
            direction_cn = "找原因（前因解释）" if question_parse.direction == "backward" else "找后果（结果预测）"
            domain_cn = question_parse.domain or "未指定"
            lines.append(f"解析方向: {direction_cn} | 目标事件: {question_parse.target_concept} | 领域: {domain_cn}")

        lines.append("")
        lines.append("检索到的因果路径（按推荐度排序，score 越高越可信）:")
        lines.append("")

        for i, p in enumerate(paths, 1):
            chain = " → ".join(p["concepts"])
            lines.append(
                f"路径{i}（score={p.get('score', '?')}, "
                f"平均置信度={p.get('avg_confidence', '?')}, "
                f"跳数={p.get('length', '?')}）:"
            )
            lines.append(f"  因果链: {chain}")
            evidences = p.get("evidences") or []
            if any(ev for ev in evidences):
                lines.append("  逐跳证据:")
                concepts = p["concepts"]
                for j, ev in enumerate(evidences):
                    if ev and j + 1 < len(concepts):
                        lines.append(f"    - {concepts[j]} → {concepts[j+1]}: {ev}")
            lines.append("")

        lines.append("请基于以上因果路径，生成机制解释。")
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    #  Fallback：无路径时的解释
    # ---------------------------------------------------------------- #
    def _fallback_no_paths(
        self,
        question: str,
        question_parse: QuestionParse | None,
    ) -> str:
        """无路径时的 fallback 解释。"""
        target = question_parse.target_concept if question_parse else "未知事件"
        direction = question_parse.direction if question_parse else "backward"
        if direction == "backward":
            return (
                f"关于「{question}」，因果知识图谱中未检索到「{target}」的直接前因路径。\n"
                f"可能原因：（1）「{target}」属于根本性原因事件（如环境胁迫源头），"
                f"图谱中无更上游的因果链；（2）图谱未覆盖该事件的相关文献。\n"
                f"建议：尝试换用「{target}会导致什么后果？」查询其下游影响。"
            )
        else:
            return (
                f"关于「{question}」，因果知识图谱中未检索到「{target}」的直接后果路径。\n"
                f"可能原因：「{target}」在图谱中作为叶节点存在，未记录其下游效应。\n"
                f"建议：尝试换用「为什么{target}？」查询其前因。"
            )

    # ---------------------------------------------------------------- #
    #  Mock 模式：规则生成（无 API key 时）
    # ---------------------------------------------------------------- #
    def _mock_explanation(
        self,
        question: str,
        paths: list[dict[str, Any]],
        question_parse: QuestionParse | None,
    ) -> str:
        """Mock 模式：基于规则拼接因果链文本，模拟 LLM 解释。"""
        top = paths[0]
        concepts = top["concepts"]
        evidences = top.get("evidences") or []
        avg_conf = top.get("avg_confidence", 0.0)

        # 构建逐步解释
        steps: list[str] = []
        for j in range(len(concepts) - 1):
            cause = concepts[j]
            effect = concepts[j + 1]
            ev = evidences[j] if j < len(evidences) and evidences[j] else ""
            if ev:
                steps.append(f"首先，{cause}会导致{effect}（证据：{ev}）")
            else:
                steps.append(f"其次，{cause}进一步引起{effect}")

        # 替换首词
        if steps:
            steps[0] = steps[0].replace("首先，", "首先，")
            if len(steps) > 1:
                steps[-1] = steps[-1].replace("其次，", "最终，")

        chain_text = " ".join(steps)
        target = question_parse.target_concept if question_parse else concepts[-1]
        direction = question_parse.direction if question_parse else "backward"

        header = f"[Mock 解释] 针对「{question}」(目标事件: {target}, 方向: {direction})"
        confidence_note = f"(平均置信度: {avg_conf}, 基于 {len(paths)} 条候选路径)"
        footer = (
            "\n\n注：当前为 Mock 模式输出，配置 LLM_API_KEY 后将获得更流畅的自然语言解释。"
        )
        return f"{header}\n{chain_text} {confidence_note}{footer}"
