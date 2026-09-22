"""研究内容2 Step 4: 端到端 CausalRAG Pipeline 封装。

将 Step1（问题解析）→ Step2（双向路径检索）→ Step3（可解释生成）封装为
统一接口 ``CausalRAGPipeline.answer(question)``，提供端到端农业因果问答能力。

设计依据（精简方案：1篇核心论文 CausalRAG + 规则后处理）：
- 整个 pipeline 仅 1 次 LLM 调用（Step 3 可解释生成）
- Step 1/2 为纯规则 + Neo4j Cypher，0 LLM 调用
- 依赖注入：可注入 AgriCausalGraph / QuestionParser / ExplanationGenerator

使用：
    pipe = CausalRAGPipeline.from_env()
    with pipe:
        ans = pipe.answer("为什么番茄落花？")
        print(ans.explanation)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any

from .neo4j_client import AgriCausalGraph
from .question_parser import QuestionParser, QuestionParse
from .explanation_generator import ExplanationGenerator


@dataclass
class CausalAnswer:
    """端到端问答结果。"""
    question: str
    parse: QuestionParse
    paths: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    elapsed_ms: int = 0
    # 阶段计时（ms）
    step1_parse_ms: int = 0
    step2_search_ms: int = 0
    step3_generate_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def brief(self) -> str:
        """返回简洁摘要（不含完整解释）。"""
        return (
            f"问题: {self.question}\n"
            f"  解析: concept={self.parse.target_concept!r}, direction={self.parse.direction}\n"
            f"  路径: {len(self.paths)} 条"
            + (f" (Top1 score={self.paths[0]['score']})" if self.paths else "")
            + f"\n  耗时: {self.elapsed_ms}ms (parse={self.step1_parse_ms}, "
            f"search={self.step2_search_ms}, generate={self.step3_generate_ms})\n"
            f"  解释长度: {len(self.explanation)} 字符"
        )


class CausalRAGPipeline:
    """端到端 CausalRAG Pipeline：问题 → 解析 → 检索 → 解释。

    整个 pipeline 仅 1 次 LLM 调用（Step 3）。

    Parameters
    ----------
    graph : AgriCausalGraph
        Neo4j 图谱客户端（需已建立连接）。
    parser : QuestionParser, optional
        问题解析器，未提供时使用默认配置。
    generator : ExplanationGenerator, optional
        解释生成器，未提供时使用默认配置（无 API key 时 Mock）。
    """

    def __init__(
        self,
        graph: AgriCausalGraph,
        parser: QuestionParser | None = None,
        generator: ExplanationGenerator | None = None,
    ) -> None:
        self.graph = graph
        self.parser = parser or QuestionParser()
        self.generator = generator or ExplanationGenerator()

    @classmethod
    def from_env(
        cls,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> "CausalRAGPipeline":
        """从环境变量便捷创建 pipeline。

        需要环境变量：
        - NEO4J_PASSWORD（必需，如 123456）
        - LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（可选，无则 Mock 模式）
        """
        graph = AgriCausalGraph.from_env(uri=uri, user=user, password=password)
        return cls(graph=graph)

    # ---------------------------------------------------------------- #
    #  上下文管理（透传 graph 的生命周期）
    # ---------------------------------------------------------------- #
    def __enter__(self) -> "CausalRAGPipeline":
        # AgriCausalGraph 的连接由 driver 管理，无需显式打开
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()

    # ---------------------------------------------------------------- #
    #  公开接口
    # ---------------------------------------------------------------- #
    def answer(
        self,
        question: str,
        max_hops: int = 4,
        top_k: int = 5,
    ) -> CausalAnswer:
        """端到端问答：问题 → 解析 → 检索 → 解释。

        Parameters
        ----------
        question : str
            用户自然语言问题。
        max_hops : int
            最大路径跳数（1-4）。
        top_k : int
            Top-K 路径数。

        Returns
        -------
        CausalAnswer
            含 parse / paths / explanation / 耗时等完整结果。
        """
        t_total = time.time()

        # Step 1: 问题解析（0 LLM）
        t0 = time.time()
        parse = self.parser.parse(question)
        step1_ms = int((time.time() - t0) * 1000)

        # Step 2: 双向路径检索（0 LLM，Neo4j Cypher）
        # 作物约束：domain 非"通用"时启用（避免混入其他作物的噪声），无结果时自动回退全图
        t0 = time.time()
        if parse.is_valid:
            crop_concept = parse.domain if parse.domain and parse.domain != "通用" else ""
            paths = self.graph.search_paths(
                parse.target_concept, parse.direction,
                max_hops=max_hops, top_k=top_k,
                crop_concept=crop_concept,
            )
        else:
            paths = []
        step2_ms = int((time.time() - t0) * 1000)

        # Step 3: 可解释生成（1 次 LLM 调用）
        t0 = time.time()
        explanation = self.generator.generate(question, paths, parse)
        step3_ms = int((time.time() - t0) * 1000)

        return CausalAnswer(
            question=question,
            parse=parse,
            paths=paths,
            explanation=explanation,
            elapsed_ms=int((time.time() - t_total) * 1000),
            step1_parse_ms=step1_ms,
            step2_search_ms=step2_ms,
            step3_generate_ms=step3_ms,
        )

    def batch_answer(
        self,
        questions: list[str],
        max_hops: int = 4,
        top_k: int = 5,
    ) -> list[CausalAnswer]:
        """批量问答。"""
        return [self.answer(q, max_hops=max_hops, top_k=top_k) for q in questions]
