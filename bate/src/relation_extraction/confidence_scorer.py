"""置信度评分模块（纯规则评分，3维）

各维度权重（和为1.0）：
    W_LLM       = 0.50  LLM 置信度（主导信号）
    W_ONTOLOGY  = 0.30  本体方向合理性（领域知识约束）
    W_EVIDENCE  = 0.20  证据片段可溯源性

输出 0.0-1.0 的综合置信度
"""

from __future__ import annotations

# 本体方向合理性（cause_class → effect_class 的合理组合，AKO-F v1.9）
_VALID_DIRECTIONS = {
    ("环境因子", "功能症状"),
    ("环境因子", "生物胁迫"),
    ("环境因子", "防治措施"),
    ("防治措施", "功能症状"),
    ("防治措施", "环境因子"),
    ("防治措施", "生物胁迫"),
    ("生物胁迫", "功能症状"),
    ("功能症状", "功能症状"),
}

_INVALID_DIRECTIONS = {
    ("功能症状", "环境因子"),
    ("功能症状", "防治措施"),
    ("功能症状", "生物胁迫"),
    ("生物胁迫", "环境因子"),
    ("生物胁迫", "防治措施"),
}


class ConfidenceScorer:
    """因果关系置信度评分器（3维评分）。

    各维度权重（和为1.0）：
        W_LLM       = 0.50  LLM 置信度（主导信号）
        W_ONTOLOGY  = 0.30  本体方向合理性（领域知识约束）
        W_EVIDENCE  = 0.20  证据片段可溯源性
    """

    W_LLM = 0.50
    W_ONTOLOGY = 0.30
    W_EVIDENCE = 0.20

    def score(
        self,
        is_causal: bool,
        evidence: str = "",
        llm_confidence: float = 0.5,
        context: str = "",
        cause_class: str = "",
        effect_class: str = "",
    ) -> float:
        """计算综合置信度。

        Parameters
        ----------
        is_causal : bool
            是否为因果关系（本体校验结果）。
        evidence : str
            证据片段。
        llm_confidence : float
            LLM 给出的置信度。
        context : str
            原文上下文（用于证据匹配校验）。
        cause_class, effect_class : str
            因果事件大类（用于本体方向校验）。

        Returns
        -------
        float
            0.0-1.0 的置信度分数。
        """
        if not is_causal:
            return 0.0

        # 1. LLM 置信度（主导信号）
        s_llm = max(0.0, min(1.0, llm_confidence))

        # 2. 本体方向合理性
        if cause_class and effect_class:
            if (cause_class, effect_class) in _VALID_DIRECTIONS:
                s_ontology = 1.0
            elif (cause_class, effect_class) in _INVALID_DIRECTIONS:
                s_ontology = 0.2
            else:
                s_ontology = 0.6  # 未知方向
        else:
            s_ontology = 0.6  # 缺失类别信息

        # 3. 证据匹配（证据非空且在上下文中可溯源）
        if evidence and context:
            s_evidence = 1.0 if evidence.strip() in context else 0.5
        elif evidence:
            s_evidence = 0.7  # 有证据但无上下文校验
        else:
            s_evidence = 0.3  # 无证据

        # 加权综合
        confidence = (
            self.W_LLM * s_llm
            + self.W_ONTOLOGY * s_ontology
            + self.W_EVIDENCE * s_evidence
        )
        return max(0.0, min(1.0, confidence))
