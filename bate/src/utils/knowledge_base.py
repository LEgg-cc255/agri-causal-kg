"""农业因果规则知识库模块

设计依据：
- 引入外部农业知识库增强因果关系确认，提升 concept 级别的因果判断精度
- 区别于通用知识图谱（如 COMET），本库仅收录农业领域有明确机制的因果规则，噪声低
- 知识库用于"增强"而非"替代"LLM 判断：
    已知因果 → 提升置信度
    已知禁止 → 降低置信度（过滤反向因果）
    未知      → 不影响置信度（避免伤害 Recall）

核心接口：
    CausalKnowledgeBase.lookup(cause_concept, effect_concept) → LookupResult
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 知识库文件默认路径
_DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "agri_causal_rules.json"


class LookupResult(dict):
    """知识库查询结果（dict 子类，便于类型提示）。

    Fields:
        known: bool        - 是否在已知因果规则中
        forbidden: bool    - 是否在禁止方向中
        strength: float    - 因果强度（0.0-1.0，已知因果>0.5，禁止<0.3，未知=0.5）
        path: list[str]    - 已知因果路径（如有）
        source: str        - 结果来源描述
    """


class CausalKnowledgeBase:
    """农业因果规则知识库。

    Parameters
    ----------
    kb_path : str or Path, optional
        知识库 JSON 文件路径。未提供时使用默认路径。
    """

    def __init__(self, kb_path: str | Path | None = None) -> None:
        self.kb_path = Path(kb_path) if kb_path else _DEFAULT_KB_PATH
        self.known_pairs: dict[str, float] = {}
        self.forbidden_pairs: dict[str, float] = {}
        self.causal_paths: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        """加载知识库 JSON 文件。"""
        if not self.kb_path.exists():
            logger.warning("知识库文件不存在: %s，知识库增强将被跳过", self.kb_path)
            return

        try:
            with open(self.kb_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("加载知识库失败: %s", e)
            return

        # 过滤 _comment 键
        self.known_pairs = {
            k: float(v) for k, v in data.get("known_causal_pairs", {}).items()
            if not k.startswith("_")
        }
        self.forbidden_pairs = {
            k: float(v) for k, v in data.get("forbidden_pairs", {}).items()
            if not k.startswith("_")
        }
        self.causal_paths = {
            k: list(v) for k, v in data.get("causal_paths", {}).items()
            if not k.startswith("_")
        }
        logger.info(
            "知识库加载完成: %d 条已知因果, %d 条禁止方向, %d 条因果路径",
            len(self.known_pairs), len(self.forbidden_pairs), len(self.causal_paths),
        )

    # ------------------------------------------------------------------ #
    #  查询接口
    # ------------------------------------------------------------------ #
    def lookup(self, cause_concept: str, effect_concept: str) -> LookupResult:
        """查询 concept 对的因果先验知识。

        Parameters
        ----------
        cause_concept : str
            原因事件概念标签。
        effect_concept : str
            结果事件概念标签。

        Returns
        -------
        LookupResult
            known: 是否已知因果
            forbidden: 是否禁止方向
            strength: 因果强度（已知>0.5，禁止<0.3，未知=0.5）
            path: 已知因果路径（如有）
            source: 结果来源描述
        """
        if not cause_concept or not effect_concept:
            return LookupResult(
                known=False, forbidden=False, strength=0.5,
                path=[], source="概念缺失",
            )

        key = f"{cause_concept}|||{effect_concept}"

        # 1. 检查是否在已知因果规则中
        if key in self.known_pairs:
            strength = self.known_pairs[key]
            # 查找因果路径
            path = self.causal_paths.get(key, [])
            return LookupResult(
                known=True, forbidden=False, strength=strength,
                path=path, source=f"已知因果规则(strength={strength})",
            )

        # 2. 检查是否在禁止方向中
        if key in self.forbidden_pairs:
            penalty = self.forbidden_pairs[key]
            return LookupResult(
                known=False, forbidden=True, strength=penalty,
                path=[], source=f"禁止方向(penalty={penalty})",
            )

        # 3. 未知（知识库未覆盖）
        return LookupResult(
            known=False, forbidden=False, strength=0.5,
            path=[], source="知识库未覆盖",
        )

    def lookup_path(self, start_concept: str, end_concept: str) -> list[str]:
        """查询两个概念之间的已知因果路径。

        Returns
        -------
        list[str]
            因果路径的概念序列，未找到则返回空列表。
        """
        key = f"{start_concept}|||{end_concept}"
        return self.causal_paths.get(key, [])

    # ------------------------------------------------------------------ #
    #  统计信息
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        """知识库是否已加载。"""
        return bool(self.known_pairs or self.forbidden_pairs)

    @property
    def stats(self) -> dict[str, int]:
        """知识库统计信息。"""
        return {
            "num_known_pairs": len(self.known_pairs),
            "num_forbidden_pairs": len(self.forbidden_pairs),
            "num_causal_paths": len(self.causal_paths),
        }
