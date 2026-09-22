"""模块1.3 因果链构建（纯图算法，无LLM判别）

设计依据（精简方案）：
- 移除 LLM 链判别（避免与模块1.1的LLM判断重复，引入噪声）
- 模块1.1 的 prompt 已让 LLM 直接输出 causal_chains，本模块优先采用
- 若 LLM 未输出 chains 或链条不完整，则用图算法 DFS 从因果对中合并出链
- 链置信度 = 路径上各边置信度的几何均值 × 长度惩罚

核心流程：
    若有 LLM 输出的 chains → 校验并直接采用
    否则：因果对集合 → 构建有向图 → DFS搜索CAUSE→EFFECT路径 → 链置信度计算
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# 因果链最大长度（节点数）
MAX_CHAIN_LENGTH = 6
# 最大返回链数（避免组合爆炸）
MAX_CHAINS_RETURNED = 10


class ChainBuilder:
    """因果链构建器（纯图算法 + LLM链优先）。

    Parameters
    ----------
    max_length : int
        因果链最大节点数。
    """

    def __init__(self, max_length: int = MAX_CHAIN_LENGTH) -> None:
        self.max_length = max_length

    # ------------------------------------------------------------------ #
    #  主接口
    # ------------------------------------------------------------------ #
    def build_chains(
        self,
        causal_pairs: list[dict[str, Any]],
        min_confidence: float = 0.5,
        llm_chains: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """从因果对集合构建因果链。

        Parameters
        ----------
        causal_pairs : list of dict
            已确认的因果关系对（含 cause/effect/concept/confidence/relation_type）。
        min_confidence : float
            参与建链的因果对最低置信度阈值。
        llm_chains : list of dict, optional
            模块1.1 LLM 直接输出的因果链（若有则优先采用并校验）。

        Returns
        -------
        list of dict
            [{"chain": [concept1, concept2, ...], "events": [text1, text2, ...],
              "confidence": float, "is_valid": bool, "reasoning": str}]
        """
        # 1. 过滤低置信度因果对
        valid_pairs = [
            p for p in causal_pairs
            if p.get("is_causal", False) and p.get("confidence", 0) >= min_confidence
        ]

        # 2. 若 LLM 已输出 chains，优先校验并采用
        if llm_chains:
            llm_validated = self._validate_llm_chains(llm_chains, valid_pairs)
            if llm_validated:
                return llm_validated[:MAX_CHAINS_RETURNED]

        # 3. LLM 未输出链或校验失败 → 用图算法从因果对构建
        if not valid_pairs:
            return []

        graph, node_meta = self._build_graph(valid_pairs)

        # 识别起点（CAUSE_EVENT）与终点（EFFECT_EVENT）节点
        start_nodes = {n for n, m in node_meta.items() if m["type"] == "CAUSE_EVENT"}
        end_nodes = {n for n, m in node_meta.items() if m["type"] == "EFFECT_EVENT"}

        # 若无明确起点，则用入度为0的节点
        if not start_nodes:
            in_degree = self._in_degree(graph)
            start_nodes = {n for n in graph if in_degree.get(n, 0) == 0}

        # DFS 搜索所有 CAUSE→EFFECT 路径
        candidate_chains: list[list[str]] = []
        for start in start_nodes:
            for path in self._dfs_paths(graph, start, end_nodes, self.max_length):
                if len(path) >= 2:  # 至少2个节点
                    candidate_chains.append(path)

        # 去重（同一概念序列只保留一条）
        seen = set()
        unique_chains: list[list[str]] = []
        for chain in candidate_chains:
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                unique_chains.append(chain)

        # 限制数量
        unique_chains = unique_chains[:MAX_CHAINS_RETURNED]

        # 4. 计算每条链的综合置信度（路径上边置信度的几何均值）
        results: list[dict[str, Any]] = []
        for chain in unique_chains:
            chain_conf = self._chain_confidence(chain, valid_pairs)
            events_text = [node_meta.get(c, {}).get("text", c) for c in chain]
            results.append({
                "chain": chain,
                "events": events_text,
                "confidence": round(chain_conf, 3),
                "is_valid": True,
                "reasoning": "图算法DFS合并构建",
            })

        # 排序：按置信度降序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  内部：LLM 输出链的校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_llm_chains(
        llm_chains: list[dict[str, Any]],
        valid_pairs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """校验 LLM 输出的因果链。

        校验规则：
        1. 链长度 ≥ 2
        2. 链上相邻概念对应在因果对集合中存在（或反向不存在）
        3. 用因果对的置信度重新计算链置信度（覆盖 LLM 自报的置信度）
        """
        # 构建 (cause_concept, effect_concept) -> confidence 索引
        edge_conf: dict[tuple[str, str], float] = {}
        for p in valid_pairs:
            c = p.get("cause", {}).get("concept", "")
            e = p.get("effect", {}).get("concept", "")
            if c and e:
                edge_conf[(c, e)] = p.get("confidence", 0.5)

        validated: list[dict[str, Any]] = []
        for lc in llm_chains:
            chain = lc.get("chain") or lc.get("concepts") or []
            if not isinstance(chain, list) or len(chain) < 2:
                continue
            chain = [str(c).strip() for c in chain if str(c).strip()]
            if len(chain) < 2:
                continue

            # 计算链置信度：路径上各边置信度的几何均值
            confs: list[float] = []
            supported_edges = 0
            for i in range(len(chain) - 1):
                conf = edge_conf.get((chain[i], chain[i + 1]))
                if conf is None:
                    conf = 0.4  # 缺失边的惩罚
                else:
                    supported_edges += 1
                confs.append(conf)

            if not confs:
                continue

            geo_mean = math.exp(sum(math.log(max(c, 0.01)) for c in confs) / len(confs))
            length_penalty = max(0.7, 1.0 - 0.05 * (len(chain) - 2))
            chain_conf = geo_mean * length_penalty

            # 边支持率：链上有多少边在因果对集合中得到支持
            edge_support = supported_edges / (len(chain) - 1)

            validated.append({
                "chain": chain,
                "events": lc.get("events", []) if isinstance(lc.get("events"), list) else [],
                "confidence": round(chain_conf, 3),
                "is_valid": True,
                "reasoning": f"LLM直接输出，边支持率{edge_support:.0%}",
            })

        # 按置信度降序
        validated.sort(key=lambda x: x["confidence"], reverse=True)
        return validated

    # ------------------------------------------------------------------ #
    #  内部：图构建与路径搜索
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_graph(pairs: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, dict]]:
        """构建有向图邻接表 + 节点元信息。

        Returns
        -------
        graph : dict
            {concept: [neighbor_concept, ...]}
        node_meta : dict
            {concept: {"type": CAUSE/EFFECT/INTERMEDIATE, "text": 原文}}
        """
        graph: dict[str, list[str]] = defaultdict(list)
        node_meta: dict[str, dict] = {}

        for p in pairs:
            cause = p.get("cause", {})
            effect = p.get("effect", {})
            c_concept = cause.get("concept", "")
            e_concept = effect.get("concept", "")
            if not c_concept or not e_concept or c_concept == "UNKNOWN" or e_concept == "UNKNOWN":
                continue
            # 避免自环
            if c_concept == e_concept:
                continue
            # 避免重复边
            if e_concept not in graph[c_concept]:
                graph[c_concept].append(e_concept)

            # 更新节点元信息
            if c_concept not in node_meta:
                node_meta[c_concept] = {"type": cause.get("type", "CAUSE_EVENT"), "text": cause.get("text", c_concept)}
            if e_concept not in node_meta:
                node_meta[e_concept] = {"type": effect.get("type", "EFFECT_EVENT"), "text": effect.get("text", e_concept)}

        return dict(graph), node_meta

    @staticmethod
    def _in_degree(graph: dict[str, list[str]]) -> dict[str, int]:
        """计算各节点入度。"""
        in_deg: dict[str, int] = defaultdict(int)
        for src, neighbors in graph.items():
            for dst in neighbors:
                in_deg[dst] += 1
        return dict(in_deg)

    def _dfs_paths(
        self,
        graph: dict[str, list[str]],
        start: str,
        end_nodes: set[str],
        max_depth: int,
    ) -> list[list[str]]:
        """DFS 搜索从 start 出发到 end_nodes 的所有路径。

        使用迭代式DFS避免递归深度限制。
        """
        paths: list[list[str]] = []
        # 栈元素: (当前节点, 路径, 访问集合)
        stack: list[tuple[str, list[str], set[str]]] = [(start, [start], {start})]

        while stack:
            node, path, visited = stack.pop()
            # 到达终点 → 记录路径
            if node in end_nodes and len(path) >= 2:
                paths.append(path[:])
                # 不continue，可能还有更长的路径（但通常终点不再扩展）
            # 深度限制
            if len(path) >= max_depth:
                continue
            # 扩展邻居
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, path + [neighbor], new_visited))

        return paths

    # ------------------------------------------------------------------ #
    #  内部：链置信度计算
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chain_confidence(chain: list[str], pairs: list[dict[str, Any]]) -> float:
        """计算因果链置信度 = 路径上各边置信度的几何均值 × 长度惩罚。"""
        # 构建 (cause_concept, effect_concept) -> confidence 索引
        edge_conf: dict[tuple[str, str], float] = {}
        for p in pairs:
            c = p.get("cause", {}).get("concept", "")
            e = p.get("effect", {}).get("concept", "")
            if c and e:
                edge_conf[(c, e)] = p.get("confidence", 0.5)

        confs: list[float] = []
        for i in range(len(chain) - 1):
            conf = edge_conf.get((chain[i], chain[i + 1]))
            if conf is None:
                conf = 0.4  # 缺失边的惩罚
            confs.append(conf)

        if not confs:
            return 0.5

        # 几何均值（对低置信度边更敏感）
        geo_mean = math.exp(sum(math.log(max(c, 0.01)) for c in confs) / len(confs))
        # 链越长置信度略降（避免过长链）
        length_penalty = max(0.7, 1.0 - 0.05 * (len(chain) - 2))
        return geo_mean * length_penalty
