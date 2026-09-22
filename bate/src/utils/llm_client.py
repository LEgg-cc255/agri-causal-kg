"""LLM 客户端封装

设计目标：
1. 支持 OpenAI 兼容 API（OpenAI / DeepSeek / 智谱 / 豆包等均可通过 base_url 配置）
2. 无 API key 时自动降级为 Mock 模式，保证代码可独立运行与测试
3. 统一接口：chat(messages) -> str，屏蔽不同SDK差异
4. 内置重试与超时，避免单次调用失败导致整个pipeline崩溃

精简原则：不引入 langchain 等重框架，直接用 openai SDK（最稳定、最通用）。
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的 LLM 客户端封装。

    Parameters
    ----------
    api_key : str, optional
        API 密钥。未提供时从环境变量 ``LLM_API_KEY`` 读取；仍未取到则进入 Mock 模式。
    base_url : str, optional
        OpenAI 兼容 API 的 base url。例如 DeepSeek: https://api.deepseek.com/v1
    model : str
        模型名称，默认 ``deepseek-chat``，可改为 ``gpt-4o-mini`` / ``glm-4-flash`` 等。
    mock : bool, optional
        强制使用 Mock 模式（用于离线测试）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "deepseek-chat",
        mock: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        # 优先级：显式参数 > 环境变量
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.model = os.environ.get("LLM_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Mock 模式判定：无 key 或显式指定 mock
        self.mock = mock or (not self.api_key)
        if self.mock:
            logger.warning(
                "LLMClient 运行在 Mock 模式（未配置 API key）。"
                "将返回基于规则的模拟结果，仅用于功能验证。"
                "配置环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 以启用真实 LLM。"
            )
        else:
            try:
                from openai import OpenAI  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "检测到 API key 但未安装 openai 包。请执行: pip install openai"
                ) from e

    # ------------------------------------------------------------------ #
    #  公开接口
    # ------------------------------------------------------------------ #
    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送对话请求，返回 assistant 文本。

        Parameters
        ----------
        messages : list of dict
            OpenAI chat messages 格式: [{"role": "system", "content": "..."}, ...]

        Returns
        -------
        str
            模型回复文本。
        """
        if self.mock:
            return self._mock_chat(messages)

        return self._real_chat(messages)

    def chat_json(self, messages: list[dict[str, str]]) -> Any:
        """发送对话请求并解析 JSON 返回。

        自动从 Markdown 代码块中提取 JSON，容错性强。
        """
        text = self.chat(messages)
        return self._parse_json(text)

    # ------------------------------------------------------------------ #
    #  真实 LLM 调用
    # ------------------------------------------------------------------ #
    def _real_chat(self, messages: list[dict[str, str]]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("LLM 调用失败 (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, e)
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # 线性退避

        raise RuntimeError(f"LLM 调用连续失败 {self.max_retries + 1} 次: {last_err}")

    # ------------------------------------------------------------------ #
    #  Mock 模式：基于规则的模拟输出
    # ------------------------------------------------------------------ #
    def _mock_chat(self, messages: list[dict[str, str]]) -> str:
        """Mock 模式：根据 system/user prompt 关键词返回结构化模拟结果。

        用于无 API 环境下的功能验证与单元测试，保证 pipeline 可运行。

        精简方案：仅支持「事件+因果对+因果链联合抽取」任务，
        不再模拟关系判别/链判别（精简方案已移除这两个 LLM 调用）。
        """
        user_content = ""
        for m in messages:
            if m["role"] == "user":
                user_content += m["content"] + "\n"

        # 仅模拟「事件+因果对+因果链联合抽取」任务
        if any(kw in user_content for kw in ("事件抽取", "causal_pairs", "因果对", "因果链", "抽取事件")):
            return self._mock_extraction(user_content)

        return json.dumps({"mock": True, "message": "Mock模式默认返回"}, ensure_ascii=False)

    def _mock_extraction(self, user_content: str) -> str:
        """模拟事件+因果对+因果链联合抽取：从输入文本中通过关键词匹配生成结果。

        精简方案：Mock 模式同时输出 events / causal_pairs / causal_chains，
        模拟 LLM 在单次调用中完成全部抽取的行为。
        """
        events: list[dict] = []
        pairs: list[dict] = []

        # 基于关键词的简化抽取（仅用于Mock演示）
        keyword_events = [
            ("高温", "高温胁迫", "CAUSE_EVENT"),
            ("干旱", "干旱胁迫", "CAUSE_EVENT"),
            ("干热风", "干热风", "CAUSE_EVENT"),
            ("花粉活力下降", "花粉活力下降", "INTERMEDIATE"),
            ("花粉稔性", "花粉活力下降", "INTERMEDIATE"),
            ("花药不开裂", "花药开裂受阻", "INTERMEDIATE"),
            ("花药开裂受制", "花药开裂受阻", "INTERMEDIATE"),
            ("授粉率降低", "授粉率下降", "INTERMEDIATE"),
            ("授粉失败", "授粉失败", "INTERMEDIATE"),
            ("落花", "落花落蕾", "EFFECT_EVENT"),
            ("空壳率", "空壳率上升", "EFFECT_EVENT"),
            ("千粒重下降", "千粒重下降", "EFFECT_EVENT"),
            ("千粒重降低", "千粒重下降", "EFFECT_EVENT"),
            ("减产", "产量下降", "EFFECT_EVENT"),
            ("着果率", "着果率下降", "EFFECT_EVENT"),
            ("光合作用", "光合作用下降", "INTERMEDIATE"),
            ("气孔关闭", "气孔关闭", "INTERMEDIATE"),
        ]
        seen = set()
        for kw, concept, etype in keyword_events:
            if kw in user_content and concept not in seen:
                events.append({"text": kw, "type": etype, "concept": concept})
                seen.add(concept)

        # 简化的因果对（基于常识规则）
        rules = [
            ("高温胁迫", "花粉活力下降"),
            ("高温胁迫", "花药开裂受阻"),
            ("花粉活力下降", "授粉率下降"),
            ("花药开裂受阻", "授粉失败"),
            ("授粉率下降", "落花落蕾"),
            ("授粉失败", "空壳率上升"),
            ("落花落蕾", "产量下降"),
            ("空壳率上升", "产量下降"),
            ("干热风", "光合作用下降"),
            ("光合作用下降", "千粒重下降"),
            ("千粒重下降", "产量下降"),
            ("干旱胁迫", "气孔关闭"),
            ("气孔关闭", "光合作用下降"),
        ]
        concept_set = {e["concept"] for e in events}
        for cause, effect in rules:
            if cause in concept_set and effect in concept_set:
                pairs.append({
                    "cause": {"concept": cause, "text": cause, "type": "CAUSE_EVENT"},
                    "effect": {"concept": effect, "text": effect, "type": "EFFECT_EVENT"},
                    "relation_type": "DIRECT",
                    "confidence": 0.85,
                    "evidence": f"{cause}导致{effect}",
                })

        # Mock 因果链：基于因果对构建简单链（CAUSE→INTERMEDIATE→EFFECT）
        chains = self._mock_build_chains(pairs, events)

        return json.dumps(
            {"events": events, "causal_pairs": pairs, "causal_chains": chains},
            ensure_ascii=False, indent=2,
        )

    @staticmethod
    def _mock_build_chains(
        pairs: list[dict], events: list[dict]
    ) -> list[dict]:
        """Mock 因果链构建：从因果对中合并简单的因果链。"""
        if not pairs:
            return []

        # 构建邻接表
        adj: dict[str, list[str]] = {}
        for p in pairs:
            c = p["cause"]["concept"]
            e = p["effect"]["concept"]
            adj.setdefault(c, []).append(e)

        # 事件类型索引
        type_map = {e["concept"]: e["type"] for e in events}

        # 从 CAUSE_EVENT 出发找路径到 EFFECT_EVENT
        start_nodes = [c for c, t in type_map.items() if t == "CAUSE_EVENT"]
        end_nodes = {c for c, t in type_map.items() if t == "EFFECT_EVENT"}

        chains: list[dict] = []
        visited_chains: set[tuple[str, ...]] = set()

        for start in start_nodes:
            # BFS 找路径（限制深度5）
            stack: list[tuple[str, list[str]]] = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) >= 5:
                    continue
                for neighbor in adj.get(node, []):
                    new_path = path + [neighbor]
                    if neighbor in end_nodes and len(new_path) >= 2:
                        key = tuple(new_path)
                        if key not in visited_chains:
                            visited_chains.add(key)
                            chains.append({
                                "concepts": new_path,
                                "events": new_path[:],  # Mock 用概念代替原文
                                "confidence": 0.8,
                            })
                    else:
                        stack.append((neighbor, new_path))

        return chains[:5]  # 限制返回5条

    # ------------------------------------------------------------------ #
    #  JSON 解析工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json(text: str) -> Any:
        """从模型回复中解析 JSON，兼容 markdown 代码块包裹与截断情况。"""
        # 1. 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 尝试从 ```json ... ``` 代码块中提取（含未闭合的截断情况）
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```?", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. 尝试提取第一个 { ... } 或 [ ... ] 块
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 4. 截断 JSON 容错：尝试补全缺失的闭合括号
        # 从 ```json 后提取内容，或直接取第一个 { 开始的内容
        bracket_match = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
        if not bracket_match:
            bracket_match = re.search(r"(\{.*)", text, re.DOTALL)
        if bracket_match:
            fragment = bracket_match.group(1).rstrip("`").rstrip()
            # 统计未闭合的括号
            opens = fragment.count("{") - fragment.count("}")
            opens_sq = fragment.count("[") - fragment.count("]")
            # 补全缺失的闭合符号
            if opens > 0 or opens_sq > 0:
                fragment = fragment + "]" * max(opens_sq, 0) + "}" * max(opens, 0)
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    pass

        logger.error("无法从 LLM 回复中解析 JSON。原始文本: %s", text[:500])
        raise ValueError(f"LLM 回复无法解析为 JSON。前200字符: {text[:200]}")
