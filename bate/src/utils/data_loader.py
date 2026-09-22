"""数据加载工具

加载本目录下的三类数据文件：
- agri_causal_events.json : 农业因果事件数据集（真实文献标注）
- agri_event_types.json   : 农业事件类型体系本体
- agri_synonym_dict.json  : 农业事件同义词典（概念聚合用）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 数据目录：本文件所在包的 ../../data/
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def get_data_dir() -> Path:
    """返回数据目录绝对路径。"""
    return _DATA_DIR


def load_causal_events(path: str | Path | None = None) -> dict[str, Any]:
    """加载农业因果事件数据集。

    Returns
    -------
    dict
        含 dataset_info / documents 两个顶层键。
    """
    p = Path(path) if path else _DATA_DIR / "agri_causal_events.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_event_types(path: str | Path | None = None) -> dict[str, Any]:
    """加载农业知识图谱本体（AKO-F v1.9）。

    Returns
    -------
    dict
        含 schema_info / entity_classes / causal_relations 三个顶层键。
    """
    p = Path(path) if path else _DATA_DIR / "agri_event_types.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_synonym_dict(path: str | Path | None = None) -> dict[str, Any]:
    """加载农业事件同义词典。

    Returns
    -------
    dict
        含 dict_info / concept_to_subtype / mention_to_concept 三个顶层键。
    """
    p = Path(path) if path else _DATA_DIR / "agri_synonym_dict.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def flatten_causal_pairs(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """将数据集中所有文档的 causal_pairs 展平为一个列表，便于评估。"""
    pairs: list[dict[str, Any]] = []
    for doc in dataset.get("documents", []):
        for pair in doc.get("causal_pairs", []):
            # 附加文档元信息
            pair_full = dict(pair)
            pair_full["doc_id"] = doc["doc_id"]
            pair_full["domain"] = doc.get("domain", "")
            pair_full["doc_source"] = doc.get("source", {})
            pairs.append(pair_full)
    return pairs


def flatten_causal_chains(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """将数据集中所有文档的 causal_chains 展平为一个列表。"""
    chains: list[dict[str, Any]] = []
    for doc in dataset.get("documents", []):
        for chain in doc.get("causal_chains", []):
            chain_full = dict(chain)
            chain_full["doc_id"] = doc["doc_id"]
            chain_full["domain"] = doc.get("domain", "")
            chains.append(chain_full)
    return chains


def get_all_concepts(event_types: dict[str, Any]) -> list[str]:
    """从本体中提取所有标准概念标签（支持多层嵌套子类）。"""
    concepts: list[str] = []

    def _collect(node: dict[str, Any]) -> None:
        concepts.extend(node.get("concepts", []))
        for sub in node.get("sub_classes", []):
            _collect(sub)

    for cls in event_types.get("entity_classes", []):
        _collect(cls)
    return concepts


def build_concept_index(event_types: dict[str, Any]) -> dict[str, str]:
    """构建 concept -> 'class_id.sub_class_id' 的索引（支持多层嵌套子类）。

    AKO-F v1.9：嵌套结构下，每个层级的 concepts 都按其所在 sub_class_id 索引。
    例如"降水增加"在 WeatherParam.PrecipitationParam 下，索引值为"EnvFactor.PrecipitationParam"。
    父类顶层 concepts（如"复合胁迫"在 WeatherParam 下）索引值为"EnvFactor.WeatherParam"。
    """
    index: dict[str, str] = {}

    def _collect(node: dict[str, Any], class_id: str) -> None:
        sub_id = node.get("sub_class_id")
        # 当前层级的 concepts 用当前 sub_class_id 索引
        if sub_id and "concepts" in node:
            for c in node["concepts"]:
                index[c] = f"{class_id}.{sub_id}"
        # 递归处理嵌套子类
        for sub in node.get("sub_classes", []):
            _collect(sub, class_id)

    for cls in event_types.get("entity_classes", []):
        cls_id = cls.get("class_id", "")
        # 无子类的实体类（如 Treatment）直接挂 concepts
        if "concepts" in cls and not cls.get("sub_classes"):
            for c in cls["concepts"]:
                index[c] = f"{cls_id}.{cls_id}"
        # 处理嵌套子类
        for sub in cls.get("sub_classes", []):
            _collect(sub, cls_id)
    return index
