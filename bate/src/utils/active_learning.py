"""基于LLM的主动学习模块（AL-LLM-Causal）

设计依据：
- 农业因果链标注稀缺（当前仅42对），需用最少人工成本扩充训练集
- 复用Pipeline已抽取结果，不额外调用LLM（符合1次LLM原则）
- LLM预标注 = pipeline的is_causal/confidence/reasoning，人工只需3级校验

核心流程：
    Pipeline抽取结果 → 价值评分 → 选Top-K → 格式化展示 → 人工校验 → 更新知识库+标注集

价值评分（4维）：
    0.35 × uncertainty      - LLM置信度接近0.5（模型不确定，标注价值高）
    0.30 × kb_uncovered     - 知识库未覆盖（扩展知识库覆盖的价值）
    0.20 × domain_novelty   - 新concept/新作物（多样性采样）
    0.15 × evidence_strength - 文本有明确因果词（证据可溯源性）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .knowledge_base import CausalKnowledgeBase

logger = logging.getLogger(__name__)

# 因果指示词（用于证据强度评分）
_CAUSAL_CUES = [
    "导致", "引起", "使得", "造成", "引发", "促使", "诱发", "造成",
    "影响", "抑制", "促进", "降低", "升高", "下降", "上升", "减少", "增加",
    "阻碍", "受阻", "障碍", "损伤", "受损", "失调", "失衡",
    "导致", "从而", "进而", "因此", "由于", "因为",
]

# 数据文件路径
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_KB_PATH = _DATA_DIR / "agri_causal_rules.json"
_DATASET_PATH = _DATA_DIR / "agri_causal_events.json"


# ---------------------------------------------------------------------- #
#  价值评分
# ---------------------------------------------------------------------- #
def value_score(
    pair: dict[str, Any],
    kb: CausalKnowledgeBase,
    existing_concepts: set[str] | None = None,
    text: str = "",
) -> dict[str, float]:
    """计算单个因果对的标注价值分数（0.0-1.0）。

    Parameters
    ----------
    pair : dict
        Pipeline输出的因果对（含cause/effect/confidence/kb_match/evidence）。
    kb : CausalKnowledgeBase
        知识库（用于判断是否已覆盖）。
    existing_concepts : set, optional
        当前标注集已覆盖的concept集合（用于多样性评分）。
    text : str
        原文上下文（用于证据强度评分）。

    Returns
    -------
    dict
        {total, uncertainty, kb_uncovered, domain_novelty, evidence_strength, reason}
    """
    existing_concepts = existing_concepts or set()
    cause = pair.get("cause", {})
    effect = pair.get("effect", {})
    cause_c = cause.get("concept", "")
    effect_c = effect.get("concept", "")
    llm_conf = pair.get("confidence", 0.5)
    kb_match = pair.get("kb_match", {})
    evidence = pair.get("evidence", "")

    # 1. 不确定性：LLM置信度越接近0.5，价值越高（0.3-0.7区间为不确定区）
    if 0.3 <= llm_conf <= 0.7:
        s_uncertainty = 1.0 - abs(llm_conf - 0.5) * 2  # 0.5时=1.0, 0.3/0.7时=0.6
    else:
        s_uncertainty = 0.2  # 高置信度样本价值低（模型已确定）

    # 2. 知识库未覆盖：不在known也不在forbidden的价值最高
    if kb_match.get("known"):
        s_kb = 0.1  # 已覆盖，扩展价值低
    elif kb_match.get("forbidden"):
        s_kb = 0.3  # 禁止方向，有一定确认价值
    else:
        s_kb = 1.0  # 未覆盖，扩展价值最高

    # 3. 领域多样性：新concept价值高
    new_concepts = 0
    if cause_c and cause_c not in existing_concepts:
        new_concepts += 1
    if effect_c and effect_c not in existing_concepts:
        new_concepts += 1
    s_novelty = new_concepts / 2.0  # 0/0.5/1.0

    # 4. 证据强度：文本中有因果指示词
    search_text = (evidence + " " + text).lower()
    cue_count = sum(1 for cue in _CAUSAL_CUES if cue in search_text)
    s_evidence = min(1.0, cue_count * 0.3)  # 1个词=0.3, 3个词=0.9

    # 加权综合
    total = (
        0.35 * s_uncertainty
        + 0.30 * s_kb
        + 0.20 * s_novelty
        + 0.15 * s_evidence
    )

    return {
        "total": round(total, 3),
        "uncertainty": round(s_uncertainty, 3),
        "kb_uncovered": round(s_kb, 3),
        "domain_novelty": round(s_novelty, 3),
        "evidence_strength": round(s_evidence, 3),
        "reason": _score_reason(s_uncertainty, s_kb, s_novelty, s_evidence),
    }


def _score_reason(s_u: float, s_k: float, s_n: float, s_e: float) -> str:
    """生成价值分数的文字说明。"""
    parts = []
    if s_u >= 0.6:
        parts.append("模型不确定")
    if s_k >= 0.8:
        parts.append("知识库未覆盖(高扩展价值)")
    elif s_k >= 0.3:
        parts.append("知识库部分覆盖")
    if s_n >= 0.5:
        parts.append("含新concept(多样性)")
    if s_e >= 0.6:
        parts.append("证据充分")
    elif s_e < 0.3:
        parts.append("证据不足")
    return "；".join(parts) if parts else "常规样本"


# ---------------------------------------------------------------------- #
#  候选选择
# ---------------------------------------------------------------------- #
def select_candidates(
    pairs: list[dict],
    kb: CausalKnowledgeBase,
    existing_concepts: set[str] | None = None,
    text: str = "",
    top_k: int = 20,
    min_score: float = 0.15,
) -> list[dict]:
    """从Pipeline抽取结果中选择Top-K高价值标注候选。

    Parameters
    ----------
    pairs : list[dict]
        Pipeline输出的因果对列表。
    kb : CausalKnowledgeBase
        知识库。
    existing_concepts : set, optional
        当前标注集已覆盖的concept集合。
    text : str
        原文上下文。
    top_k : int
        选取的候选数量上限。
    min_score : float
        最低价值分数阈值。

    Returns
    -------
    list[dict]
        候选样本列表（含价值分数），按total降序。
    """
    scored = []
    for i, pair in enumerate(pairs):
        score = value_score(pair, kb, existing_concepts, text)
        if score["total"] >= min_score:
            scored.append({
                "index": i,
                "pair": pair,
                "score": score,
            })

    scored.sort(key=lambda x: x["score"]["total"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------- #
#  格式化展示（供人工校验）
# ---------------------------------------------------------------------- #
def format_for_review(candidate: dict, doc_id: str = "") -> str:
    """格式化单个候选样本，供人工校验。

    输出格式：
        [DOC-001 #03] 价值=0.82 (模型不确定；知识库未覆盖；含新concept)
        原因: 高温胁迫 → 花粉活力下降
        LLM预标注: is_causal=True, confidence=0.45
        证据: 高温条件下花粉稔性低下
        知识库: 未覆盖
        请校验 [✓确认 / ✗否定 / ✎修改关系方向]:
    """
    pair = candidate["pair"]
    score = candidate["score"]
    cause = pair.get("cause", {})
    effect = pair.get("effect", {})
    kb_match = pair.get("kb_match", {})

    lines = [
        f"[{doc_id} #{candidate['index']:02d}] 价值={score['total']} ({score['reason']})",
        f"  原因: {cause.get('concept', '?')} ({cause.get('text', '')})",
        f"  结果: {effect.get('concept', '?')} ({effect.get('text', '')})",
        f"  LLM预标注: is_causal={pair.get('is_causal', '?')}, confidence={pair.get('confidence', '?')}",
        f"  证据: {pair.get('evidence', '无')}",
        f"  知识库: {kb_match.get('source', '未查询')}",
        f"  评分明细: 不确定={score['uncertainty']} KB未覆盖={score['kb_uncovered']} "
        f"多样性={score['domain_novelty']} 证据={score['evidence_strength']}",
        f"  请校验 [y=确认 / n=否定 / r=反向 / s=跳过]:",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
#  应用标注结果
# ---------------------------------------------------------------------- #
def apply_annotations(
    annotations: list[dict],
    kb: CausalKnowledgeBase,
    kb_path: str | Path | None = None,
) -> dict[str, int]:
    """应用人工校验结果，更新知识库。

    Parameters
    ----------
    annotations : list[dict]
        人工校验结果列表，每项含:
        {
          "cause_concept": str,
          "effect_concept": str,
          "label": "yes" | "no" | "reverse",  # 确认/否定/反向
          "strength": float,  # 可选，人工指定的因果强度
        }
    kb : CausalKnowledgeBase
        知识库实例（将被原地更新）。
    kb_path : optional
        知识库文件路径（用于持久化）。

    Returns
    -------
    dict
        更新统计 {added_known, added_forbidden, updated}
    """
    kb_path = Path(kb_path) if kb_path else _KB_PATH
    added_known = 0
    added_forbidden = 0

    for ann in annotations:
        c = ann["cause_concept"]
        e = ann["effect_concept"]
        label = ann["label"]
        strength = ann.get("strength", 0.85 if label == "yes" else 0.15)

        if label == "yes":
            key = f"{c}|||{e}"
            if key not in kb.known_pairs:
                kb.known_pairs[key] = strength
                added_known += 1
        elif label == "no":
            key = f"{c}|||{e}"
            if key not in kb.forbidden_pairs:
                kb.forbidden_pairs[key] = strength
                added_forbidden += 1
        elif label == "reverse":
            # 反向：原方向禁止，反方向已知
            key_forbid = f"{c}|||{e}"
            key_known = f"{e}|||{c}"
            if key_forbid not in kb.forbidden_pairs:
                kb.forbidden_pairs[key_forbid] = 0.15
                added_forbidden += 1
            if key_known not in kb.known_pairs:
                kb.known_pairs[key_known] = strength
                added_known += 1

    # 持久化到文件
    _save_kb(kb, kb_path)

    logger.info("知识库更新: +%d 已知, +%d 禁止", added_known, added_forbidden)
    return {"added_known": added_known, "added_forbidden": added_forbidden}


def _save_kb(kb: CausalKnowledgeBase, kb_path: Path) -> None:
    """将知识库写回JSON文件。"""
    # 读取原始文件保留元信息
    try:
        with open(kb_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data["known_causal_pairs"] = {
        k: v for k, v in kb.known_pairs.items() if not k.startswith("_")
    }
    data["forbidden_pairs"] = {
        k: v for k, v in kb.forbidden_pairs.items() if not k.startswith("_")
    }
    data["causal_paths"] = {
        k: v for k, v in kb.causal_paths.items() if not k.startswith("_")
    }

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------- #
#  写回增长集（与金标集分离，人工复核后再并入）
# ---------------------------------------------------------------------- #
_GROWTH_PATH = _DATA_DIR / "agri_causal_events_growth.json"


def apply_to_growth_set(
    annotations: list[dict],
    growth_path: str | Path | None = None,
    doc_meta: dict[str, Any] | None = None,
) -> dict[str, int]:
    """把人工确认的因果对写回增长集。

    仅写 label == "yes" 或 "reverse" 的对（reverse 写反转后方向）。
    写入 data/agri_causal_events_growth.json，与金标集同 schema，每条带
    source="active_learning" / verified=true，便于人工复核后并入金标集。

    Parameters
    ----------
    annotations : list[dict]
        人工校验结果（每项含 cause_concept/effect_concept/label，以及扩展的
        pair 原始引用与 doc_id）。
    growth_path : optional
        增长集文件路径，默认 data/agri_causal_events_growth.json。
    doc_meta : optional
        文档元信息 {domain, title, source, text}，首次出现该 doc_id 时用于
        填充文档条目；缺失字段用占位值。

    Returns
    -------
    dict
        {added_pairs, docs_touched}
    """
    growth_path = Path(growth_path) if growth_path else _GROWTH_PATH
    doc_meta = doc_meta or {}

    # 读取或初始化增长集
    try:
        with open(growth_path, encoding="utf-8") as f:
            growth = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        growth = {
            "dataset_info": {
                "name": "农业因果事件数据集 - 主动学习增长集 (ACED-Growth)",
                "version": "1.0",
                "description": "由主动学习模块产出、待人工复核后并入金标集的因果对。",
                "construction_method": "主动学习：Pipeline抽取 + 人工校验(y/n/r)。",
                "statistics": {
                    "num_documents": 0,
                    "num_causal_pairs": 0,
                    "num_causal_chains": 0,
                    "domains": [],
                },
            },
            "documents": [],
        }

    docs = growth.setdefault("documents", [])
    doc_index = {d["doc_id"]: d for d in docs if "doc_id" in d}

    added = 0
    docs_touched: set[str] = set()

    for ann in annotations:
        if ann.get("label") not in ("yes", "reverse"):
            continue
        pair = ann.get("pair")
        if not pair:
            continue

        doc_id = ann.get("doc_id", "") or doc_meta.get("doc_id", "") or "AL-UNKNOWN"
        cause = dict(pair.get("cause", {}))
        effect = dict(pair.get("effect", {}))
        # 反向：写反转后的方向
        if ann["label"] == "reverse":
            cause, effect = effect, cause

        c_text = (cause.get("text") or "").strip()
        e_text = (effect.get("text") or "").strip()
        c_concept = (cause.get("concept") or "").strip()
        e_concept = (effect.get("concept") or "").strip()
        if not c_concept or not e_concept:
            continue

        # 取或创建文档条目
        if doc_id not in doc_index:
            new_doc = {
                "doc_id": doc_id,
                "domain": doc_meta.get("domain", "未分类"),
                "title": doc_meta.get("title", f"主动学习标注-{doc_id}"),
                "source": doc_meta.get("source", {"type": "主动学习标注", "note": "由AL模块产出"}),
                "text": doc_meta.get("text", ""),
                "causal_pairs": [],
                "causal_chains": [],
            }
            docs.append(new_doc)
            doc_index[doc_id] = new_doc

        doc = doc_index[doc_id]
        existing = doc["causal_pairs"]

        # 去重：同 doc 下相同 concept 对 + 相同 evidence 不重复写
        ev = pair.get("evidence", "") or ""
        dup = any(
            p["cause_event"].get("concept") == c_concept
            and p["effect_event"].get("concept") == e_concept
            and p.get("evidence", "") == ev
            for p in existing
        )
        if dup:
            continue

        seq = len(existing) + 1
        pair_id = f"{doc_id}-G{seq:02d}"  # G 前缀标识增长集条目
        entry = {
            "pair_id": pair_id,
            "cause_event": {"text": c_text, "type": "CAUSE_EVENT", "concept": c_concept},
            "effect_event": {"text": e_text, "type": "EFFECT_EVENT", "concept": e_concept},
            "relation_type": "DIRECT",
            "confidence": round(float(pair.get("confidence", 0.85)), 3),
            "evidence": ev,
            "evidence_source": doc_id,
            "source": "active_learning",
            "verified": True,
        }
        existing.append(entry)
        added += 1
        docs_touched.add(doc_id)

    # 更新统计
    stats = growth["dataset_info"]["statistics"]
    stats["num_documents"] = len(docs)
    stats["num_causal_pairs"] = sum(len(d.get("causal_pairs", [])) for d in docs)
    stats["domains"] = sorted({d.get("domain", "未分类") for d in docs})

    with open(growth_path, "w", encoding="utf-8") as f:
        json.dump(growth, f, ensure_ascii=False, indent=2)

    logger.info("增长集更新: +%d 因果对, %d 文档", added, len(docs_touched))
    return {"added_pairs": added, "docs_touched": len(docs_touched)}


# ---------------------------------------------------------------------- #
#  主动学习主循环（一轮）
# ---------------------------------------------------------------------- #
def active_learning_round(
    pairs: list[dict],
    kb: CausalKnowledgeBase,
    existing_concepts: set[str] | None = None,
    text: str = "",
    doc_id: str = "",
    top_k: int = 20,
    interactive: bool = True,
    doc_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行一轮主动学习。

    标注结果双写：知识库（known/forbidden）+ 增长集（agri_causal_events_growth.json，
    仅 y/reverse）。增长集与金标集分离，人工复核后再并入。

    Parameters
    ----------
    pairs : list[dict]
        Pipeline抽取的因果对列表。
    kb : CausalKnowledgeBase
        知识库（会被更新）。
    existing_concepts : set[str], optional
        当前标注集已覆盖的concept。
    text : str
        原文上下文。
    doc_id : str
        文档ID（用于展示与写回增长集）。
    top_k : int
        候选数量上限。
    interactive : bool
        True=交互式人工校验；False=仅返回候选（供程序化处理）。
    doc_meta : dict, optional
        文档元信息 {domain, title, source, text}，首次写入增长集时填充文档条目。

    Returns
    -------
    dict
        {candidates, annotations, stats}
    """
    # 1. 选择候选
    candidates = select_candidates(pairs, kb, existing_concepts, text, top_k=top_k)

    if not interactive:
        return {"candidates": candidates, "annotations": [], "stats": {"num_candidates": len(candidates)}}

    # 2. 交互式校验
    annotations: list[dict] = []
    print(f"\n{'='*60}")
    print(f"主动学习 - {doc_id} ({len(candidates)} 个候选)")
    print(f"{'='*60}")

    for cand in candidates:
        print(format_for_review(cand, doc_id=doc_id))
        try:
            user_input = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input in ("y", "yes", "确认"):
            annotations.append({
                "cause_concept": cand["pair"]["cause"]["concept"],
                "effect_concept": cand["pair"]["effect"]["concept"],
                "label": "yes",
                "strength": cand["pair"].get("confidence", 0.85),
                "pair": cand["pair"],
                "doc_id": doc_id,
            })
            print("  → 已确认为因果关系\n")
        elif user_input in ("n", "no", "否定"):
            annotations.append({
                "cause_concept": cand["pair"]["cause"]["concept"],
                "effect_concept": cand["pair"]["effect"]["concept"],
                "label": "no",
                "pair": cand["pair"],
                "doc_id": doc_id,
            })
            print("  → 已标记为非因果\n")
        elif user_input in ("r", "reverse", "反向"):
            annotations.append({
                "cause_concept": cand["pair"]["cause"]["concept"],
                "effect_concept": cand["pair"]["effect"]["concept"],
                "label": "reverse",
                "pair": cand["pair"],
                "doc_id": doc_id,
            })
            print("  → 已标记为反向因果\n")
        else:
            print("  → 跳过\n")
            continue

    # 3. 应用标注，更新知识库
    stats = apply_annotations(annotations, kb)

    # 4. 写回增长集（仅 y/reverse，与金标集分离）
    growth_stats = apply_to_growth_set(annotations, doc_meta=doc_meta)

    return {
        "candidates": candidates,
        "annotations": annotations,
        "stats": {
            "num_candidates": len(candidates),
            "num_annotated": len(annotations),
            **stats,
            "growth_added": growth_stats["added_pairs"],
        },
    }
