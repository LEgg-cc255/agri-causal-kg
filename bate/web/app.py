"""农业因果知识图谱系统 — FastAPI Web 应用。

功能：
1. 因果事件抽取（研究内容1：论文文本 → 事件/因果对/因果链）
2. 因果图可视化（Neo4j 119节点/236边 vis.js 渲染）
3. 因果路径搜索（Step 2 双向检索 + 4维排序）
4. 可解释结果生成（Step 3 CausalRAG LLM 生成）
5. 端到端问答（Step 1→2→3 pipeline）

启动：
    python -m bate.web.app
    或
    uvicorn bate.web.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保项目根目录在 path 中
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.pipeline import AgriCausalPipeline
from bate.src.kg import AgriCausalGraph, CausalRAGPipeline, QuestionParser, EntityNode, CausalEdge, determine_edge_type
from bate.src.utils.knowledge_base import CausalKnowledgeBase
from bate.src.utils.active_learning import value_score

logger = logging.getLogger(__name__)

# 设置 Neo4j 密码默认值
if not os.environ.get("NEO4J_PASSWORD"):
    os.environ["NEO4J_PASSWORD"] = "123456"

# -------------------------------------------------------------------- #
#  全局单例（懒加载）
# -------------------------------------------------------------------- #
_extract_pipeline: AgriCausalPipeline | None = None
_graph: AgriCausalGraph | None = None
_rag_pipeline: CausalRAGPipeline | None = None
_parser: QuestionParser | None = None
_kb: CausalKnowledgeBase | None = None


def get_extract_pipeline() -> AgriCausalPipeline:
    global _extract_pipeline
    if _extract_pipeline is None:
        _extract_pipeline = AgriCausalPipeline()
    return _extract_pipeline


def get_graph() -> AgriCausalGraph:
    global _graph
    if _graph is None:
        _graph = AgriCausalGraph.from_env()
    return _graph


def get_rag_pipeline() -> CausalRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = CausalRAGPipeline.from_env()
    return _rag_pipeline


def get_parser() -> QuestionParser:
    global _parser
    if _parser is None:
        _parser = QuestionParser()
    return _parser


def get_kb() -> CausalKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = CausalKnowledgeBase()
    # 每次获取都重新加载（确保 confirm 写入后文件变更被读到）
    _kb._load()
    return _kb


# -------------------------------------------------------------------- #
#  请求模型
# -------------------------------------------------------------------- #
class ExtractRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    question: str
    direction: str = "auto"  # auto/backward/forward；auto 由 parser 决定
    max_hops: int = 4
    top_k: int = 5


class AnswerRequest(BaseModel):
    question: str
    direction: str = "auto"  # auto 由 pipeline 内部 parser 决定
    max_hops: int = 4
    top_k: int = 5


# -------------------------------------------------------------------- #
#  FastAPI 应用
# -------------------------------------------------------------------- #
app = FastAPI(title="农业因果知识图谱系统", version="1.0")

# 静态文件
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# -------------------------------------------------------------------- #
#  页面路由
# -------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    """主页面。"""
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>静态文件未找到，请检查 bate/web/static/ 目录</h1>")


# -------------------------------------------------------------------- #
#  API 路由
# -------------------------------------------------------------------- #
@app.get("/api/stats")
async def stats():
    """图谱统计信息。"""
    try:
        g = get_graph()
        info = g.test_connection()
        if not info["ok"]:
            return {"ok": False, "error": "Neo4j 连接失败"}
        with g._open_session() as s:
            node_count = s.run("MATCH (n:Entity) RETURN count(*) AS c").data()[0]["c"]
            edge_count = s.run(
                "MATCH ()-[r:TRIGGERS|PROPAGATES|ALLEVIATES]->() RETURN count(*) AS c"
            ).data()[0]["c"]
            forbid_count = s.run(
                "MATCH ()-[r:FORBIDDEN_CAUSES]->() RETURN count(*) AS c"
            ).data()[0]["c"]
            class_dist = s.run(
                "MATCH (n:Entity) WHERE n.entity_class IS NOT NULL "
                "RETURN n.entity_class AS ec, count(*) AS c ORDER BY c DESC"
            ).data()
        return {
            "ok": True,
            "neo4j_version": info.get("neo4j_version"),
            "nodes": node_count,
            "causal_edges": edge_count,
            "forbidden_edges": forbid_count,
            "class_distribution": {r["ec"]: r["c"] for r in class_dist},
        }
    except Exception as e:
        logger.exception("stats error")
        raise HTTPException(500, str(e))


@app.post("/api/extract")
async def extract(req: ExtractRequest):
    """因果事件抽取（研究内容1：1次LLM + 规则后处理）。"""
    try:
        if not req.text.strip():
            raise HTTPException(400, "文本不能为空")
        pipeline = get_extract_pipeline()
        result = pipeline.run(req.text)
        return result
    except Exception as e:
        logger.exception("extract error")
        raise HTTPException(500, str(e))


@app.get("/api/graph")
async def graph(limit: int = 200):
    """获取因果图数据（节点+边），供 vis.js 渲染。"""
    try:
        g = get_graph()
        with g._open_session() as s:
            nodes = s.run(
                "MATCH (n:Entity) "
                "RETURN n.concept AS id, n.concept AS label, "
                "       n.entity_class AS group, "
                "       n.sub_class_id AS subtype, "
                "       coalesce(n.sub_class_name, '') AS sub_class_name "
                f"LIMIT {limit}"
            ).data()
            edges = s.run(
                "MATCH (a)-[r:TRIGGERS|PROPAGATES|ALLEVIATES|INFLUENCE|AFFECTS]->(b) "
                "RETURN a.concept AS from, b.concept AS to, "
                "       r.confidence AS conf, r.evidence AS ev, "
                "       type(r) AS rel_type "
                f"LIMIT {limit * 2}"
            ).data()
        # vis.js 格式
        vis_nodes = []
        for n in nodes:
            sid = n.get("subtype", "") or ""
            sub_name = n.get("sub_class_name") or ""
            # 降级：如果 Neo4j 里没 sub_class_name，用前端兜底映射
            cls = n.get("group") or "未分类"
            # 用于分组着色：优先用 sub_class_name（子类），回退用 group（大类）
            color_group = sub_name if sub_name else cls
            vis_nodes.append({
                "id": n["id"],
                "label": n["label"],
                "group": cls,          # 大类（用于颜色映射）
                "sub_class_name": sub_name,  # 子类中文名（用于 tooltip + 细分颜色）
                "subtype": sid,        # 子类ID（英文）
                "title": (
                    f"概念: {n['label']}\n"
                    f"大类: {cls}\n"
                    f"子类: {sub_name or sid}"
                ),
            })
        vis_edges = [{
            "from": e["from"], "to": e["to"],
            "label": f"{e['conf']:.2f}" if e.get("conf") else "",
            "title": f"关系: {e.get('rel_type','')}\n置信度: {e.get('conf','?')}\n证据: {(e.get('ev') or '')[:80]}",
            "arrows": "to",
        } for e in edges]
        return {"nodes": vis_nodes, "edges": vis_edges}
    except Exception as e:
        logger.exception("graph error")
        raise HTTPException(500, str(e))


@app.post("/api/search")
async def search(req: SearchRequest):
    """因果路径搜索（Step 2 双向检索 + 4维排序）。"""
    try:
        parser = get_parser()
        parse = parser.parse(req.question)
        # 前端显式覆盖 direction（auto 时使用 parser 结果）
        if req.direction in ("backward", "forward"):
            parse.direction = req.direction
        if not parse.is_valid:
            return {
                "ok": False,
                "parse": parse.to_dict(),
                "paths": [],
                "message": "未能从问题中识别到图谱中的目标事件",
            }
        g = get_graph()
        crop_concept = parse.domain if parse.domain and parse.domain != "通用" else ""
        paths = g.search_paths(
            parse.target_concept, parse.direction,
            max_hops=req.max_hops, top_k=req.top_k,
            crop_concept=crop_concept,
        )
        return {
            "ok": True,
            "parse": parse.to_dict(),
            "paths": paths,
        }
    except Exception as e:
        logger.exception("search error")
        raise HTTPException(500, str(e))


@app.post("/api/answer")
async def answer(req: AnswerRequest):
    """端到端问答（Step 1→2→3 pipeline，仅 1 次 LLM 调用）。"""
    try:
        pipe = get_rag_pipeline()
        ans = pipe.answer(req.question, max_hops=req.max_hops, top_k=req.top_k)
        return {
            "ok": True,
            "question": ans.question,
            "parse": ans.parse.to_dict(),
            "paths": ans.paths,
            "explanation": ans.explanation,
            "elapsed_ms": ans.elapsed_ms,
            "step1_parse_ms": ans.step1_parse_ms,
            "step2_search_ms": ans.step2_search_ms,
            "step3_generate_ms": ans.step3_generate_ms,
        }
    except Exception as e:
        logger.exception("answer error")
        raise HTTPException(500, str(e))


@app.get("/api/sample-texts")
async def sample_texts():
    """示例农业文本。"""
    return {
        "samples": [
            {
                "title": "高温对水稻花粉活力的影响",
                "text": "在水稻抽穗扬花期，连续3天以上日最高温度超过35℃的高温胁迫会严重影响花粉活力，"
                        "导致花粉萌发率显著下降。花粉活力下降使得授粉受精过程受阻，"
                        "最终导致空壳率上升，产量大幅降低。",
            },
            {
                "title": "土壤过湿对根系的影响",
                "text": "土壤过湿会导致土壤中氧气扩散受阻，根系长期处于缺氧环境，"
                        "根系活力下降。根系缺氧进一步引起水分供需失衡，"
                        "植株出现萎蔫症状，严重时导致青枯。",
            },
        ]
    }


# -------------------------------------------------------------------- #
#  主动学习 API（与因果抽取面板联动）
# -------------------------------------------------------------------- #
class ActiveReviewRequest(BaseModel):
    """抽取后的因果对审核请求。"""
    causal_pairs: list[dict]  # 来自 extract 的 causal_pairs
    events: list[dict] = []  # 来自 extract 的 events（用于 Neo4j 节点属性）
    source_text: str = ""    # 原文（用于价值评分 evidence_strength）


class ActiveConfirmRequest(BaseModel):
    """确认写入知识库的审核结果。"""
    decisions: list[dict]  # [{index, cause_concept, effect_concept, decision: "y"/"n"/"r"/"s", confidence?}]


KB_STATUS_LABELS = {
    "KNOWN": "知识库已知（跳过）",
    "FORBIDDEN": "知识库禁止（跳过）",
    "NEW_UNKNOWN": "新增候选（等待确认）",
    "SELF_LOOP": "概念自环（自动过滤）",
    "LOW_CONF": "置信度过低（conf<0.3，自动过滤）",
    "CONFLICT": "极性冲突（负向→负向，需关注）",
}


@app.post("/api/active/review")
async def active_review(req: ActiveReviewRequest):
    """审核抽取的因果对：与知识库对比、分类、计算价值分数。

    每个 pair 返回状态：
    - KNOWN: 已知因果（已在KB中，无需确认）
    - FORBIDDEN: 禁止方向（需人工确认是否保留禁止）
    - NEW_UNKNOWN: KB未覆盖（需y/n/r/s确认）
    - SELF_LOOP: 概念自环（cause==effect，自动过滤）
    - LOW_CONF: conf<0.3（自动过滤）
    """
    kb = get_kb()
    reviewed: list[dict] = []

    for i, pair in enumerate(req.causal_pairs):
        cause = pair.get("cause", {})
        effect = pair.get("effect", {})
        cause_c = cause.get("concept", "") if isinstance(cause, dict) else str(cause)
        effect_c = effect.get("concept", "") if isinstance(effect, dict) else str(effect)
        cause_cls = cause.get("entity_class", "") or cause.get("event_class", "") if isinstance(cause, dict) else ""
        effect_cls = effect.get("entity_class", "") or effect.get("event_class", "") if isinstance(effect, dict) else ""
        cause_subtype = cause.get("sub_class_id", "") or cause.get("subtype_id", "") if isinstance(cause, dict) else ""
        effect_subtype = effect.get("sub_class_id", "") or effect.get("subtype_id", "") if isinstance(effect, dict) else ""
        cause_text = cause.get("text", "") if isinstance(cause, dict) else ""
        effect_text = effect.get("text", "") if isinstance(effect, dict) else ""
        conf = float(pair.get("confidence") or pair.get("conf") or 0.0)
        evidence = pair.get("evidence", "") or ""

        # 判定优先级：自环 > 低置信 > 禁止 > 已知 > 新候选
        status = "NEW_UNKNOWN"
        kb_info = pair.get("kb_match") or {}
        if not kb_info:
            kb_info = kb.lookup(cause_c, effect_c).__dict__

        if cause_c == effect_c:
            status = "SELF_LOOP"
        elif conf < 0.3:
            status = "LOW_CONF"
        elif kb_info.get("forbidden"):
            status = "FORBIDDEN"
        elif kb_info.get("known"):
            status = "KNOWN"

        # 价值评分（仅 NEW_UNKNOWN 需要）
        value = None
        if status == "NEW_UNKNOWN":
            pair_copy = {
                "cause": {"concept": cause_c, "entity_class": cause_cls},
                "effect": {"concept": effect_c, "entity_class": effect_cls},
                "confidence": conf,
                "kb_match": kb_info,
                "evidence": evidence,
            }
            value = value_score(pair_copy, kb, existing_concepts=set(), text=req.source_text)

        reviewed.append({
            "index": i,
            "cause_concept": cause_c,
            "effect_concept": effect_c,
            "cause_class": cause_cls,
            "effect_class": effect_cls,
            "cause_subtype": cause_subtype,
            "effect_subtype": effect_subtype,
            "cause_text": cause_text,
            "effect_text": effect_text,
            "confidence": conf,
            "evidence": evidence,
            "status": status,
            "status_label": KB_STATUS_LABELS.get(status, status),
            "kb_info": kb_info,
            "value_score": value,  # None 或 {total, uncertainty, kb_uncovered, domain_novelty, evidence_strength, reason}
            "decision": "s" if status in ("KNOWN", "SELF_LOOP", "LOW_CONF") else None,  # 默认决策：已知类自动skip
        })

    summary = {s: sum(1 for r in reviewed if r["status"] == s) for s in KB_STATUS_LABELS}
    return {
        "ok": True,
        "kb_stats": kb.stats,
        "summary": summary,
        "reviewed_pairs": reviewed,
    }


@app.post("/api/active/confirm")
async def active_confirm(req: ActiveConfirmRequest):
    """应用审核结果：更新知识库 JSON + 同步写入 Neo4j。

    decisions: [{index, cause_concept, effect_concept, decision, confidence?}]
      decision:
        "y" → yes：加入 known_causal_pairs + Neo4j CAUSES 边
        "n" → no：加入 forbidden_pairs + Neo4j FORBIDDEN_CAUSES 边
        "r" → reverse：原方向禁止，反方向 known
        "s" → skip：跳过（不做任何改动）
    """
    kb = get_kb()
    g = get_graph()
    annotations_known: list[tuple[str, str, float]] = []  # (c, e, strength)
    annotations_forbid: list[tuple[str, str, float]] = []
    event_nodes: list[EntityNode] = []
    seen_nodes: set[str] = set()

    for d in req.decisions:
        idx = d.get("index", -1)
        c = str(d.get("cause_concept", "")).strip()
        e = str(d.get("effect_concept", "")).strip()
        dec = str(d.get("decision", "s")).lower()
        strength = float(d.get("confidence") or 0.85)
        strength_forbid = 0.15
        if strength < 0.5:
            strength_forbid = strength

        if not c or not e or dec == "s":
            continue
        if c == e:
            continue  # 自环跳过

        # 获取节点分类信息
        cause_sub = d.get("cause_subtype", "") or ""
        effect_sub = d.get("effect_subtype", "") or ""
        cause_cls = cause_sub.split(".", 1)[0] if cause_sub and "." in cause_sub else ""
        effect_cls = effect_sub.split(".", 1)[0] if effect_sub and "." in effect_sub else ""

        if dec == "y":
            annotations_known.append((c, e, strength))
        elif dec == "n":
            annotations_forbid.append((c, e, strength_forbid))
        elif dec == "r":
            annotations_forbid.append((c, e, strength_forbid))
            annotations_known.append((e, c, strength))
        else:
            continue

    # ------------------------------------------------------------------ #
    # 1. 更新知识库 JSON
    # ------------------------------------------------------------------ #
    kb_before = (len(kb.known_pairs), len(kb.forbidden_pairs))
    added_known = 0
    added_forbid = 0
    for c, e, s in annotations_known:
        key = f"{c}|||{e}"
        if key not in kb.known_pairs:
            kb.known_pairs[key] = s
            added_known += 1
    for c, e, s in annotations_forbid:
        key = f"{c}|||{e}"
        if key not in kb.forbidden_pairs:
            kb.forbidden_pairs[key] = s
            added_forbid += 1

    # 持久化 JSON
    kb_path = kb.kb_path
    try:
        with open(kb_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data["known_causal_pairs"] = kb.known_pairs
    data["forbidden_pairs"] = kb.forbidden_pairs
    if "causal_paths" not in data:
        data["causal_paths"] = kb.causal_paths
    # 版本/时间戳
    meta = data.get("_meta", {})
    meta["last_active_learning"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["version"] = str(float(meta.get("version", "3.2")) + 0.01)[:4]
    data["_meta"] = meta
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    kb_after = (len(kb.known_pairs), len(kb.forbidden_pairs))

    # ------------------------------------------------------------------ #
    # 2. 同步写入 Neo4j
    # ------------------------------------------------------------------ #
    neo4j_edges = 0
    neo4j_forbid = 0
    # 收集节点属性（从 decisions 中提取）
    node_attrs: dict[str, dict[str, str]] = {}
    for d in req.decisions:
        for key in ("cause_", "effect_"):
            concept = d.get(f"{key}concept")
            sub = d.get(f"{key}subtype") or ""
            cls = sub.split(".", 1)[0] if sub and "." in sub else (d.get(f"{key}class") or "")
            if concept and concept not in node_attrs:
                node_attrs[concept] = {"entity_class": cls, "sub_class_id": sub}
    # 构造节点
    all_concepts: set[str] = set()
    for c, e, s in annotations_known:
        all_concepts.add(c); all_concepts.add(e)
    for c, e, s in annotations_forbid:
        all_concepts.add(c); all_concepts.add(e)
    for concept in all_concepts:
        attr = node_attrs.get(concept, {})
        if concept not in seen_nodes:
            seen_nodes.add(concept)
            event_nodes.append(EntityNode(
                concept=concept,
                entity_class=attr.get("entity_class", ""),
                sub_class_id=attr.get("sub_class_id", ""),
            ))

    # 注意：get_graph() 返回的是全局单例，不能用 with g（会关闭 driver）
    # upsert_events/upsert_edges/upsert_forbidden 内部已独立管理 session
    if event_nodes:
        g.upsert_events(event_nodes)
    # 已知边（自动判定边类型 TRIGGERS/PROPAGATES/ALLEVIATES）
    known_edges = []
    for c, e, s in annotations_known:
        c_sub = node_attrs.get(c, {}).get("sub_class_id", "")
        e_sub = node_attrs.get(e, {}).get("sub_class_id", "")
        c_cls = c_sub.split(".", 1)[0] if c_sub and "." in c_sub else ""
        e_cls = e_sub.split(".", 1)[0] if e_sub and "." in e_sub else ""
        etype = determine_edge_type(c_cls, e_cls)
        known_edges.append(CausalEdge(
            cause_concept=c, effect_concept=e,
            confidence=s, evidence="主动学习审核",
            edge_type=etype,
        ))
    if known_edges:
        neo4j_edges = g.upsert_edges(known_edges)
    # 禁止边（FORBIDDEN_CAUSES）
    if annotations_forbid:
        neo4j_forbid = g.upsert_forbidden(
            {f"{c}|||{e}": s for c, e, s in annotations_forbid}
        )

    return {
        "ok": True,
        "kb": {
            "before": {"known": kb_before[0], "forbidden": kb_before[1]},
            "after":  {"known": kb_after[0],  "forbidden": kb_after[1]},
            "added":  {"known": added_known,   "forbidden": added_forbid},
            "kb_path": str(kb_path),
        },
        "neo4j": {
            "nodes_upserted": len(event_nodes),
            "causes_edges": neo4j_edges,
            "forbidden_edges": neo4j_forbid,
        },
        "decisions_processed": len(req.decisions),
    }


# -------------------------------------------------------------------- #
#  启动
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bate.web.app:app", host="0.0.0.0", port=8000, reload=False)
