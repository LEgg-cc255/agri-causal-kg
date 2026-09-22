"""研究内容2：把金标数据集 + 规则库导入 Neo4j 图谱（AKO-F v1.9 本体）。

用法：
    # 1) 先设置密码
    $env:NEO4J_PASSWORD='123456'
    # 2) 从数据集 + 规则库 导入（自动去重 concept 节点）
    python -m bate.scripts.import_to_neo4j
    # 3) 只做连通性测试，不写入
    python -m bate.scripts.import_to_neo4j --dry-run

产出：
  - Entity 节点：所有 concept 去重，带 entity_class/sub_class_id + 多 label
    如 :Entity:EnvFactor:WeatherParam
  - 因果边 3 种子关系（自动判定）：
    TRIGGERS    (EnvFactor → FunctionalSymptom)
    PROPAGATES  (Entity → Entity)
    ALLEVIATES  (Treatment → Symptom/EnvFactor/BioticStress, 抑制型)
  - FORBIDDEN_CAUSES 边：规则库 forbid 70
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.kg import AgriCausalGraph, CausalEdge, EntityNode, determine_edge_type
from bate.src.utils.data_loader import build_concept_index, load_event_types
from bate.src.event_extraction.concept_aggregator import ConceptAggregator

DATA = Path(__file__).resolve().parent.parent / "data"
EVENTS_PATH = DATA / "agri_causal_events.json"
RULES_PATH = DATA / "agri_causal_rules.json"

# 全局 ConceptAggregator
_AGG = ConceptAggregator()


def _get_entity_class(sub_class_id: str) -> str:
    """从 "EnvFactor.WeatherParam" 提取 "EnvFactor"。"""
    if "." in sub_class_id:
        return sub_class_id.split(".", 1)[0]
    return sub_class_id


# 效果概念中含这些关键词时，判定为缓解型（ALLEVIATES）
_ALLEVIATE_KEYWORDS = {"缓解", "减轻", "降低", "下降", "减少", "抑制", "改善", "提升", "增强"}


def _infer_polarity(effect_concept: str) -> str:
    """根据效果概念名推断极性。含缓解/减轻等词 → inhibit，否则 promote。"""
    for kw in _ALLEVIATE_KEYWORDS:
        if kw in effect_concept:
            return "inhibit"
    return "promote"


def _determine_edge_type(cause_concept: str, effect_concept: str, cause_sub: str, effect_sub: str) -> str:
    """判定边类型：用英文 class_id + 效果概念名推断极性。"""
    cause_cls = cause_sub.split(".", 1)[0] if cause_sub and "." in cause_sub else ""
    effect_cls = effect_sub.split(".", 1)[0] if effect_sub and "." in effect_sub else ""
    polarity = _infer_polarity(effect_concept)
    return determine_edge_type(cause_cls, effect_cls, polarity)


def collect_from_events_dataset() -> tuple[dict[str, EntityNode], list[CausalEdge]]:
    """从标注数据集取 (concept节点, 因果边)。"""
    ds = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    events_index: dict[str, EntityNode] = {}
    edges: list[CausalEdge] = []

    for doc in ds.get("documents", []):
        did = doc.get("doc_id", "")
        for pair in doc.get("causal_pairs", []):
            conf = float(pair.get("confidence", 0.85))
            rel_type = pair.get("relation_type", "DIRECT")
            evidence = pair.get("evidence", "")
            for side, is_cause in (("cause_event", True), ("effect_event", False)):
                ev = pair.get(side, {})
                concept = ev.get("concept", "") or ""
                if not concept:
                    continue
                # 用 ConceptAggregator 查新本体 sub_class_id / entity_class / sub_class_name
                ev_class = ev.get("entity_class", "") or ev.get("event_class", "")
                ev_sub = ev.get("sub_class_id", "") or ev.get("subtype_id", "")
                ev_sub_name = ev.get("sub_class_name", "")
                # 旧数据可能是旧格式，用 ConceptAggregator 重新查
                agg_r = _AGG.aggregate(concept, concept)
                if agg_r.get("sub_class_id") and agg_r["sub_class_id"] != "UNKNOWN":
                    ev_sub = agg_r["sub_class_id"]
                if agg_r.get("entity_class") and agg_r["entity_class"] != "UNKNOWN":
                    ev_class = agg_r["entity_class"]
                # 从 aggregator 的 subtype_meta 查 sub_class_name
                if not ev_sub_name and ev_sub and ev_sub != "UNKNOWN":
                    meta = _AGG.subtype_meta.get(ev_sub, {})
                    ev_sub_name = meta.get("sub_class_name", "")
                labels: list[str] = []
                if ev_sub and "." in ev_sub:
                    cls_id, sub_id = ev_sub.split(".", 1)
                    labels.extend([cls_id, sub_id])
                elif ev_class:
                    labels.append(ev_class)
                if concept not in events_index:
                    events_index[concept] = EntityNode(
                        concept=concept, entity_class=ev_class,
                        sub_class_id=ev_sub, sub_class_name=ev_sub_name, labels=labels,
                    )
                else:
                    old = events_index[concept]
                    if not old.entity_class and ev_class:
                        old.entity_class = ev_class
                    if not old.sub_class_id and ev_sub:
                        old.sub_class_id = ev_sub
                    if not old.sub_class_name and ev_sub_name:
                        old.sub_class_name = ev_sub_name
                    for lb in labels:
                        if lb and (not old.labels or lb not in old.labels):
                            old.labels = list(old.labels or []) + [lb]

            c_c = pair["cause_event"].get("concept", "")
            e_c = pair["effect_event"].get("concept", "")
            if not c_c or not e_c or c_c == e_c:
                continue
            # 自动判定边类型
            _cn = events_index.get(c_c, EntityNode(concept=c_c))
            _en = events_index.get(e_c, EntityNode(concept=e_c))
            etype = _determine_edge_type(c_c, e_c, _cn.sub_class_id, _en.sub_class_id)
            edges.append(CausalEdge(
                cause_concept=c_c, effect_concept=e_c,
                confidence=conf, relation_type=rel_type,
                evidence=evidence, evidence_source=did,
                edge_type=etype,
            ))
    return events_index, edges


def collect_from_rules(events_index: dict[str, EntityNode]) -> tuple[list[CausalEdge], dict[str, float]]:
    """从规则库取 known_pairs(补边) + forbidden_pairs。"""
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    known_edges: list[CausalEdge] = []
    for key, strength in rules.get("known_causal_pairs", {}).items():
        parts = key.split("|||")
        if len(parts) != 2:
            continue
        c, e = parts
        if not c or not e or c == e:
            continue
        # concept 节点登记
        for concept in (c, e):
            if concept not in events_index:
                ar = _AGG.aggregate(concept, concept)
                sub = ar.get("sub_class_id", "") or ""
                ec = ar.get("entity_class", "") or ""
                sub_name = ""
                if sub and sub != "UNKNOWN":
                    meta = _AGG.subtype_meta.get(sub, {})
                    sub_name = meta.get("sub_class_name", "")
                labels: list[str] = []
                if sub and "." in sub:
                    cls_id, sub_id = sub.split(".", 1)
                    labels.extend([cls_id, sub_id])
                elif ec:
                    labels.append(ec)
                events_index[concept] = EntityNode(
                    concept=concept, entity_class=ec, sub_class_id=sub,
                    sub_class_name=sub_name, labels=labels,
                )
        # 自动判定边类型
        _cn = events_index.get(c, EntityNode(concept=c))
        _en = events_index.get(e, EntityNode(concept=e))
        etype = _determine_edge_type(c, e, _cn.sub_class_id, _en.sub_class_id)
        known_edges.append(CausalEdge(
            cause_concept=c, effect_concept=e,
            confidence=float(strength), relation_type="KNOWN_RULE",
            evidence_source="agri_causal_rules.json",
            edge_type=etype,
        ))
    forbidden_pairs = dict(rules.get("forbidden_pairs", {}))
    return known_edges, forbidden_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只做连通性测试 + 打印统计，不写入图谱")
    ap.add_argument("--uri", default=None, help="NEO4J_URI")
    ap.add_argument("--user", default=None, help="NEO4J_USER")
    args = ap.parse_args()

    if not EVENTS_PATH.exists() or not RULES_PATH.exists():
        print("[FATAL] 数据文件缺失:", EVENTS_PATH, RULES_PATH)
        sys.exit(1)
    ds_events, ds_edges = collect_from_events_dataset()
    rl_edges, forbid_pairs = collect_from_rules(ds_events)

    # 边类型分布
    etype_counter: Counter = Counter()
    all_edges = ds_edges + rl_edges
    for e in all_edges:
        etype_counter[e.edge_type] += 1

    print("=" * 70)
    print("【研究内容2 KG 导入摘要 (AKO-F v1.9)】")
    print("=" * 70)
    print(f"  [数据集]  concept 节点:     {len(ds_events)}")
    print(f"  [数据集]  causal 边:       {len(ds_edges)}")
    print(f"  [规则库]  known 补边:      {len(rl_edges)}")
    print(f"  [规则库]  forbidden 边:    {len(forbid_pairs)}")
    print(f"  [边类型分布]")
    for et, cnt in etype_counter.most_common():
        print(f"    {et:<15} {cnt:>5}")

    # 节点大类分布
    cls_counter: Counter = Counter()
    for ev in ds_events.values():
        cls = ev.entity_class or _get_entity_class(ev.sub_class_id)
        if cls:
            cls_counter[cls] += 1
    print(f"  [节点大类分布]")
    for k, v in sorted(cls_counter.items(), key=lambda x: -x[1]):
        print(f"    {k:<25} {v:>5}")
    no_sub = sum(1 for ev in ds_events.values() if not ev.sub_class_id)
    print(f"  无 sub_class_id 节点: {no_sub}")

    if args.dry_run:
        print("\n[DRY-RUN] 不连接 Neo4j，仅校验数据构建完成。")
        return

    # 连接 + 连通性测试
    try:
        g = AgriCausalGraph.from_env(uri=args.uri, user=args.user)
    except ValueError as e:
        print(f"[FATAL] {e}")
        sys.exit(2)
    with g:
        info = g.test_connection()
        print(f"\n[连通性测试] {info}")
        if not info["ok"]:
            print("[FATAL] 连接失败。")
            sys.exit(3)

        # Schema
        schema = g.init_schema()
        print(f"\n[Schema初始化] 版本={schema['version']}, 新建/确认约束={len(schema['ddl_created'])}")

        # 写入节点
        n_nodes = g.upsert_events(list(ds_events.values()))
        print(f"[节点写入] MERGE {n_nodes} 个 Entity 节点")

        # 写入边
        n_ds = g.upsert_edges(ds_edges)
        print(f"[边写入-数据集] 追加 {n_ds} 因果边 (数据集 {len(ds_edges)} 对)")
        n_rl = g.upsert_edges(rl_edges)
        print(f"[边写入-规则库known] 追加 {n_rl} 因果边 (规则库 {len(rl_edges)} 对补)")
        n_fb = g.upsert_forbidden(forbid_pairs)
        print(f"[边写入-forbidden] 追加 {n_fb} FORBIDDEN_CAUSES 边 ({len(forbid_pairs)} 对)")

        # 最终统计
        s = g.stats()
        print("\n" + "=" * 70)
        print("【图谱统计 (After Import)】")
        print("=" * 70)
        for k, v in s.items():
            if k == "per_class_distribution":
                print(f"  {k}:")
                for row in v:
                    print(f"    {row.get('ec','(未知)'):<25s} count={row.get('c',0):>5}")
            else:
                print(f"  {k}: {v}")

        # 演示因果路径
        sample = "高温胁迫"
        paths = g.find_paths(start_concept=sample, max_hops=4, top_k=3)
        print(f"\n[Demo] 起节点={sample!r} top-3路径:")
        if paths:
            for i, p in enumerate(paths, 1):
                print(f"  P{i:02d} score={p['score']:.2f} avg_conf={p['avg_confidence']:.2f} len={p['length']}")
                print(f"       {' → '.join(p['concepts'])}")
                if p.get("rel_types"):
                    print(f"       边类型: {' → '.join(p['rel_types'])}")
        else:
            print("  (无路径 — start_concept 不在图谱或边不足)")

    print("\n[完成] 图谱可视化: http://localhost:7474")
    print("      查询示例: MATCH (n:Entity) RETURN count(n);")
    print("      路径示例: MATCH p=(:Entity{concept:'高温胁迫'})-[:TRIGGERS|PROPAGATES|ALLEVIATES*1..4]->() RETURN p LIMIT 20")


if __name__ == "__main__":
    main()
