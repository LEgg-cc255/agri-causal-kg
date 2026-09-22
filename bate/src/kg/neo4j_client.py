"""研究内容2：农业因果知识图谱 Neo4j 客户端模块（AKO-F v1.9 本体）。

设计要点：
1. 连接信息从环境变量读取，不硬编码：NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
2. 先识别 Neo4j 版本，再按版本选约束 DDL
3. 批量写入用 UNWIND 参数化 + execute_write 事务
4. 节点多 label：:Entity + :{class_id} + :{sub_class_id}（如 :Entity:EnvFactor:WeatherParam）
5. 因果边 3 种子关系：TRIGGERS / PROPAGATES / ALLEVIATES（替代旧单一 CAUSES）
6. Cypher 路径查询匹配 3 种子关系：[:TRIGGERS|PROPAGATES|ALLEVIATES]
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD_ENV_NAME = "NEO4J_PASSWORD"

# -------------------------------------------------------------------- #
#  Schema (AKO-F v1.9)
# -------------------------------------------------------------------- #
LABEL_ENTITY = "Entity"
LABEL_CROP = "Crop"           # 作物主体节点（项目组 f:Crop）

# 3 种因果子关系（项目组本体已定义 triggers/propagates/alleviates，暂保留大写不改）
REL_TRIGGERS = "TRIGGERS"      # EnvFactor → FunctionalSymptom
REL_PROPAGATES = "PROPAGATES"  # Entity → Entity（通用传导）
REL_ALLEVIATES = "ALLEVIATES"  # Treatment → Symptom/EnvFactor/BioticStress（缓解）
REL_INFLUENCE = "INFLUENCE"    # EnvFactor → Crop（项目组 f:influence，作物做尾节点）
REL_AFFECTS = "AFFECTS"        # BioticStress → Crop（项目组 f:affects，病害/虫害→作物）
REL_FORBIDDEN = "FORBIDDEN_CAUSES"

# 路径查询时匹配的所有因果边类型（不含 INFLUENCE/AFFECTS，这俩是作物关联边不是因果传导）
ALL_CAUSAL_RELS = f"{REL_TRIGGERS}|{REL_PROPAGATES}|{REL_ALLEVIATES}"
# 包含作物的查询用 ALL_CAUSAL_RELS_WITH_CROP（如需查"环境→作物"或"病害→作物"链）
ALL_CAUSAL_RELS_WITH_CROP = f"{ALL_CAUSAL_RELS}|{REL_INFLUENCE}|{REL_AFFECTS}"


@dataclass
class EntityNode:
    """实体节点（AKO-F v1.9）。

    labels 示例：["Entity", "EnvFactor", "WeatherParam"]
    注意：作物主体通过 Crop 节点 + INFLUENCE 边表达（项目组 f:influence），
    不在症状节点上加 crops 属性，避免破坏症状节点的通用复用性
    （同一症状对不同作物可有不同表现）。
    """
    concept: str
    entity_class: str = ""        # 大类中文名，如 "环境因子"
    sub_class_id: str = ""        # "class_id.sub_class_id"，如 "EnvFactor.WeatherParam"
    sub_class_name: str = ""      # 子类中文名，如 "天气"
    labels: list[str] | None = None  # 额外标签（如 "EnvFactor", "WeatherParam"）

    @property
    def all_labels(self) -> list[str]:
        ls = [LABEL_ENTITY]
        extra = list(self.labels or [])
        # 从 sub_class_id 拆出 class_id 和 sub_class_id 作为 label
        if self.sub_class_id and "." in self.sub_class_id:
            cls_id, sub_id = self.sub_class_id.split(".", 1)
            if cls_id and cls_id not in extra:
                extra.append(cls_id)
            if sub_id and sub_id != cls_id and sub_id not in extra:
                extra.append(sub_id)
        elif self.entity_class and self.entity_class not in extra:
            extra.append(self.entity_class)
        for e in extra:
            if e and e not in ls:
                ls.append(e)
        return ls

    def to_crop_node(self) -> "EntityNode":
        """转换为 Crop 节点（用于建作物节点）。"""
        return EntityNode(
            concept=self.concept,
            entity_class="Crop",
            sub_class_id="Crop.Crop",
            sub_class_name=self.sub_class_name or "作物",
            labels=[LABEL_CROP],
        )


@dataclass
class CausalEdge:
    """因果边（3 种子关系之一）。"""
    cause_concept: str
    effect_concept: str
    confidence: float = 0.85
    relation_type: str = "DIRECT"  # DIRECT/INDIRECT/CONDITIONAL/KNOWN_RULE
    evidence: str = ""
    evidence_source: str = ""
    temporal_order: int | None = None
    # 新本体：edge_type = TRIGGERS / PROPAGATES / ALLEVIATES
    edge_type: str = REL_PROPAGATES


def determine_edge_type(cause_class: str, effect_class: str, polarity: str = "promote") -> str:
    """根据起点/终点实体类 + 极性，自动判定因果子关系。

    - EnvFactor → FunctionalSymptom = TRIGGERS
    - Treatment + inhibit → ALLEVIATES
    - 其余 = PROPAGATES
    """
    if cause_class == "EnvFactor" and effect_class == "FunctionalSymptom":
        return REL_TRIGGERS
    if cause_class == "Treatment" and polarity == "inhibit":
        return REL_ALLEVIATES
    return REL_PROPAGATES


# ====================================================================== #
#  Neo4j 客户端封装
# ====================================================================== #
class AgriCausalGraph:
    """农业因果知识图谱 Neo4j 客户端（AKO-F v1.9）。

    使用：
        g = AgriCausalGraph.from_env()
        g.init_schema()
        g.upsert_events([EntityNode(concept="高温胁迫", entity_class="EnvFactor",
                                    sub_class_id="EnvFactor.WeatherParam")])
        g.upsert_edges([CausalEdge(cause_concept="高温胁迫", effect_concept="光合作用下降",
                                   confidence=0.90, edge_type="TRIGGERS")])
        print(g.stats())
        g.close()
    """

    def __init__(self, driver: Driver) -> None:
        self.driver: Driver = driver
        self._neo4j_version: tuple[int, ...] | None = None

    # ----------------------------------------------------------------- #
    #  构造
    # ----------------------------------------------------------------- #
    @classmethod
    def from_env(cls, uri: str | None = None, user: str | None = None, password: str | None = None):
        uri = uri or os.environ.get("NEO4J_URI", DEFAULT_URI)
        user = user or os.environ.get("NEO4J_USER", DEFAULT_USER)
        password = password or os.environ.get(DEFAULT_PASSWORD_ENV_NAME, "")
        if not password:
            raise ValueError(
                f"缺少 Neo4j 密码。请设置环境变量 {DEFAULT_PASSWORD_ENV_NAME}=<密码>，"
                "或启动 Docker 时用 -e NEO4J_AUTH=neo4j/<密码>"
            )
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return cls(driver)

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass

    def __enter__(self) -> "AgriCausalGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----------------------------------------------------------------- #
    #  会话工厂（版本兼容：3.x Bolt3.0 不支持 database 参数）
    # ----------------------------------------------------------------- #
    def _open_session(self):
        self.detect_version()
        if self._neo4j_version and self._neo4j_version[0] >= 4:
            return self.driver.session(database="neo4j")
        return self.driver.session()

    # ----------------------------------------------------------------- #
    #  版本探测 + 约束DDL
    # ----------------------------------------------------------------- #
    def detect_version(self) -> tuple[int, ...]:
        if self._neo4j_version is not None:
            return self._neo4j_version
        with self.driver.session() as sess:
            row = sess.run("CALL dbms.components() YIELD name, versions, edition "
                           "WHERE name = 'Neo4j Kernel' RETURN versions[0] AS v").single()
            v_str = str(row["v"])
        parts = []
        for x in v_str.split("."):
            try:
                parts.append(int(x))
            except ValueError:
                break
        self._neo4j_version = tuple(parts) if parts else (5, 0, 0)
        logger.info("Neo4j 版本探测: %s -> %s", v_str, self._neo4j_version)
        return self._neo4j_version

    def _constraint_ddl(self, major: int) -> list[str]:
        if major >= 5:
            return [
                f"CREATE CONSTRAINT entity_concept_unique IF NOT EXISTS "
                f"FOR (n:{LABEL_ENTITY}) REQUIRE n.concept IS UNIQUE",
            ]
        if major == 4:
            return [
                f"CREATE CONSTRAINT entity_concept_unique IF NOT EXISTS "
                f"ON (n:{LABEL_ENTITY}) ASSERT n.concept IS UNIQUE",
            ]
        # 3.x
        return [
            f"CREATE CONSTRAINT ON (n:{LABEL_ENTITY}) ASSERT n.concept IS UNIQUE",
        ]

    def _index_ddl(self, major: int) -> list[str]:
        if major >= 4:
            return [
                f"CREATE INDEX entity_class_idx IF NOT EXISTS "
                f"FOR (n:{LABEL_ENTITY}) ON (n.entity_class)",
                f"CREATE INDEX entity_subclass_idx IF NOT EXISTS "
                f"FOR (n:{LABEL_ENTITY}) ON (n.sub_class_id)",
            ]
        # 3.x
        return [
            f"CREATE INDEX ON :{LABEL_ENTITY}(entity_class)",
        ]

    def init_schema(self) -> dict[str, Any]:
        major = self.detect_version()[0]
        created = []
        with self._open_session() as sess:
            for ddl in self._constraint_ddl(major):
                try:
                    sess.run(ddl)
                    created.append(ddl)
                    logger.info("约束创建成功: %s", ddl)
                except Exception as e:
                    logger.info("约束DDL已存在或版本不支持，跳过: %s (%s)", ddl, e)
            for ddl in self._index_ddl(major):
                try:
                    sess.run(ddl)
                    created.append(ddl)
                except Exception as e:
                    logger.info("索引DDL跳过: %s (%s)", ddl, e)
        return {"version": self._neo4j_version, "ddl_created": created}

    # ----------------------------------------------------------------- #
    #  批量写入
    # ----------------------------------------------------------------- #
    def upsert_events(self, events: Iterable[EntityNode], batch_size: int = 200) -> int:
        events = list(events)
        total = 0
        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]
            rows = [
                {
                    "concept": ev.concept,
                    "entity_class": ev.entity_class or "",
                    "sub_class_id": ev.sub_class_id or "",
                    "sub_class_name": ev.sub_class_name or "",
                }
                for ev in batch
            ]
            with self._open_session() as sess:
                def _upsert(tx):
                    q = (
                        "UNWIND $rows AS r "
                        f"MERGE (n:{LABEL_ENTITY} {{concept: r.concept}}) "
                        "SET n.entity_class = CASE WHEN coalesce(r.entity_class,'') = '' "
                        "                        THEN n.entity_class ELSE r.entity_class END, "
                        "    n.sub_class_id = CASE WHEN coalesce(r.sub_class_id,'') = '' "
                        "                        THEN n.sub_class_id ELSE r.sub_class_id END, "
                        "    n.sub_class_name = CASE WHEN coalesce(r.sub_class_name,'') = '' "
                        "                          THEN n.sub_class_name ELSE r.sub_class_name END "
                    )
                    tx.run(q, rows=rows)
                    # 多 label 追加
                    for ev in batch:
                        extra = [lb for lb in ev.all_labels if lb != LABEL_ENTITY]
                        if not extra:
                            continue
                        lbls = "".join(f":`{lb}`" for lb in extra)
                        tx.run(
                            f"MERGE (n:{LABEL_ENTITY} {{concept: $c}}) SET n{lbls}",
                            c=ev.concept,
                        )
                    return len(batch)
                total += sess.execute_write(_upsert)
        return total

    def upsert_edges(
        self,
        edges: Iterable[CausalEdge],
        batch_size: int = 200,
    ) -> int:
        """批量写入因果边。每条边的 edge_type 决定关系类型。"""
        edges = list(edges)
        total = 0
        # 按 edge_type 分组
        by_type: dict[str, list[CausalEdge]] = {}
        for e in edges:
            et = e.edge_type or REL_PROPAGATES
            by_type.setdefault(et, []).append(e)

        for edge_type, group in by_type.items():
            q = (
                "UNWIND $rows AS row "
                f"MERGE (a:{LABEL_ENTITY} {{concept: row.cause}}) "
                f"MERGE (b:{LABEL_ENTITY} {{concept: row.effect}}) "
                f"MERGE (a)-[rel:`{edge_type}`]->(b) "
                "SET rel.confidence = CASE "
                "      WHEN coalesce(rel.confidence, -1.0) < toFloat(row.conf) THEN toFloat(row.conf) "
                "      ELSE coalesce(rel.confidence, toFloat(row.conf)) END, "
                "    rel.relation_type  = coalesce(rel.relation_type, row.rel_type), "
                "    rel.evidence       = coalesce(rel.evidence, row.ev), "
                "    rel.evidence_source = coalesce(rel.evidence_source, row.src) "
            )
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                rows = [
                    {
                        "cause": e.cause_concept,
                        "effect": e.effect_concept,
                        "conf": float(e.confidence),
                        "rel_type": e.relation_type,
                        "ev": (e.evidence or "")[:500],
                        "src": e.evidence_source or "",
                    }
                    for e in batch
                ]
                with self._open_session() as sess:
                    total += sess.execute_write(
                        lambda tx: tx.run(q, rows=rows).consume().counters.relationships_created or 0
                    )
        return total

    def upsert_forbidden(self, pairs: Mapping[str, float], batch_size: int = 200) -> int:
        edges: list[CausalEdge] = []
        for key, strength in pairs.items():
            parts = key.split("|||")
            if len(parts) != 2:
                continue
            edges.append(CausalEdge(
                cause_concept=parts[0], effect_concept=parts[1],
                confidence=float(strength), relation_type="FORBIDDEN",
                edge_type=REL_FORBIDDEN,
            ))
        # FORBIDDEN_CAUSES 作为单独边类型
        if not edges:
            return 0
        q = (
            "UNWIND $rows AS row "
            f"MERGE (a:{LABEL_ENTITY} {{concept: row.cause}}) "
            f"MERGE (b:{LABEL_ENTITY} {{concept: row.effect}}) "
            f"MERGE (a)-[rel:`{REL_FORBIDDEN}`]->(b) "
            "SET rel.confidence = toFloat(row.conf), "
            "    rel.relation_type = 'FORBIDDEN' "
        )
        total = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            rows = [
                {"cause": e.cause_concept, "effect": e.effect_concept, "conf": float(e.confidence)}
                for e in batch
            ]
            with self._open_session() as sess:
                total += sess.execute_write(
                    lambda tx: tx.run(q, rows=rows).consume().counters.relationships_created or 0
                )
        return total

    # ----------------------------------------------------------------- #
    #  路径检索（匹配 3 种因果子关系）
    # ----------------------------------------------------------------- #
    def find_paths(
        self,
        start_concept: str | None = None,
        end_concept: str | None = None,
        max_hops: int = 5,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """在因果边上找路径（start→…→end），按平均置信度+节点覆盖度排序。"""
        if not start_concept and not end_concept:
            return []
        where: list[str] = []
        params: dict[str, Any] = {"limit": top_k * 5}
        if start_concept:
            where.append("s.concept = $s_c")
            params["s_c"] = start_concept
        if end_concept:
            where.append("e.concept = $e_c")
            params["e_c"] = end_concept
        q = (
            f"MATCH path = (s)-[r:{ALL_CAUSAL_RELS}*1..{int(max_hops)}]->(e) "
            "WHERE " + (" AND ".join(where) if where else "true") + " "
            "RETURN [node IN nodes(path) | node.concept] AS concepts, "
            "       [rel IN relationships(path) | rel.confidence] AS confs, "
            "       [rel IN relationships(path) | type(rel)] AS rel_types, "
            "       length(path) AS len "
            "LIMIT $limit"
        )
        rows = []
        with self._open_session() as sess:
            for rec in sess.run(q, params):
                cs = rec["concepts"]
                confs = rec["confs"] or []
                rel_types = rec["rel_types"] or []
                l = int(rec["len"])
                if not confs:
                    continue
                conf_vals = [float(x) if x is not None else 0.0 for x in confs]
                avg_conf = sum(conf_vals) / len(conf_vals)
                coverage_score = len(set(cs)) / max(len(cs), 1)
                score = avg_conf * 0.6 + coverage_score * 0.2 + (1.0 / (l + 1)) * 0.2
                rows.append({"concepts": cs, "avg_confidence": round(avg_conf, 3),
                             "length": l, "score": round(score, 3),
                             "confidences": [round(c, 2) for c in conf_vals],
                             "rel_types": rel_types})
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:top_k]

    # ----------------------------------------------------------------- #
    #  Step 2: CausalRAG 双向路径检索 + 4维规则排序
    # ----------------------------------------------------------------- #
    def search_paths(
        self,
        target_concept: str,
        direction: str = "backward",
        max_hops: int = 4,
        top_k: int = 5,
        crop_concept: str = "",
    ) -> list[dict[str, Any]]:
        """CausalRAG 核心：双向因果路径检索 + 4维规则加权排序。

        匹配 3 种因果子关系：TRIGGERS | PROPAGATES | ALLEVIATES

        Parameters
        ----------
        crop_concept : str
            作物主体约束（如 "番茄"）。提供时：
            - backward 模式：起点（上游 EnvFactor）必须是影响该作物的 EnvFactor 之一
              （通过 INFLUENCE 边约束，避免混入其他作物的噪声）
            - forward 模式：起点（target 本身）必须是影响该作物的 EnvFactor 之一
            - 约束无结果时自动回退到全图查询（兜底，避免漏召回）
        """
        if not target_concept or target_concept == "UNKNOWN":
            return []

        if direction == "forward":
            start_c, end_c = target_concept, None
        else:
            start_c, end_c = None, target_concept

        candidates: list[dict[str, Any]] = []
        seen_paths: set[tuple] = set()  # 去重：按 concepts 序列去重

        def _merge(extra: list[dict[str, Any]]) -> None:
            """去重合并路径到 candidates。"""
            for p in extra:
                key = tuple(p.get("concepts", []))
                if key not in seen_paths:
                    seen_paths.add(key)
                    candidates.append(p)

        # 第一层：目标作物精确约束（INFLUENCE/AFFECTS → 该作物）
        if crop_concept:
            first = self._find_paths_with_evidence(
                start_concept=start_c, end_concept=end_c,
                max_hops=max_hops, limit=top_k * 3,
                crop_concept=crop_concept,
            )
            for p in first:
                p["crop_match"] = "target"  # 目标作物精确匹配
            _merge(first)

        # 第二层：不够 top_k 时，用同作物类（如茄果类）的其他作物补充
        if crop_concept and len(candidates) < top_k:
            crop_class = self._get_crop_class(crop_concept)
            if crop_class:
                second = self._find_paths_with_evidence(
                    start_concept=start_c, end_concept=end_c,
                    max_hops=max_hops, limit=top_k * 3,
                    crop_class=crop_class,
                )
                for p in second:
                    p["crop_match"] = "class"  # 同作物类匹配（放宽）
                _merge(second)

        # 第三层：再不够，回退全图查询（兜底，避免漏召回）
        if len(candidates) < top_k:
            third = self._find_paths_with_evidence(
                start_concept=start_c, end_concept=end_c,
                max_hops=max_hops, limit=top_k * 3,
            )
            for p in third:
                p["crop_match"] = "fallback"  # 全图兜底
            _merge(third)

        for p in candidates:
            p["score"] = self._score_path(p)
        # 排序：优先 crop_match=target > class > fallback，同层级按 score 降序
        match_order = {"target": 0, "class": 1, "fallback": 2}
        candidates.sort(key=lambda x: (match_order.get(x.get("crop_match", "fallback"), 2), -x["score"]))
        return candidates[:top_k]

    def _get_crop_class(self, crop_concept: str) -> str:
        """查 Crop 节点的 crop_class 属性（如 "番茄" → "茄果类"）。"""
        if not crop_concept:
            return ""
        with self._open_session() as sess:
            row = sess.run(
                f"MATCH (c:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $c}}) "
                "RETURN c.crop_class AS cc LIMIT 1",
                {"c": crop_concept},
            ).single()
        return row["cc"] if row and row["cc"] else ""

    def _find_paths_with_evidence(
        self,
        start_concept: str | None,
        end_concept: str | None,
        max_hops: int = 4,
        limit: int = 15,
        crop_concept: str = "",
        crop_class: str = "",
    ) -> list[dict[str, Any]]:
        if not start_concept and not end_concept:
            return []
        where: list[str] = []
        params: dict[str, Any] = {"limit": limit * 5}
        if start_concept:
            where.append("s.concept = $s_c")
            params["s_c"] = start_concept
        if end_concept:
            where.append("e.concept = $e_c")
            params["e_c"] = end_concept
        # 作物约束：起点 s 必须是影响该作物的 EnvFactor 或病害该作物的 BioticStress
        # - EnvFactor 通过 INFLUENCE 边（对齐项目组 f:influence）
        # - BioticStress 通过 AFFECTS 边（对齐项目组 f:affects）
        # 用 pattern filter（兼容 Neo4j 3.5）
        if crop_concept:
            where.append(
                f"((s)-[:`{REL_INFLUENCE}`]->(:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $crop}}) "
                f"OR (s)-[:`{REL_AFFECTS}`]->(:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $crop}}))"
            )
            params["crop"] = crop_concept
        elif crop_class:
            # 作物类放宽：同 crop_class 的其他作物（排除已查的精确作物）
            where.append(
                f"((s)-[:`{REL_INFLUENCE}`]->(:{LABEL_ENTITY}:{LABEL_CROP} {{crop_class: $cc}}) "
                f"OR (s)-[:`{REL_AFFECTS}`]->(:{LABEL_ENTITY}:{LABEL_CROP} {{crop_class: $cc}}))"
            )
            params["cc"] = crop_class
        q = (
            f"MATCH path = (s)-[r:{ALL_CAUSAL_RELS}*1..{int(max_hops)}]->(e) "
            "WHERE " + (" AND ".join(where) if where else "true") + " "
            "RETURN [node IN nodes(path) | node.concept] AS concepts, "
            "       [rel IN relationships(path) | rel.confidence] AS confs, "
            "       [rel IN relationships(path) | rel.evidence] AS evidences, "
            "       [rel IN relationships(path) | type(rel)] AS rel_types, "
            "       length(path) AS len "
            "LIMIT $limit"
        )
        rows: list[dict[str, Any]] = []
        with self._open_session() as sess:
            for rec in sess.run(q, params):
                cs = rec["concepts"]
                confs = rec["confs"] or []
                evidences = rec["evidences"] or []
                rel_types = rec["rel_types"] or []
                l = int(rec["len"])
                if not confs:
                    continue
                conf_vals = [float(x) if x is not None else 0.0 for x in confs]
                avg_conf = sum(conf_vals) / len(conf_vals)
                coverage = len(set(cs)) / max(len(cs), 1)
                evi_count = sum(1 for ev in evidences if ev and str(ev).strip())
                evidence_score = min(evi_count, 5) / 5.0
                length_penalty = 1.0 / (l + 1)
                rows.append({
                    "concepts": cs,
                    "avg_confidence": round(avg_conf, 3),
                    "length": l,
                    "confidences": [round(c, 2) for c in conf_vals],
                    "evidences": [str(ev) if ev else "" for ev in evidences],
                    "evidence_count": evi_count,
                    "rel_types": rel_types,
                    "_avg_conf": avg_conf,
                    "_coverage": coverage,
                    "_length_penalty": length_penalty,
                    "_evidence_score": evidence_score,
                })
        return rows

    @staticmethod
    def _score_path(p: dict) -> float:
        return round(
            p["_avg_conf"] * 0.4
            + p["_coverage"] * 0.3
            + p["_length_penalty"] * 0.2
            + p["_evidence_score"] * 0.1,
            3,
        )

    # ----------------------------------------------------------------- #
    #  统计
    # ----------------------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        with self._open_session() as sess:
            n_evt = sess.run(f"MATCH (n:{LABEL_ENTITY}) RETURN count(n) AS c").single()["c"]
            n_trig = sess.run(f"MATCH ()-[r:{REL_TRIGGERS}]->() RETURN count(r) AS c").single()["c"]
            n_prop = sess.run(f"MATCH ()-[r:{REL_PROPAGATES}]->() RETURN count(r) AS c").single()["c"]
            n_allev = sess.run(f"MATCH ()-[r:{REL_ALLEVIATES}]->() RETURN count(r) AS c").single()["c"]
            n_forb = sess.run(f"MATCH ()-[r:{REL_FORBIDDEN}]->() RETURN count(r) AS c").single()["c"]
            cls_row = sess.run(
                f"MATCH (n:{LABEL_ENTITY}) "
                "WHERE n.entity_class IS NOT NULL AND n.entity_class <> '' "
                "RETURN n.entity_class AS ec, count(*) AS c ORDER BY c DESC"
            ).data()
        return {
            "num_entity_nodes": n_evt,
            "num_triggers_edges": n_trig,
            "num_propagates_edges": n_prop,
            "num_alleviates_edges": n_allev,
            "num_forbidden_edges": n_forb,
            "per_class_distribution": cls_row,
            "neo4j_version": self._neo4j_version,
        }

    # ----------------------------------------------------------------- #
    #  连通性测试
    # ----------------------------------------------------------------- #
    def test_connection(self, timeout: int = 5) -> dict[str, Any]:
        t0 = time.time()
        try:
            with self._open_session() as sess:
                rec = sess.run("RETURN 'pong' AS pong, datetime() AS now").single()
            self.detect_version()
            return {"ok": True, "latency_ms": int((time.time() - t0) * 1000),
                    "pong": rec.get("pong"), "neo4j_version": self._neo4j_version}
        except (ServiceUnavailable, AuthError) as e:
            return {"ok": False, "error": type(e).__name__ + ": " + str(e)}
        except Exception as e:
            return {"ok": False, "error": type(e).__name__ + ": " + str(e)}
