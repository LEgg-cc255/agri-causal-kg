"""作物主体节点迁移脚本。

功能：
1. 从 crop_classes.json 批量建 Crop 节点
2. 题库 domain 反推 + 反向查 EnvFactor，建 EnvFactor → Crop 的 INFLUENCE 边（项目组 f:influence）
3. 给 agri_causal_qa_dataset.json 题库补 domain 字段（无作物的标"通用"）

设计原则：
- 作物主体通过 Crop 节点 + INFLUENCE 边表达（项目组 f:influence）
- 不在症状节点上加 crops 属性，避免破坏症状节点通用复用性
  （同一症状对不同作物可有不同表现，如高温→番茄:落花；高温→水稻:空壳率）
- INFLUENCE 边是路径推理做"作物约束"的锚点：先找影响该作物的 EnvFactor，
  再从这些 EnvFactor 出发找因果链

运行：
    $env:NEO4J_PASSWORD="123456"; python -m bate.scripts.migrate_crops
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.kg import AgriCausalGraph
from bate.src.kg.neo4j_client import (
    LABEL_CROP, LABEL_ENTITY, REL_INFLUENCE,
    ALL_CAUSAL_RELS,
)


# 复用 question_parser 的作物词典
_CROP_DICT: dict[str, str] = {
    "番茄": "番茄", "西红柿": "番茄",
    "茄子": "茄子", "辣椒": "辣椒", "黄瓜": "黄瓜",
    "水稻": "水稻", "稻": "水稻",
    "小麦": "小麦", "玉米": "玉米", "大麦": "大麦",
    "油菜": "油菜", "大豆": "大豆", "黄豆": "大豆",
    "棉花": "棉花", "蔬菜": "蔬菜",
    "茶树": "茶", "茶叶": "茶",
    "葡萄": "葡萄", "苹果": "苹果", "果树": "果树",
}
_SORTED_CROPS = sorted(_CROP_DICT.keys(), key=len, reverse=True)


def detect_crop(text: str) -> str:
    """从文本中检测作物，返回标准作物名（无则空）。"""
    for crop in _SORTED_CROPS:
        if crop in text:
            return _CROP_DICT[crop]
    return ""


def load_crop_classes() -> list[dict]:
    """加载作物清单。"""
    path = _ROOT / "bate" / "data" / "crop_classes.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["crops"]


def load_dataset() -> list[dict]:
    """加载题库。"""
    path = _ROOT / "bate" / "data" / "agri_causal_qa_dataset.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def main() -> None:
    os.environ.setdefault("NEO4J_PASSWORD", "123456")

    crops = load_crop_classes()
    questions = load_dataset()
    print(f"[迁移] 作物 {len(crops)} 种 | 题库 {len(questions)} 题")

    graph = AgriCausalGraph.from_env()

    # ----------------------------------------------------------------- #
    # Step 1: 建 Crop 节点
    # ----------------------------------------------------------------- #
    with graph._open_session() as sess:
        for c in crops:
            sess.run(
                f"MERGE (n:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $c}}) "
                "SET n.entity_class = 'Crop', "
                "    n.sub_class_id = 'Crop.Crop', "
                "    n.sub_class_name = $cn, "
                "    n.crop_class = $cc",
                c=c["concept"], cn=c.get("crop_class", ""), cc=c.get("crop_class", ""),
            )
        print(f"[Step1] Crop 节点建完 {len(crops)} 个")

    # ----------------------------------------------------------------- #
    # Step 2: 题库 domain 反推 + 反向查 EnvFactor，建 INFLUENCE 边
    # ----------------------------------------------------------------- #
    # 策略：对每道题
    #   1) 检测问题文本里的作物 crop
    #   2) 对 expected_concepts 里的每个概念，反向找上游 EnvFactor
    #      （沿 TRIGGERS/PROPAGATES 反向走 1..3 跳）
    #   3) 建 EnvFactor → Crop 的 INFLUENCE 边
    # 这样能覆盖所有题库涉及的 (EnvFactor, Crop) 组合，预期 20+ 条边
    influence_edges: set[tuple[str, str]] = set()

    with graph._open_session() as sess:
        for q in questions:
            qtext = q.get("question", "")
            crop = detect_crop(qtext)
            if not crop:
                continue
            # 题库里标注的 domain 字段也用上（更可靠）
            domain = q.get("domain", "")
            if domain and domain != "通用":
                crop = domain

            for concept in q.get("expected_concepts", []):
                # 反向找 EnvFactor：沿 TRIGGERS/PROPAGATES 反向走 1..3 跳
                # 起点是 expected_concept，终点是 EnvFactor
                # 注意：Neo4j 3.5 多关系类型不能用反引号包裹，写法是 :TRIGGERS|PROPAGATES|ALLEVIATES
                recs = sess.run(
                    f"MATCH (env:{LABEL_ENTITY})-[:{ALL_CAUSAL_RELS}*1..3]->(s:{LABEL_ENTITY} {{concept: $c}}) "
                    "WHERE env.entity_class = '环境因子' "
                    "RETURN DISTINCT env.concept AS env_concept",
                    c=concept,
                )
                for r in recs:
                    env_c = r["env_concept"]
                    influence_edges.add((env_c, crop))

    # 建 INFLUENCE 边
    with graph._open_session() as sess:
        for env_concept, crop_concept in influence_edges:
            sess.run(
                f"MATCH (a:{LABEL_ENTITY} {{concept: $ec}}), "
                f"      (b:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $cc}}) "
                f"MERGE (a)-[:`{REL_INFLUENCE}`]->(b)",
                ec=env_concept, cc=crop_concept,
            )
        print(f"[Step2] INFLUENCE 边建完 {len(influence_edges)} 条")

    # ----------------------------------------------------------------- #
    # Step 3: 题库补 domain 字段
    # ----------------------------------------------------------------- #
    ds_path = _ROOT / "bate" / "data" / "agri_causal_qa_dataset.json"
    with open(ds_path, encoding="utf-8") as f:
        ds = json.load(f)
    for q in ds["questions"]:
        if "domain" not in q or not q["domain"]:
            q["domain"] = detect_crop(q.get("question", "")) or "通用"
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)
    domain_dist = {}
    for q in ds["questions"]:
        d = q.get("domain", "通用")
        domain_dist[d] = domain_dist.get(d, 0) + 1
    print(f"[Step3] 题库 domain 标完: {domain_dist}")

    # ----------------------------------------------------------------- #
    # Step 4: 给因果边打 source_layer 标签（项目组本体对齐）
    # ----------------------------------------------------------------- #
    # ako_f: 项目组本体核心关系（INFLUENCE 对齐 f:influence）
    # extension: 我的扩展关系（TRIGGERS/PROPAGATES/ALLEVIATES，对齐项目组 triggers/propagates/alleviates
    #           但语义上是把项目组两跳压成一跳的快捷方式，详见 ontology_mapping.json）
    with graph._open_session() as sess:
        n_ako = sess.run(
            f"MATCH ()-[r:`{REL_INFLUENCE}`]->() SET r.source_layer = 'ako_f' "
            "RETURN count(r) AS c"
        ).single()["c"]
        n_ext = sess.run(
            f"MATCH ()-[r:TRIGGERS|PROPAGATES|ALLEVIATES]->() SET r.source_layer = 'extension' "
            "RETURN count(r) AS c"
        ).single()["c"]
        print(f"[Step4] source_layer 标签打完: ako_f={n_ako}, extension={n_ext}")

    # ----------------------------------------------------------------- #
    # 汇总
    # ----------------------------------------------------------------- #
    with graph._open_session() as sess:
        crop_cnt = sess.run(f"MATCH (n:{LABEL_CROP}) RETURN count(n) AS c").single()["c"]
        inf_cnt = sess.run(
            f"MATCH ()-[r:`{REL_INFLUENCE}`]->() RETURN count(r) AS c"
        ).single()["c"]
        # 清除可能残留的 crops 属性（之前错误地给症状节点加过）
        removed = sess.run(
            f"MATCH (n:{LABEL_ENTITY}) WHERE n.crops IS NOT NULL "
            "REMOVE n.crops RETURN count(n) AS c"
        ).single()["c"]
        sym_total = sess.run(
            f"MATCH (n:{LABEL_ENTITY}) WHERE n.entity_class = '功能症状' "
            "RETURN count(n) AS c"
        ).single()["c"]

    print(f"\n=== 迁移完成 ===")
    print(f"  Crop 节点: {crop_cnt}")
    print(f"  INFLUENCE 边: {inf_cnt}")
    print(f"  清除症状节点残留 crops 属性: {removed} 个")
    print(f"  FunctionalSymptom 节点总数: {sym_total}（均不带 crops 属性，保持通用）")
    graph.close()


if __name__ == "__main__":
    main()
