"""具体病害/虫害节点 + AFFECTS 边迁移脚本。

功能：
1. 从 disease_list.json 批量建具体病害/虫害节点（对齐项目组 f:Disease / f:Pest）
2. 建 BioticStress →(AFFECTS)→ Crop 边（对齐项目组 f:affects）
3. 打 source_layer='ako_f' 标签（项目组本体核心关系）

设计原则：
- 项目组本体 f:affects 的 domain 是 Disease/Pest/WeedCompetition，range 是 Crop
- 我的图谱用 AFFECTS 边对齐：BioticStress(具体病害/虫害) → Crop
- 这样路径推理的作物约束支持两种起点：
  - EnvFactor →(INFLUENCE)→ Crop（环境因子路径）
  - BioticStress →(AFFECTS)→ Crop（生物胁迫路径）

运行：
    $env:NEO4J_PASSWORD="123456"; python -m bate.scripts.migrate_diseases
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
    LABEL_ENTITY, LABEL_CROP, REL_AFFECTS, REL_PROPAGATES,
)


def load_diseases() -> list[dict]:
    """加载病害清单。"""
    path = _ROOT / "bate" / "data" / "disease_list.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["diseases"]


def main() -> None:
    os.environ.setdefault("NEO4J_PASSWORD", "123456")

    diseases = load_diseases()
    print(f"[迁移] 病害/虫害 {len(diseases)} 个")

    graph = AgriCausalGraph.from_env()

    # ----------------------------------------------------------------- #
    # Step 1: 建具体病害/虫害节点
    # ----------------------------------------------------------------- #
    with graph._open_session() as sess:
        for d in diseases:
            # 拆 sub_class_id 取 class_id 和 sub_id 作为 label
            # 如 "BioticStress.Disease" → labels: ["BioticStress", "Disease"]
            cls_id, sub_id = d["sub_class_id"].split(".", 1)
            sess.run(
                f"MERGE (n:{LABEL_ENTITY}:{cls_id}:{sub_id} {{concept: $c}}) "
                "SET n.entity_class = $ec, "
                "    n.sub_class_id = $sc, "
                "    n.sub_class_name = $sn, "
                "    n.pathogen = $pg",
                c=d["concept"],
                ec=d["entity_class"],
                sc=d["sub_class_id"],
                sn=d["sub_class_name"],
                pg=d.get("pathogen", ""),
            )
        print(f"[Step1] 病害/虫害节点建完 {len(diseases)} 个")

    # ----------------------------------------------------------------- #
    # Step 2: 建 AFFECTS 边（病害 → 作物）
    # ----------------------------------------------------------------- #
    affects_edges = 0
    missing_crops: list[str] = []
    with graph._open_session() as sess:
        for d in diseases:
            for crop in d.get("affects_crops", []):
                # 检查 Crop 节点是否存在
                rec = sess.run(
                    f"MATCH (c:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $cc}}) RETURN c.concept AS c",
                    cc=crop,
                ).single()
                if not rec:
                    missing_crops.append(f"{d['concept']}→{crop}（Crop 节点 {crop} 不存在）")
                    continue
                sess.run(
                    f"MATCH (a:{LABEL_ENTITY} {{concept: $dc}}), "
                    f"      (b:{LABEL_ENTITY}:{LABEL_CROP} {{concept: $cc}}) "
                    f"MERGE (a)-[:`{REL_AFFECTS}`]->(b)",
                    dc=d["concept"], cc=crop,
                )
                affects_edges += 1
        print(f"[Step2] AFFECTS 边建完 {affects_edges} 条")
        if missing_crops:
            print(f"  跳过（Crop 节点不存在）: {missing_crops}")

    # ----------------------------------------------------------------- #
    # Step 3: 打 source_layer 标签
    # ----------------------------------------------------------------- #
    with graph._open_session() as sess:
        n_ako = sess.run(
            f"MATCH ()-[r:`{REL_AFFECTS}`]->() SET r.source_layer = 'ako_f' "
            "RETURN count(r) AS c"
        ).single()["c"]
        print(f"[Step3] source_layer 标签打完: ako_f(AFFECTS)={n_ako}")

    # ----------------------------------------------------------------- #
    # Step 4: 建病害完整因果链（方案B）
    # ----------------------------------------------------------------- #
    # 链结构：病害 →(PROPAGATES)→ 病害专属症状 →(PROPAGATES)→ 通用症状 → ... → 产量下降
    # 病害专属症状是新建的（如叶片病斑/穗腐/果实腐烂等）
    # 通用症状尽量复用已有的（如光合作用下降/灌浆不良/产量下降等）
    chains_path = _ROOT / "bate" / "data" / "disease_chains.json"
    with open(chains_path, encoding="utf-8") as f:
        chain_data = json.load(f)

    # 4.1 建新症状节点（病害专属症状，如叶片病斑/穗腐等）
    new_syms = chain_data["new_symptoms"]
    with graph._open_session() as sess:
        for sym in new_syms:
            cls_id, sub_id = sym["sub_class_id"].split(".", 1)
            sess.run(
                f"MERGE (n:{LABEL_ENTITY}:{cls_id}:{sub_id} {{concept: $c}}) "
                "SET n.entity_class = $ec, "
                "    n.sub_class_id = $sc, "
                "    n.sub_class_name = $sn",
                c=sym["concept"],
                ec=sym["entity_class"],
                sc=sym["sub_class_id"],
                sn=sym["sub_class_name"],
            )
        print(f"[Step4.1] 新症状节点建完 {len(new_syms)} 个（叶片病斑/穗腐/果实腐烂等）")

    # 4.2 建因果链边：病害 → path[0] → path[1] → ... → path[n]，每段 PROPAGATES
    chain_edges = 0
    with graph._open_session() as sess:
        for chain in chain_data["chains"]:
            disease = chain["disease"]
            path = chain["path"]
            full_seq = [disease] + path  # 完整节点序列
            # 建相邻节点之间的 PROPAGATES 边
            for i in range(len(full_seq) - 1):
                cause = full_seq[i]
                effect = full_seq[i + 1]
                sess.run(
                    f"MATCH (a:{LABEL_ENTITY} {{concept: $c}}), "
                    f"      (b:{LABEL_ENTITY} {{concept: $e}}) "
                    f"MERGE (a)-[rel:`{REL_PROPAGATES}`]->(b) "
                    "ON CREATE SET rel.confidence = 0.85, "
                    "             rel.source_layer = 'extension', "
                    "             rel.evidence = '病害因果链（方案B）'",
                    c=cause, e=effect,
                )
                chain_edges += 1
        print(f"[Step4.2] 病害因果链边建完 {chain_edges} 条（{len(chain_data['chains'])} 条链）")

    # ----------------------------------------------------------------- #
    # 汇总
    # ----------------------------------------------------------------- #
    with graph._open_session() as sess:
        dis_cnt = sess.run(
            f"MATCH (n:{LABEL_ENTITY}) WHERE n.entity_class = '生物胁迫' "
            "RETURN count(n) AS c"
        ).single()["c"]
        aff_cnt = sess.run(
            f"MATCH ()-[r:`{REL_AFFECTS}`]->() RETURN count(r) AS c"
        ).single()["c"]
        # 按 sub_class_name 分组
        rows = sess.run(
            f"MATCH (n:{LABEL_ENTITY}) WHERE n.entity_class = '生物胁迫' "
            "RETURN n.sub_class_name AS sn, count(n) AS c ORDER BY c DESC"
        ).data()

    print(f"\n=== 迁移完成 ===")
    print(f"  生物胁迫节点总数: {dis_cnt}")
    for r in rows:
        print(f"    {r['sn']}: {r['c']}")
    print(f"  AFFECTS 边: {aff_cnt}")
    graph.close()


if __name__ == "__main__":
    main()
