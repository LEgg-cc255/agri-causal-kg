"""构建农业因果推理测试集（50-100 问答对）。

从 Neo4j 图谱反向构造问答对，覆盖：
- 3 类意图：why（原因解释）/ what_effect（后果预测）/ how（机制说明）
- 2 个方向：backward（找前因）/ forward（找后果）
- 5 大事件类：环境胁迫 / 生理生化 / 发育与产量 / 管理措施 / 生物胁迫
- 3 个难度：easy（1-2跳）/ medium（3-4跳）/ hard（边界用例）

输出：bate/data/agri_causal_qa_dataset.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bate.src.kg import AgriCausalGraph, QuestionParser

# -------------------------------------------------------------------- #
#  问题模板
# -------------------------------------------------------------------- #
WHY_TEMPLATES = [
    "为什么{crop}{effect}？",
    "{effect}是什么原因导致的？",
    "造成{effect}的原因是什么？",
]

WHAT_EFFECT_TEMPLATES = [
    "{cause}会导致什么后果？",
    "{cause}的影响是什么？",
    "{cause}会带来什么影响？",
]

HOW_TEMPLATES = [
    "{cause}如何影响{effect}？",
    "{cause}影响{effect}的机制是什么？",
]

# 作物领域映射（根据目标事件概念）
CROP_MAP = {
    "落花落蕾": "番茄", "着果率下降": "番茄", "蕾铃脱落": "棉花",
    "结荚数减少": "大豆", "稻米品质下降": "水稻",
    "产量下降": "水稻", "千粒重下降": "水稻",
    "空壳率上升": "水稻", "灌浆不良": "水稻", "灌浆期缩短": "水稻",
    "植株矮化": "小麦", "颖壳闭合障碍": "水稻",
    "病害加重": "油菜", "虫害加重": "棉花", "感病性增加": "油菜",
    "萌发受阻": "蔬菜",
}

# 难度判定
def _difficulty(chain_len: int, is_boundary: bool = False) -> str:
    if is_boundary:
        return "hard"
    if chain_len <= 2:
        return "easy"
    if chain_len <= 4:
        return "medium"
    return "hard"


def main() -> None:
    if not os.environ.get("NEO4J_PASSWORD"):
        os.environ["NEO4J_PASSWORD"] = "123456"

    random.seed(42)
    parser = QuestionParser()
    g = AgriCausalGraph.from_env()

    questions: list[dict] = []
    qa_id = 0

    def add_q(question: str, category: str, intent: str, direction: str,
              domain: str, target_concept: str, expected_concepts: list[str],
              difficulty: str, chain_length: int, notes: str) -> None:
        nonlocal qa_id
        qa_id += 1
        # 验证 target_concept 能被 QuestionParser 归一化
        parse = parser.parse(question)
        questions.append({
            "id": f"QA-{qa_id:03d}",
            "question": question,
            "category": category,
            "intent": intent,
            "direction": direction,
            "domain": domain,
            "target_concept": target_concept,
            "parsed_concept": parse.target_concept,
            "parse_match": parse.target_concept == target_concept,
            "expected_concepts": expected_concepts,
            "difficulty": difficulty,
            "chain_length": chain_length,
            "notes": notes,
        })

    with g:
        s = g._open_session()

        # ============================================================
        # 1. 从典型因果链构造（backward + forward）
        # ============================================================
        print("[1/5] 从典型因果链构造问答对...")
        chains = s.run(
            "MATCH p=(a)-[:CAUSES*3..5]->(b) "
            "WITH p, length(p) AS L, "
            "  REDUCE(c=1.0, r IN relationships(p) | c*coalesce(r.confidence,0.5)) AS prod "
            "WITH p, L, prod, [n IN nodes(p) | n.concept] AS cs, "
            "  [n IN nodes(p) | coalesce(n.entity_class,'')] AS ecs "
            "WITH cs, ecs, L, prod WHERE prod > 0.8 "
            "RETURN cs, ecs, L ORDER BY prod DESC LIMIT 30"
        ).data()

        seen_targets: set[str] = set()
        for c in chains:
            cs = c["cs"]
            L = c["L"]
            if len(cs) < 3:
                continue
            start, end = cs[0], cs[-1]
            mid_concepts = cs[1:-1]

            # backward: 为什么[end]？ 期望路径含 start + mid
            crop = CROP_MAP.get(end, "")
            if end not in seen_targets:
                tpl = random.choice(WHY_TEMPLATES).format(effect=end, crop=crop + "会" if crop else "")
                # 清理"会会"双重
                tpl = tpl.replace("会会", "会").replace("为什么会", "为什么")
                expected = [start] + mid_concepts[:2]
                add_q(tpl, "why", "cause_explanation", "backward",
                      crop, end, expected, _difficulty(L), L,
                      f"因果链 backward from {start}→{end}")
                seen_targets.add(end)

            # forward: [start]会导致什么？ 期望路径含 mid + end
            if start not in seen_targets:
                tpl = random.choice(WHAT_EFFECT_TEMPLATES).format(cause=start)
                expected = mid_concepts[:2] + [end]
                add_q(tpl, "what_effect", "effect_prediction", "forward",
                      "", start, expected, _difficulty(L), L,
                      f"因果链 forward from {start}→{end}")
                seen_targets.add(start)

        # ============================================================
        # 2. 从2跳因果对构造（easy 难度）
        # ============================================================
        print("[2/5] 从2跳因果对构造 easy 问答对...")
        pairs = s.run(
            "MATCH (a)-[:CAUSES]->(m)-[:CAUSES]->(b) "
            "WITH a, m, b, count(*) AS cnt "
            "WITH a, m, b WHERE cnt >= 1 "
            "RETURN a.concept AS a, m.concept AS m, b.concept AS b "
            "LIMIT 20"
        ).data()
        seen_2hop: set[tuple] = set()
        for p in pairs:
            a, m, b = p["a"], p["m"], p["b"]
            key = (a, b)
            if key in seen_2hop:
                continue
            seen_2hop.add(key)
            crop = CROP_MAP.get(b, "")
            # backward
            tpl = f"为什么{crop}{b}？" if crop else f"为什么{b}？"
            add_q(tpl, "why", "cause_explanation", "backward",
                  crop, b, [a, m], "easy", 2, f"2跳 backward {a}→{m}→{b}")

        # ============================================================
        # 3. 根本原因边界用例（backward 返回 0 路径）
        # ============================================================
        print("[3/5] 构造根本原因边界用例...")
        roots = s.run(
            "MATCH (n:Event) WHERE NOT ()-[:CAUSES]->(n) AND (n)-[:CAUSES]->() "
            "RETURN n.concept AS c, n.entity_class AS ec LIMIT 15"
        ).data()
        for r in roots[:8]:
            concept = r["c"]
            # backward: 为什么[根本原因]？ → 期望 0 路径
            add_q(f"为什么{concept}？", "why", "cause_explanation", "backward",
                  "", concept, [], "hard", 0,
                  f"根本原因边界 backward（期望0路径）")
            # forward: [根本原因]会导致什么？ → 正常检索
            if qa_id % 3 == 0:  # 抽样避免过多
                add_q(f"{concept}会导致什么后果？", "what_effect", "effect_prediction",
                      "forward", "", concept, [], "medium", 0,
                      f"根本原因 forward 正常")

        # ============================================================
        # 4. 叶节点边界用例（forward 返回 0 路径）
        # ============================================================
        print("[4/5] 构造叶节点边界用例...")
        leaves = s.run(
            "MATCH (n:Event) WHERE NOT (n)-[:CAUSES]->() AND ()-[:CAUSES]->(n) "
            "RETURN n.concept AS c, n.entity_class AS ec LIMIT 15"
        ).data()
        for r in leaves[:8]:
            concept = r["c"]
            crop = CROP_MAP.get(concept, "")
            # forward: [叶节点]会导致什么？ → 期望 0 路径
            add_q(f"{concept}会导致什么？", "what_effect", "effect_prediction",
                  "forward", crop, concept, [], "hard", 0,
                  f"叶节点边界 forward（期望0路径）")
            # backward: 为什么[叶节点]？ → 正常检索
            if qa_id % 3 == 1:
                tpl = f"为什么{crop}{concept}？" if crop else f"为什么{concept}？"
                add_q(tpl, "why", "cause_explanation", "backward",
                      crop, concept, [], "medium", 0,
                      f"叶节点 backward 正常")

        # ============================================================
        # 5. 机制说明类（how）
        # ============================================================
        print("[5/5] 构造机制说明 how 问答对...")
        how_pairs = s.run(
            "MATCH p=(a)-[:CAUSES*2..3]->(b) "
            "WITH a, b, count(*) AS cnt, length(p) AS L, "
            "  [n IN nodes(p) | n.concept] AS cs "
            "RETURN a.concept AS a, b.concept AS b, cs ORDER BY cnt DESC LIMIT 15"
        ).data()
        seen_how: set[tuple] = set()
        for p in how_pairs:
            a, b = p["a"], p["b"]
            cs = p["cs"]
            if (a, b) in seen_how or a == b:
                continue
            seen_how.add((a, b))
            mid = [c for c in cs if c != a and c != b][:2]
            crop = CROP_MAP.get(b, "")
            tpl = f"{a}如何影响{b}？"
            add_q(tpl, "how", "mechanism", "backward",
                  crop, b, [a] + mid, "medium", len(cs) - 1,
                  f"机制说明 {a}→...→{b}")

    # ================================================================
    # 去重 + 输出
    # ================================================================
    # 按 question 去重
    seen_q: set[str] = set()
    deduped: list[dict] = []
    for q in questions:
        if q["question"] in seen_q:
            continue
        seen_q.add(q["question"])
        deduped.append(q)
    questions = deduped
    # 重新编号
    for i, q in enumerate(questions, 1):
        q["id"] = f"QA-{i:03d}"

    # 统计
    by_intent = {}
    by_direction = {}
    by_difficulty = {}
    by_class_target = {}
    parse_match_count = 0
    for q in questions:
        by_intent[q["intent"]] = by_intent.get(q["intent"], 0) + 1
        by_direction[q["direction"]] = by_direction.get(q["direction"], 0) + 1
        by_difficulty[q["difficulty"]] = by_difficulty.get(q["difficulty"], 0) + 1
        if q["parse_match"]:
            parse_match_count += 1

    dataset = {
        "dataset_info": {
            "name": "agri_causal_qa_dataset",
            "version": "v1.0",
            "description": "农业因果推理测试集 - 用于评估 CausalRAG pipeline",
            "num_questions": len(questions),
            "source": "基于 Neo4j 图谱(119节点/236边)反向构造",
            "statistics": {
                "by_intent": by_intent,
                "by_direction": by_direction,
                "by_difficulty": by_difficulty,
                "parse_match_rate": round(parse_match_count / len(questions), 3),
            },
        },
        "questions": questions,
    }

    out_path = _ROOT / "bate" / "data" / "agri_causal_qa_dataset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"测试集已生成: {out_path}")
    print(f"问答对总数: {len(questions)}")
    print(f"意图分布: {by_intent}")
    print(f"方向分布: {by_direction}")
    print(f"难度分布: {by_difficulty}")
    print(f"Step1 解析匹配率: {parse_match_count}/{len(questions)} "
          f"({parse_match_count/len(questions)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
