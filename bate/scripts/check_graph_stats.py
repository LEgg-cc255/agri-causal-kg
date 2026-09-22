"""查询图谱统计信息（用于报告撰写）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ["NEO4J_PASSWORD"] = "123456"
from src.kg.neo4j_client import AgriCausalGraph

g = AgriCausalGraph.from_env()
s = g._open_session()

# 基本统计
node_cnt = s.run("MATCH (n:Entity) RETURN count(n) AS cnt").data()[0]["cnt"]
edge_cnt = s.run("MATCH ()-[r:TRIGGERS|PROPAGATES|ALLEVIATES]->() RETURN count(r) AS cnt").data()[0]["cnt"]
forbid_cnt = s.run("MATCH ()-[r:FORBIDDEN]->() RETURN count(r) AS cnt").data()[0]["cnt"]

print(f"=== 图谱统计 ===")
print(f"节点数: {node_cnt}")
print(f"因果边: {edge_cnt}")
print(f"禁止边: {forbid_cnt}")

# 大类分布
print(f"\n=== 大类分布 ===")
for r in s.run("MATCH (n:Entity) RETURN n.entity_class AS cls, count(n) AS cnt ORDER BY cnt DESC").data():
    print(f"  {r['cls']}: {r['cnt']}")

# 子类分布
print(f"\n=== 子类分布 ===")
for r in s.run("MATCH (n:Entity) WHERE n.sub_class_name IS NOT NULL RETURN n.entity_class AS cls, n.sub_class_name AS sub, count(n) AS cnt ORDER BY cls, cnt DESC").data():
    print(f"  {r['cls']} / {r['sub']}: {r['cnt']}")

# 关系类型分布
print(f"\n=== 关系类型分布 ===")
for r in s.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC").data():
    print(f"  {r['type']}: {r['cnt']}")

# 样例节点
print(f"\n=== 样例节点（前10） ===")
for r in s.run("MATCH (n:Entity) RETURN n.concept AS concept, n.entity_class AS cls, n.sub_class_name AS sub LIMIT 10").data():
    print(f"  {r['concept']} | {r['cls']} | {r['sub']}")

# 样例因果链
print(f"\n=== 样例因果链 ===")
for r in s.run("MATCH path = (a)-[:TRIGGERS|PROPAGATES*1..3]->(b) RETURN [node in nodes(path) | node.concept] AS chain LIMIT 5").data():
    print(f"  {' → '.join(r['chain'])}")

s.close()
