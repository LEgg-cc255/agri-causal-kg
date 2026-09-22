"""将 4 篇 MDPI 新论文整理为 DOC-017~DOC-020 追加到 agri_causal_events.json 末尾。

元数据来自 WebSearch 命中的论文页面，text 为摘要的中文翻译（保留因果信号词）。
causal_pairs / causal_chains 留空，等待回填脚本标注。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DS_PATH = _ROOT / "bate" / "data" / "agri_causal_events.json"


NEW_DOCS = [
    {
        "doc_id": "DOC-017",
        "domain": "玉米-盐碱胁迫",
        "title": "蚯蚓粪和高粱水提物缓解玉米幼苗盐碱胁迫",
        "source": {
            "type": "sci_paper",
            "publisher": "MDPI Plants",
            "year": 2022,
            "url": "https://www.mdpi.com/2223-7747/11/19/2548",
            "doi": "10.3390/plants11192548",
        },
        "text": (
            "非生物胁迫是制约全球作物生产力的重要障碍，盐碱化是破坏作物生产最严重的环境因素之一，"
            "迫切需要寻找环境安全且可持续的方法以减轻盐碱对作物的有害影响。因此，施用蚯蚓粪与低浓度高粱水提物"
            "为缓解盐碱对玉米幼苗的不利后果提供了切实可行的解决方案。试验设置三因素：盐碱梯度（对照、6 dS·m⁻¹、12 dS·m⁻¹）、"
            "蚯蚓粪用量（对照、5%、10%）、高粱水提物浓度（对照、1%、2%）。较高的盐碱胁迫对玉米形态与生理性状造成了负面影响；"
            "然而，分别施用 10% 蚯蚓粪与 2% 高粱水提物提高了幼苗对盐碱的耐受性。2% 高粱水提物与 10% 蚯蚓粪的配施显著改善了"
            "玉米的形态指标、叶绿素含量、抗氧化酶活性以及叶片与根系中的 K⁺/Na⁺ 比值和 K⁺ 含量，并在较高盐碱水平下"
            "降低了 Na⁺ 浓度、H₂O₂ 和丙二醛含量。研究表明，土施蚯蚓粪与叶面喷施高粱水提物共同作用，通过激活抗氧化防御系统、"
            "提高叶绿素含量并减少盐碱下 Na⁺ 的积累，从而缓解了盐碱对玉米的伤害。"
        ),
        "causal_pairs": [],
        "causal_chains": [],
    },
    {
        "doc_id": "DOC-018",
        "domain": "水稻-盐碱胁迫",
        "title": "水稻盐碱胁迫可持续耐受性的多层策略综述",
        "source": {
            "type": "sci_paper",
            "publisher": "MDPI Int. J. Mol. Sci.",
            "year": 2025,
            "url": "https://www.mdpi.com/1422-0067/26/13/6025",
            "doi": "10.3390/ijms26136025",
        },
        "text": (
            "耕地中的盐分积累造成严重的非生物胁迫，导致全球耕地面积减少 10%，直接威胁粮食生产与农业可持续发展。"
            "为实现高且可持续的粮食生产，必须将传统农作措施与现代技术结合，以修复受盐碱侵害的耕地。本综述系统整合了近期"
            "针对水稻的研究进展，从综合策略层面提升水稻对盐碱胁迫的耐受力。我们系统论述了初生代谢与次生代谢途径、"
            "有机改良剂、微生物共生及植物生长调节剂在减轻盐害负面效应中的作用；并重点介绍了基因编辑与转录调控等新兴"
            "遗传和表观遗传技术在培育耐盐碱水稻品种中的意义。生理学研究揭示了水稻植株对盐碱胁迫的响应过程，生化分析鉴定了"
            "与胁迫相关的代谢物，微生物学研究发现了有益植物-微生物互作关系，分子方法则筛选出关键耐盐基因，"
            "这些层面共同为培育耐盐碱水稻品种提供了不可或缺的科学依据。本文全面梳理了从农艺管理、生理适应到分子育种与微生物应用"
            "的多层策略体系，这些策略在近几十年间不断完善与发展，为深入理解并改进水稻的耐盐碱机制奠定了基础框架，"
            "也为未来抗逆水稻栽培系统的研究与落地实施提供了重要的理论支撑。"
        ),
        "causal_pairs": [],
        "causal_chains": [],
    },
    {
        "doc_id": "DOC-019",
        "domain": "水稻-灌溉管理",
        "title": "控制灌溉通过调节生长与生物量分配缓解拔节期淹涝诱导的水稻产量损失",
        "source": {
            "type": "sci_paper",
            "publisher": "MDPI Plants",
            "year": 2026,
            "url": "https://www.mdpi.com/2223-7747/15/15/2405",
            "doi": "10.3390/plants15152405",
        },
        "text": (
            "水稻拔节期的淹涝威胁着降雨集中、排水受限地区的产量稳定性，然而淹涝之前的水分管理制度如何改变水稻的生长状况"
            "与产量形成过程，目前认识仍然不足。本研究开展两年盆栽试验，在拔节期设置三种淹涝深度和两种淹涝时长，"
            "比较控制灌溉（CI）与传统淹灌（CF）的表现。淹涝处理使得水稻分蘖数增加、株高升高、总叶面积增大，"
            "但净光合速率降低、籽粒产量下降；产量损失随淹涝深度的增加和淹涝时长的延长而加剧。在相同淹涝处理下，"
            "控制灌溉（CI）抑制了营养器官的过度扩张，维持了更高的净光合速率、更大的根系干物质积累和更高的根冠比，"
            "均优于传统淹灌（CF）。相对于各自无淹涝的对照，CI 下产量损失范围为 4.11%～39.33%，而 CF 下损失高达 5.63%～52.50%。"
            "CI 较低的产量损失与其保持了更高的有效穗数、更高的结实率，以及更好的光合活性与根系生物量密切相关。"
            "研究表明，淹涝之前的水分管理方式能够影响拔节期淹涝下的生物量分配与产量形成过程；采用控制灌溉并结合及时排涝"
            "可有效降低暴露于短期淹涝的水稻系统中的产量风险。"
        ),
        "causal_pairs": [],
        "causal_chains": [],
    },
    {
        "doc_id": "DOC-020",
        "domain": "水稻-高温胁迫",
        "title": "灌浆期高温下不同施氮方式对水稻产量与品质的影响",
        "source": {
            "type": "sci_paper",
            "publisher": "MDPI Agronomy",
            "year": 2024,
            "url": "https://www.mdpi.com/2073-4395/14/1/216",
            "doi": "10.3390/agronomy14010216",
        },
        "text": (
            "灌浆早期高温频发于中国南方稻区，对水稻产量和品质造成负面影响，已成为当地水稻生产的主要威胁。"
            "本研究以大面积推广的粳稻品种“武运粳31”为材料，在2020和2022两年设置灌浆前20天增温3.5℃的处理，"
            "重点研究抽穗期三种施氮方式——低量撒施氮肥（LBN）、高量撒施氮肥（HBN）与叶面喷施氮肥（FN）——对水稻器官温度、"
            "叶片光合、叶绿素荧光、产量与稻米品质及糊化和热特性的影响，以探索减轻高温对水稻生产不利后果的切实缓解措施。"
            "结果表明，灌浆期高温使得水稻株温升高、破坏叶绿素荧光系统并降低净光合速率，进而导致结实率和千粒重下降，"
            "2020年与2022年分别造成7.0%和13.9%的产量损失。此外，高温引起整精米率下降、垩白增加和糊化温度升高，从而使稻米品质恶化。"
            "在高温条件下，高量撒施氮肥（HBN）在2020年和2022年分别使水稻产量提高3.6%和13.0%，而叶面喷氮（FN）在2022年使产量提高11.5%，"
            "增产与结实率和千粒重的提高密切相关。低量撒施（LBN）对高温下水稻产量无显著影响。氮肥措施的增产效果与其降低株温、"
            "抵御高温对叶绿素荧光系统的破坏有关；三种施氮措施总体上还改善了稻米的加工品质与外观品质。"
        ),
        "causal_pairs": [],
        "causal_chains": [],
    },
]


def main():
    with open(DS_PATH, encoding="utf-8") as f:
        ds = json.load(f)

    existing_ids = {d["doc_id"] for d in ds["documents"]}
    for doc in NEW_DOCS:
        if doc["doc_id"] in existing_ids:
            print(f"[跳过] {doc['doc_id']} 已存在于金标集，不重复插入")
            continue
        ds["documents"].append(doc)
        existing_ids.add(doc["doc_id"])
        print(f"[追加] {doc['doc_id']} {doc['domain']:<14} {doc['title'][:20]}...  DOI: {doc['source'].get('doi', '')}")

    # 重算 statistics 中配对数量
    docs = ds["documents"]
    total_pairs = sum(len(d.get("causal_pairs", [])) for d in docs)
    total_chains = sum(len(d.get("causal_chains", [])) for d in docs)
    stats = ds["dataset_info"].setdefault("statistics", {})
    stats["num_documents"] = len(docs)
    stats["num_causal_pairs"] = total_pairs
    stats["num_causal_chains"] = total_chains
    # domains：保持原有顺序，新出现的大类追加
    current = stats.get("domains", [])
    present = set()
    for d in docs:
        big = d.get("domain", "").split("-")[0] if d.get("domain") else ""
        if big and big not in present:
            present.add(big)
    new_domains = [d for d in current if d in present]
    new_domains += [d for d in sorted(present) if d not in new_domains]
    stats["domains"] = new_domains

    with open(DS_PATH, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)

    print(f"\n写入完成: {DS_PATH}")
    print(f"  文档数: {len(docs)}   causal_pairs: {total_pairs}   causal_chains: {total_chains}")
    print(f"  domains: {new_domains}")


if __name__ == "__main__":
    main()
