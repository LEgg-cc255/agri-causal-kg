# 农业因果知识获取与解释推理方法研究

基于大语言模型（LLM）的农业因果知识图谱构建与解释推理系统。系统从农业文献中抽取因果事件，构建 Neo4j 因果知识图谱，并通过 CausalRAG（因果检索增强生成）实现可解释的因果问答。

## 功能特性

- **因果事件抽取**：一次 LLM 调用完成事件识别 + 因果对抽取 + 因果链构建
- **知识图谱构建**：Neo4j 存储，支持 3 种因果子关系（TRIGGERS/PROPAGATES/ALLEVIATES）
- **作物主体约束**：通过 INFLUENCE/AFFECTS 边实现路径推理的作物维度过滤
- **CausalRAG 因果问答**：基于因果图谱的检索增强生成，减少幻觉
- **可视化界面**：因果图谱可视化 + 路径推理 + 端到端问答
- **本体对齐**：与 AKO-F v1.9 本体对齐（f:influence/f:triggers/f:propagates/f:alleviates）

## 目录结构

```
.
├── bate/                          # 核心代码
│   ├── src/
│   │   ├── event_extraction/      # 模块1: 因果事件抽取
│   │   │   ├── event_identifier.py   # 事件识别器
│   │   │   ├── concept_aggregator.py # 概念聚合校准
│   │   │   └── prompts.py            # LLM Prompt 模板
│   │   ├── relation_extraction/  # 模块2: 因果关系抽取
│   │   │   ├── causal_extractor.py  # 因果对抽取
│   │   │   ├── chain_builder.py      # 因果链构建
│   │   │   └── confidence_scorer.py  # 置信度评分
│   │   ├── kg/                   # 模块3: 知识图谱
│   │   │   ├── neo4j_client.py       # Neo4j 客户端（图谱CRUD+路径推理）
│   │   │   ├── pipeline.py           # CausalRAG 管线
│   │   │   ├── question_parser.py    # 问题意图解析
│   │   │   └── explanation_generator.py # 解释生成
│   │   ├── utils/
│   │   │   ├── llm_client.py         # LLM 客户端（OpenAI兼容）
│   │   │   ├── knowledge_base.py     # 知识库
│   │   │   ├── data_loader.py        # 数据加载
│   │   │   └── active_learning.py    # 主动学习
│   │   └── pipeline.py             # 总管线
│   ├── data/                     # 数据文件
│   │   ├── agri_causal_events.json      # 因果事件库
│   │   ├── agri_causal_qa_dataset.json   # QA 数据集
│   │   ├── agri_causal_rules.json        # 因果规则库
│   │   ├── agri_event_types.json         # 事件类型本体
│   │   ├── agri_synonym_dict.json        # 同义词词典
│   │   ├── crop_classes.json            # 作物分类
│   │   ├── disease_list.json            # 病害清单
│   │   ├── disease_chains.json          # 病害因果链
│   │   ├── ontology_mapping.json        # 本体映射表
│   │   └── experiment_results.json      # 实验结果
│   ├── scripts/                  # 工具脚本
│   │   ├── import_to_neo4j.py        # 导入Neo4j
│   │   ├── migrate_crops.py          # 作物迁移
│   │   ├── migrate_diseases.py       # 病害迁移
│   │   ├── build_qa_dataset.py       # 构建QA数据集
│   │   ├── run_experiment.py         # 运行实验
│   │   ├── start_neo4j.ps1           # 启动Neo4j
│   │   └── ...
│   ├── web/                      # Web 可视化
│   │   ├── app.py                # FastAPI 后端
│   │   └── static/               # 前端（vis.js 图谱可视化）
│   ├── examples/                 # 示例代码
│   └── tests/                    # 测试
├── agri-kg-ontology.ttl          # AKO-F v1.9 本体（项目组）
├── 本体构建/                      # 本体设计文档
├── requirements.txt              # Python 依赖
└── .gitignore
```

## 环境要求

- **Python** >= 3.10
- **Neo4j** >= 3.5（推荐 5.x）
- **LLM API**：OpenAI 兼容 API（DeepSeek / 智谱 / 豆包 / OpenAI 均可）

## 安装步骤

### 1. 克隆仓库

```bash
git clone <仓库地址>
cd <仓库名>
```

### 2. 安装 Python 依赖

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 安装并启动 Neo4j

**方式一：Docker（推荐）**

```bash
docker run -d \
  --name neo4j-agri \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/123456 \
  -v $(pwd)/neo4j_data:/data \
  neo4j:5
```

**方式二：本地安装**

从 [neo4j.com/download](https://neo4j.com/download/) 下载安装，设置密码为 `123456`。

或用项目自带脚本：
```powershell
.\bate\scripts\start_neo4j.ps1
```

### 4. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# Neo4j
NEO4J_PASSWORD=123456

# LLM API（选一个）
# DeepSeek
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 或 智谱
# LLM_API_KEY=your_zhipu_api_key
# LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# LLM_MODEL=glm-4

# 或 OpenAI
# LLM_API_KEY=your_openai_api_key
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
```

### 5. 导入数据到 Neo4j

```bash
# 导入因果事件和关系
python -m bate.scripts.import_to_neo4j

# 迁移作物节点和 INFLUENCE 边
python -m bate.scripts.migrate_crops

# 迁移病害节点和 AFFECTS 边 + 病害因果链
python -m bate.scripts.migrate_diseases
```

## 使用方法

### 1. 启动 Web 可视化界面

```bash
python -m bate.web.app
```

浏览器访问 `http://localhost:8000`，可看到：
- **因果图**：图谱可视化（节点+边），支持按类别筛选
- **因果抽取**：输入农业文本，LLM 抽取因果对和因果链
- **路径推理**：输入问题，查找因果路径
- **端到端问答**：输入问题，生成因果解释

### 2. 命令行使用

```python
from bate.src.pipeline import AgriCausalPipeline

# 因果抽取
pipeline = AgriCausalPipeline()
result = pipeline.extract("高温胁迫使得番茄花粉量减少，造成落花。")
print(result["causal_pairs"])

# 路径推理
from bate.src.kg import AgriCausalGraph
g = AgriCausalGraph.from_env()
paths = g.search_paths("落花落蕾", direction="backward", crop_concept="番茄")
for p in paths:
    print(" -> ".join(p["concepts"]))
```

### 3. 运行实验

```bash
# 主实验
python -m bate.scripts.run_experiment

# 消融实验
python -m bate.scripts.run_ablation_experiment_v2

# 幻觉分析
python -m bate.scripts.run_hallucination_analysis
```

## 知识图谱设计

### 节点类型

| 类型 | 标签 | 数量 | 说明 |
|------|------|------|------|
| 环境因子 | EnvFactor | 20 | 高温胁迫、干旱胁迫等 |
| 生物胁迫 | BioticStress | 22 | 病害（稻瘟病等）、虫害（稻飞虱等） |
| 功能症状 | FunctionalSymptom | 106 | 生理/生殖/产量/品质症状 |
| 作物 | Crop | 16 | 番茄、水稻、小麦等 |
| 防治措施 | Treatment | 4 | 补光、保水剂等 |

### 边类型

| 关系 | 方向 | 对齐本体 | 说明 |
|------|------|----------|------|
| TRIGGERS | EnvFactor→Symptom | f:triggers | 环境因子触发症状 |
| PROPAGATES | Entity→Entity | f:propagates | 通用因果传导 |
| ALLEVIATES | Treatment→Symptom | f:alleviates | 防治措施缓解症状 |
| INFLUENCE | EnvFactor→Crop | f:influence | 环境影响作物（作物约束） |
| AFFECTS | BioticStress→Crop | f:affects | 病害影响作物（作物约束） |

### 路径推理的作物约束

查询时支持作物维度过滤（三层补充）：

1. **target**（精确匹配）：起点必须是 `INFLUENCE/AFFECTS → 该作物` 的节点
2. **class**（同类放宽）：不够时用同作物类补充（如番茄不够→茄果类的茄子/辣椒）
3. **fallback**（全图兜底）：再不够时回退全图查询

## 本体对齐

本项目图谱与 AKO-F v1.9 本体（`agri-kg-ontology.ttl`）对齐：

- **层1（ako_f 核心关系）**：INFLUENCE → f:influence
- **层2（extension 扩展关系）**：TRIGGERS/PROPAGATES/ALLEVIATES → f:triggers/f:propagates/f:alleviates
- **结构差异**：TRIGGERS 是项目组 `r_favor ⊕ r_exhibit` 两跳的快捷方式

映射表：`bate/data/ontology_mapping.json`

## 数据文件说明

| 文件 | 说明 |
|------|------|
| `agri_causal_events.json` | 因果事件库（220KB） |
| `agri_causal_qa_dataset.json` | QA 数据集（61 题：番茄16+水稻10+通用35） |
| `agri_causal_rules.json` | 因果规则库 |
| `agri_event_types.json` | 事件类型本体（AKO-F v1.9） |
| `agri_synonym_dict.json` | 同义词词典（概念聚合用） |
| `crop_classes.json` | 16 种作物 + 作物分类 |
| `disease_list.json` | 18 种病害/虫害清单 |
| `disease_chains.json` | 18 条病害因果链 |
| `ontology_mapping.json` | 本体映射表 |

## 技术栈

- **后端**：Python 3.10+ / FastAPI / Neo4j
- **前端**：原生 JS + vis.js（图谱可视化）
- **LLM**：OpenAI 兼容 API（DeepSeek/智谱/豆包/OpenAI）
- **本体**：AKO-F v1.9（OWL/Turtle）

## 许可证

仅用于学术研究。
