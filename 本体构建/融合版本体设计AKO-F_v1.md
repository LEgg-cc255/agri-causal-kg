# 融合版本体设计（AKO-F v1.0）
### ——以 B《农业知识图谱本体构建报告》四层+七元组+Mask 为骨架，吸收 A《AKO 本体》细粒度与关系拆分，并修订 B 的形式化漏洞

> 配套文件：`agri-kg-ontology-fused.ttl`（机器可读本体）、`knowledge-unit.schema.json`（修订版 JSON Schema）、`M_type_热力图.png`（合法配对 Mask）
> 域：以水稻/玉米-病害为主线（兼容 B 的语料与实验设计），保留虫害与全生育期扩展（吸收 A）

---

## 0. 设计原则与修订总览

**骨架（来自 B，原样保留）**：四层结构（实体/关系/事件/时空+因果规则库）、知识单元七元组 `ku=(s,r,o,t,l,c,m)`、约束 Mask 机制（M_type / M_rel 软惩罚）、五维约束算子定义来源、低资源可计算设计。

**增强（来自 A，选择性吸收）**：细粒度实体（病原、虫害、症状部位化）、病三角关系拆分、事件类型/实例分层与 n 元槽位、观测与多模态媒体挂载、时间/空间独立建模、OWL 形式化。

**对 B 形式化漏洞的修订（10 项）**：

| # | B 的问题 | 融合版修订 |
|---|---|---|
| 1 | `occur_in` 约束主体写 EVENT，但 8 类实体中无 EVENT 类，示例主体又为 DISEASE，自相矛盾 | `occur_in` 主体改为 **DISEASE/PEST → GROWTH_STAGE/TIME**；事件不再充当关系主体，改由事件槽位展开 |
| 2 | `affect` 语义过粗（同一关系承担"环境→影响→作物"与"环境→促进→病害"） | 拆分为 **`r_influence`（ENV→CROP）** 与 **`r_favor`（ENV→DISEASE/PEST）** |
| 3 | 缺失"病害/虫害危害作物"这一最核心关系（B 无 DISEASE→CROP） | 新增 **`r_affects`（DISEASE/PEST → CROP）** |
| 4 | 病三角缺"病原"角（只有病害-环境两角） | 新增实体 **PATHOGEN** + 关系 **`r_caused_by`（DISEASE→PATHOGEN）** |
| 5 | SYMPTOM 为扁平实体，无部位/形态/图像挂载 | SYMPTOM 增加**部位子类型**（叶部/茎部/果实），并接入观测/媒体 |
| 6 | 事件类型与关系/实体重叠（"防治"事件=treat 关系=TREATMENT 实体；"扩散"事件=spread_in 关系） | 明确**关系=图谱边（一对），事件=知识单元复合载体（r 取事件类型时按槽位展开）**，知识单元加 `ku_type` 判别 |
| 7 | TIME/LOCATION 角色双重（既是 8 类实体又"不作为主客体"） | **双轨制**：七元组内嵌 t/l 三元组（轻量）+ 图谱层独立 TIME/LOCATION 类（吸收 A）；明确二者只做客体与边属性 |
| 8 | 事件仅 5 类，覆盖不了虫害/全生育期 | 扩展为 **8 类事件**（+虫害发生/生长发育/品质形成，可继续扩展） |
| 9 | 实体层无虫害（PEST），病害主线外无法表达虫害 | 新增实体 **PEST** |
| 10 | 稀疏性 ρ=18/64≈0.28 | 扩展后 10 类实体合法配对 20/100，**ρ=0.20**，假设空间进一步压缩 |

---

## 1. 本体总体构成（四层结构）

```
O_F = 实体层(10类语义实体 + 观测增强) + 关系层(10类核心 + 4类观测)
    + 事件层(8类事件类型) + 时空规范与因果规则库
```

| 构成层 | 内容 | 在图谱/流程中的作用 |
|---|---|---|
| 实体层 | 10 类实体（CROP/DISEASE/PEST/PATHOGEN/SYMPTOM/TREATMENT/ENV_FACTOR/GROWTH_STAGE/LOCATION/TIME）+ OBSERVATION 观测增强 | 决定图谱节点类型（V） |
| 关系层 | 10 类核心关系（affects/influence/favor/exhibit/caused_by/treat/occur_in/depend_on/spread_in/resist）+ 4 类观测关系 | 决定边（E），并生成 10×10 兼容 Mask |
| 事件层 | 8 类事件类型（病害发生/虫害发生/环境影响/防治/物候/扩散/生长发育/品质形成） | 多模态融合的统一语义载体，决定槽位对齐 |
| 时空规范与因果规则库 | 时间三元组、空间三元组、生育期/物候映射；作物-病害-环境因果规则（NY/T） | 时空编码与因果先验，支撑约束算子 |

---

## 2. 实体层：10 类实体类型体系

```
T_entity = {CROP, DISEASE, PEST, PATHOGEN, SYMPTOM, TREATMENT,
            ENV_FACTOR, GROWTH_STAGE, LOCATION, TIME}
```

| 类型 | 含义 | 示例 | 角色/约束 |
|---|---|---|---|
| CROP | 受危害/栽培的作物 | 水稻、玉米、番茄 | 语义主体；可作 r_affects/r_influence 客体、r_resist 主体 |
| DISEASE | 病害 | 稻瘟病、纹枯病 | 语义主体；r_affects 主体、r_exhibit 主体、r_caused_by 主体等 |
| PEST ★新增 | 虫害/害虫 | 稻飞虱、二化螟 | 与 DISEASE 对称；r_affects/r_exhibit 主体等 |
| PATHOGEN ★新增 | 病原（独立实体） | 稻瘟病菌(Pyricularia) | 仅作 r_caused_by 客体 |
| SYMPTOM | 症状（子类型按部位） | 叶斑、枯心、白穗 | 部位子类型：叶部/茎部/穗部/根部/果实 |
| TREATMENT | 防治措施（药剂/农艺） | 三环唑、排水晒田 | 仅作 r_treat 主体 |
| ENV_FACTOR | 环境因子（子类型按参数） | 高温、高湿、干旱 | 作 r_influence/r_favor 主体、r_depend_on 客体 |
| GROWTH_STAGE | 生育期 | 分蘖期、孕穗期、坐果期 | 作 r_occur_in 客体；物候事件槽位 |
| LOCATION | 空间（省/稻区/地块/设施） | 广东省、华南稻区 | 只作 r_spread_in/观测 客体与边属性 |
| TIME | 时间（点/区间/物候期） | 6月、2026-06-01 | 只作 r_occur_in/观测 客体与边属性 |

**图谱角色划分**（修订 B 的第 7 项）：
- **前 8 类**为领域语义主体类（可作 v_s / v_o），承载 Ontology 角色约束；
- **LOCATION/TIME** 为时空属性类，双重角色明确为——七元组中作边属性（t/l），图谱中作实体节点（仅作客体），**不参与语义关系的主体侧**；
- **OBSERVATION** 为图谱增强实体（多模态挂载），承载环境观测、图像/文本证据，独立于核心语义 Mask。

**ENV_FACTOR / SYMPTOM 子类型**（吸收 A）：

```
ENV_FACTOR ─┬─ 温度（日均/昼夜温差/高温/低温）
            ├─ 湿度（空气相对湿度）
            ├─ 光照（时长/强度）
            ├─ 降水（雨量/连阴雨）
            ├─ 土壤（质地/肥力N-P-K/酸碱度/有机质）
            └─ 水质（盐度/硬度/重金属）
SYMPTOM ────┬─ 叶部症状 / 茎部症状 / 穗部症状 / 根部症状 / 果实症状
```

---

## 3. 关系层：10 类核心关系 + 4 类观测关系

### 3.1 核心关系（进入 M_type / M_rel 的合法性判定）

| 关系 | 主体 → 客体约束 | 示例 |
|---|---|---|
| `r_affects` ★危害（新增） | DISEASE/PEST → CROP | 稻瘟病 → 危害 → 水稻 |
| `r_influence` 影响（拆分①） | ENV_FACTOR → CROP | 低温 → 影响 → 水稻 |
| `r_favor` 促进（拆分②，病三角） | ENV_FACTOR → DISEASE/PEST | 高湿 → 促进 → 稻瘟病 |
| `r_exhibit` 表现 | DISEASE/PEST → SYMPTOM | 稻瘟病 → 表现 → 叶斑 |
| `r_caused_by` 由病原引起（新增） | DISEASE → PATHOGEN | 稻瘟病 → 由病原引起 → 稻瘟病菌 |
| `r_treat` 防治 | TREATMENT → DISEASE/PEST | 三环唑 → 防治 → 稻瘟病 |
| `r_occur_in` 发生于（修订） | DISEASE/PEST → GROWTH_STAGE/TIME | 稻瘟病 → 发生于 → 孕穗期 |
| `r_depend_on` 依赖 | DISEASE/PEST → ENV_FACTOR | 稻瘟病 → 依赖 → 高湿（与 r_favor 互为逆语义） |
| `r_spread_in` 扩散于 | DISEASE/PEST → LOCATION | 稻瘟病 → 扩散于 → 华南 |
| `r_resist` 抗性（新增，可选） | CROP → DISEASE/PEST | 汕优63 → 抗 → 稻瘟病 |

### 3.2 观测关系（图谱增强层，不进入核心语义 Mask）

| 关系 | 主体 → 客体约束 | 示例 |
|---|---|---|
| `r_measures` 测量 | OBSERVATION → ENV_FACTOR | 观测 → 测量 → 温度 |
| `r_observed_at` 观测时间 | OBSERVATION → TIME | 观测 → 发生于 → 2026-06-01 |
| `r_observed_in` 观测地点 | OBSERVATION → LOCATION | 观测 → 位于 → 华南稻区 |
| `r_depicts` 描绘 | OBSERVATION(媒体) → SYMPTOM/DISEASE/CROP | 症状图 → 描绘 → 叶斑 |

### 3.3 修订说明

- **r_affects 的补入**是最关键修订：B 的关系体系中作物只作为环境影响的客体，缺失"病害→危害→作物"这一农业图谱最核心的边，将导致图谱无法表达"谁危害了谁"；
- **r_favor 与 r_depend_on 互为逆语义**：`ENV favor DISEASE` ⇔ `DISEASE depend_on ENV`，实现时可作为一对逆关系，仅存一侧即可；
- 主客体合法性约束同时构成 M_type 的构造依据（见 §8）。

---

## 4. 事件层：8 类事件类型体系

### 4.1 事件定义（修订 B 的事件/关系边界）

事件不再作为"实体类型"，而是**知识单元的复合载体**：当七元组中 `r` 取事件类型时，`s/o/t/l` 按该事件类型的槽位填充，形成一个事件假设。事件定义：

```
E = (event_type, 槽位集合, 模态约束, 因果角色 R)
```

**事件类型与关系/实体的边界**（修订 B 第 6 项）：

| 概念 | 定义 | 例子 |
|---|---|---|
| 关系（relation） | 图谱**边**，一对主客体 | `treat(三环唑, 稻瘟病)` |
| 实体（entity） | 图谱**节点** | TREATMENT=三环唑 |
| 事件（event） | **知识单元**的复合载体（r 取事件类型，槽位含多实体+时空+因果角色） | "2026年6月华南稻区水稻孕穗期稻瘟病发生" |

"防治/扩散"在 B 中既是事件又是关系又是实体——融合版明确：**treat/spread_in 是关系**，"防治/扩散"是**事件类型**（事件描述一段含时间地点的防治或扩散过程），二者作用于不同抽象层级，通过 `ku_type` 判别。

### 4.2 事件类型体系（8 类，可扩展枚举）

| 事件类型 | 论元槽位 | 主要模态 | 需对齐关键槽位 |
|---|---|---|---|
| 病害发生 | (C, D, S, τ, λ, A) | 文本+图像(+环境) | C, D, S, τ, λ, A |
| 虫害发生 ★新增 | (C, P, S, τ, λ, A) | 文本+图像(+环境) | C, P, S, τ, λ, A |
| 环境影响 | (A, C, D, τ) | 环境+文本 | A, C, D, τ |
| 防治 | (C, D/P, TREATMENT, τ) | 文本 | C, D/P, TREATMENT, τ |
| 物候 | (C, GROWTH_STAGE, τ) | 文本+环境 | C, GROWTH_STAGE, τ |
| 扩散 | (D/P, LOCATION, τ) | 文本 | D/P, LOCATION, τ |
| 生长发育 ★新增 | (C, GROWTH_STAGE/器官, τ) | 文本(+图像) | C, GROWTH_STAGE, τ |
| 品质形成 ★新增 | (C, 性状, τ) | 文本(+环境) | C, 性状, τ |

### 4.3 事件槽位约束（修订 B 第 8 项）

- 槽位值域由实体类型体系限定（如 D 槽 ∈ DISEASE、A 槽 ∈ ENV_FACTOR）；
- 每个模态独立生成"事件假设"（槽位填充+槽位置信度），作为融合基本单元（沿用 B 的三模态假设）；
- 槽位空间有界：7 作物 × 约 12 病害 × 8 事件类型，可在小规模下精确求解或轻量评分近似（沿用 B）。

---

## 5. 时空规范与时序建模

### 5.1 时间/空间三元组（保留 B，内嵌七元组）

```
t = (t_start, t_end, stage)          # 起止时间 + 生育期
l = (province, lon, lat)             # 省级行政区 + 经纬度
```

时序性通过每条知识单元内嵌时间三元组实现，不单独建立一套独立时序本体（保留 B 的轻量决策）；图谱层补充 TIME/LOCATION 独立类（吸收 A，供观测/事件挂载）：

```
TIME 类 ── TimePoint(时间点) / TimeInterval(区间) / PhenologicalPeriod(物候期)
LOCATION 类 ── Region(产区) / Plot(地块) / Facility(设施：日光温室/大棚/连栋)
```

### 5.2 生育期（GROWTH_STAGE）时序锚点（保留 B）

支撑"病害-生育期匹配"（稻瘟病→发生于→孕穗期）与"病害-季节/区域匹配"时空合理性校验（6月华南-稻瘟病 √、冬季北方-稻瘟病 ✗）。物候事件承载物候/季节-病害映射。

### 5.3 环境时序数据编码（保留 B）

实验站温/湿/光/降雨时序数据经双层 GRU 编码；环境主线数据 500 组 = 真实传感器 200 + 文献 180 + 规则构造 120（来源透明化，实验 3 对比"仅真实+文献"与"含构造"）。

---

## 6. 因果规则库（保留 B）

- 以 NY/T 标准为支撑，形式为"环境条件 → 病害风险"因果链；
- 示例：高温高湿 → 稻瘟病风险↑（√）；冬季北方 → 稻瘟病（✗）；
- **与关系层打通**：因果规则可自动实例化为 `r_favor`（ENV→DISEASE）或用于校验 `r_depend_on` 配对，同时服务于"农业因果合理性"约束维度与事件级"因果一致性"对齐。

---

## 7. 知识单元七元组（修订）

### 7.1 定义

```
ku = (s, r, o, t, l, c, m)
s  主体实体        ∈ 前 8 类语义实体
r  关系类型(10类) 或 事件类型(8类)
o  客体实体        ∈ 前 8 类语义实体 ∪ {LOCATION, TIME}
t  (t_start, t_end, stage)     时间三元组
l  (province, lon, lat)        空间三元组
c  [0,1]                       融合置信度
m  (m_text, m_img, m_env)      模态可用性标记
```

### 7.2 修订：r 的双态判别（解决 B"r 既是关系又是事件"的歧义）

B 的七元组中 r 同时承载关系与事件类型但未显式判别。融合版新增判别字段：

```
ku_type ∈ {REL, EVENT}
  REL   → r 为关系类型，ku 即三元组 (s, r, o) + 时空/置信度/模态
  EVENT → r 为事件类型，s/o/t/l 按事件槽位填充（§4.2），即事件假设
```

同一语义在两种知识单元下的表达示例：
- REL：`ku_type=REL, r=treat, s=三环唑, o=稻瘟病`
- EVENT：`ku_type=EVENT, r=防治, 槽位{C=水稻, D=稻瘟病, TREATMENT=三环唑, τ=2026-06}`
- 关系是"边"，事件是"一段带时空的复合过程"，二者可在图谱中互转（事件主体展开即关系组）。

### 7.3 与已有时空知识图谱三元组的区别（保留 B 表述）

传统方式为"三元组+时间戳"；七元组内嵌时间/空间三元组、融合置信度 c 与模态标记 m，支持溯源与消融；r 承载关系/事件双态；可无损映射为图谱节点、边与属性。

---

## 8. 修订后的合法配对表与 Mask

### 8.1 实体类型兼容性 M_type（修订后）

**10 类实体 × 10 类实体 = 100 种可能配对，合法 20 种，ρ = 20/100 = 0.20**（较 B 的 0.28 更稀疏）。

| 主体 \ 客体 | CROP | DISEASE | PEST | PATHOGEN | SYMPTOM | TREATMENT | ENV | STAGE | LOC | TIME |
|---|---|---|---|---|---|---|---|---|---|---|
| **CROP** | – | r_resist | r_resist | – | – | – | – | – | – | – |
| **DISEASE** | r_affects | – | – | r_caused_by | r_exhibit | – | r_depend_on | r_occur_in | r_spread_in | r_occur_in |
| **PEST** | r_affects | – | – | – | r_exhibit | – | r_depend_on | r_occur_in | r_spread_in | r_occur_in |
| **PATHOGEN** | – | – | – | – | – | – | – | – | – | – |
| **SYMPTOM** | – | – | – | – | – | – | – | – | – | – |
| **TREATMENT** | – | r_treat | r_treat | – | – | – | – | – | – | – |
| **ENV_FACTOR** | r_influence | r_favor | r_favor | – | – | – | – | – | – | – |
| **GROWTH_STAGE** | – | – | – | – | – | – | – | – | – | – |
| **LOCATION** | – | – | – | – | – | – | – | – | – | – |
| **TIME** | – | – | – | – | – | – | – | – | – | – |

**合法配对清单（20 条，逐条对应关系）**：

| # | 主体→客体 | 关系 | # | 主体→客体 | 关系 |
|---|---|---|---|---|---|
| 1 | DISEASE→CROP | r_affects | 11 | PEST→SYMPTOM | r_exhibit |
| 2 | PEST→CROP | r_affects | 12 | DISEASE→PATHOGEN | r_caused_by |
| 3 | ENV→CROP | r_influence | 13 | TREATMENT→DISEASE | r_treat |
| 4 | ENV→DISEASE | r_favor | 14 | TREATMENT→PEST | r_treat |
| 5 | ENV→PEST | r_favor | 15 | DISEASE→GROWTH_STAGE | r_occur_in |
| 6 | DISEASE→SYMPTOM | r_exhibit | 16 | PEST→GROWTH_STAGE | r_occur_in |
| 7 | DISEASE→ENV | r_depend_on | 17 | DISEASE→TIME | r_occur_in |
| 8 | PEST→ENV | r_depend_on | 18 | PEST→TIME | r_occur_in |
| 9 | DISEASE→LOCATION | r_spread_in | 19 | CROP→DISEASE | r_resist |
| 10 | PEST→LOCATION | r_spread_in | 20 | CROP→PEST | r_resist |

> 注：`r_favor`（ENV→DISEASE/PEST）与 `r_depend_on`（DISEASE/PEST→ENV）为逆语义，实现时可只存一侧，M_type 两向对称置 1（#4/#5 与 #7/#8）。

### 8.2 关系合法性 M_rel（保留 B 机制，扩展关系集）

```
M_rel[i, j] = 1   （当 rel(i,j) ∈ R_valid，10 类核心关系）
M_rel[i, j] = 0.1 （否则，软惩罚——保留长尾容错，不直接否决）
```

### 8.3 Mask 自动构造与稀疏性

- M_type 由 §3.1 关系主客体约束**自动生成**（10×10 常数矩阵，可预计算缓存）；
- 稀疏性 ρ=0.20，较 B 的 0.28 进一步提升：实体类型扩展（8→10）后合法配对仅 20/100，LLM 生成与跨模态对齐的候选假设空间被进一步压缩；
- OBSERVATION 观测实体不进入核心语义 Mask（其 4 类观测关系单独定义），保证语义 Mask 的稀疏性不被观测层稀释。

---

## 9. JSON Schema（修订版）

完整文件见 `knowledge-unit.schema.json`。相对 B 附录 B Schema 的主要修订：

| 修订点 | 说明 |
|---|---|
| `entity_type` 枚举 | 8 类 → **10 类**（+PEST、PATHOGEN） |
| `relation` 枚举 | 6 类 → **10 类**（+affects、influence、favor、caused_by、resist；occur_in 语义修订） |
| `event_type` 枚举 | 5 类 → **8 类**（+虫害发生/生长发育/品质形成） |
| 新增 `ku_type` | `REL / EVENT` 判别（解决 r 双态歧义） |
| 新增 `slot` 槽位对象 | 事件知识单元必填，按 §4.2 校验（如 EVENT=病害发生 → 必须含 D、S 槽） |
| 新增 `evidence` 证据数组 | 多模态溯源：图像/文本/环境证据 URI + 模态标记（吸收 A 的 hasEvidence） |
| 新增 `uri` / `aliases` | 实体唯一标识与同义归一（对齐 AGROVOC/CO） |
| 槽位值域约束 | C 槽 ∈ CROP、D 槽 ∈ DISEASE、A 槽 ∈ ENV_FACTOR 等（`$ref` 校验） |

---

## 10. 与 B、A 的对应关系总表

| 模块 | 保留 B | 吸收 A | 融合新增/修订 |
|---|---|---|---|
| 实体层 | 8 类框架、角色划分、图谱角色 | 病原/虫害/症状部位/时空类 | PEST、PATHOGEN、OBSERVATION 增强 |
| 关系层 | 6 类骨架、主客体约束、Mask | 病三角拆分、46 属性精选 | affects、caused_by、resist、influence/favor 拆分、occur_in 修复 |
| 事件层 | 5 类、三模态假设、槽位对齐 | 8 类事件类型/实例分层 | 事件与关系/实体边界、ku_type、8 类模板 |
| 时空 | 时间/空间三元组、生育期锚点 | TimeEntity/SpatialLocation 类 | 双轨制（内嵌三元组+图谱类） |
| 七元组 | (s,r,o,t,l,c,m) 定义 | – | ku_type 判别、evidence/uri 字段 |
| Mask | M_type/M_rel、软惩罚、预计算 | – | 10×10、ρ=0.20、自动构造 |
| 因果规则库 | NY/T 规则 | – | 与 r_favor/r_depend_on 打通 |

---

## 11. 落图与工程接入

1. **Mask 预计算**：按 §8 生成 10×10 M_type 与 M_rel 常数矩阵，缓存；
2. **标注映射**：80 篇语料按 10 实体 / 10 关系 / 8 事件 / 时空槽位标注，产出证据库 E；
3. **抽取对齐**：GLiNER2 实体锚点（10 类）→ 关系抽取映射 10 类 R_valid → 事件抽取按 §4.2 模板生成 Event 型知识单元；
4. **三模态融合**：文本（Qwen3-8B+BGE-M3）、图像（DINOv3+Qwen3-VL-2B）、环境（双层 GRU）各自生成事件假设，在五维约束下求最大一致赋值（沿用 B 创新2）；
5. **图谱构建**：知识单元经 φ 映射为 `v_s ─[e_r, a_t, a_l, a_c]─ v_o`，供 GraphRAG 下游检索与推理；
6. **Hard Negative**：实体混淆、关系混淆、时空偏移、模态缺失四类结构化负例（对应修订后关系集重新生成，如"香蕉-稻瘟病""疾病-生长阶段"）。
