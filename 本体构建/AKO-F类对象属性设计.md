# AKO-F 类对象属性设计（数据属性，供审查）

> 区分清楚：本文件设计的是**类对象的属性（数据属性 / Data Property）**——描述每个类对象的特征字段（如"作物.品种"、"病害.严重度"、"温度.数值"）。
> **不是**实例间关系（对象属性，如稻瘟病→危害→水稻）。对象属性（r_affects 等）是另一层，本次不涉及，待你确认后再单独设计。
> 原则：精炼、不冗余、每个字段服务于农业 KG 的查询/推理/展示。

---

## 一、通用属性（所有实体类 + 事件类共用）

这三个属性所有类都继承，后面各类**不重复列**。

| 属性 | 中文 | range | 用途 |
|---|---|---|---|
| `name` | 名称 | string | 规范名（水稻、稻瘟病、三环唑） |
| `aliases` | 别名 | string | 同义归一（稻瘟病=稻热病=稻瘟） |
| `description` | 描述 | string | 该对象的文字说明 |

> 这样避免每类各设一个 `crop_name`/`disease_name`/`pest_name` 的冗余；如你希望每类有独立命名属性，告诉我改。

---

## 二、实体层各类特有属性

### 2.1 CROP 作物
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `variety` | 品种 | string | 具体品种（汕优63、扬麦16） |
| `growth_days` | 全生育期天数 | int | 全生育期长度（120 天） |

### 2.2 DISEASE 病害
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `severity_grade` | 严重度 | string | 轻/中/重 |
| `peak_season` | 易发季节 | string | 春/夏/秋 |

### 2.3 PEST 虫害
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `pest_order` | 分类目 | string | 鳞翅目/同翅目/鞘翅目 |
| `peak_period` | 发生高峰期 | string | 6 月下旬—7 月上旬 |

### 2.4 PATHOGEN 病原
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `pathogen_kingdom` | 类群 | string | 真菌/细菌/病毒/线虫 |

### 2.5 SYMPTOM 症状
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `symptom_form` | 形态 | string | 斑点/枯萎/腐烂/畸形 |
| `color` | 颜色 | string | 褐、黑、黄 |
> 部位（叶/茎/穗/根/果）**已由子类 FoliarSymptom 等表达**，不再设 `position` 属性，避免冗余。

### 2.6 TREATMENT 防治措施
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `treatment_category` | 类别 | string | 化学/农艺/生物 |
| `dosage` | 用药量 | string | 20 g/亩 |
| `usage_method` | 用法 | string | 喷雾/拌种/灌根 |

### 2.7 ENV_FACTOR 环境因子
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `factor_value` | 数值 | string | 25（温度值） |
| `unit` | 单位 | string | ℃/%/mm |
> 温度/湿度/光照等 6 个子类**共用** `factor_value`+`unit`，不再各设 `temp_value`/`humidity_value`，避免冗余。

### 2.8 GROWTH_STAGE 生育期
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `stage_code` | 阶段编码 | string | GS-03（分蘖期） |
| `typical_days` | 典型天数 | int | 该阶段通常持续天数 |

### 2.9 LOCATION 空间
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `province` | 省份 | string | 广东省 |
| `longitude` | 经度 | decimal | 113.27 |
| `latitude` | 纬度 | decimal | 23.13 |
> 子类 Region/Plot/Facility 继承以上属性；Plot 可加 `plot_id` 地块编号、`area` 面积（见 §四待确认）。

### 2.10 TIME 时间
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `time_value` | 时间值 | dateTime/string | 2026-06-15 |
| `start_time` | 开始 | dateTime | 区间起点 |
| `end_time` | 结束 | dateTime | 区间终点 |
> TimePoint 用 `time_value`；TimeInterval 用 `start_time`+`end_time`；PhenologicalPeriod 用通用 `name`。类型由子类表达，不设 `time_type` 字段。

### 2.11 OBSERVATION 观测
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `obs_id` | 观测编号 | string | OBS-1023 |
| `obs_timestamp` | 时间戳 | dateTime | 观测发生时间 |
| `data_source` | 数据源 | string | 田间传感器/文献/人工记录 |

**MediaAsset 媒体资产**（OBSERVATION 子类）
| `asset_uri` | 资产地址 | string | 文件路径/URL |

**ImageAsset 图像**
| `image_format` | 格式 | string | jpg/png/tif |
| `resolution` | 分辨率 | string | 4032×3024 |

**SensorSeries 传感器时序**
| `sensor_type` | 传感器类型 | string | 温湿度计/雨量计 |
| `sampling_rate` | 采样率 | string | 1 次/小时 |

**TextDocument 文本资料**
| `text_content` | 文本内容 | string | 报告/记录正文 |
| `source_url` | 来源URL | string | 文献来源链接 |

---

## 三、事件层属性

### EventType 事件类型（及其 8 个子类共用）
| 属性 | 中文 | range | 说明/示例 |
|---|---|---|---|
| `event_id` | 事件编号 | string | EVT-2026-001 |
| `occurrence_time` | 发生时间 | dateTime | 事件发生时间 |
| `event_description` | 事件描述 | string | 文字摘要 |

> 事件的"槽位"（作物/病害/症状等关联实体）本质是**对象属性**（事件→实体），不是数据属性，本版不设。待你确认后再单独设计。

---

## 四、属性统计与待确认点

**统计**：通用 3 + 实体特有约 30 + 事件 3，共约 36 个数据属性（多数类 2–3 个字段，精炼）。

**请你确认：**
1. **通用 `name`** vs **每类专用名**（如 `crop_name`）：同意共用 `name`？还是每类独立命名？
2. **症状部位**：同意靠子类表达、不设 `position` 字段？
3. **环境 6 子类**：同意共用 `factor_value`+`unit`？还是每个子类要专属字段（如温度加 `temp_range`）？
4. **Plot 地块**：是否加 `plot_id` 地块编号、`area` 面积两个子类专属属性？
5. **是否要溯源字段**：每个对象是否加 `confidence` 置信度、`source` 来源（即使不建知识单元类）？
6. **事件槽位**：是否让我接着设计事件→实体的对象属性（r_affects 那套关系）？

逐条告诉我取舍，我再据此把数据属性写入 OWL 本体。
