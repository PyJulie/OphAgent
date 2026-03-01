# OphAgent — 详细技术参考文档（中文）

<div align="center">

**Language / 语言**

[**🇨🇳 中文**](./README_zh.md) | [🇬🇧 English](./README_en.md) | [📖 搭建指南](./GUIDE_zh.md)

</div>

---

## 目录

1. [概述](#1-概述)
2. [系统架构](#2-系统架构)
3. [核心组件](#3-核心组件)
4. [工具池](#4-工具池)
5. [模型调度](#5-模型调度)
6. [知识库](#6-知识库)
7. [创新策略](#7-创新策略)
8. [LLM 主干](#8-llm-主干)
9. [安装](#9-安装)
10. [配置](#10-配置)
11. [构建知识库](#11-构建知识库)
12. [训练模型](#12-训练模型)
13. [运行服务](#13-运行服务)
14. [使用智能体](#14-使用智能体)
15. [评估](#15-评估)
16. [完整项目结构](#16-完整项目结构)
17. [功能清单与差距分析](#17-功能清单与差距分析)
18. [扩展指南](#18-扩展指南)

---

## 1. 概述

**OphAgent** 是一个以大型语言模型（LLM）为核心驱动的眼科临床 AI 智能体。它将专用眼科视觉模型池与结构化的 **Planner → Executor → Verifier** 流水线相结合，从视网膜图像和自然语言查询中生成可靠、有据可查的临床报告。

### 关键设计目标

| 设计目标 | 实现机制 |
|---------|---------|
| 多任务眼科分析 | 21+ 个跨 5 种模态的专用工具封装 |
| 可靠的医疗输出 | RAG 引导的 Verifier，具备冲突解决能力 |
| 零样本可扩展性 | 证据引导 CLIP（§2.4.1）|
| 无分割标签的区域 VQA | 可组合工具式 VQA（§2.4.2）|
| 无标注图像库检索 | 多尺度 RAG（§2.4.3）|
| 依赖冲突模型的插拔化 | 双模式调度（Conda / Docker+FastAPI）|
| 长期临床记忆 | FAISS 持久化病例存储 |

### 支持的图像模态

| 缩写 | 全称 | 代表性疾病 |
|------|------|----------|
| CFP | 彩色眼底照相（Colour Fundus Photography）| 糖尿病视网膜病变、AMD、青光眼、RVO、近视 |
| OCT | 光学相干断层扫描（Optical Coherence Tomography）| AMD、DME、CNV、玻璃膜疣 |
| UWF | 超广角眼底（Ultra-Widefield Fundus）| 周边变性、糖网病 |
| FFA | 荧光素眼底血管造影（Fundus Fluorescein Angiography）| PDR、RVO、CNV |

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                       用户 / 临床医生                         │
│            "分析这张眼底图像中的糖尿病视网膜病变"              │
└───────────────────────────┬──────────────────────────────────┘
                            │  查询 + 图像路径
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    OphAgent 编排器                            │
│  ┌──────────┐  计划  ┌──────────┐ 结果  ┌──────────────┐   │
│  │ Planner  │───────▶│ Executor │──────▶│  Verifier    │   │
│  │  (LLM)   │        │          │       │  (LLM + RAG) │   │
│  └──────────┘        └────┬─────┘       └──────┬───────┘   │
│       ▲                   │                     │ 裁定结果   │
│       │        ┌──────────────────────────────────────────┐ │
│       └────────│           记忆管理器                      │ │
│                │  短期记忆（会话缓冲区）长期记忆（FAISS）   │ │
│                └──────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────────────────────┘
               │ 工具调用
               ▼
┌──────────────────────────────────────────────────────────────┐
│                         工具池                               │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐ │
│  │ 分类工具   │  │  分割工具  │  │ 检测工具 │  │ VQA工具 │ │
│  │  (9个)     │  │   (2个)    │  │  (2个)   │  │  (2个)  │ │
│  └────────────┘  └────────────┘  └──────────┘  └─────────┘ │
│  ┌────────────┐  ┌────────────┐                             │
│  │ CLIP 模型  │  │  辅助工具  │  调度器：inline|FastAPI|Conda│
│  │   (2个)    │  │   (4个)    │                             │
│  └────────────┘  └────────────┘                             │
└──────────────────┬───────────────────────────────────────────┘
                   │
       ┌───────────┴──────────────┐
       ▼                          ▼
┌────────────────┐      ┌───────────────────────┐
│    知识库      │      │       创新策略          │
│  ├ 本地档案    │      │  ├ 证据引导 CLIP         │
│  ├ 教科书      │      │  ├ 可组合工具 VQA        │
│  ├ 搜索引擎    │      │  └ 多尺度 RAG            │
│  └ 交互输入    │      └───────────────────────┘
│  (FAISS 统一)  │
└────────────────┘
```

---

## 3. 核心组件

### 3.1 Planner（规划器）

**文件：** `ophagent/core/planner.py`

Planner 接收自然语言查询，将其转换为结构化的 `ExecutionPlan`——一个有序的、考虑依赖关系的工具调用步骤列表。

#### 关键类

| 类名 | 作用 |
|------|------|
| `PlanStep` | 单个步骤：`tool_name`、`inputs`、`depends_on` |
| `ExecutionPlan` | `PlanStep` 的有序列表，支持拓扑排序 |
| `Planner` | LLM 驱动的计划生成器 |

#### 规划流程

1. 从 `config/tool_registry.yaml` 加载工具描述。
2. 构建包含所有可用工具列表的系统提示词。
3. 携带用户查询和图像上下文调用 LLM。
4. 将返回的 JSON 解析为 `PlanStep` 对象。
5. 验证依赖图；使用 Kahn 拓扑排序确定执行顺序。
6. 若 LLM 未返回任何步骤，则回退到 `web_search`。

#### 输出 JSON 格式

```json
{
  "steps": [
    {"step_id": 1, "tool_name": "cfp_quality",  "inputs": {"image_path": "fundus.jpg"}, "description": "评估图像质量", "depends_on": []},
    {"step_id": 2, "tool_name": "cfp_disease",  "inputs": {"image_path": "fundus.jpg"}, "description": "分类视网膜疾病", "depends_on": [1]},
    {"step_id": 3, "tool_name": "gradcam",      "inputs": {"image_path": "fundus.jpg", "model": "cfp_disease"}, "description": "生成可视化热图", "depends_on": [2]},
    {"step_id": 4, "tool_name": "synthesise",   "inputs": {"summary": "综合报告"}, "description": "生成最终报告", "depends_on": [1,2,3]}
  ]
}
```

`synthesise` 步骤始终位于最后，由编排器处理，不由 Executor 直接执行。

---

### 3.2 Executor（执行器）

**文件：** `ophagent/core/executor.py`

按拓扑顺序执行每个 `PlanStep`。自动跳过依赖步骤失败的后续步骤，支持动态输入引用。

#### 动态输入引用

步骤可通过 `${step_N.output.key}` 语法引用前序步骤的输出：

```json
{"image_path": "${step_1.output.cropped_path}"}
```

`_resolve_inputs()` 方法在运行时遍历结果存储并替换这些引用，实现链式流水线（例如：裁剪 → 分类 → 解释）。

#### 步骤结果格式

```python
StepResult(
    step_id=2,
    tool_name="cfp_disease",
    output={"labels": ["DR"], "probabilities": {"DR": 0.94, ...}},
    error=None,
    duration_s=0.82,
    success=True,
)
```

---

### 3.3 Verifier（验证器）

**文件：** `ophagent/core/verifier.py`

在生成报告之前，验证 Executor 输出的医疗一致性。

#### 验证工作流

```
Executor 输出结果
      │
      ▼
LLM（验证器系统提示 + RAG 上下文）
      │
      ▼
裁定结果：{valid, confidence, conflicts, resolution, verified_result}
      │
      ├── confidence < 0.6 → 标记 needs_human_review = True
      │
      └── 存在冲突 → 扩展 KB 搜索 → 重新验证（最多 2 次）
```

#### LLM 执行的医疗一致性检查
- 疾病分级与分类不相矛盾（例如"无 DR"与"PDR 活跃"不能同时出现）
- 质量评分与分析置信度匹配
- 分割面积与报告结果合理吻合
- 若提供双眼图像，检查双侧一致性

#### 图像质量门控

`Verifier.check_quality_gate(quality_result)` 在以下情况返回 `False`（阻断后续分析）：
- `quality_score < 0.4`
- `quality_label` 包含 "ungradable"（不可分级）

---

### 3.4 Memory Manager（记忆管理器）

**文件：** `ophagent/core/memory.py`

两级记忆架构：

| 层级 | 类 | 作用域 | 存储方式 |
|------|-----|-------|---------|
| 短期记忆 | `ShortTermMemory` | 会话 | 内存环形缓冲区（可配置最大轮次数） |
| 长期记忆 | `LongTermMemory` | 持久化 | FAISS 索引 + JSONL 元数据文件 |

#### 长期记忆流程

```
已完成的会话
      │
      ▼
LLM（记忆整合提示词）
      │
      ▼
MemoryEntry: {entry_id, summary, key_findings, tools_used, modalities, tags}
      │
      ▼
向量化嵌入 → FAISS 索引 → 保存至磁盘
```

未来的查询通过对嵌入摘要的余弦相似度搜索来检索相关历史病例。

---

### 3.5 OphAgent（编排器）

**文件：** `ophagent/core/agent.py`

`OphAgent` 类整合所有组件，是主要的用户接口。

#### 完整运行循环

```
run(query, image_paths)
  │
  ├── 1. add_turn("user", query)           # 存入短期记忆
  ├── 2. 从长期记忆检索历史病例             # RAG 上下文
  ├── 3. planner.plan(query, images)        # 生成执行计划
  ├── 4. executor.execute(plan)             # 调用工具
  ├── 5. verifier.verify(results, query)   # 验证结果
  │        └── 若存在冲突 → planner.replan() → 重新执行
  ├── 6. check_quality_gate(quality)        # 拦截低质量图像
  ├── 7. synthesise_report(...)             # LLM 生成临床报告
  ├── 8. add_turn("assistant", report)     # 存储响应
  └── 9. consolidate_session()             # 持久化至长期记忆
```

#### `AgentResponse` 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | str | 原始查询 |
| `report` | str | 最终临床报告 |
| `verdict` | Verdict | Verifier 输出 |
| `plan` | ExecutionPlan | 生成的执行步骤 |
| `raw_results` | dict | 各步骤原始工具输出 |
| `duration_s` | float | 墙钟时间（秒） |
| `needs_human_review` | bool | 低置信度标记 |

---

## 4. 工具池

### 4.1 复用模型（9 个）

| 工具 ID | 名称 | 模态 | 任务 | 调度方式 | 文件 |
|---------|------|------|------|---------|------|
| `fmue` | FMUE | OCT | B-scan 分类（正常/AMD/DME/CNV/玻璃膜疣）| Conda | `tools/classification/oct_fmue.py` |
| `uwf_mdd` | UWF-MDD | UWF | 多病变检测 | Conda | `tools/classification/uwf_mdd.py` |
| `uwf_multi_abnormality` | UWF 多异常 | UWF | 多标签异常筛查 | Conda | `tools/classification/uwf_multi_abnormality.py` |
| `retizero` | RetiZero | CFP | CLIP 零样本（英文）| Inline | `tools/clip_models/retizero.py` |
| `vilref` | ViLReF | CFP | CLIP 零样本（中文）| Inline | `tools/clip_models/vilref.py` |
| `fundus_expert` | FundusExpert | CFP | VQA 问答 | FastAPI :8101 | `tools/vqa/fundus_expert.py` |
| `vision_unite` | VisionUnite | CFP | VQA 问答 | FastAPI :8102 | `tools/vqa/vision_unite.py` |
| `automorph` | AutoMorph | CFP | 血管分割 + 形态定量 | Conda | `tools/segmentation/automorph.py` |
| `retsam` | RetSAM | CFP | SAM 适配视网膜分割 | Inline | `tools/segmentation/retsam.py` |

> **AutoMorph 输出指标**：血管掩膜、分形维数、血管密度、动静脉比、迂曲度。

---

### 4.2 新开发模型（8 个）

| 工具 ID | 模态 | 任务 | 端口 | 主干网络 | 训练器 |
|---------|------|------|------|---------|-------|
| `cfp_quality` | CFP | 图像质量评估（差/中/好 + 推理说明）| 8110 | EfficientNet-B4 | `CFPQualityTrainer` |
| `cfp_disease` | CFP | 多标签视网膜疾病分类 | 8111 | ConvNeXt-Base | `CFPDiseaseTrainer` |
| `cfp_ffa_multimodal` | CFP+FFA | 双模态联合分类 | 8112 | 双 ViT-Small + 交叉注意力 | `CFPFFAMultimodalTrainer` |
| `uwf_quality_disease` | UWF | 质量评估 + 疾病分类（多任务）| 8113 | ResNet-50，双头 | `UWFQualityDiseaseTrainer` |
| `cfp_glaucoma` | CFP | 可转诊青光眼 + 结构特征（CDR/视杯面积）| 8114 | Swin-Transformer-Small | `CFPGlaucomaTrainer` |
| `cfp_pdr` | CFP | PDR 活动性分级（4 级）| 8115 | EfficientNet-B5 | `CFPPDRTrainer` |
| `ffa_lesion` | FFA | 多病灶目标检测 | 8120 | Faster R-CNN ResNet-50-FPN | `FFALesionTrainer` |
| `disc_fovea` | CFP | 视盘 + 中心凹关键点定位 | 8121 | HRNet-W32 + 回归头 | `DiscFoveaTrainer` |

#### 输出示例

**CFP 质量评估：**
```json
{
  "quality_label": "Good",
  "quality_score": 0.91,
  "probabilities": {"Bad": 0.02, "Moderate": 0.07, "Good": 0.91}
}
```

**FFA 多病灶检测：**
```json
{
  "detections": [
    {"label": "microaneurysm", "bbox": [120, 340, 145, 365], "confidence": 0.87},
    {"label": "haemorrhage",   "bbox": [220, 180, 290, 230], "confidence": 0.79}
  ],
  "lesion_types": ["microaneurysm", "haemorrhage"]
}
```

**视盘与中心凹定位：**
```json
{
  "disc_center": [512, 490],
  "disc_radius": 88.5,
  "fovea_center": [820, 502]
}
```

---

### 4.3 辅助工具（4 个）

| 工具 ID | 任务 | 调度方式 | 文件 |
|---------|------|---------|------|
| `gradcam` | Grad-CAM 显著性热图（适用于任意 CNN 模型）| Inline | `tools/auxiliary/gradcam.py` |
| `roi_cropping` | 图像感兴趣区域裁剪 | Inline | `tools/auxiliary/roi_cropping.py` |
| `ocr_detector` | 图像文字识别（EasyOCR / Tesseract）| Inline | `tools/auxiliary/ocr_detector.py` |
| `web_search` | 领域受限文献检索（PubMed、arXiv）| Inline | `tools/auxiliary/web_search.py` |

---

## 5. 模型调度

三种调度模式由 `ToolScheduler`（`ophagent/tools/scheduler.py`）统一管理：

```
工具调用请求
      │
      ├── scheduling_mode == "inline"
      │       └── 在主进程内一次性加载，缓存实例
      │
      ├── scheduling_mode == "fastapi"
      │       └── HTTP POST 到预加载的 Docker 微服务
      │           （等待 /health 端点，然后 POST /run）
      │
      └── scheduling_mode == "conda"
              └── 子进程：source conda.sh → activate env → python script.py '{json}'
                  （从 stdout 解析 JSON 结果）
```

### 为什么需要两种非 inline 模式？

| 模式 | 适用场景 | 优势 |
|------|---------|------|
| **FastAPI / Docker** | 新开发的大型模型（≥6 GB 显存）| 始终热启动、低延迟、隔离 GPU 内存 |
| **Conda 子进程** | 依赖冲突的复用模型（不同 PyTorch/CUDA 版本）| 真正的依赖隔离，无容器开销 |

### 端口分配

| 服务 | 端口 |
|------|------|
| FundusExpert | 8101 |
| VisionUnite | 8102 |
| cfp_quality | 8110 |
| cfp_disease | 8111 |
| cfp_ffa_multimodal | 8112 |
| uwf_quality_disease | 8113 |
| cfp_glaucoma | 8114 |
| cfp_pdr | 8115 |
| ffa_lesion | 8120 |
| disc_fovea | 8121 |

---

## 6. 知识库

**文件：** `ophagent/knowledge/knowledge_base.py`

统一的 RAG 存储，将四类来源聚合到同一个 FAISS 索引中。

### 数据来源

| 来源 | 类 | 内容 | 检索类型 |
|------|----|------|---------|
| 本地档案 | `ImageReportArchive` | 历史病例：图像 + 临床报告对 | 文本（报告）+ 图像（CLIP）|
| 操作规范 | `OperationalStandards` | 临床 SOP、筛查指南 | 文本 |
| 教科书 | `TextbookSource` | 眼科教科书（PDF/TXT）| 文本 |
| 搜索引擎 | `SearchEngineSource` | PubMed/arXiv 实时检索结果 | 文本（检索后索引）|
| 交互输入 | `InteractiveSource` | 临床医生提供的会话上下文 | 文本 |

### 向量存储详情（`ophagent/knowledge/vector_store.py`）

| 方面 | 实现 |
|------|------|
| 索引类型 | FAISS `IndexFlatIP`（单位向量上的内积 = 余弦相似度）|
| 文本嵌入 | `sentence-transformers/all-MiniLM-L6-v2`（384维 → 零填充至 512维）|
| 图像嵌入 | OpenCLIP `ViT-B/32`（512维）|
| 统一维度 | 512（所有向量在同一索引中）|
| 持久化 | `.index` 文件（FAISS）+ `.jsonl` 元数据文件 |

### 本地档案目录结构

```
data/local_archive/
├── cases/
│   ├── case_001/
│   │   ├── fundus.jpg
│   │   └── report.txt
│   └── case_002/
│       ├── oct.jpg
│       └── report.txt
└── standards/
    ├── dr_screening_guideline.pdf
    └── amd_treatment_protocol.txt
```

---

## 7. 创新策略

### 7.1 证据引导 CLIP 零样本分类（§2.4.1）

**文件：** `ophagent/strategies/clip_evidence.py`

**问题：** 标准 CLIP 零样本分类使用原始标签名称（如"diabetic retinopathy"）作为文本提示，粒度粗，缺乏视觉特异性。

**解决方案：** 对每个候选标签，由 LLM 生成 K 个（默认 7 个）临床精准的视觉证据描述词，每个描述词是一段简短的短语，描述该疾病在图像中的具体视觉表现。

**算法流程：**

```
对 candidate_labels 中的每个标签 L：
  1. LLM 生成 K 个证据描述词：
     ["黄斑区明亮的黄色硬性渗出",
      "中周部点状和片状出血",
      "视盘新生血管", ...]

  2. CLIP 文本编码器嵌入所有 K 个描述词 → T_k  (K × D)

  3. CLIP 图像编码器嵌入查询图像 → I  (1 × D)

  4. 逐描述词相似度：S = I · T_k^T  (K,)

  5. 标签得分 = mean(S) 或 max(S)

对所有标签完成后：
  6. 对标签得分做 Softmax → 概率分布
  7. 返回最高概率标签 + 每标签的证据列表
```

---

### 7.2 可组合工具式 VQA（§2.4.2）

**文件：** `ophagent/strategies/vqa_composable.py`

**问题：** VQA 模型回答整图问题时，对于特定视网膜结构的针对性问题（在没有专门分割模型时）表现较差。

**解决方案：** 基于关键词路由，选择多步组合流水线：定位 → 裁剪 → 对特定区域执行 VQA。

**内置组合规则：**

| 触发关键词 | 流水线 |
|-----------|-------|
| "optic disc"、"cup disc"、"rim"、"RNFL" | `disc_fovea` → `roi_cropping` → `fundus_expert` |
| "macula"、"fovea"、"drusen" | `disc_fovea` → `roi_cropping`（黄斑区域）→ `vision_unite` |
| *（默认）* | `fundus_expert`（全图） |

**示例（问题："What is the cup-to-disc ratio?"）：**

```
步骤 1: disc_fovea(fundus.jpg) → {disc_center: [512,490], disc_radius: 88.5}
步骤 2: roi_cropping(fundus.jpg, x=380, y=358, w=265, h=265) → {cropped_b64: "..."}
步骤 3: fundus_expert(cropped_b64, "What is the cup-to-disc ratio?") → {answer: "~0.6"}
```

---

### 7.3 多尺度 RAG（无标注数据库）（§2.4.3）

**文件：** `ophagent/strategies/multiscale_rag.py`

**问题：** 现有 RAG 系统需要文本标签或报告才能进行检索，而大多数临床图像档案没有标注。

**解决方案：** 多尺度检索结合三种粒度，实现无需任何标签的图像到图像相似度检索：

| 尺度 | 查询方式 | 说明 |
|------|---------|------|
| 全局图像 | 查询图像的完整 CLIP 嵌入 | 患者级相似性：查找视觉上最相似的历史病例 |
| 区域图像 | 裁剪 ROI 的 CLIP 嵌入（通过 `disc_fovea` 获取视盘区域）| 结构级相似性：查找视盘外观相似的病例 |
| 文本 | 查询字符串的文本嵌入 | 概念级相似性：匹配报告和指南 |

**得分融合：**

```
fused_score(doc) = Σ (各尺度相似度 × 尺度权重)

默认权重：global_image=0.5, regional_image=0.3, text=0.2
```

**索引无标注图像：**
```bash
python scripts/build_knowledge_base.py --unlabelled /path/to/image/archive
```

无需任何文本标签，查询时可通过 CLIP 视觉相似度检索相似图像。

---

## 8. LLM 主干

**文件：** `ophagent/llm/backbone.py`

提供统一的跨 LLM 提供商接口：

| 提供商 | 类 | 支持模型 | 默认 |
|--------|-----|---------|------|
| OpenAI | `OpenAILLM` | `gpt-5`、`gpt-4o` | ✅ `gpt-5` |
| Google Gemini | `GeminiLLM` | `gemini-3.0-pro`、`gemini-1.5-pro`、`gemini-2.0-flash` | — |
| 本地（Ollama）| `OpenAILLM`（含 `base_url`）| `qwen2.5:72b`、`llama3.1:70b` | — |

### 提示词库（`ophagent/llm/prompts.py`）

| 提示词 | 使用者 | 用途 |
|--------|--------|------|
| `PLANNER_SYSTEM` | Planner | 工具列表 + 计划格式规范 |
| `PLANNER_USER` | Planner | 查询 + 图像路径 + 会话历史 |
| `VERIFIER_SYSTEM` | Verifier | 医疗一致性检查角色定义 |
| `VERIFIER_USER` | Verifier | 工具输出 + KB 上下文 |
| `SYNTHESIS_SYSTEM` | OphAgent | 资深眼科医生报告撰写者角色 |
| `SYNTHESIS_USER` | OphAgent | 验证后结果 + KB 上下文 |
| `MEMORY_CONSOLIDATION_SYSTEM` | Memory | 会话 → 结构化记忆条目 |
| `CLIP_EVIDENCE_SYSTEM` | EvidenceGuidedCLIP | 视觉证据描述词生成器 |

---

## 9. 安装

### 环境要求

- Python ≥ 3.10
- CUDA 11.8+（推荐，用于 GPU 加速）
- Conda（用于按需 Conda 工具）
- Docker + NVIDIA Container Toolkit（用于 FastAPI 服务）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/OphAgent.git
cd OphAgent

# 2. 创建并激活虚拟环境
conda create -n ophagent python=3.10
conda activate ophagent

# 3. 安装依赖
pip install -r requirements.txt

# 4. 以开发模式安装包
pip install -e .

# 5. 创建环境变量文件
cp .env.example .env
# 编辑 .env 填入您的 API 密钥
```

### 按需 Conda 工具环境

```bash
# FMUE 环境
conda create -n fmue_env python=3.9
conda activate fmue_env
pip install torch==1.13.0 torchvision==0.14.0 timm==0.6.12 Pillow numpy

# AutoMorph 环境
conda create -n automorph_env python=3.8
conda activate automorph_env
pip install torch==1.11.0 torchvision==0.12.0 scikit-image opencv-python Pillow
```

---

## 10. 配置

### 配置层级

```
.env 文件（OPHAGENT_* 前缀）
   └── config/settings.py（pydantic-settings）
         ├── LLMSettings           # LLM 设置
         ├── KnowledgeBaseSettings # 知识库设置
         ├── ModelSchedulerSettings # 调度器设置
         ├── ToolSettings          # 工具设置
         └── TrainingSettings      # 训练设置
```

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OPHAGENT_LLM__PROVIDER` | `openai` | LLM 提供商（`openai` / `gemini` / `local`） |
| `OPHAGENT_LLM__MODEL_ID` | `gpt-5` | 模型 ID |
| `OPHAGENT_LLM__API_KEY` | — | API 密钥 |
| `OPHAGENT_SCHEDULER__CONDA_BASE` | `/opt/conda` | Conda 安装路径 |
| `OPHAGENT_SCHEDULER__FASTAPI_BASE_URL` | `http://localhost` | 服务基础 URL |
| `OPHAGENT_SESSION_HISTORY_LIMIT` | `50` | 短期记忆最大轮次 |
| `OPHAGENT_DEBUG` | `false` | 启用调试日志 |

---

## 11. 构建知识库

```bash
# 首次索引所有本地来源
python scripts/build_knowledge_base.py

# 强制重新索引
python scripts/build_knowledge_base.py --force

# 同时索引无标注图像目录
python scripts/build_knowledge_base.py --unlabelled /data/fundus_archive

# 测试检索效果
python scripts/build_knowledge_base.py --search "糖尿病视网膜病变黄斑水肿" --top-k 5
```

**离线索引输出：**
```
data/
└── vector_store/
    ├── ophagent.index      # FAISS 二进制索引
    └── ophagent_meta.jsonl # 文档元数据（每行一个 JSON）
```

---

## 12. 训练模型

```bash
# 训练 CFP 质量模型
python scripts/train_model.py \
  --model cfp_quality \
  --data-root data/cfp_quality \
  --epochs 50 --batch-size 32 --lr 1e-4

# 训练 FFA 病灶检测器
python scripts/train_model.py \
  --model ffa_lesion \
  --annotation-file data/ffa_lesion/annotations.json \
  --epochs 100 --batch-size 8

# 从断点继续训练
python scripts/train_model.py \
  --model cfp_disease \
  --annotation-file data/cfp_disease/ann.json \
  --resume models_weights/checkpoints/cfp_disease_ep25_metric0.8420.pth
```

### 多标签分类标注格式

```json
[
  {"image_path": "data/cfp/img001.jpg", "labels": [1, 0, 0, 1, 0, 0, 0, 0]},
  {"image_path": "data/cfp/img002.jpg", "labels": [0, 1, 0, 0, 1, 0, 0, 0]}
]
```

### 目标检测标注格式（FFA 病灶）

```json
[
  {
    "image_path": "data/ffa/img001.jpg",
    "boxes": [[120,340,145,365], [220,180,290,230]],
    "labels": [1, 2]
  }
]
```

标签映射：`0=背景, 1=微血管瘤, 2=出血, 3=硬性渗出, 4=新生血管`

---

## 13. 运行服务

### 方式 A：Docker Compose（推荐用于生产环境）

```bash
# 构建镜像
docker build \
  --build-arg MODEL_ID=cfp_quality \
  --build-arg MODEL_PORT=8110 \
  -t ophagent/cfp_quality:latest \
  -f services/docker/Dockerfile.template .

# 启动所有服务
docker-compose -f services/docker/docker-compose.yml up -d

# 健康检查
curl http://localhost:8110/health
# {"status": "ok", "model_id": "cfp_quality"}
```

### 方式 B：Python 进程管理器

```bash
# 启动所有服务（子进程模式）
python services/fastapi_service.py --all

# 启动单个服务
python services/fastapi_service.py --model cfp_quality --port 8110
```

### 方式 C：直接启动 uvicorn（开发环境）

```bash
MODEL_ID=cfp_quality MODEL_WEIGHT=models_weights/cfp_quality/best.pth \
  uvicorn ophagent.models.inference.service:app --host 0.0.0.0 --port 8110 --reload
```

---

## 14. 使用智能体

### Python API

```python
from ophagent.core.agent import OphAgent

agent = OphAgent()

# 单次查询
response = agent.run(
    query="分析这张眼底图像中的糖尿病视网膜病变，请提供临床报告。",
    image_paths=["patient_cfp.jpg"],
)
print(response.report)
print(f"需人工审核: {response.needs_human_review}")
print(f"耗时: {response.duration_s:.2f}s")

# 访问原始工具输出
for step_id, result in response.raw_results.items():
    print(f"步骤 {step_id} ({result['tool_name']}): {result['output']}")

# 交互式会话
agent.chat("视盘外观如何？")
agent.chat("是否存在黄斑水肿？", image_paths=["oct.jpg"])
agent.reset_session()  # 清除短期记忆
```

### 命令行（CLI）

```bash
# 交互式 REPL
python scripts/run_agent.py --interactive

# 单次查询
python scripts/run_agent.py \
  --query "评估这张 CFP 的糖网病分级" \
  --images fundus.jpg

# JSON 格式输出
python scripts/run_agent.py \
  --query "检测 FFA 图像中的病灶" \
  --images angio.jpg \
  --json-out

# 批量处理
python scripts/run_agent.py \
  --batch queries.json \
  --output results.json
```

### 批量输入格式（`queries.json`）

```json
[
  {"query": "评估糖网病分级", "images": ["patient001_cfp.jpg"]},
  {"query": "检查青光眼", "images": ["patient002_cfp.jpg"]},
  {"query": "分析 OCT 中的 AMD", "images": ["patient003_oct.jpg"]}
]
```

### 直接使用创新策略

```python
# 证据引导 CLIP
from ophagent.strategies.clip_evidence import EvidenceGuidedCLIP
from ophagent.tools.clip_models.retizero import RetiZeroTool
from ophagent.tools.registry import ToolRegistry

registry = ToolRegistry()
clip_tool = RetiZeroTool(registry.get("retizero"))
clf = EvidenceGuidedCLIP(clip_tool=clip_tool)

result = clf.classify(
    image_path="fundus.jpg",
    candidate_labels=["diabetic retinopathy", "age-related macular degeneration", "glaucoma", "normal"],
    modality="CFP",
)
print(result["label"], result["probabilities"])

# 可组合 VQA
from ophagent.strategies.vqa_composable import ComposableVQA
vqa = ComposableVQA()
answer = vqa.answer(image_path="fundus.jpg", question="Describe the cup-to-disc ratio.")
print(answer["answer"])

# 多尺度 RAG
from ophagent.strategies.multiscale_rag import MultiScaleRAG
from ophagent.knowledge.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
rag = MultiScaleRAG(vector_store=kb.vs)
context = rag.retrieve(
    query="增殖性糖尿病视网膜病变治疗",
    image_path="fundus.jpg",
    image_modality="CFP",
)
print(context)
```

---

## 15. 评估

```bash
python scripts/evaluate.py \
  --tool cfp_disease \
  --test-file data/cfp_disease/test.json \
  --output results/cfp_disease_eval.json
```

### 自动计算的评估指标

| 任务类型 | 指标 |
|---------|------|
| 二分类 | 准确率（Accuracy）、AUC-ROC、敏感性（Sensitivity）、特异性（Specificity）|
| 多分类 | 准确率、宏平均 F1、Cohen's Kappa |
| 多标签 | 各类别宏平均 F1 |
| 分割 | Dice 系数、IoU |
| 回归（关键点）| MSE、MAE |

---

## 16. 完整项目结构

```
OphAgent/
├── .env.example                         # 环境变量模板
├── README.md                            # 主 README（英文，含语言切换链接）
├── requirements.txt                     # Python 依赖
├── setup.py                             # 包安装配置
│
├── config/
│   ├── settings.py                      # 全局 pydantic-settings 配置
│   ├── tool_registry.yaml               # 所有工具元数据（21 个工具）
│   └── deployment.yaml                  # Docker + Conda 部署规格
│
├── docs/
│   ├── README_en.md                     # 英文详细技术文档
│   ├── README_zh.md                     # 本文件（中文）
│   ├── README_ja.md                     # 日文
│   ├── README_ko.md                     # 韩文
│   ├── README_fr.md                     # 法文
│   ├── README_es.md                     # 西班牙文
│   └── README_de.md                     # 德文
│
├── ophagent/
│   ├── core/                            # 智能体编排层
│   │   ├── agent.py                     # 主 OphAgent 类
│   │   ├── planner.py                   # 查询 → ExecutionPlan
│   │   ├── executor.py                  # 计划 → 工具调用 → StepResults
│   │   ├── verifier.py                  # 医疗一致性验证
│   │   └── memory.py                   # 短期 + 长期记忆
│   │
│   ├── tools/                           # 工具池
│   │   ├── base.py                      # BaseTool、ToolMetadata、混入类
│   │   ├── registry.py                  # ToolRegistry（YAML 加载器）
│   │   ├── scheduler.py                 # ToolScheduler（路由→执行）
│   │   ├── classification/              # 分类工具封装（9 个）
│   │   ├── segmentation/                # 分割工具（2 个）
│   │   ├── detection/                   # 检测工具（2 个）
│   │   ├── vqa/                         # VQA 工具（2 个）
│   │   ├── clip_models/                 # CLIP 模型（2 个）
│   │   └── auxiliary/                   # 辅助工具（4 个）
│   │
│   ├── knowledge/                       # 知识库
│   │   ├── knowledge_base.py            # 统一知识库 API
│   │   ├── vector_store.py             # 多模态 FAISS 存储
│   │   ├── local_data.py               # 图像档案 + 操作规范
│   │   ├── textbook.py                 # 教科书/指南索引
│   │   └── search_engine.py            # 实时网络搜索 + 索引
│   │
│   ├── strategies/                      # §2.4 创新策略
│   │   ├── clip_evidence.py            # §2.4.1 证据引导 CLIP
│   │   ├── vqa_composable.py           # §2.4.2 可组合工具 VQA
│   │   └── multiscale_rag.py           # §2.4.3 多尺度 RAG
│   │
│   ├── models/
│   │   ├── training/                    # 模型训练器（BaseTrainer + 8 个专用训练器）
│   │   └── inference/
│   │       └── service.py              # FastAPI 服务工厂
│   │
│   ├── llm/
│   │   ├── backbone.py                  # LLM 提供商抽象
│   │   └── prompts.py                  # 所有提示词模板
│   │
│   └── utils/
│       ├── image_utils.py               # 图像处理工具
│       ├── text_utils.py               # 文本处理工具
│       ├── metrics.py                  # 评估指标
│       └── logger.py                   # 日志工具
│
├── services/
│   ├── fastapi_service.py              # 多服务进程启动器
│   └── docker/
│       ├── Dockerfile.template         # 通用模型服务 Dockerfile
│       └── docker-compose.yml          # 10 个服务编排
│
├── scripts/
│   ├── run_agent.py                    # CLI：交互式/单次/批量
│   ├── train_model.py                  # 训练任意新开发模型
│   ├── evaluate.py                     # 在测试集上评估工具
│   └── build_knowledge_base.py         # 索引所有知识库来源
│
└── tests/
    ├── test_planner.py
    ├── test_executor.py
    ├── test_verifier.py
    └── test_tools.py
```

---

## 17. 功能清单与差距分析

### 核心流水线
- [x] Planner：LLM 驱动的计划生成
- [x] Planner：基于依赖关系的拓扑排序
- [x] Planner：无步骤时回退到 `web_search`
- [x] Planner：Verifier 冲突触发的重规划
- [x] Executor：顺序步骤执行
- [x] Executor：动态输入引用解析（`${step_N.output.key}`）
- [x] Executor：失败依赖传播（跳过后续相关步骤）
- [x] Executor：三种调度模式（inline / FastAPI / Conda）
- [x] Verifier：LLM 一致性检查
- [x] Verifier：RAG 上下文注入
- [x] Verifier：迭代冲突解决（最多 2 次）
- [x] Verifier：人工审核标记（置信度 < 0.6）
- [x] Verifier：图像质量门控
- [x] Memory：短期会话缓冲区（环形缓冲）
- [x] Memory：长期 FAISS 持久化病例存储
- [x] Memory：LLM 驱动的会话整合
- [x] OphAgent：完整运行循环
- [ ] OphAgent：异步/流式响应模式 *（尚未实现）*
- [ ] OphAgent：并行批量多患者处理 *（当前为单线程）*

### 工具池
- [x] 全部 9 个复用模型已封装
- [x] 全部 8 个新开发模型已封装
- [x] 全部 4 个辅助工具
- [x] `ToolRegistry` 支持 YAML 热重载
- [x] `ToolScheduler` 线程安全实例缓存
- [ ] FastAPI 服务健康监控面板 *（尚未实现）*
- [ ] 崩溃后自动重启服务 *（依赖 Docker restart policy）*
- [ ] DICOM 文件格式支持 *（尚未实现）*

### 知识库
- [x] 统一多模态 FAISS 向量存储
- [x] 文本嵌入（sentence-transformers）
- [x] 图像嵌入（CLIP）
- [x] 本地档案索引（图像 + 报告对）
- [x] 操作规范索引（PDF、TXT）
- [x] 教科书索引（PDF、TXT）
- [x] 实时网络搜索（DuckDuckGo、PubMed via Biopython）
- [x] 交互式会话上下文注入
- [ ] 增量索引更新（当前重建完整索引）*（性能待优化）*
- [ ] 超大规模档案分片索引（>100 万图像）*（尚未实现）*

### 创新策略
- [x] §2.4.1 LLM 生成视觉证据描述词
- [x] §2.4.1 逐标签嵌入 + 得分聚合
- [x] §2.4.1 证据缓存机制
- [x] §2.4.2 关键词路由流水线
- [x] §2.4.2 视盘/黄斑/默认组合规则
- [x] §2.4.2 自定义组合规则注册 API
- [x] §2.4.3 三尺度加权检索
- [x] §2.4.3 无标注图像 CLIP 索引
- [x] §2.4.3 多尺度得分融合
- [ ] §2.4.3 按模态配置尺度权重 *（当前硬编码）*

### 模型训练
- [x] BaseTrainer：AMP、早停、余弦退火 LR
- [x] BaseTrainer：断点保存与加载
- [x] 全部 8 个模型专用训练器
- [ ] 分布式训练（DDP）*（尚未实现）*
- [ ] 超参数搜索集成 *（尚未实现）*

### 服务部署
- [x] FastAPI 服务工厂（单代码库，通过 ENV 配置）
- [x] Docker Compose（10 个服务）
- [x] Dockerfile 模板
- [x] 多服务 Python 进程启动器
- [ ] Kubernetes/Helm Charts *（尚未实现）*
- [ ] 模型权重自动下载脚本 *（尚未实现）*
- [ ] CI/CD 流水线（GitHub Actions）*（尚未实现）*

### 测试
- [x] `test_planner.py`（5 个测试）
- [x] `test_executor.py`（5 个测试）
- [x] `test_verifier.py`（6 个测试）
- [x] `test_tools.py`（6 个测试）
- [ ] 端到端集成测试（带 mock 工具的完整智能体运行）*（尚未实现）*
- [ ] 测试覆盖率报告 *（尚未配置）*
- [ ] 性能基准测试 *（尚未实现）*

---

## 18. 扩展指南

### 添加新工具

**第 1 步**：在 `config/tool_registry.yaml` 中添加条目：
```yaml
my_new_tool:
  name: "我的新工具"
  description: "..."
  modality: "CFP"
  task: "classification"
  scheduling_mode: "fastapi"
  fastapi_port: 8130
  model_weight: "models_weights/my_new_tool/"
  newly_developed: true
```

**第 2 步**：创建 `ophagent/tools/classification/my_new_tool.py`：
```python
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata

class MyNewTool(FastAPIToolMixin, BaseTool):
    def run(self, inputs):
        from ophagent.utils.image_utils import image_to_base64
        return self._post(self.metadata.fastapi_port, "/run",
                          {"image_b64": image_to_base64(inputs["image_path"])})
```

**第 3 步**：在 `ophagent/tools/scheduler.py` 的 `_TOOL_CLASS_MAP` 中注册：
```python
"my_new_tool": "ophagent.tools.classification.my_new_tool:MyNewTool",
```

### 切换 LLM 提供商

```bash
# 在 .env 文件中修改：
OPHAGENT_LLM__PROVIDER=openai
OPHAGENT_LLM__MODEL_ID=gpt-4o
OPHAGENT_LLM__API_KEY=sk-...

# 使用本地模型（Ollama）：
OPHAGENT_LLM__PROVIDER=local
OPHAGENT_LLM__MODEL_ID=qwen2.5:72b
OPHAGENT_LLM__LOCAL_MODEL_URL=http://localhost:11434
```

### 添加新知识来源

```python
# 1. 在 ophagent/knowledge/ 中创建新类
class MyDataSource:
    def __init__(self, vector_store):
        self.vs = vector_store

    def index_all(self):
        # 读取数据并调用 self.vs.add_text() 或 self.vs.add_image()
        pass

# 2. 在 KnowledgeBase.__init__() 中注册
self.my_source = MyDataSource(vector_store=self.vs)
```
