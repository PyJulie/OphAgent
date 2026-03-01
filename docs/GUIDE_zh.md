# OphAgent 搭建指南（中文）

> 本文档手把手带你从零开始搭建 OphAgent，构建本地知识库，并教你如何添加自己的知识来源（Knowledge）和工具技能（Tools/Skills）。

---

## 目录

1. [环境准备](#1-环境准备)
2. [克隆项目与安装依赖](#2-克隆项目与安装依赖)
3. [配置 API Key 与参数](#3-配置-api-key-与参数)
4. [构建本地知识库](#4-构建本地知识库)
5. [启动模型服务](#5-启动模型服务)
6. [运行 Agent](#6-运行-agent)
7. [添加自己的知识来源（Knowledge）](#7-添加自己的知识来源knowledge)
8. [添加自己的工具技能（Tools/Skills）](#8-添加自己的工具技能toolsskills)
9. [训练自己的模型](#9-训练自己的模型)
10. [常见问题](#10-常见问题)

---

## 1. 环境准备

### 1.1 系统要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| 操作系统 | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 |
| Python | 3.10 | 3.10 |
| GPU | 无（仅 CPU 模式） | NVIDIA ≥ 16 GB VRAM（A100/3090/4090） |
| CUDA | — | 11.8 或 12.1 |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 50 GB | 500 GB（存放影像数据） |

### 1.2 安装 Conda（管理独立模型环境）

```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Windows（在 PowerShell 中运行安装包后执行）
conda init
```

### 1.3 安装 Docker（运行 FastAPI 模型服务）

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 安装 NVIDIA Container Toolkit（GPU 支持）
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

> **仅使用 CPU？** 跳过 NVIDIA Container Toolkit，后续 Docker 命令去掉 `--gpus` 参数即可。

---

## 2. 克隆项目与安装依赖

### 2.1 克隆仓库

```bash
git clone https://github.com/PyJulie/OphAgent.git
cd OphAgent
```

### 2.2 创建主环境

```bash
conda create -n ophagent python=3.10 -y
conda activate ophagent
```

### 2.3 安装 Python 依赖

```bash
pip install -r requirements.txt
pip install -e .   # 以可编辑模式安装，方便修改代码
```

### 2.4 创建复用模型的独立 Conda 环境

部分复用模型（FMUE、AutoMorph 等）依赖旧版 PyTorch，需隔离环境：

```bash
# FMUE（OCT B-scan 分类）
conda create -n fmue_env python=3.9 -y
conda activate fmue_env
pip install torch==1.13.0 torchvision==0.14.0 timm==0.6.12 Pillow numpy
conda deactivate

# AutoMorph（眼底血管分割）
conda create -n automorph_env python=3.8 -y
conda activate automorph_env
pip install torch==1.11.0 torchvision==0.12.0 scikit-image opencv-python Pillow
conda deactivate

# UWF-MDD（广角眼底多病检测）
conda create -n uwf_mdd_env python=3.9 -y
conda activate uwf_mdd_env
pip install torch==2.0.0 torchvision==0.15.0 timm Pillow numpy
conda deactivate
```

完成后切回主环境：

```bash
conda activate ophagent
```

---

## 3. 配置 API Key 与参数

### 3.1 复制配置模板

```bash
cp .env.example .env
```

### 3.2 编辑 `.env`

用任意文本编辑器打开 `.env`，填入你的配置：

```dotenv
# ── LLM 骨干配置 ──────────────────────────────────────────
# 默认：OpenAI GPT-5
OPHAGENT_LLM__PROVIDER=openai
OPHAGENT_LLM__MODEL_ID=gpt-5
OPHAGENT_LLM__API_KEY=sk-your-openai-key-here

# 可选：Google Gemini
# OPHAGENT_LLM__PROVIDER=gemini
# OPHAGENT_LLM__MODEL_ID=gemini-3.0-pro
# OPHAGENT_LLM__API_KEY=AIza-your-gemini-key-here

# 可选：本地 Ollama（无需 API Key）
# OPHAGENT_LLM__PROVIDER=local
# OPHAGENT_LLM__MODEL_ID=qwen2.5:72b
# OPHAGENT_LLM__LOCAL_MODEL_URL=http://localhost:11434/v1

# ── 模型调度 ───────────────────────────────────────────────
OPHAGENT_SCHEDULER__CONDA_BASE=/opt/conda        # Conda 安装路径（Linux）
# OPHAGENT_SCHEDULER__CONDA_BASE=C:/ProgramData/miniconda3  # Windows
OPHAGENT_SCHEDULER__FASTAPI_BASE_URL=http://localhost

# ── 知识库路径 ─────────────────────────────────────────────
OPHAGENT_KB__ARCHIVE_DIR=data/local_archive
OPHAGENT_KB__TEXTBOOK_DIR=data/textbooks
OPHAGENT_KB__VECTOR_STORE_DIR=data/vector_store

# ── 其他 ───────────────────────────────────────────────────
OPHAGENT_DEBUG=false
OPHAGENT_SESSION_HISTORY_LIMIT=50
```

### 3.3 验证配置

```bash
python -c "from config.settings import get_settings; s = get_settings(); print('LLM provider:', s.llm.provider)"
```

输出 `LLM provider: openai`（或你设置的值）则表示配置正确。

---

## 4. 构建本地知识库

知识库是 OphAgent 做医学核查、RAG 检索的核心数据来源。建库之前先准备好数据目录结构。

### 4.1 数据目录规范

```
OphAgent/
└── data/
    ├── local_archive/          # 本地病例档案
    │   ├── cases/              # 病例：每个子文件夹是一个病例
    │   │   ├── case_001/
    │   │   │   ├── fundus.jpg      # 眼底影像（CFP/OCT/UWF/FFA 均可）
    │   │   │   └── report.txt      # 对应的临床报告（纯文本）
    │   │   ├── case_002/
    │   │   │   ├── oct.jpg
    │   │   │   └── report.txt
    │   │   └── ...
    │   └── standards/          # 操作规范、筛查指南
    │       ├── dr_screening_guideline.pdf
    │       └── amd_protocol.txt
    │
    ├── textbooks/              # 眼科教材、指南文献
    │   ├── duanes_ophthalmology.pdf
    │   └── aao_guidelines/
    │       ├── dr_guidelines.pdf
    │       └── glaucoma_guideline.pdf
    │
    └── vector_store/           # 自动生成，无需手动创建
        ├── ophagent.index
        └── ophagent_meta.jsonl
```

### 4.2 准备病例数据

**report.txt 的内容格式**（自由文本即可，越详细越好）：

```
患者：45岁，男性，右眼
主诉：视力模糊 3 个月
检查发现：眼底照相显示视盘颞侧盘沿变薄，C/D 比约 0.7，
          黄斑区可见少量硬性渗出，未见新生血管。
诊断：可疑青光眼；轻度非增殖性糖尿病性视网膜病变（NPDR）
建议：OCT 视神经纤维层检查，3 个月随访。
```

> 英文报告同样支持，系统会自动统一向量化。

### 4.3 准备教材/指南

支持 **PDF** 和 **TXT** 格式，直接放入 `data/textbooks/` 即可。
PDF 会自动按页分块（chunk），每块约 400 词。

### 4.4 运行建库脚本

```bash
# 首次建库（索引所有来源）
python scripts/build_knowledge_base.py

# 强制重建（数据有更新时使用）
python scripts/build_knowledge_base.py --force

# 同时索引一批无标注影像（无需报告，仅用 CLIP 图像向量）
python scripts/build_knowledge_base.py --unlabelled /path/to/image_folder

# 建完后测试检索
python scripts/build_knowledge_base.py \
    --search "diabetic retinopathy macular oedema" \
    --top-k 5
```

**建库输出示例：**

```
[INFO] Indexing local archive...  200 cases indexed.
[INFO] Indexing standards...      5 documents indexed.
[INFO] Indexing textbooks...      1240 chunks indexed.
[INFO] Total vectors: 1445
[INFO] FAISS index saved → data/vector_store/ophagent.index
[INFO] Metadata saved  → data/vector_store/ophagent_meta.jsonl
```

### 4.5 增量更新知识库

目前知识库为全量重建；若数据量大，可只对新增文件单独调用：

```python
# 在 Python 中单独添加一篇新报告
from ophagent.knowledge.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
kb.add_text(
    text="新病例报告内容...",
    metadata={"source": "case_003", "modality": "CFP"}
)
kb.save()   # 持久化到磁盘
```

---

## 5. 启动模型服务

OphAgent 中的「新开发模型」（8 个）需要以 FastAPI 微服务形式运行。

### 5.1 方式 A：Docker Compose（推荐，生产环境）

**前提：** 已将模型权重放入 `models_weights/<model_id>/best.pth`。

```bash
# 构建所有镜像（首次，较慢）
bash services/docker/build_all.sh

# 或逐个构建
docker build \
  --build-arg MODEL_ID=cfp_quality \
  --build-arg MODEL_PORT=8110 \
  -t ophagent/cfp_quality:latest \
  -f services/docker/Dockerfile.template .

# 启动所有服务
docker-compose -f services/docker/docker-compose.yml up -d

# 查看服务状态
docker-compose -f services/docker/docker-compose.yml ps

# 验证某个服务是否正常
curl http://localhost:8110/health
# 返回：{"status": "ok", "model_id": "cfp_quality"}
```

### 5.2 方式 B：直接用 Python 启动（开发调试）

```bash
# 启动全部服务（后台进程）
python services/fastapi_service.py --all

# 仅启动某一个
python services/fastapi_service.py --model cfp_quality --port 8110
```

### 5.3 端口一览

| 服务 | 端口 | 说明 |
|------|------|------|
| FundusExpert VQA | 8101 | 复用模型 |
| VisionUnite VQA | 8102 | 复用模型 |
| cfp_quality | 8110 | 新开发：CFP 图像质量 |
| cfp_disease | 8111 | 新开发：CFP 多标签疾病 |
| cfp_ffa_multimodal | 8112 | 新开发：CFP+FFA 联合 |
| uwf_quality_disease | 8113 | 新开发：UWF 质量+疾病 |
| cfp_glaucoma | 8114 | 新开发：青光眼筛查 |
| cfp_pdr | 8115 | 新开发：PDR 分级 |
| ffa_lesion | 8120 | 新开发：FFA 病灶检测 |
| disc_fovea | 8121 | 新开发：视盘/黄斑定位 |

> **仅测试核心流程？** 可以先不启动 FastAPI 服务，把 `config/tool_registry.yaml` 中对应工具的 `scheduling_mode` 改为 `inline`（需本地有权重文件）。

---

## 6. 运行 Agent

### 6.1 交互式命令行（最简单）

```bash
python scripts/run_agent.py --interactive
```

```
OphAgent > 你好，请分析这张眼底照片是否有糖尿病视网膜病变
Image paths (comma separated, or Enter to skip): patient_cfp.jpg
...
[Planner] 生成执行计划: cfp_quality → cfp_disease → gradcam → synthesise
[Executor] 步骤 1/4: cfp_quality ...
[Verifier] 医学一致性检查通过，置信度 0.87
...
报告：该眼底照片质量良好（Good, 0.91）。多标签疾病分类显示糖尿病视网膜病变（DR）
     阳性（概率 0.94），未见明显增殖性病变...
```

### 6.2 单次查询

```bash
python scripts/run_agent.py \
  --query "分析这张 FFA 影像中的病灶类型" \
  --images angio.jpg
```

### 6.3 Python API

```python
from ophagent.core.agent import OphAgent

agent = OphAgent()

response = agent.run(
    query="请对这张 CFP 图像进行全面的眼底分析",
    image_paths=["fundus.jpg"],
)

print(response.report)                    # 最终临床报告
print(response.needs_human_review)        # 是否需要人工复核
for step_id, r in response.raw_results.items():
    print(f"Step {step_id}: {r['tool_name']} → {r['output']}")
```

---

## 7. 添加自己的知识来源（Knowledge）

### 知识来源存放位置

```
ophagent/knowledge/          ← 所有知识来源的代码都放这里
├── knowledge_base.py        ← 统一入口（在这里注册新来源）
├── vector_store.py          ← FAISS 向量库（一般不需要改）
├── local_data.py            ← 病例档案 + 操作规范（可参考）
├── textbook.py              ← 教材/指南（可参考）
├── search_engine.py         ← 在线检索（PubMed/DuckDuckGo）
└── my_custom_source.py      ← 你新建的自定义来源 ← 放这里
```

### 7.1 示例：添加「微信公众号文章」知识来源

**第一步：新建文件 `ophagent/knowledge/wechat_articles.py`**

```python
"""
自定义知识来源：微信公众号眼科科普文章。
文章以 JSON 格式存放在 data/wechat/ 目录下。
"""
import json
from pathlib import Path
from typing import List, Dict, Any

class WechatArticleSource:
    """
    数据格式：data/wechat/*.json
    每个 JSON 文件结构：
    {
        "title": "糖尿病视网膜病变的早期筛查",
        "author": "XX医院眼科",
        "date": "2024-01-15",
        "content": "全文内容..."
    }
    """

    def __init__(self, article_dir: str = "data/wechat"):
        self.article_dir = Path(article_dir)

    def load_documents(self) -> List[Dict[str, Any]]:
        """返回可被 KnowledgeBase 索引的文档列表。"""
        docs = []
        if not self.article_dir.exists():
            return docs

        for json_file in self.article_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                article = json.load(f)

            # 按段落分块，每块最多 300 字
            paragraphs = [p.strip() for p in article["content"].split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "metadata": {
                        "source": "wechat_article",
                        "title": article["title"],
                        "author": article.get("author", ""),
                        "date": article.get("date", ""),
                        "chunk_id": i,
                        "file": json_file.name,
                    }
                })
        return docs
```

**第二步：在 `ophagent/knowledge/knowledge_base.py` 中注册**

打开 `knowledge_base.py`，找到 `__init__` 方法，加入你的来源：

```python
# knowledge_base.py 的 __init__ 中（约第 30 行左右）
from ophagent.knowledge.wechat_articles import WechatArticleSource  # ← 新增

class KnowledgeBase:
    def __init__(self):
        self.vs = VectorStore(...)
        self.sources = [
            ImageReportArchive(...),
            OperationalStandards(...),
            TextbookSource(...),
            SearchEngineSource(...),
            WechatArticleSource(),   # ← 新增这一行
        ]
```

**第三步：重建知识库**

```bash
python scripts/build_knowledge_base.py --force
```

**验证：**

```bash
python scripts/build_knowledge_base.py \
    --search "糖尿病视网膜病变筛查" \
    --top-k 3
```

---

### 7.2 示例：添加「医院内网 API」知识来源

```python
# ophagent/knowledge/hospital_api.py
import requests
from typing import List, Dict, Any

class HospitalAPISource:
    """从医院内网 REST API 拉取历史病例摘要。"""

    def __init__(self, api_url: str, token: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def load_documents(self) -> List[Dict[str, Any]]:
        resp = requests.get(
            f"{self.api_url}/cases/summaries",
            headers=self.headers,
            timeout=30
        )
        resp.raise_for_status()
        cases = resp.json()["data"]

        docs = []
        for case in cases:
            docs.append({
                "text": case["summary"],
                "metadata": {
                    "source": "hospital_api",
                    "case_id": case["id"],
                    "date": case["date"],
                    "diagnosis": case["diagnosis"],
                }
            })
        return docs
```

在 `knowledge_base.py` 中注册时从 `.env` 读取配置：

```python
from config.settings import get_settings

settings = get_settings()
HospitalAPISource(
    api_url=settings.hospital_api_url,
    token=settings.hospital_api_token,
)
```

---

### 7.3 知识来源接口规范（summary）

你自定义的知识来源只需满足一个条件：

```python
class MySource:
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        返回 list，每个元素是：
        {
            "text": str,              # 必须，文档文本内容
            "metadata": dict,         # 推荐，存储来源信息
            "image_path": str,        # 可选，有图像时提供（用 CLIP 编码）
        }
        """
        ...
```

---

## 8. 添加自己的工具技能（Tools/Skills）

### 工具存放位置

```
ophagent/tools/
├── base.py                   ← BaseTool 基类（必读）
├── registry.py               ← 工具注册表（加载 YAML）
├── scheduler.py              ← 工具调度器（需添加类映射）
│
├── classification/           ← 分类工具
├── segmentation/             ← 分割工具
├── detection/                ← 检测工具
├── vqa/                      ← VQA 工具
├── clip_models/              ← CLIP 模型
├── auxiliary/                ← 辅助工具
└── my_category/              ← 你的新分类 ← 可以新建文件夹
    └── my_tool.py
```

---

### 8.1 示例 A：添加「本地 FastAPI 服务」工具（新开发模型）

**场景**：你训练了一个 OCT 黄斑分级模型，已封装成 FastAPI 服务在 8130 端口运行。

**第一步：在 `config/tool_registry.yaml` 中声明工具**

```yaml
oct_macular_grade:
  name: "OCT Macular Grading"
  description: >
    对 OCT B-scan 影像进行黄斑病变分级，输出分级标签（Normal/Early/Intermediate/Late AMD）
    及各级别概率。适用于 AMD 随访和筛查场景。
  modality: "OCT"
  task: "classification"
  scheduling_mode: "fastapi"
  fastapi_port: 8130
  model_weight: "models_weights/oct_macular_grade/best.pth"
  newly_developed: true
  input_schema:
    image_path: "string"
  output_schema:
    grade_label: "string"
    probabilities: "dict"
```

**第二步：新建工具文件 `ophagent/tools/classification/oct_macular_grade.py`**

```python
"""
OCT 黄斑分级工具：调用 FastAPI 服务（端口 8130）。
"""
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata
from ophagent.utils.image_utils import image_to_base64


class OCTMacularGradeTool(FastAPIToolMixin, BaseTool):
    """OCT 黄斑病变分级（Normal / Early / Intermediate / Late AMD）。"""

    def run(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {"image_path": "path/to/oct.jpg"}
        Returns:
            {"grade_label": "Early AMD", "probabilities": {...}}
        """
        image_b64 = image_to_base64(inputs["image_path"])
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={"image_b64": image_b64},
        )
```

**第三步：在 `ophagent/tools/scheduler.py` 中注册类映射**

打开 `scheduler.py`，找到 `_TOOL_CLASS_MAP` 字典，添加一行：

```python
_TOOL_CLASS_MAP = {
    # 已有工具...
    "cfp_quality":          "ophagent.tools.classification.cfp_quality:CFPQualityTool",
    "cfp_disease":          "ophagent.tools.classification.cfp_disease:CFPDiseaseTool",
    # ...
    # ↓ 新增这一行
    "oct_macular_grade":    "ophagent.tools.classification.oct_macular_grade:OCTMacularGradeTool",
}
```

**第四步：验证工具注册**

```python
from ophagent.tools.registry import ToolRegistry
from ophagent.tools.scheduler import ToolScheduler

registry = ToolRegistry()
scheduler = ToolScheduler(registry)

# 查看是否注册成功
print(registry.list_tools())  # 应包含 "oct_macular_grade"

# 直接调用测试（需服务已启动）
result = scheduler.run("oct_macular_grade", {"image_path": "test_oct.jpg"})
print(result)
```

---

### 8.2 示例 B：添加「Conda 隔离环境」工具（复用第三方模型）

**场景**：你想接入开源的 `RETFound` 眼底基础模型，但它依赖特定版本的 `timm`。

**第一步：创建独立 Conda 环境**

```bash
conda create -n retfound_env python=3.9 -y
conda activate retfound_env
pip install torch==2.0.1 torchvision==0.15.2 timm==0.9.2 Pillow numpy
# 安装 RETFound 本身
pip install git+https://github.com/rmaphoh/RETFound_MAE.git
conda deactivate
```

**第二步：创建子进程调用脚本 `ophagent/tools/adapters/retfound_runner.py`**

> 这个脚本会在 `retfound_env` 内被 subprocess 调用，读取 stdin JSON，输出 JSON。

```python
#!/usr/bin/env python
"""
在 retfound_env 中被 Conda 调度器以 subprocess 调用。
从 sys.argv[1] 读取 JSON 输入，结果输出到 stdout。
"""
import sys
import json
import torch
from PIL import Image

def main():
    inputs = json.loads(sys.argv[1])
    image_path = inputs["image_path"]
    task = inputs.get("task", "feature_extraction")

    # 加载 RETFound 模型（首次加载后 OS 会缓存）
    # 此处为示意，具体 API 以 RETFound 文档为准
    model = load_retfound_model("models_weights/retfound/RETFound_cfp_weights.pth")
    model.eval()

    image = Image.open(image_path).convert("RGB")
    features = extract_features(model, image)

    result = {
        "features": features.tolist(),  # 1024-dim 特征向量
        "task": task,
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

**第三步：在 `config/tool_registry.yaml` 中声明**

```yaml
retfound:
  name: "RETFound Foundation Model"
  description: >
    使用 RETFound 眼底基础模型提取 1024 维图像特征，可用于下游迁移学习或
    zero-shot 相似度检索。支持 CFP 输入。
  modality: "CFP"
  task: "feature_extraction"
  scheduling_mode: "conda"
  conda_env: "retfound_env"
  conda_script: "ophagent/tools/adapters/retfound_runner.py"
  newly_developed: false
```

**第四步：在 `scheduler.py` 中注册**

```python
_TOOL_CLASS_MAP = {
    # ...
    "retfound": "ophagent.tools.clip_models.retfound:RETFoundTool",
}
```

**新建 `ophagent/tools/clip_models/retfound.py`：**

```python
from ophagent.tools.base import BaseTool, CondaToolMixin, ToolMetadata

class RETFoundTool(CondaToolMixin, BaseTool):
    """通过 Conda 子进程调用 RETFound 提取特征。"""

    def run(self, inputs: dict) -> dict:
        return self._run_conda(inputs)  # CondaToolMixin 负责子进程调度
```

---

### 8.3 示例 C：添加纯内联辅助工具（无需模型）

**场景**：你想加一个「DICOM → JPEG 转换」工具，直接在主进程运行。

**`ophagent/tools/auxiliary/dicom_converter.py`：**

```python
from ophagent.tools.base import BaseTool, ToolMetadata

class DICOMConverterTool(BaseTool):
    """将 DICOM 文件转换为 JPEG，供后续工具处理。"""

    def run(self, inputs: dict) -> dict:
        """
        Args:
            inputs: {"dicom_path": "scan.dcm", "output_path": "scan.jpg"}
        Returns:
            {"output_path": "scan.jpg", "success": True}
        """
        import pydicom
        import numpy as np
        from PIL import Image

        dicom_path = inputs["dicom_path"]
        output_path = inputs.get("output_path", dicom_path.replace(".dcm", ".jpg"))

        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        # 归一化到 0-255
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
        img = Image.fromarray(pixel_array.astype(np.uint8))
        img.save(output_path)

        return {"output_path": output_path, "success": True}
```

在 `tool_registry.yaml` 中：

```yaml
dicom_converter:
  name: "DICOM to JPEG Converter"
  description: "将 DICOM 格式的眼科影像转换为 JPEG，以便后续分析工具使用。"
  modality: "ANY"
  task: "preprocessing"
  scheduling_mode: "inline"
  newly_developed: false
```

在 `scheduler.py` 中注册：

```python
"dicom_converter": "ophagent.tools.auxiliary.dicom_converter:DICOMConverterTool",
```

---

### 8.4 工具开发速查

| 调度模式 | 继承的 Mixin | 主要方法 | 适用场景 |
|---------|------------|---------|---------|
| `inline` | 无 | `run(inputs)` 直接实现 | 轻量工具、无 GPU 要求 |
| `fastapi` | `FastAPIToolMixin` | `self._post(port, endpoint, payload)` | 重型模型、需持久化 GPU 占用 |
| `conda` | `CondaToolMixin` | `self._run_conda(inputs)` | 依赖有冲突的第三方模型 |

---

## 9. 训练自己的模型

### 9.1 数据准备

**分类任务标注格式**（JSON 列表）：

```json
[
  {"image_path": "data/cfp/img001.jpg", "labels": [1, 0, 0, 1, 0, 0, 0, 0]},
  {"image_path": "data/cfp/img002.jpg", "labels": [0, 1, 0, 0, 1, 0, 0, 0]}
]
```

**目标检测标注格式**（适用于 `ffa_lesion`）：

```json
[
  {
    "image_path": "data/ffa/img001.jpg",
    "boxes": [[120, 340, 145, 365], [220, 180, 290, 230]],
    "labels": [1, 2]
  }
]
```

### 9.2 训练命令

```bash
# 训练 CFP 质量评估模型（50 epoch，batch 32）
python scripts/train_model.py \
  --model cfp_quality \
  --data-root data/cfp_quality \
  --epochs 50 --batch-size 32 --lr 1e-4

# 训练 FFA 病灶检测模型
python scripts/train_model.py \
  --model ffa_lesion \
  --annotation-file data/ffa_lesion/annotations.json \
  --epochs 100 --batch-size 8

# 从断点恢复训练
python scripts/train_model.py \
  --model cfp_disease \
  --annotation-file data/cfp_disease/ann.json \
  --resume models_weights/checkpoints/cfp_disease_ep25_metric0.8420.pth
```

### 9.3 训练完成后

```
models_weights/
├── cfp_quality/
│   └── best.pth          ← 最佳模型（自动保存）
└── checkpoints/
    └── cfp_quality_ep45_metric0.9120.pth
```

将 `best.pth` 放入 `models_weights/<model_id>/best.pth`，重启对应 FastAPI 服务即可生效。

---

## 10. 常见问题

### Q1：运行时报 `ModuleNotFoundError: No module named 'ophagent'`

```bash
# 确保以可编辑模式安装
pip install -e .
# 或将项目根目录加入 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/OphAgent
```

---

### Q2：FastAPI 服务健康检查失败

```bash
# 查看服务日志
docker logs ophagent_cfp_quality_1

# 常见原因：模型权重文件不存在
ls models_weights/cfp_quality/best.pth
```

---

### Q3：Conda 工具报 `conda: command not found`

在 `.env` 中配置正确的 Conda 安装路径：

```dotenv
OPHAGENT_SCHEDULER__CONDA_BASE=/opt/conda      # Linux
# OPHAGENT_SCHEDULER__CONDA_BASE=C:/ProgramData/miniconda3  # Windows
```

---

### Q4：知识库检索结果不相关

- 数据量太少（< 50 条）时，语义检索效果有限，可降低 `top_k` 并提升 `score_threshold`
- 检查报告文本是否过短（建议每份报告 ≥ 100 字）
- 尝试重建知识库：`python scripts/build_knowledge_base.py --force`

---

### Q5：如何切换成本地 LLM（不联网）

```dotenv
# .env
OPHAGENT_LLM__PROVIDER=local
OPHAGENT_LLM__MODEL_ID=qwen2.5:72b
OPHAGENT_LLM__BASE_URL=http://localhost:11434/v1
```

需先安装并运行 [Ollama](https://ollama.com)，然后拉取模型：

```bash
ollama pull qwen2.5:72b
ollama serve
```

---

### Q6：我的新工具 Planner 从来不调用它

原因是 Planner 基于 `config/tool_registry.yaml` 中的 `description` 字段来决定用哪个工具。
**解决方案**：把 `description` 写得具体，明确说明适用的影像模态、疾病和任务类型：

```yaml
# ❌ 太模糊
description: "分析眼底图像"

# ✅ 具体
description: >
  对 OCT B-scan 影像进行黄斑病变分级，输出 Normal/Early/Intermediate/Late AMD 标签
  及各级别置信度。适用于 AMD 筛查、随访监测场景。输入：OCT 影像路径。
```

---

*如遇其他问题，请在 [GitHub Issues](https://github.com/PyJulie/OphAgent/issues) 反馈。*
