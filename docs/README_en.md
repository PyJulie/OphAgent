# OphAgent — Detailed Technical Reference (English)

<div align="center">

**Language / 语言**

[🇨🇳 中文](./README_zh.md) | **🇬🇧 English**

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Core Components](#3-core-components)
   - 3.1 [Planner](#31-planner)
   - 3.2 [Executor](#32-executor)
   - 3.3 [Verifier](#33-verifier)
   - 3.4 [Memory Manager](#34-memory-manager)
   - 3.5 [OphAgent Orchestrator](#35-ophagent-orchestrator)
4. [Tool Pool](#4-tool-pool)
   - 4.1 [Reused Models](#41-reused-models)
   - 4.2 [Newly Developed Models](#42-newly-developed-models)
   - 4.3 [Auxiliary Tools](#43-auxiliary-tools)
5. [Model Scheduling](#5-model-scheduling)
6. [Knowledge Base](#6-knowledge-base)
7. [Innovative Strategies](#7-innovative-strategies)
   - 7.1 [Evidence-Guided CLIP](#71-evidence-guided-clip-zero-shot-classification)
   - 7.2 [Composable Tool-Based VQA](#72-composable-tool-based-vqa)
   - 7.3 [Multi-Scale RAG](#73-multi-scale-rag-with-unlabelled-databases)
8. [LLM Backbone](#8-llm-backbone)
9. [Installation](#9-installation)
10. [Configuration](#10-configuration)
11. [Building the Knowledge Base](#11-building-the-knowledge-base)
12. [Training Models](#12-training-models)
13. [Running Services](#13-running-services)
14. [Using the Agent](#14-using-the-agent)
15. [Evaluation](#15-evaluation)
16. [Full Project Structure](#16-full-project-structure)
17. [Feature Checklist & Gap Analysis](#17-feature-checklist--gap-analysis)
18. [Extension Guide](#18-extension-guide)

---

## 1. Overview

**OphAgent** is an LLM-driven clinical AI agent specialised for ophthalmology. It integrates a curated pool of ophthalmic vision models through a structured Planner → Executor → Verifier pipeline, producing reliable, evidence-grounded clinical reports from retinal images and natural-language queries.

### Key design goals

| Goal | Mechanism |
|------|-----------|
| Multi-task ophthalmic analysis | 21+ specialised tool wrappers across 5 modalities |
| Reliable medical outputs | RAG-guided Verifier with conflict resolution |
| Zero-shot extensibility | Evidence-Guided CLIP (§2.4.1) |
| Region-specific VQA without segmentation labels | Composable Tool-Based VQA (§2.4.2) |
| Retrieval over unlabelled image archives | Multi-Scale RAG (§2.4.3) |
| Pluggable models with conflicting dependencies | Two-mode scheduling (Conda / Docker+FastAPI) |
| Long-term clinical memory | FAISS-backed persistent case store |

### Supported image modalities

| Abbreviation | Full name | Example diseases |
|---|---|---|
| CFP | Colour Fundus Photography | DR, AMD, Glaucoma, RVO, Myopia |
| OCT | Optical Coherence Tomography | AMD, DME, CNV, DRUSEN |
| UWF | Ultra-Widefield Fundus | Peripheral degeneration, DR |
| FFA | Fundus Fluorescein Angiography | PDR, RVO, CNV |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User / Clinician                       │
│          "Analyse this CFP for diabetic retinopathy"         │
└───────────────────────────┬──────────────────────────────────┘
                            │  query + image paths
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    OphAgent Orchestrator                      │
│  ┌──────────┐  plan  ┌──────────┐ results ┌──────────────┐  │
│  │ Planner  │───────▶│ Executor │────────▶│  Verifier    │  │
│  │ (LLM)   │        │          │         │  (LLM + RAG) │  │
│  └──────────┘        └────┬─────┘         └──────┬───────┘  │
│       ▲                   │                       │ verdict  │
│       │                   ▼                       ▼          │
│  ┌────┴────────────────────────────────────────────────────┐ │
│  │                   Memory Manager                        │ │
│  │  Short-term (session buffer)  Long-term (FAISS cases)  │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────────────────────────────┘
               │ tool calls
               ▼
┌──────────────────────────────────────────────────────────────┐
│                        Tool Pool                             │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ Classification │  │  Segmentation  │  │   Detection   │  │
│  │ (9 tools)      │  │  (2 tools)     │  │  (2 tools)    │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │    VQA         │  │  CLIP Models   │  │  Auxiliary    │  │
│  │  (2 tools)     │  │  (2 tools)     │  │  (4 tools)    │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
│                                                              │
│           Scheduler: inline | FastAPI | Conda                │
└──────────────────────────────────────────────────────────────┘
               │                        │
               ▼                        ▼
┌────────────────────┐      ┌───────────────────────┐
│   Knowledge Base   │      │   Innovative Strategies│
│  ├ Local Archive   │      │  ├ Evidence-CLIP        │
│  ├ Textbooks       │      │  ├ Composable VQA       │
│  ├ Search Engine   │      │  └ Multi-Scale RAG      │
│  └ Interactive     │      └───────────────────────┘
│   (FAISS unified)  │
└────────────────────┘
```

---

## 3. Core Components

### 3.1 Planner

**File:** `ophagent/core/planner.py`

The Planner receives a natural-language query and converts it into a structured `ExecutionPlan` — an ordered, dependency-aware list of tool-call steps.

#### Key classes

| Class | Purpose |
|-------|---------|
| `PlanStep` | Single step: `tool_name`, `inputs`, `depends_on` |
| `ExecutionPlan` | Ordered list of `PlanStep`, with topological sort |
| `Planner` | LLM-backed plan generator |

#### Planning process

1. Load tool descriptions from `config/tool_registry.yaml`.
2. Build a system prompt listing all available tools.
3. Call the LLM with the user query and image context.
4. Parse the returned JSON into `PlanStep` objects.
5. Validate dependency graph; use Kahn's topological sort for execution order.
6. If the LLM returns no steps, fall back to `web_search`.

#### Output JSON format

```json
{
  "steps": [
    {"step_id": 1, "tool_name": "cfp_quality",  "inputs": {"image_path": "fundus.jpg"}, "description": "...", "depends_on": []},
    {"step_id": 2, "tool_name": "cfp_disease",  "inputs": {"image_path": "fundus.jpg"}, "description": "...", "depends_on": [1]},
    {"step_id": 3, "tool_name": "gradcam",      "inputs": {"image_path": "fundus.jpg", "model": "cfp_disease"}, "description": "...", "depends_on": [2]},
    {"step_id": 4, "tool_name": "synthesise",   "inputs": {"summary": "..."}, "description": "Report", "depends_on": [1,2,3]}
  ]
}
```

The `synthesise` step is always last and is handled by the Orchestrator, not the Executor.

---

### 3.2 Executor

**File:** `ophagent/core/executor.py`

Executes each `PlanStep` in topological order. Skips steps whose dependencies failed. Supports dynamic input references.

#### Dynamic input references

Steps can reference outputs of prior steps using `${step_N.output.key}` syntax:

```json
{"image_path": "${step_1.output.cropped_path}"}
```

The `_resolve_inputs()` method traverses the result store and substitutes these at runtime, enabling chained pipelines (e.g., crop → classify → explain).

#### Step result format

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

### 3.3 Verifier

**File:** `ophagent/core/verifier.py`

Validates the medical consistency of Executor outputs before report generation.

#### Verification workflow

```
Executor results
      │
      ▼
LLM (verifier system prompt + RAG context)
      │
      ▼
Verdict: {valid, confidence, conflicts, resolution, verified_result}
      │
      ├── confidence < 0.6 → flag needs_human_review = True
      │
      └── conflicts → expand KB search → re-verify (max 2 attempts)
```

#### Medical consistency checks (performed by LLM)
- Disease grade and classification do not contradict (e.g. "no DR" + "PDR active")
- Quality score aligns with analysis confidence
- Segmentation area plausible for reported findings
- Bilateral consistency if both eyes provided

#### Quality gate

`Verifier.check_quality_gate(quality_result)` returns `False` (blocking further analysis) if:
- `quality_score < 0.4`
- `quality_label` contains "ungradable"

---

### 3.4 Memory Manager

**File:** `ophagent/core/memory.py`

Two-tier memory architecture:

| Tier | Class | Scope | Storage |
|------|-------|-------|---------|
| Short-term | `ShortTermMemory` | Session | In-memory ring buffer (configurable max turns) |
| Long-term | `LongTermMemory` | Persistent | FAISS index + JSONL metadata file |

#### Long-term memory flow

```
Completed session
      │
      ▼
LLM (consolidation prompt)
      │
      ▼
MemoryEntry: {entry_id, summary, key_findings, tools_used, modalities, tags}
      │
      ▼
Embedded → FAISS index → saved to disk
```

Future queries retrieve relevant past cases via cosine similarity search over embedded summaries.

---

### 3.5 OphAgent Orchestrator

**File:** `ophagent/core/agent.py`

The `OphAgent` class ties all components together and is the primary user-facing API.

#### Full run loop

```
run(query, image_paths)
  │
  ├── 1. add_turn("user", query)                    # short-term memory
  ├── 2. retrieve past cases from long-term memory  # RAG context
  ├── 3. planner.plan(query, images, context)        # generate plan
  ├── 4. executor.execute(plan)                      # call tools
  ├── 5. verifier.verify(results, query)             # validate
  │        └── if conflicts → planner.replan() → executor again
  ├── 6. check_quality_gate(quality_result)          # block bad images
  ├── 7. synthesise_report(query, verdict, context)  # LLM report
  ├── 8. add_turn("assistant", report)               # store response
  └── 9. consolidate_session() → long-term memory    # persist case
```

#### `AgentResponse` fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Original query |
| `report` | str | Final clinical report |
| `verdict` | Verdict | Verifier output |
| `plan` | ExecutionPlan | Steps generated |
| `raw_results` | dict | Per-step tool outputs |
| `duration_s` | float | Wall-clock time |
| `needs_human_review` | bool | Low-confidence flag |

---

## 4. Tool Pool

### 4.1 Reused Models

| Tool ID | Name | Modality | Task | Scheduling | File |
|---------|------|----------|------|-----------|------|
| `fmue` | FMUE | OCT | B-scan classification (Normal/AMD/DME/CNV/Drusen) | Conda | `tools/classification/oct_fmue.py` |
| `uwf_mdd` | UWF-MDD | UWF | Multi-disease detection | Conda | `tools/classification/uwf_mdd.py` |
| `uwf_multi_abnormality` | UWF Multi-Abnormality | UWF | Multi-label abnormality screening | Conda | `tools/classification/uwf_multi_abnormality.py` |
| `retizero` | RetiZero | CFP | CLIP zero-shot (English) | Inline | `tools/clip_models/retizero.py` |
| `vilref` | ViLReF | CFP | CLIP zero-shot (Chinese) | Inline | `tools/clip_models/vilref.py` |
| `fundus_expert` | FundusExpert | CFP | VQA | FastAPI :8101 | `tools/vqa/fundus_expert.py` |
| `vision_unite` | VisionUnite | CFP | VQA | FastAPI :8102 | `tools/vqa/vision_unite.py` |
| `automorph` | AutoMorph | CFP | Vessel seg + morphometric quantification | Conda | `tools/segmentation/automorph.py` |
| `retsam` | RetSAM | CFP | SAM-adapted retinal segmentation | Inline | `tools/segmentation/retsam.py` |

**AutoMorph outputs:** vessel mask, fractal dimension, vessel density, A/V ratio, tortuosity.

**CLIP models:** Both `retizero` and `vilref` are enhanced by the Evidence-Guided CLIP strategy (§7.1).

---

### 4.2 Newly Developed Models

| Tool ID | Modality | Task | Port | Architecture | Trainer |
|---------|----------|------|------|-------------|---------|
| `cfp_quality` | CFP | Image quality reasoning (Bad/Moderate/Good + explanation) | 8110 | EfficientNet-B4 | `CFPQualityTrainer` |
| `cfp_disease` | CFP | Multi-label retinal disease classification | 8111 | ConvNeXt-Base | `CFPDiseaseTrainer` |
| `cfp_ffa_multimodal` | CFP+FFA | Joint dual-modality classification | 8112 | Dual ViT-Small + cross-attention | `CFPFFAMultimodalTrainer` |
| `uwf_quality_disease` | UWF | Quality + disease multi-task | 8113 | ResNet-50, dual-head | `UWFQualityDiseaseTrainer` |
| `cfp_glaucoma` | CFP | Referable glaucoma + structural features (CDR, rim area) | 8114 | Swin-Transformer-Small | `CFPGlaucomaTrainer` |
| `cfp_pdr` | CFP | PDR activity grading (4-level) | 8115 | EfficientNet-B5 | `CFPPDRTrainer` |
| `ffa_lesion` | FFA | Multi-lesion object detection | 8120 | Faster R-CNN ResNet-50-FPN | `FFALesionTrainer` |
| `disc_fovea` | CFP | Optic disc + fovea landmark localisation | 8121 | HRNet-W32 + regression head | `DiscFoveaTrainer` |

#### CFP quality output example
```json
{
  "quality_label": "Good",
  "quality_score": 0.91,
  "probabilities": {"Bad": 0.02, "Moderate": 0.07, "Good": 0.91}
}
```

#### FFA lesion detection output example
```json
{
  "detections": [
    {"label": "microaneurysm", "bbox": [120, 340, 145, 365], "confidence": 0.87},
    {"label": "haemorrhage",   "bbox": [220, 180, 290, 230], "confidence": 0.79}
  ],
  "lesion_types": ["microaneurysm", "haemorrhage"]
}
```

#### Disc/fovea localisation output example
```json
{
  "disc_center": [512, 490],
  "disc_radius": 88.5,
  "fovea_center": [820, 502]
}
```

---

### 4.3 Auxiliary Tools

| Tool ID | Task | Scheduling | File |
|---------|------|-----------|------|
| `gradcam` | Grad-CAM saliency heatmap for any CNN model | Inline | `tools/auxiliary/gradcam.py` |
| `roi_cropping` | Crop a rectangular region from any image | Inline | `tools/auxiliary/roi_cropping.py` |
| `ocr_detector` | Extract text from images (EasyOCR / Tesseract) | Inline | `tools/auxiliary/ocr_detector.py` |
| `web_search` | Domain-restricted literature search (PubMed, arXiv) | Inline | `tools/auxiliary/web_search.py` |

**Grad-CAM** is applied to any CNN model already loaded by the Scheduler by referencing `model_name` in its inputs. It extracts attention maps from the last convolutional layer.

---

## 5. Model Scheduling

Three scheduling modes managed by `ToolScheduler` (`ophagent/tools/scheduler.py`):

```
Tool call request
      │
      ├── scheduling_mode == "inline"
      │       └── load once into main process memory, cache instance
      │
      ├── scheduling_mode == "fastapi"
      │       └── HTTP POST to pre-loaded Docker microservice
      │           (wait for /health endpoint, then POST /run)
      │
      └── scheduling_mode == "conda"
              └── subprocess: source conda.sh → activate env → python script.py '{json}'
                  (parse JSON from stdout)
```

### Why two non-inline modes?

| Mode | Use case | Benefit |
|------|----------|---------|
| **FastAPI / Docker** | Newly developed large models (≥6 GB VRAM) | Always-hot, low latency, isolated GPU memory |
| **Conda subprocess** | Reused models with conflicting PyTorch/CUDA versions | True dependency isolation, no container overhead |

### Scheduler thread safety

- One `threading.Lock` per `tool_id` prevents concurrent model loading.
- A global lock protects the lock dictionary itself.
- FastAPI services use uvicorn workers for concurrent requests.

### Port allocation

| Service | Port |
|---------|------|
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

## 6. Knowledge Base

**File:** `ophagent/knowledge/knowledge_base.py`

A unified RAG store that aggregates four source types into a single FAISS index.

### Sources

| Source | Class | Content | Retrieval type |
|--------|-------|---------|---------------|
| Local Archive | `ImageReportArchive` | Past cases: image + clinical report pairs | Text (report) + Image (CLIP) |
| Operational Standards | `OperationalStandards` | Clinical SOPs, screening guidelines | Text |
| Textbooks | `TextbookSource` | Ophthalmology textbooks (PDF/TXT) | Text |
| Search Engine | `SearchEngineSource` | Live PubMed/arXiv results | Text (indexed after retrieval) |
| Interactive | `InteractiveSource` | Clinician-provided session context | Text |

### Vector store details (`ophagent/knowledge/vector_store.py`)

| Aspect | Implementation |
|--------|---------------|
| Index type | FAISS `IndexFlatIP` (inner product = cosine on unit vectors) |
| Text embedding | `sentence-transformers/all-MiniLM-L6-v2` (384-dim → zero-padded to 512) |
| Image embedding | OpenCLIP `ViT-B/32` (512-dim) |
| Unified dim | 512 (all vectors in one index) |
| Persistence | `.index` file (FAISS) + `.jsonl` metadata file |

### Local archive directory structure

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

### Textbook directory structure

```
data/textbooks/
├── duanes_ophthalmology.pdf
└── aao_guidelines/
    ├── dr_guidelines.pdf
    └── glaucoma_guideline.pdf
```

---

## 7. Innovative Strategies

### 7.1 Evidence-Guided CLIP Zero-Shot Classification

**File:** `ophagent/strategies/clip_evidence.py`
**Reference:** Paper §2.4.1

**Problem:** Standard CLIP zero-shot uses raw label names (e.g. "diabetic retinopathy") as text prompts, which are coarse and miss visual specificity.

**Solution:** For each candidate label, the LLM generates K (default 7) clinically precise visual evidence descriptors. Each descriptor is a short phrase describing what a specific visual finding looks like in the image.

**Algorithm:**

```
For each label L in candidate_labels:
  1. LLM generates K evidence descriptors:
     ["bright yellow hard exudates near the macula",
      "dot and blot haemorrhages in mid-periphery",
      "neovascularisation at the disc", ...]

  2. CLIP text encoder embeds all K descriptors → T_k  (K × D)

  3. CLIP image encoder embeds query image → I  (1 × D)

  4. Per-descriptor similarities: S = I · T_k^T  (K,)

  5. Label score = mean(S) or max(S)

After all labels:
  6. Softmax over label scores → probabilities
  7. Return top label + evidence per label
```

**Benefits:**
- Improved sensitivity for rare or subtle findings
- Interpretable: the evidence list explains what the model attended to
- Language-agnostic: ViLReF (Chinese CLIP) receives Chinese evidence descriptors

---

### 7.2 Composable Tool-Based VQA

**File:** `ophagent/strategies/vqa_composable.py`
**Reference:** Paper §2.4.2

**Problem:** VQA models answer questions about entire images but struggle with questions about specific retinal structures when no dedicated segmentation model exists.

**Solution:** Keyword-based routing selects a multi-step composition pipeline that localises → crops → applies VQA to the specific region.

**Built-in compositions:**

| Keywords | Pipeline |
|----------|----------|
| "optic disc", "cup disc", "rim", "RNFL" | `disc_fovea` → `roi_cropping` → `fundus_expert` |
| "macula", "fovea", "drusen" | `disc_fovea` → `roi_cropping` (fovea region) → `vision_unite` |
| *(default)* | `fundus_expert` (full image) |

**Example for "What is the cup-to-disc ratio?":**

```
Step 1: disc_fovea(fundus.jpg) → {disc_center: [512,490], disc_radius: 88.5}
Step 2: roi_cropping(fundus.jpg, x=380, y=358, w=265, h=265) → {cropped_b64: "..."}
Step 3: fundus_expert(cropped_b64, "What is the cup-to-disc ratio?") → {answer: "~0.6"}
```

**Adding custom compositions:**
```python
vqa = ComposableVQA(scheduler)
vqa.register_composition(
    keywords=["vessel", "artery", "vein"],
    steps=[
        {"tool_id": "automorph", "input_template": {"image_path": "{image_path}"}, "output_ref": "vessel_result", "description": "..."},
        {"tool_id": "vision_unite", "input_template": {"image_path": "{image_path}", "question": "{question}"}, "output_ref": "vqa_answer", "description": "..."},
    ]
)
```

---

### 7.3 Multi-Scale RAG with Unlabelled Databases

**File:** `ophagent/strategies/multiscale_rag.py`
**Reference:** Paper §2.4.3

**Problem:** Existing RAG systems require text labels or reports for retrieval. Most clinical image archives are unlabelled.

**Solution:** Multi-scale retrieval combines three granularities, enabling image-to-image similarity search without any labels:

| Scale | Query | Description |
|-------|-------|-------------|
| Global image | Full CLIP embedding of query image | Patient-level similarity; finds the most visually similar past cases |
| Regional image | CLIP embedding of cropped ROI (disc region via `disc_fovea`) | Structure-level similarity; finds cases with similar optic disc appearance |
| Textual | Text embedding of query string | Concept-level similarity; matches reports and guidelines |

**Score fusion:**

```
fused_score(doc) = Σ (similarity_at_scale × scale_weight)

Default weights: global_image=0.5, regional_image=0.3, text=0.2
```

Duplicate documents are deduplicated by `doc_id`; scores from multiple scales are summed.

**Indexing unlabelled images:**
```bash
python scripts/build_knowledge_base.py --unlabelled /path/to/image/archive
```

This indexes all images using CLIP embeddings — **no text labels required**. At query time, visually similar images are retrieved even if they have no associated reports.

---

## 8. LLM Backbone

**File:** `ophagent/llm/backbone.py`

Provides a unified interface over three LLM providers:

| Provider | Class | Model examples | Default |
|----------|-------|---------------|---------|
| OpenAI | `OpenAILLM` | `gpt-5`, `gpt-4o` | ✅ `gpt-5` |
| Google Gemini | `GeminiLLM` | `gemini-3.0-pro`, `gemini-1.5-pro`, `gemini-2.0-flash` | — |
| Local (Ollama) | `OpenAILLM` (with `base_url`) | `qwen2.5:72b`, `llama3.1:70b` | — |

### Prompt Library (`ophagent/llm/prompts.py`)

| Prompt | Used by | Purpose |
|--------|---------|---------|
| `PLANNER_SYSTEM` | Planner | Tool list + plan format specification |
| `PLANNER_USER` | Planner | Query + image paths + session history |
| `VERIFIER_SYSTEM` | Verifier | Medical consistency checker role |
| `VERIFIER_USER` | Verifier | Tool outputs + KB context |
| `SYNTHESIS_SYSTEM` | OphAgent | Senior ophthalmologist report writer |
| `SYNTHESIS_USER` | OphAgent | Verified findings + KB context |
| `MEMORY_CONSOLIDATION_SYSTEM` | Memory | Session → structured memory entry |
| `CLIP_EVIDENCE_SYSTEM` | EvidenceGuidedCLIP | Evidence descriptor generator |

---

## 9. Installation

### Requirements

- Python ≥ 3.10
- CUDA 11.8+ (recommended for GPU acceleration)
- Conda (for on-demand Conda tools)
- Docker + NVIDIA Container Toolkit (for FastAPI services)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/OphAgent.git
cd OphAgent

# 2. Create and activate a virtual environment
conda create -n ophagent python=3.10
conda activate ophagent

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in development mode
pip install -e .

# 5. Create environment file
cp .env.example .env
# Edit .env with your API key
```

### Conda environments for on-demand tools

```bash
# FMUE environment
conda create -n fmue_env python=3.9
conda activate fmue_env
pip install torch==1.13.0 torchvision==0.14.0 timm==0.6.12 Pillow numpy

# AutoMorph environment
conda create -n automorph_env python=3.8
conda activate automorph_env
pip install torch==1.11.0 torchvision==0.12.0 scikit-image opencv-python Pillow
```

---

## 10. Configuration

### Settings hierarchy

```
.env file (OPHAGENT_* prefix)
   └── config/settings.py (pydantic-settings)
         ├── LLMSettings
         ├── KnowledgeBaseSettings
         ├── ModelSchedulerSettings
         ├── ToolSettings
         └── TrainingSettings
```

### Key settings

| Setting | Default | Description |
|---------|---------|-------------|
| `OPHAGENT_LLM__PROVIDER` | `openai` | LLM provider (`openai` / `gemini` / `local`) |
| `OPHAGENT_LLM__MODEL_ID` | `gpt-5` | Model ID |
| `OPHAGENT_LLM__API_KEY` | — | API key |
| `OPHAGENT_SCHEDULER__CONDA_BASE` | `/opt/conda` | Conda installation path |
| `OPHAGENT_SCHEDULER__FASTAPI_BASE_URL` | `http://localhost` | Service base URL |
| `OPHAGENT_SESSION_HISTORY_LIMIT` | `50` | Max turns in short-term memory |
| `OPHAGENT_DEBUG` | `false` | Enable debug logging |

### Tool registry (`config/tool_registry.yaml`)

Each tool entry specifies:
```yaml
cfp_quality:
  name: "CFP Quality Reasoning"
  description: "..."
  modality: "CFP"
  task: "quality_assessment"
  scheduling_mode: "fastapi"  # inline | fastapi | conda
  fastapi_port: 8110
  model_weight: "models_weights/cfp_quality/"
  newly_developed: true
```

---

## 11. Building the Knowledge Base

```bash
# Index all local sources (first-time setup)
python scripts/build_knowledge_base.py

# Force re-indexing
python scripts/build_knowledge_base.py --force

# Also index an unlabelled image directory
python scripts/build_knowledge_base.py --unlabelled /data/fundus_archive

# Test retrieval
python scripts/build_knowledge_base.py --search "diabetic retinopathy macular oedema" --top-k 5
```

**Offline indexing output:**
```
data/
└── vector_store/
    ├── ophagent.index      # FAISS binary index
    └── ophagent_meta.jsonl # Document metadata (one JSON per line)
```

---

## 12. Training Models

```bash
# Train CFP quality model
python scripts/train_model.py \
  --model cfp_quality \
  --data-root data/cfp_quality \
  --epochs 50 --batch-size 32 --lr 1e-4

# Train FFA lesion detector
python scripts/train_model.py \
  --model ffa_lesion \
  --annotation-file data/ffa_lesion/annotations.json \
  --epochs 100 --batch-size 8

# Resume from checkpoint
python scripts/train_model.py \
  --model cfp_disease \
  --annotation-file data/cfp_disease/ann.json \
  --resume models_weights/checkpoints/cfp_disease_ep25_metric0.8420.pth
```

### Annotation format for multi-label classification
```json
[
  {"image_path": "data/cfp/img001.jpg", "labels": [1, 0, 0, 1, 0, 0, 0, 0]},
  {"image_path": "data/cfp/img002.jpg", "labels": [0, 1, 0, 0, 1, 0, 0, 0]}
]
```

### Annotation format for detection (FFA lesion)
```json
[
  {
    "image_path": "data/ffa/img001.jpg",
    "boxes": [[120,340,145,365], [220,180,290,230]],
    "labels": [1, 2]
  }
]
```

### Label mapping for detection
```
0: background
1: microaneurysm
2: haemorrhage
3: hard exudate
4: neovascularisation
```

---

## 13. Running Services

### Option A: Docker Compose (recommended for production)

```bash
# Build images (one per model)
docker build \
  --build-arg MODEL_ID=cfp_quality \
  --build-arg MODEL_PORT=8110 \
  -t ophagent/cfp_quality:latest \
  -f services/docker/Dockerfile.template .

# Start all services
docker-compose -f services/docker/docker-compose.yml up -d

# Check health
curl http://localhost:8110/health
# {"status": "ok", "model_id": "cfp_quality"}
```

### Option B: Python process manager

```bash
# Start all services as subprocesses
python services/fastapi_service.py --all

# Start a single service
python services/fastapi_service.py --model cfp_quality --port 8110
```

### Option C: Direct uvicorn (development)

```bash
MODEL_ID=cfp_quality MODEL_WEIGHT=models_weights/cfp_quality/best.pth \
  uvicorn ophagent.models.inference.service:app --host 0.0.0.0 --port 8110 --reload
```

---

## 14. Using the Agent

### Python API

```python
from ophagent.core.agent import OphAgent

agent = OphAgent()

# Single query
response = agent.run(
    query="Analyse this fundus image for diabetic retinopathy and provide a clinical report.",
    image_paths=["patient_cfp.jpg"],
)
print(response.report)
print(f"Needs review: {response.needs_human_review}")
print(f"Duration: {response.duration_s:.2f}s")

# Access raw tool outputs
for step_id, result in response.raw_results.items():
    print(f"Step {step_id} ({result['tool_name']}): {result['output']}")

# Interactive session
agent.chat("What is the optic disc appearance?")
agent.chat("Is there any macular oedema?", image_paths=["oct.jpg"])
agent.reset_session()  # clear short-term memory
```

### CLI

```bash
# Interactive REPL
python scripts/run_agent.py --interactive

# Single query
python scripts/run_agent.py \
  --query "Grade this CFP for diabetic retinopathy" \
  --images fundus.jpg

# JSON output
python scripts/run_agent.py \
  --query "Detect lesions in this FFA image" \
  --images angio.jpg \
  --json-out

# Batch processing
python scripts/run_agent.py \
  --batch queries.json \
  --output results.json
```

### Batch input format (`queries.json`)

```json
[
  {"query": "Grade DR severity", "images": ["patient001_cfp.jpg"]},
  {"query": "Check for glaucoma", "images": ["patient002_cfp.jpg"]},
  {"query": "Analyse OCT for AMD", "images": ["patient003_oct.jpg"]}
]
```

### Using innovative strategies directly

```python
# Evidence-Guided CLIP
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

# Composable VQA
from ophagent.strategies.vqa_composable import ComposableVQA
vqa = ComposableVQA()
answer = vqa.answer(image_path="fundus.jpg", question="Describe the cup-to-disc ratio.")
print(answer["answer"])

# Multi-Scale RAG
from ophagent.strategies.multiscale_rag import MultiScaleRAG
from ophagent.knowledge.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
rag = MultiScaleRAG(vector_store=kb.vs)
context = rag.retrieve(
    query="proliferative diabetic retinopathy treatment",
    image_path="fundus.jpg",
    image_modality="CFP",
)
print(context)
```

---

## 15. Evaluation

```bash
# Evaluate a single tool on a test set
python scripts/evaluate.py \
  --tool cfp_disease \
  --test-file data/cfp_disease/test.json \
  --output results/cfp_disease_eval.json

# Test annotation format
[
  {"image_path": "data/test/img001.jpg", "label": 1},
  {"image_path": "data/test/img002.jpg", "label": 0}
]
```

### Metrics computed automatically

| Task | Metrics |
|------|---------|
| Binary classification | Accuracy, AUC-ROC, Sensitivity, Specificity |
| Multi-class | Accuracy, F1-macro, Cohen's Kappa |
| Multi-label | F1-macro per class |
| Segmentation | Dice, IoU |
| Regression (landmarks) | MSE, MAE |

---

## 16. Full Project Structure

```
OphAgent/
├── .env.example                         # Environment variables template
├── README.md                            # Main README (English, with language links)
├── requirements.txt                     # Python dependencies
├── setup.py                             # Package installation
│
├── config/
│   ├── __init__.py
│   ├── settings.py                      # Global pydantic-settings config
│   ├── tool_registry.yaml               # All tool metadata (21 tools)
│   └── deployment.yaml                  # Docker + Conda deployment specs
│
├── docs/
│   ├── README_en.md                     # This file
│   ├── README_zh.md                     # 中文
│   ├── README_ja.md                     # 日本語
│   ├── README_ko.md                     # 한국어
│   ├── README_fr.md                     # Français
│   ├── README_es.md                     # Español
│   └── README_de.md                     # Deutsch
│
├── ophagent/
│   ├── __init__.py
│   │
│   ├── core/                            # Agent orchestration layer
│   │   ├── __init__.py
│   │   ├── agent.py                     # Main OphAgent class
│   │   ├── planner.py                   # Query → ExecutionPlan
│   │   ├── executor.py                  # Plan → tool calls → StepResults
│   │   ├── verifier.py                  # Medical consistency verification
│   │   └── memory.py                   # Short-term + long-term memory
│   │
│   ├── tools/                           # Tool pool
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseTool, ToolMetadata, mixins
│   │   ├── registry.py                  # ToolRegistry (YAML loader)
│   │   ├── scheduler.py                 # ToolScheduler (route → execute)
│   │   │
│   │   ├── classification/              # Classification tool wrappers
│   │   │   ├── cfp_quality.py           # NEW: CFP image quality reasoning
│   │   │   ├── cfp_disease.py           # NEW: multi-label retinal disease
│   │   │   ├── cfp_ffa_multimodal.py    # NEW: CFP+FFA joint classification
│   │   │   ├── uwf_quality_disease.py   # NEW: UWF quality + disease
│   │   │   ├── cfp_glaucoma.py          # NEW: glaucoma + structural features
│   │   │   ├── cfp_pdr.py              # NEW: PDR activity grading
│   │   │   ├── oct_fmue.py             # Reused: OCT B-scan (FMUE)
│   │   │   ├── uwf_mdd.py              # Reused: UWF multi-disease (UWF-MDD)
│   │   │   └── uwf_multi_abnormality.py # Reused: UWF multi-abnormality
│   │   │
│   │   ├── segmentation/
│   │   │   ├── retsam.py               # Reused: SAM-adapted retinal seg
│   │   │   └── automorph.py            # Reused: vessel seg + quantification
│   │   │
│   │   ├── detection/
│   │   │   ├── ffa_lesion.py           # NEW: FFA multi-lesion detection
│   │   │   └── disc_fovea.py           # NEW: disc + fovea localisation
│   │   │
│   │   ├── vqa/
│   │   │   ├── fundus_expert.py        # Reused: FundusExpert VQA
│   │   │   └── vision_unite.py         # Reused: VisionUnite VQA
│   │   │
│   │   ├── clip_models/
│   │   │   ├── retizero.py             # Reused: CLIP English
│   │   │   └── vilref.py               # Reused: CLIP Chinese
│   │   │
│   │   └── auxiliary/
│   │       ├── gradcam.py              # Grad-CAM explainability
│   │       ├── roi_cropping.py         # Region cropping
│   │       ├── ocr_detector.py         # OCR (EasyOCR / Tesseract)
│   │       └── web_search.py           # PubMed / arXiv search
│   │
│   ├── knowledge/                       # Knowledge base
│   │   ├── __init__.py
│   │   ├── knowledge_base.py            # Unified KB API
│   │   ├── vector_store.py             # Multimodal FAISS store
│   │   ├── local_data.py               # Image archive + standards
│   │   ├── textbook.py                 # Textbook/guideline indexing
│   │   ├── search_engine.py            # Live web search + indexing
│   │   └── interactive.py             # Session context injection
│   │      (note: InteractiveSource is in search_engine.py)
│   │
│   ├── strategies/                      # §2.4 innovative strategies
│   │   ├── __init__.py
│   │   ├── clip_evidence.py            # §2.4.1 Evidence-Guided CLIP
│   │   ├── vqa_composable.py           # §2.4.2 Composable Tool-Based VQA
│   │   └── multiscale_rag.py           # §2.4.3 Multi-Scale RAG
│   │
│   ├── models/
│   │   ├── training/                    # Model trainers
│   │   │   ├── base_trainer.py         # BaseTrainer (AMP, checkpointing)
│   │   │   ├── cfp_quality_trainer.py
│   │   │   ├── cfp_disease_trainer.py
│   │   │   ├── cfp_ffa_multimodal_trainer.py
│   │   │   ├── uwf_quality_disease_trainer.py
│   │   │   ├── cfp_glaucoma_trainer.py
│   │   │   ├── cfp_pdr_trainer.py
│   │   │   ├── ffa_lesion_trainer.py
│   │   │   └── disc_fovea_trainer.py
│   │   │
│   │   └── inference/
│   │       └── service.py              # FastAPI service factory
│   │
│   ├── llm/
│   │   ├── backbone.py                  # LLM provider abstraction
│   │   └── prompts.py                  # All system + user prompt templates
│   │
│   └── utils/
│       ├── image_utils.py               # Load, resize, crop, CLAHE, base64
│       ├── text_utils.py               # Chunk, truncate, JSON extract
│       ├── metrics.py                  # Accuracy, AUC, F1, Dice, IoU, Kappa
│       └── logger.py                   # loguru-based logger
│
├── services/
│   ├── fastapi_service.py              # Multi-service process launcher
│   └── docker/
│       ├── Dockerfile.template         # Generic model service Dockerfile
│       └── docker-compose.yml          # All 10 services orchestration
│
├── scripts/
│   ├── run_agent.py                    # CLI: interactive / single / batch
│   ├── train_model.py                  # Train any newly-developed model
│   ├── evaluate.py                     # Evaluate a tool on test set
│   └── build_knowledge_base.py         # Index all KB sources
│
└── tests/
    ├── test_planner.py
    ├── test_executor.py
    ├── test_verifier.py
    └── test_tools.py
```

---

## 17. Feature Checklist & Gap Analysis

Use this checklist to verify implementation completeness and identify areas for future work.

### Core pipeline
- [x] Planner: LLM-driven plan generation
- [x] Planner: topological sort for dependency-aware execution
- [x] Planner: fallback to `web_search` on empty plan
- [x] Planner: re-planning on Verifier conflicts
- [x] Executor: sequential step execution
- [x] Executor: dynamic input reference resolution (`${step_N.output.key}`)
- [x] Executor: failed-dependency propagation (skip dependent steps)
- [x] Executor: three scheduling modes (inline / FastAPI / Conda)
- [x] Verifier: LLM-based consistency checking
- [x] Verifier: RAG-context injection
- [x] Verifier: iterative conflict resolution (max 2 attempts)
- [x] Verifier: human-review flagging (confidence < 0.6)
- [x] Verifier: image quality gate (blocks ungradable images)
- [x] Memory: short-term session buffer (ring buffer)
- [x] Memory: long-term FAISS-backed case store
- [x] Memory: LLM-driven session consolidation
- [x] OphAgent: complete run loop
- [x] OphAgent: re-plan on conflict
- [x] OphAgent: structured clinical report generation
- [ ] OphAgent: async/streaming response mode *(not yet implemented)*
- [ ] OphAgent: multi-patient batch processing with parallelism *(single-threaded)*

### Tool pool
- [x] All 9 reused models wrapped
- [x] All 8 newly developed models wrapped
- [x] All 4 auxiliary tools
- [x] `ToolRegistry` with YAML hot-reload
- [x] `ToolScheduler` with thread-safe instance caching
- [ ] Health monitoring dashboard for all FastAPI services *(not yet implemented)*
- [ ] Automatic service restart on crash *(relies on Docker restart policy)*
- [ ] Dynamic port allocation for on-demand scaling *(port range reserved but not used)*

### Knowledge base
- [x] Unified multimodal FAISS vector store
- [x] Text embedding (sentence-transformers)
- [x] Image embedding (CLIP)
- [x] Local archive indexing (image + report pairs)
- [x] Operational standards indexing (PDF, TXT)
- [x] Textbook indexing (PDF, TXT)
- [x] Live web search (DuckDuckGo, PubMed via Biopython)
- [x] Interactive session context injection
- [ ] Incremental index updates (currently rebuilds full index) *(performance gap)*
- [ ] Index sharding for very large archives (>1M images) *(not yet implemented)*
- [ ] DICOM file support *(not yet implemented)*

### Innovative strategies
- [x] §2.4.1 Evidence-Guided CLIP: LLM evidence generation
- [x] §2.4.1 Evidence-Guided CLIP: per-label embedding + score aggregation
- [x] §2.4.1 Evidence-Guided CLIP: evidence caching
- [x] §2.4.2 Composable VQA: keyword-based pipeline routing
- [x] §2.4.2 Composable VQA: disc/macula/default compositions
- [x] §2.4.2 Composable VQA: custom composition registration API
- [x] §2.4.3 Multi-Scale RAG: 3-scale weighted retrieval
- [x] §2.4.3 Multi-Scale RAG: unlabelled image indexing (CLIP only)
- [x] §2.4.3 Multi-Scale RAG: score fusion with per-scale weights
- [ ] §2.4.3 Multi-Scale RAG: configurable scale weights per modality *(hardcoded)*
- [ ] §2.4.3 Multi-Scale RAG: additional scales (e.g. vessel-region crop) *(extensible but not added)*

### Model training
- [x] BaseTrainer: AMP, early stopping, LR scheduling (CosineAnnealingLR)
- [x] BaseTrainer: checkpoint save/load
- [x] All 8 model-specific trainers implemented
- [ ] Distributed training (DDP) *(not yet implemented)*
- [ ] Hyperparameter sweep integration *(not yet implemented)*
- [ ] Pre-training on fundus foundation models *(not yet implemented)*

### Services & deployment
- [x] FastAPI service factory (single codebase, ENV-configured per model)
- [x] Docker Compose (10 services)
- [x] Dockerfile template
- [x] Multi-service Python process launcher
- [ ] Kubernetes/Helm charts *(not yet implemented)*
- [ ] Model weight auto-download script *(not yet implemented)*
- [ ] CI/CD pipeline (GitHub Actions) *(not yet implemented)*

### Testing
- [x] `test_planner.py` (5 tests)
- [x] `test_executor.py` (5 tests)
- [x] `test_verifier.py` (6 tests)
- [x] `test_tools.py` (6 tests)
- [ ] Integration tests (end-to-end agent run with mock tools) *(not yet implemented)*
- [ ] Test coverage reporting *(not configured)*
- [ ] Performance benchmarks *(not yet implemented)*

---

## 18. Extension Guide

### Adding a new tool

1. Add an entry to `config/tool_registry.yaml`:
```yaml
my_new_tool:
  name: "My New Tool"
  description: "..."
  modality: "CFP"
  task: "classification"
  scheduling_mode: "fastapi"
  fastapi_port: 8130
  model_weight: "models_weights/my_new_tool/"
  newly_developed: true
```

2. Create `ophagent/tools/classification/my_new_tool.py`:
```python
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata

class MyNewTool(FastAPIToolMixin, BaseTool):
    def run(self, inputs):
        from ophagent.utils.image_utils import image_to_base64
        return self._post(self.metadata.fastapi_port, "/run",
                          {"image_b64": image_to_base64(inputs["image_path"])})
```

3. Register in `ophagent/tools/scheduler.py` `_TOOL_CLASS_MAP`:
```python
"my_new_tool": "ophagent.tools.classification.my_new_tool:MyNewTool",
```

### Adding a new LLM provider

Subclass `BaseLLM` in `ophagent/llm/backbone.py` and add a branch in `LLMBackbone.__init__()`.

### Switching the default LLM

```bash
# In .env:
OPHAGENT_LLM__PROVIDER=openai
OPHAGENT_LLM__MODEL_ID=gpt-4o
OPHAGENT_LLM__API_KEY=sk-...
```

### Adding a new knowledge source

1. Create a class similar to `TextbookSource` in `ophagent/knowledge/`.
2. Add it to `KnowledgeBase.__init__()`.
3. Call `self.vs.add_text()` or `self.vs.add_image()` to index documents.
