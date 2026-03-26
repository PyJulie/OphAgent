# OphAgent

<div align="center">

**Language / 语言**

[🇨🇳 中文](./docs/README_zh.md) | [**🇬🇧 English**](./docs/README_en.md)

> **Full technical reference** is in [`docs/README_en.md`](./docs/README_en.md).
> **中文搭建指南** in [`docs/GUIDE_zh.md`](./docs/GUIDE_zh.md).
> This page is a quick-start overview.

</div>

---

An LLM-driven ophthalmic AI agent that orchestrates a pool of specialised vision models to analyse retinal images and answer clinical queries.

OphAgent supports two runtime modes:

- `graceful`: local-development mode. If an LLM, model service, weight file, or optional dependency is unavailable, the pipeline can fall back to heuristic execution so end-to-end smoke tests still run.
- `strict`: production-oriented mode. Real backends are required; degraded execution fails fast instead of silently falling back.

## Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌──────────┐
│ Planner │────▶│ Executor │────▶│ Verifier │
│  (LLM)  │     │          │     │ (LLM+RAG)│
└─────────┘     └──────────┘     └────┬─────┘
      ▲                               │ verdict
      │        Memory Manager         ▼
      └──── (short-term + FAISS) ─ Report
```

## Tool Pool — 21 tools across 6 capability groups

| Category | Tools |
|----------|-------|
| Classification (9) | `cfp_quality` `cfp_disease` `cfp_ffa_multimodal` `uwf_quality_disease` `cfp_glaucoma` `cfp_pdr` `fmue` `uwf_mdd` `uwf_multi_abnormality` |
| Segmentation (2) | `retsam` `automorph` |
| Detection (2) | `ffa_lesion` `disc_fovea` |
| VQA (2) | `fundus_expert` `vision_unite` |
| CLIP (2) | `retizero` `vilref` |
| Auxiliary (4) | `gradcam` `roi_cropping` `ocr_detector` `web_search` |

## Innovative Strategies (§2.4)

| Strategy | File | Description |
|----------|------|-------------|
| Evidence-Guided CLIP | `strategies/clip_evidence.py` | LLM generates visual evidence descriptors → better zero-shot accuracy |
| Composable Tool VQA | `strategies/vqa_composable.py` | Auto localise → crop → VQA for region-specific questions |
| Multi-Scale RAG | `strategies/multiscale_rag.py` | 3-scale retrieval over unlabelled image archives |

## Environment Setup

### Minimum requirements

- Python `>= 3.10`
- Conda or Miniconda recommended
- Docker recommended if you want FastAPI model services
- NVIDIA GPU + CUDA 11.8+ recommended for real-model inference

### What works in each setup

| Setup | What you can do |
|------|------------------|
| Python-only | Explore the codebase, run tests, run the agent in `graceful` mode with heuristic fallback |
| Python + API key | Use the real LLM backbone for planning / verification / report synthesis |
| Python + Docker services + model weights | Run FastAPI-backed vision tools |
| Python + Docker + extra Conda envs + model weights | Run the full mixed scheduling stack, including reused Conda tools |

### Recommended install steps

```bash
# 1. Clone
git clone https://github.com/PyJulie/OphAgent.git
cd OphAgent

# 2. Create the main environment
conda create -n ophagent python=3.10 -y
conda activate ophagent

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project in editable mode
pip install -e .

# 5. Create local config
cp .env.example .env
```

### Optional Conda environments for reused tools

Some reused models run in isolated Conda environments because they depend on older PyTorch/CUDA stacks.

```bash
# FMUE
conda create -n fmue_env python=3.9 -y
conda activate fmue_env
pip install torch==1.13.0 torchvision==0.14.0 timm==0.6.12 Pillow numpy
conda deactivate

# AutoMorph
conda create -n automorph_env python=3.8 -y
conda activate automorph_env
pip install torch==1.11.0 torchvision==0.12.0 scikit-image opencv-python Pillow
conda deactivate

# UWF-MDD
conda create -n uwf_mdd_env python=3.9 -y
conda activate uwf_mdd_env
pip install torch==2.0.0 torchvision==0.15.0 timm Pillow numpy
conda deactivate

conda activate ophagent
```

## Quick Start

```bash
# Add your API key or local-model settings to .env
# Then optionally choose a runtime mode:

# Optional: force production-style execution
# export OPHAGENT_RUNTIME__MODE=strict
#
# Default local-development mode is graceful:
# - missing LLM/model services can fall back to heuristic execution
# - reports are marked for human review when degraded execution is used

# Build knowledge base
python scripts/build_knowledge_base.py

# Start model services (optional for lightweight exploration,
# required for FastAPI-backed real vision tools)
docker-compose -f services/docker/docker-compose.yml up -d

# Run (interactive)
python scripts/run_agent.py --interactive

# Run (single query)
python scripts/run_agent.py \
  --query "Analyse this fundus image for diabetic retinopathy." \
  --images patient_cfp.jpg
```

## Runtime Modes

Use `graceful` for local smoke tests and repository exploration:

```bash
export OPHAGENT_RUNTIME__MODE=graceful
```

Use `strict` when you want real-model execution only:

```bash
export OPHAGENT_RUNTIME__MODE=strict
```

In `strict` mode, OphAgent fails fast if the configured LLM, FastAPI service, Conda tool, or model weights are unavailable or degraded.

## Python API

```python
from ophagent.core.agent import OphAgent

agent = OphAgent()
response = agent.run(
    query="Grade this CFP for diabetic retinopathy.",
    image_paths=["fundus.jpg"],
)
print(response.report)
```

## Train a Model

```bash
python scripts/train_model.py --model cfp_quality --data-root data/cfp_quality
```

## Documentation

| Document | Language |
|----------|----------|
| [`docs/README_en.md`](./docs/README_en.md) | 🇬🇧 English (full technical reference) |
| [`docs/README_zh.md`](./docs/README_zh.md) | 🇨🇳 中文 |
| [`docs/GUIDE_zh.md`](./docs/GUIDE_zh.md) | 🇨🇳 中文搭建指南 |
| [`docs/model_card.md`](./docs/model_card.md) | Hugging Face-style model card for the OphAgent system |

## Notes

- The main quick-start path is the root README plus [`docs/README_en.md`](./docs/README_en.md) or [`docs/README_zh.md`](./docs/README_zh.md).
- `pip install -e .` alone is not enough for the full stack; use `pip install -r requirements.txt` first.
- Some backends require external services, model weights, GPU support, or isolated Conda environments.
- If you are validating the repository in a lightweight environment, prefer `graceful` mode first; switch to `strict` once all real dependencies are present.
- For a fuller step-by-step setup guide, see [`docs/GUIDE_zh.md`](./docs/GUIDE_zh.md) or the installation sections in [`docs/README_en.md`](./docs/README_en.md).

## Tests

```bash
pytest tests/ -v
```
