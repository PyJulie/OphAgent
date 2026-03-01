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

## Tool Pool — 21 tools across 5 modalities

| Category | Tools |
|----------|-------|
| Classification (9) | `cfp_quality` `cfp_disease` `cfp_ffa_multimodal` `uwf_quality_disease` `cfp_glaucoma` `cfp_pdr` `oct_fmue` `uwf_mdd` `uwf_multi_abnormality` |
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

## Quick Start

```bash
# Install
pip install -e .
cp .env.example .env   # add your API key

# Build knowledge base
python scripts/build_knowledge_base.py

# Start model services
docker-compose -f services/docker/docker-compose.yml up -d

# Run (interactive)
python scripts/run_agent.py --interactive

# Run (single query)
python scripts/run_agent.py \
  --query "Analyse this fundus image for diabetic retinopathy." \
  --images patient_cfp.jpg
```

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
| [`docs/README_ja.md`](./docs/README_ja.md) | 🇯🇵 日本語 |
| [`docs/README_ko.md`](./docs/README_ko.md) | 🇰🇷 한국어 |
| [`docs/README_fr.md`](./docs/README_fr.md) | 🇫🇷 Français |
| [`docs/README_es.md`](./docs/README_es.md) | 🇪🇸 Español |
| [`docs/README_de.md`](./docs/README_de.md) | 🇩🇪 Deutsch |

## Tests

```bash
pytest tests/ -v
```
