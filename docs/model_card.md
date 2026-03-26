# OphAgent Model Card

This document follows the spirit of the [Hugging Face model card guidance](https://huggingface.co/docs/hub/model-cards) and summarizes the OphAgent system described in the accompanying manuscript and repository documentation.

## Model Details

**Model name:** OphAgent

**Version:** 0.1.0

**Model type:** Multi-model ophthalmic AI agent system

**Paper title:** *OphAgent: A Generalisable Ophthalmic Agentic System for Global Eye Care*

**Authors:** Lie Ju, Jinyuan Wang, Zhonghua Wang, and collaborators from the OphAgent Global Reader Study Group

**Reference status:** Manuscript described in the accompanying PDF draft bundled with this project context

**Summary:** OphAgent is not a single checkpoint. It is an LLM-driven orchestration system that plans, executes, verifies, and synthesizes ophthalmic image analysis by routing user queries to a pool of specialized vision models and auxiliary tools.

**Primary modalities:** CFP, OCT, UWF, FFA, and natural-language queries

**Core architecture:** Planner -> Executor -> Verifier -> Memory Manager

**Execution backends:** Inline Python tools, FastAPI model services, and Conda-isolated tools

## Model Description

OphAgent is designed for multimodal ophthalmic image analysis and clinical question answering. Given a retinal image and a natural-language request, the system uses an LLM-backed planner to select the most relevant tools, executes them with dependency-aware scheduling, checks the consistency of intermediate findings with a verifier and retrieval-augmented context, and then generates a structured clinical-style report.

The system combines reused ophthalmic models with newly developed task-specific models. It also includes three paper-level strategies that shape its behavior:

- Evidence-Guided CLIP for zero-shot retinal classification
- Composable Tool-Based VQA for region-specific question answering
- Multi-Scale RAG for retrieval over text and unlabelled image archives

## Supported Tasks and Modalities

| Modality | Example tasks |
|---|---|
| CFP | Image quality assessment, multi-label disease classification, glaucoma screening, PDR grading, vessel analysis, disc/fovea localisation, VQA |
| OCT | B-scan classification |
| UWF | Quality assessment, disease detection, multi-abnormality screening |
| FFA | Lesion detection |
| CFP + FFA | Joint multimodal classification |
| Text | Clinical querying, evidence generation, retrieval, report synthesis |

## Component Models

OphAgent currently exposes 21 tools in total. Among them, 17 are model-backed tools and 4 are auxiliary tools.

### Reused models

| Tool ID | Model | Modality | Task | Backend | Link |
|---|---|---|---|---|---|
| `fmue` | FMUE | OCT | B-scan classification | Conda | https://github.com/yuanyuanpeng0129/FMUE |
| `uwf_mdd` | UWF-MDD | UWF | Multi-disease detection | Conda | https://github.com/justinengelmann/UWF_multiple_disease_detection |
| `uwf_multi_abnormality` | UWF Multi-Abnormality | UWF | Multi-label screening | Conda | https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1584378/full |
| `retizero` | RetiZero | CFP | CLIP zero-shot classification | Inline | https://github.com/LooKing9218/RetiZero |
| `vilref` | ViLReF | CFP | CLIP zero-shot classification | Inline | https://github.com/T6Yang/ViLReF |
| `fundus_expert` | FundusExpert | CFP | Visual question answering | FastAPI | https://github.com/MeteorElf/FundusExpert |
| `vision_unite` | VisionUnite | CFP | Visual question answering | FastAPI | https://github.com/HUANGLIZI/VisionUnite |
| `automorph` | AutoMorph | CFP | Vessel segmentation and morphometrics | Conda | https://github.com/rmaphoh/AutoMorph |
| `retsam` | RetSAM | CFP | Retinal segmentation | Inline | https://github.com/Wzhjerry/retsam |

### Newly developed models

| Tool ID | Modality | Task | Architecture | Model Weights |
|---|---|---|---|---|
| `cfp_quality` | CFP | Quality reasoning | RetFound-DINOv2 | coming soon |
| `cfp_disease` | CFP | Multi-label retinal disease classification | RetFound-DINOv2 | coming soon |
| `cfp_ffa_multimodal` | CFP + FFA | Dual-modality classification | Modified RetFound-DINOv2 | coming soon |
| `uwf_quality_disease` | UWF | Joint quality and disease prediction | RetFound-DINOv2 | coming soon |
| `cfp_glaucoma` | CFP | Referable glaucoma and structural analysis | RetFound-DINOv2 | coming soon |
| `cfp_pdr` | CFP | PDR activity grading | RetFound-DINOv2 | coming soon |
| `ffa_lesion` | FFA | Multi-lesion detection | RT-DETR | coming soon |
| `disc_fovea` | CFP | Optic disc and fovea localisation | YOLOv8m | coming soon |

### Auxiliary tools

| Tool ID | Purpose |
|---|---|
| `gradcam` | Saliency explanation |
| `roi_cropping` | Region extraction |
| `ocr_detector` | OCR from images |
| `web_search` | Literature and web search |

## Intended Use

OphAgent (including all models) is intended for:

- Research on agentic orchestration of ophthalmic AI tools
- Prototyping clinical decision-support workflows for retinal imaging
- Educational use in multimodal medical AI systems
- Assisted analysis of supported image modalities under clinician oversight

## Out-of-Scope Use

OphAgent is not intended for:

- Any commercial purpose
- Fully autonomous diagnosis or treatment decisions
- Emergency triage or safety-critical use without human review
- Use as a certified medical device
- Modalities, patient populations, or acquisition settings not represented by the component models
- Real-model benchmarking when the system is running in degraded or fallback mode

## Inputs and Outputs

### Inputs

- A natural-language user query
- One or more images from supported modalities
- Optional session history and retrieved knowledge-base context

### Outputs

- Tool-level intermediate predictions
- A verifier verdict with confidence and conflict notes
- A synthesized clinical-style report
- A `needs_human_review` flag when confidence is low or degraded execution is used

## Training and Evaluation

### Training

The repository includes task-specific training entry points and trainer classes for newly developed models. Public documentation currently describes model architectures and evaluation interfaces, but does not yet provide a complete per-model training-data manifest in a machine-readable form.

The project should ideally publish, for each component model:

- Training and validation datasets
- Inclusion and exclusion criteria
- Device and site diversity
- Preprocessing pipelines
- Class definitions and label maps
- Exact checkpoint provenance

### Evaluation

The repository provides an evaluation script at `scripts/evaluate.py` and supports the following metric families:

| Task type | Metrics |
|---|---|
| Binary classification | Accuracy, AUC-ROC, Sensitivity, Specificity |
| Multi-class classification | Accuracy, F1-macro, Cohen's Kappa |
| Multi-label classification | Per-class and macro F1 |
| Segmentation | Dice, IoU |
| Landmark regression | MSE, MAE |

Noted that we did not fully evaluate these metrics in our orginal manuscript. More results will be updated.

## Limitations

- OphAgent is a system-level orchestrator, so end-to-end behavior depends on both the LLM backbone and the availability of model services.
- Some components rely on external FastAPI services or isolated Conda environments, which can complicate reproducibility.
- The repository supports a `graceful` mode with fallback behavior for local development. That mode is useful for smoke tests, but should not be treated as real-model performance.
- `strict` mode is the appropriate setting for real-model validation because it fails fast on unavailable or degraded backends.
- Knowledge-base retrieval can influence verification and final report wording.

## Bias, Risks, and Safety

- Performance may vary across devices, institutions, image quality levels, and patient populations.
- LLM-based planning, verification, and report synthesis can introduce language bias or overconfident phrasing.
- Retrieval-augmented outputs depend on the quality and recency of the knowledge base.
- Poor-quality or unsupported images can lead to unreliable outputs even when the pipeline completes.
- Clinical use should require qualified human oversight, especially for low-confidence, conflicting, or safety-relevant findings.

## Responsible Use

- Review outputs with an ophthalmologist or qualified clinician before acting on them.
- Avoid uploading patient-identifiable data to external services unless governance and consent requirements are satisfied.
- Document whether results were produced in `graceful` or `strict` mode when reporting experiments.
- Treat the generated report as decision support, not as a standalone diagnosis.

## Citation

Use the following provisional citation until a final DOI, venue, or preprint identifier is available:

```bibtex
@misc{ju2026ophagent,
  title={OphAgent: A Generalisable Ophthalmic Agentic System for Global Eye Care},
  author={Lie Ju and Jinyuan Wang and Zhonghua Wang and others},
  year={2026},
  note={Manuscript draft and repository documentation}
}
```

