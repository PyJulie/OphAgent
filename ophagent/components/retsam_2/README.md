# ReT-SAM 2.0 Inference Component

This directory contains the inference and quantitative post-processing source
used by `cfp_retsam_segmentation`. It intentionally excludes model weights,
training data, cached masks, prior runs, and generated visualisations.

The separately distributed checkpoint is recorded as `cfp_retsam` in
`ophagent/resources/model_assets.yaml`. OphAgent passes its path explicitly to
`scripts/infer.py`.

The Swin UNETR implementation under `models/` contains MONAI-derived,
Apache-2.0-licensed portions. See the file headers and the repository-level
`THIRD_PARTY.md`. Remaining project-authored integration and post-processing
source is distributed under the OphAgent project license.
