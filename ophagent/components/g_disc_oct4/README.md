# G-DISC OCT Inference Component

This directory contains the `octseg` orchestration layer and the numerical
inference code used by `oct_volume_disc`. It excludes checkpoints, calibration
files, input metadata, RDSS/SAN operations, prior runs, and generated figures.

Four separately distributed assets are required under
`<OPHAGENT_RUNTIME_DIR>/checkpoints/oct_volume/g_disc/`:

- retinal-layer segmentation checkpoint;
- W-Net optic-disc localisation checkpoint;
- 3D OCT histogram calibration; and
- Triton histogram calibration.

Their exact paths, sizes, and SHA-256 values are recorded in
`ophagent/resources/model_assets.yaml`. Environment overrides are documented
in `.env.example`.

The pipeline includes attributed model implementations from its research
development lineage. Distribution is approved under the OphAgent project
release; see `THIRD_PARTY.md` for attribution and dependency notes.
