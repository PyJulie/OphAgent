# Source Components and Model Assets

OphAgent uses a code release with separately distributed model assets:

- The repository contains the agent, Web UI, adapters, training utilities, and
  inference source that OphAgent owns or is authorised to redistribute.
- ReT-SAM 2.0 and G-DISC inference source is bundled under
  `ophagent/components/`; their weights and calibration assets remain external.
- Public third-party source is installed at a locked Git revision.
- Model weights, datasets, credentials, clinical examples, reports, and caches
  stay outside the repository.
- A source component is not considered reproducible until its revision and
  license status are known.

The default runtime layout is:

```text
<OPHAGENT_RUNTIME_DIR>/
  .env
  checkpoints/
  external/
  reports/
  cache/
```

When `OPHAGENT_RUNTIME_DIR` is unset, this root defaults to `~/.ophagent`.

## Install Public Source

Install the project first:

```bash
pip install -e .
```

For the G-DISC OCT optic-disc pipeline, install its CUDA and conversion
dependencies as well:

```bash
pip install -e ".[oct-disc]"
```

Then install all public components whose upstream repositories declare a
redistribution license:

```bash
ophagent-components install --all
```

The command clones each repository into
`<OPHAGENT_RUNTIME_DIR>/external/`, checks out the exact revision in
`ophagent/resources/components.yaml`, verifies marker files, and never
downloads model weights.

Inspect the current source state with:

```bash
ophagent-components status --profile manuscript-full
```

Use `--json` for a machine-readable report.

## Component Status

| Component | Integration | License status | Default action |
|---|---|---|---|
| Chinese-CLIP | Locked upstream source | MIT | Installed by `--all` |
| FLAIR | Locked upstream source | Apache-2.0 | Installed by `--all` |
| EFIQA inference glue | Built into OphAgent | EFIQA upstream MIT; OphAgent glue uses project license | No source install |
| Glaucoma inference architecture | Built into OphAgent | Project license | No source install |
| PDR cascade inference architecture | Built into OphAgent | Project license | No source install |
| ReT-SAM 2.0 | Bundled inference and quantitative post-processing source | Project release; MONAI-derived files retain Apache-2.0 notices | No source install |
| G-DISC OCT | Bundled orchestration and checkpoint-compatible inference source | Approved under project release | No source install |
| OCTCubeM | Locked upstream source | BSD-2-Clause | Installed by `--all` |
| RetiZero | Locked upstream source | No upstream license declared | Explicit acknowledgement required |
| FMUE | Locked upstream source | No upstream license declared | Explicit acknowledgement required |

RetiZero or FMUE may be cloned for an authorised local environment only after
reviewing their upstream terms:

```bash
ophagent-components install retizero fmue --allow-unlicensed
```

This flag requires an explicit operator decision; it does not grant a license
or permit redistribution.

## Model Assets

Weights are intentionally absent from Git. The public inventory is
`ophagent/resources/model_assets.yaml`, which records the expected relative
path, environment override, size, SHA-256 digest, role, and release profile.

Place separately supplied assets under
`<OPHAGENT_RUNTIME_DIR>/checkpoints/`, or set the corresponding environment
variable. For example:

```text
OPHAGENT_GLAUCOMA_WEIGHTS=/secure/path/glaucoma.pth
OPHAGENT_EFIQA_WEIGHTS=/secure/path/efiqa_adapter.npz
```

The EFIQA DINOv3 backbone is a gated Hugging Face model and must be obtained
under its own access terms. OphAgent pins the backbone revision used by the
adapter.

## Readiness Levels

`ophagent-components status` verifies source code only. It does not imply that
weights, API credentials, or every manuscript tool are available.

Use both checks for a full local deployment:

```bash
ophagent-assets verify --profile manuscript-full
ophagent-components status --profile manuscript-full
ophagent-preflight
```

ReT-SAM 2.0 and G-DISC use the bundled, checkpoint-compatible source and must
not be replaced with similarly named repositories. RetiZero and FMUE remain
upstream-linked components whose audited revisions declare no license; users
must review their terms before the explicit `--allow-unlicensed` installation.

The source/runtime boundary and complete validation workflow are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and
[`VALIDATION.md`](VALIDATION.md).
