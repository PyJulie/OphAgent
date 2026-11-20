# Reproducibility and runtime boundaries

OphAgent separates the public source tree from a private, mutable runtime. A
source checkout contains the code and metadata required to verify an
authorised set of model assets, while clinical inputs, sessions, logs, reports,
credentials, and machine-specific state remain local.

## Source and runtime boundary

| Public source release | Private runtime |
|---|---|
| OphAgent source and tests | API credentials and Web passwords |
| Configuration templates | Model weights |
| Approved bundled component source and locked source manifest | Other installed upstream source trees |
| Model-asset paths, sizes, and SHA-256 values | Datasets and clinical inputs |
| Tutorials and reproducibility instructions | Uploads, sessions, logs, reports, and caches |
| Source version and asset integrity metadata | Local path overrides |

The recommended runtime is outside the Git checkout:

```text
ophagent-runtime/
  .env
  checkpoints/
  external/
  reports/
  cache/
  config/
```

OphAgent defaults to `~/.ophagent` when `OPHAGENT_RUNTIME_DIR` is not set.

## Reproducing from source and separately supplied weights

Create an isolated environment and a separate runtime:

```bash
git clone https://github.com/PyJulie/OphAgent.git
cd OphAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[oct-disc,dev]"
export OPHAGENT_RUNTIME_DIR="$HOME/ophagent-runtime"
mkdir -p "$OPHAGENT_RUNTIME_DIR/checkpoints"
cp .env.example "$OPHAGENT_RUNTIME_DIR/.env"
```

PowerShell:

```powershell
git clone https://github.com/PyJulie/OphAgent.git
Set-Location OphAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[oct-disc,dev]"
$env:OPHAGENT_RUNTIME_DIR = "$HOME\ophagent-runtime"
New-Item -ItemType Directory -Force "$env:OPHAGENT_RUNTIME_DIR\checkpoints"
Copy-Item .env.example "$env:OPHAGENT_RUNTIME_DIR\.env"
```

The `dev` extra installs `pytest`, which the test suite below requires. The
commands in the rest of this guide assume that the virtual environment is
active.

Install redistributable external source at the revisions recorded in
`ophagent/resources/components.yaml`:

```bash
ophagent-components install --all
```

For the manuscript-full profile, RetiZero and FMUE must also be cloned from
their locked upstream revisions after reviewing their undeclared license
status:

```bash
ophagent-components install retizero fmue --allow-unlicensed
```

The verified archive installer and complete cross-platform instructions are
provided in [`RUNTIME_ASSETS.md`](RUNTIME_ASSETS.md). For manual installation,
place each authorised weight under `<OPHAGENT_RUNTIME_DIR>/checkpoints/` at the
relative path recorded in `ophagent/resources/model_assets.yaml`. An
environment-variable override listed for an asset may be used when that file
is stored elsewhere.

Check presence and expected size:

```bash
ophagent-assets status --profile manuscript-full
```

Perform the slower byte-level verification before a benchmark or formal
deployment:

```bash
ophagent-assets verify --profile manuscript-full
ophagent-components status --profile manuscript-full
python -m pytest -q
ophagent-preflight --json --no-save-json
```

Run the test suite from the repository root. It needs no checkpoints, API
keys, or network access, and reports 82 passing tests at release 0.1.0. The
plain `pytest -q` command gives the same result, because the test
configuration in `pyproject.toml` adds the repository root to the import
path.

The Web checkpoint page uses the same packaged asset manifest, so a new clone
does not require a developer-generated local manifest. A runtime
`checkpoints/MANIFEST.yaml` remains an optional authorised override.

## Software environments

Release 0.1.0 installs under Python 3.9 or newer. Dependencies are declared
in `pyproject.toml` with minimum versions. PyTorch 2.6 or newer is required,
because the checkpoint loaders read the released weights with
`weights_only=True` and an explicit allow-list of permitted classes
(`torch.serialization.safe_globals`), which earlier PyTorch versions do not
provide.

Two fully resolved environments in which the test suite and the offline
safety checks pass are recorded in
[`environments/`](../environments/README.md):

| File | Python | PyTorch | Torchvision |
|---|---|---|---|
| `environments/tested-python3.9.txt` | 3.9 | 2.8.0 | 0.23.0 |
| `environments/tested-python3.10.txt` | 3.10 | 2.14.1 | 0.29.2 |

Install one of them by passing the file as a constraints file:

```bash
python -m pip install -e ".[dev]" -c environments/tested-python3.9.txt
```

The experiments reported in the manuscript were run before this release was
prepared, with Python 3.9, PyTorch 2.2.1 (CUDA 11.8), Torchvision 0.17.1,
NumPy 1.26.4, Pandas 2.2.2 and Pillow 10.3.0, and each integrated model ran in
its native software environment. The release reads the same checkpoint files,
which are verified by SHA-256, and differs from that environment in the
PyTorch version for the reason given above.

## Meaning of each check

- `ophagent-assets verify` proves that the supplied files match the released
  byte-level asset inventory. It does not grant permission to use or
  redistribute those files.
- `ophagent-components status` proves that required external source is present
  at the locked revision, or reports a release blocker.
- `python -m pytest -q` checks deterministic code behaviour without claiming
  clinical performance.
- The full `ophagent-preflight` loads the configured runtime stack and reports
  missing or failed tools. It must pass before a run is described as using the
  full configured pipeline.

At release version 0.1.0, the checkpoint-compatible ReT-SAM 2.0 and G-DISC
inference source ships with OphAgent. Their model and calibration assets remain
separate and are verified through the public asset manifest. The
manuscript-full profile additionally requires the pinned RetiZero and FMUE
upstream checkouts described above.

## Data-safe validation

Use only examples whose redistribution has been explicitly approved. Keep all
local validation images and DICOM files under the private runtime or another
ignored directory. Do not commit clinical inputs, generated reports, API
credentials, or runtime logs to the source repository. See
[`VALIDATION.md`](VALIDATION.md) for the distinct scopes of asset verification,
offline safety tests, registration checks, and full runtime preflight.
