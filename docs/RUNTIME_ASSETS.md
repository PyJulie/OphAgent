# Runtime assets and full setup

The OphAgent source repository does not contain model checkpoints. A complete
local configuration combines the public source tree with a separately
distributed runtime-asset archive:

`OphAgent-runtime-assets-0.1.0.zip`

The expected archive name, size, SHA-256, directory layout, and asset count are
recorded in [`reviewer/assets.json`](../reviewer/assets.json). Obtain the
archive through the authorised distribution channel for the relevant release.
The download location is intentionally not stored in the repository.

## Distribution boundary

| Location | Contents |
|---|---|
| Source repository | Source code, tests, configuration templates, component manifests, checkpoint metadata, and the asset installer |
| Runtime-asset archive | 37 checkpoint files for the `manuscript-full` profile and an asset manifest |
| Local runtime | API credentials, installed external components, authorised inputs, sessions, reports, logs, and caches |

Keep the runtime outside the Git checkout. The runtime-asset archive contains
no API credentials, Web passwords, datasets, clinical inputs, sessions, logs,
or generated reports.

## Requirements

- Python 3.10 or newer and Git.
- A CUDA-capable NVIDIA GPU is recommended for the complete local tool set.
- A filesystem that supports files larger than 4 GB.
- At least 50 GiB free when the downloaded ZIP and extracted runtime share a
  disk, with additional space for Python and model caches.
- Internet access for Python packages, pinned external components, and the
  selected hosted language-model API.

## 1. Clone and install the code

PowerShell:

```powershell
git clone https://github.com/PyJulie/OphAgent.git
Set-Location .\OphAgent
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[oct-disc]"
```

bash or zsh:

```bash
git clone https://github.com/PyJulie/OphAgent.git
cd OphAgent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[oct-disc]"
```

The code can be inspected and its offline safety tests can run without
checkpoints. Missing checkpoint-dependent tools in a code-only checkout are
expected and do not represent the complete configured pipeline.

## 2. Verify and install the runtime assets

Keep the downloaded ZIP outside the Git checkout.

PowerShell:

```powershell
.\.venv\Scripts\python reviewer\install_assets.py `
  --archive "$HOME\Downloads\OphAgent-runtime-assets-0.1.0.zip" `
  --runtime-dir "$HOME\ophagent-runtime"
$env:OPHAGENT_RUNTIME_DIR = (Resolve-Path "$HOME\ophagent-runtime").Path
```

bash or zsh:

```bash
.venv/bin/python reviewer/install_assets.py \
  --archive "$HOME/Downloads/OphAgent-runtime-assets-0.1.0.zip" \
  --runtime-dir "$HOME/ophagent-runtime"
export OPHAGENT_RUNTIME_DIR="$HOME/ophagent-runtime"
```

The installer verifies the archive size and SHA-256, rejects unsafe ZIP paths,
confirms all 37 checkpoint files, and extracts the runtime outside the
repository. It refuses to overwrite an existing runtime directory.

To verify the download without extracting it:

```bash
python reviewer/install_assets.py \
  --archive /path/to/OphAgent-runtime-assets-0.1.0.zip \
  --verify-only
```

## 3. Install pinned external components

ReT-SAM 2.0 and G-DISC inference code is bundled with the repository. Install
the other declared-license components at their locked revisions:

```bash
ophagent-components install --all
```

RetiZero and FMUE are fetched only after the user explicitly accepts the
upstream terms associated with revisions that do not declare a license:

```bash
ophagent-components install retizero fmue --allow-unlicensed
```

Use the executable under `.venv/Scripts/` on Windows when the virtual
environment is not activated.

## 4. Configure a provider locally

The asset installer creates `<OPHAGENT_RUNTIME_DIR>/.env` from `.env.example`.
Add the API key for the selected provider there. Do not put credentials in the
Git checkout.

Supported official providers and gateways include OpenAI, Anthropic Claude,
Google Gemini, DashScope, AIGCBest, and OpenRouter. The Web UI also supports
per-user credentials under **Personalize > API**. Hosted API usage may incur
charges under the user's provider account.

## 5. Validate the complete configuration

Run the checks in this order:

```bash
ophagent-assets verify --profile manuscript-full
ophagent-components status --profile manuscript-full
python -m ophagent.reviewer_smoke
ophagent-preflight --quick --json --no-save-json
ophagent-preflight --json --no-save-json
```

Expected results:

- Asset verification reports 37 verified checkpoint files.
- Component status reports the required inference components as ready.
- The offline safety smoke test returns JSON containing `"ok": true`.
- Full preflight exits with code `0` and reports `strict_ready=true` only when
  the selected provider, checkpoints, component sources, and four advertised
  modalities are operational.

The full preflight loads local model components and probes the configured API,
so it takes longer than the quick check. See
[`VALIDATION.md`](VALIDATION.md) for the scope of each check.

## 6. Start OphAgent

Web UI:

```bash
ophagent-web
```

Open `http://127.0.0.1:8765`.

CLI:

```bash
python demos/chat.py
```

For a reproducible run, record the exact provider, resolved model identifier,
effort level, task definition, input, date, source commit, and asset profile.
Hosted model aliases can change over time.

