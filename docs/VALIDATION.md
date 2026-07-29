# Runtime validation

The checks below answer different reproducibility questions. Run them from the
repository root and interpret each result within its stated scope.

## 1. Asset integrity

```bash
ophagent-assets verify --profile manuscript-full
```

This performs byte-level verification against the packaged asset inventory. A
passing result establishes that the supplied files match the expected runtime
assets; it does not test model execution or grant permission to use or
redistribute those files.

## 2. Offline safety smoke test

```bash
python -m ophagent.reviewer_smoke
```

This command removes API keys from its own process and does not load model
weights. It checks deterministic input and finalization gates. Success is a
zero exit code and JSON containing `"ok": true`, with one passing result for
each of these cases:

- missing file refusal;
- non-image file refusal;
- non-ophthalmic input refusal;
- refusal when ophthalmic scope cannot be verified;
- complete core-observer failure suppression;
- unchanged single-modality finalization.

This result verifies those control paths only. It does not establish model
availability or diagnostic performance.

## 3. Registration preflight

```bash
python -m ophagent.preflight --quick --json --no-save-json
```

Quick preflight resolves the effective `OPH_WEB_*` configuration, checks that
the selected providers have credentials, imports the adapter registry, and
reports core tools registered for CFP, OCT, UWF, and FFA. It does not make an
LLM request or load model checkpoints. The JSON fields
`summary.quick=true`, `summary.strict_ready=false`, and
`summary.strict_stack_probed=false` make that scope explicit.

## 4. Full component preflight

```bash
python -m ophagent.preflight --json --no-save-json
```

Full preflight probes the exact planner provider/model selected by
`OPH_WEB_BACKEND` and `OPH_WEB_MODEL`. When a distinct
`OPH_WEB_VISION_BACKEND` or `OPH_WEB_VISION_MODEL` is configured, that exact
provider/model pair is probed separately. It then loads the modality detector,
loads registered adapters one at a time, checks external source directories,
and reports per-modality core-observer coverage.

Arguments can make the validated configuration explicit:

```bash
python -m ophagent.preflight \
  --backend dashscope \
  --model qwen3-vl-plus \
  --vision-backend aigcbest \
  --vision-model gpt-5-mini \
  --effort high \
  --json --no-save-json
```

Argument values override environment values. The JSON `runtime` object records
the effective planner, vision, and effort values and whether each came from an
argument, environment variable, default, or planner inheritance. Credentials
are represented only by source/status and are never printed.

Exit code `0` and `summary.strict_ready=true` mean that every check in that
full run passed and each advertised modality had an operational core observer.
Exit code `1` identifies failed components in `checks` and `modalities`. A
code-only checkout can therefore be inspected without being mistaken for a
fully configured runtime.

## 5. Web configuration inspection

Start the service and inspect a newly created session:

```bash
ophagent-web
curl -X POST http://127.0.0.1:8765/api/sessions
```

The response includes `backend`, `model`, `effort`, and a `runtime` object. Its
`components` section resolves the planner, vision, verifier, and debate roles;
the vision entry includes availability and the resolution reason. The runtime
object reports configuration, not component readiness. Use full preflight for
the readiness decision.

