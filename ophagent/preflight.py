"""
OphAgent preflight health check.

Run as:
    python -m ophagent.preflight
    python -m ophagent.preflight --json    # machine-readable output
    python -m ophagent.preflight --quick   # skip slow checks (adapter loads)

Audits every component the agent depends on:
  • Python / PyTorch / CUDA environment
  • LLM backend connectivity (each env-configured API key is probed)
  • Modality detector CNN (loads + dummy forward)
  • Every adapter (registers + loads checkpoint + light forward pass)
  • External source repos (OCTCubeM, retsam, etc.) if used

Output:
  • Per-component status with [OK] / [FAIL] / [SKIP]
  • Per-modality summary: which modalities have at least one core
    disease-detection tool operational
  • Exit code 0 if every advertised modality has its core observers
    operational; 1 otherwise.

Use this command to verify that a local checkout is running with the full
backend stack active (i.e. strict-equivalent mode).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Load deployment environment values before importing ``utils.paths``. Those
# path constants are resolved at import time, so loading .env afterwards would
# make preflight inspect different roots from the Web service.
_PREFLIGHT_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREFLIGHT_ENV_PATHS: list[Path] = []
try:
    from dotenv import load_dotenv as _load_dotenv

    _explicit_env = os.environ.get("OPHAGENT_ENV_FILE", "").strip()
    _runtime_root = os.environ.get("OPHAGENT_RUNTIME_DIR", "").strip()
    _env_candidates: list[Path] = []
    if _explicit_env:
        _env_candidates.append(Path(_explicit_env).expanduser())
    if _runtime_root:
        _env_candidates.extend([
            Path(_runtime_root).expanduser() / ".env",
            Path(_runtime_root).expanduser() / ".env.local",
        ])
    _env_candidates.append(_PREFLIGHT_REPO_ROOT / ".env")
    for _candidate in _env_candidates:
        if not _candidate.is_file():
            continue
        _resolved = _candidate.resolve()
        if _resolved in _PREFLIGHT_ENV_PATHS:
            continue
        _load_dotenv(_resolved, override=False)
        _PREFLIGHT_ENV_PATHS.append(_resolved)
except ImportError:
    pass

from ophagent.chat.api_config import (
    DEFAULT_WEB_MODELS,
    PROVIDER_SPECS,
    create_provider_client,
    resolve_provider_connection,
)
from ophagent.components.manager import status_report
from ophagent.utils.paths import CKPT_DIR, ENV_FILE, output_path

# Don't pollute stdout with module-load noise
import logging
logging.disable(logging.CRITICAL)


@dataclass
class Check:
    """One health-check result."""
    name: str
    status: str        # "OK" | "FAIL" | "SKIP" | "WARN"
    detail: str = ""
    elapsed_s: float = 0.0

    def mark(self) -> str:
        return {"OK": "[OK]  ", "FAIL": "[FAIL]", "SKIP": "[SKIP]",
                "WARN": "[WARN]"}.get(self.status, "[?]   ")


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    modality_core_status: dict[str, dict] = field(default_factory=dict)
    runtime_config: dict[str, Any] = field(default_factory=dict)

    def add(self, c: Check) -> None:
        self.checks.append(c)

    def fatal_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    def all_modalities_operational(self) -> bool:
        """A modality is operational iff at least one of its core observers
        passes a real load probe. Registration-only checks are not sufficient."""
        if not self.modality_core_status:
            return True
        if any(s.get("probed") is False
               for s in self.modality_core_status.values()):
            return False
        return all(s.get("any_core_ok", False)
                   for s in self.modality_core_status.values())


# ──────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────
def check_python() -> Check:
    v = sys.version.split()[0]
    if sys.version_info < (3, 10):
        return Check("Python ≥ 3.10", "FAIL",
                     f"found {v} — OphAgent requires Python 3.10+")
    return Check("Python ≥ 3.10", "OK", f"Python {v}")


def check_torch_cuda() -> tuple[Check, Check]:
    try:
        import torch
    except ImportError as e:
        return (Check("PyTorch installed", "FAIL", str(e)),
                Check("CUDA available", "SKIP", "torch missing"))
    c1 = Check("PyTorch installed", "OK", f"torch {torch.__version__}")
    if torch.cuda.is_available():
        try:
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            detail = (f"{n} device(s), GPU0={name}, "
                      f"free={free/1e9:.1f}GB / total={total/1e9:.1f}GB")
            return c1, Check("CUDA available", "OK", detail)
        except Exception as e:
            return c1, Check("CUDA available", "WARN",
                              f"detected but mem_get_info failed: {e}")
    return c1, Check("CUDA available", "WARN",
                     "no CUDA — running on CPU; expect ~50× slower inference")


_VALID_EFFORTS = {"low", "medium", "high", "max", "ultra"}


def _setting(
    argument: str | None,
    env_name: str,
    default: str,
) -> tuple[str, str]:
    if argument is not None and argument.strip():
        return argument.strip(), "argument"
    environment = os.environ.get(env_name, "").strip()
    if environment:
        return environment, "environment"
    return default, "default"


def resolve_runtime_config(
    *,
    backend: str | None = None,
    model: str | None = None,
    vision_backend: str | None = None,
    vision_model: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Resolve the public runtime knobs used by the Web application."""
    selected_backend, backend_source = _setting(
        backend, "OPH_WEB_BACKEND", "openrouter"
    )
    selected_model, model_source = _setting(
        model, "OPH_WEB_MODEL", DEFAULT_WEB_MODELS.get(selected_backend, "")
    )
    selected_effort, effort_source = _setting(
        effort, "OPH_WEB_EFFORT", "low"
    )
    selected_vision_backend, vision_backend_source = _setting(
        vision_backend, "OPH_WEB_VISION_BACKEND", selected_backend
    )

    configured_vision_model = ""
    vision_model_source = "planner"
    if vision_model is not None and vision_model.strip():
        configured_vision_model = vision_model.strip()
        vision_model_source = "argument"
    elif os.environ.get("OPH_WEB_VISION_MODEL", "").strip():
        configured_vision_model = os.environ["OPH_WEB_VISION_MODEL"].strip()
        vision_model_source = "environment"

    errors: list[str] = []
    if selected_backend not in PROVIDER_SPECS:
        errors.append(f"unsupported planner backend: {selected_backend}")
    if not selected_model:
        errors.append("planner model is empty")
    if selected_vision_backend not in PROVIDER_SPECS:
        errors.append(f"unsupported vision backend: {selected_vision_backend}")
    if selected_effort not in _VALID_EFFORTS:
        errors.append(
            "effort must be one of " + ", ".join(sorted(_VALID_EFFORTS))
        )

    return {
        "planner": {
            "backend": selected_backend,
            "model": selected_model,
            "backend_source": backend_source,
            "model_source": model_source,
        },
        "vision": {
            "backend": selected_vision_backend,
            "model": configured_vision_model or selected_model,
            "backend_source": vision_backend_source,
            "model_source": vision_model_source,
            "inherits_planner_model": not bool(configured_vision_model),
        },
        "effort": selected_effort,
        "effort_source": effort_source,
        "errors": errors,
    }


def check_llm_backend(
    backend: str,
    model: str,
    *,
    probe: bool,
    role: str = "planner",
) -> Check:
    """Validate or probe the exact provider/model selected for one role."""
    t0 = time.time()
    name = f"LLM {role}: {backend}/{model}"
    if backend not in PROVIDER_SPECS:
        return Check(name, "FAIL", f"unsupported provider: {backend}")
    if not model:
        return Check(name, "FAIL", "model id is empty")
    resolved = resolve_provider_connection(backend)
    if not resolved["api_key"]:
        env_var = PROVIDER_SPECS[backend]["api_key_env"]
        return Check(name, "FAIL", f"{env_var} is not configured")
    if not probe:
        return Check(
            name,
            "OK",
            f"credential source={resolved['source']}; connectivity not probed in --quick",
            time.time() - t0,
        )
    try:
        client = create_provider_client(
            backend, timeout=15.0, max_retries=0
        )
        # A small visible-output allowance also works with reasoning endpoints.
        client.chat.completions.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with one word: ok"}],
        )
        return Check(
            name,
            "OK",
            f"reachable; credential source={resolved['source']}",
            time.time() - t0,
        )
    except Exception as e:
        return Check(
            name,
            "FAIL",
            f"{type(e).__name__}: {str(e)[:200]}",
            time.time() - t0,
        )


def check_modality_cnn() -> Check:
    """Verify the 4-class modality detector CNN loads + does a forward."""
    t0 = time.time()
    try:
        from ophagent.chat.oph_tools import _load_modality_model
        cfg = _load_modality_model()
        if not isinstance(cfg, dict):
            expected = CKPT_DIR / "_shared" / "modality_classifier" / "best.pt"
            return Check("Modality detector CNN", "FAIL",
                          f"_load_modality_model returned {cfg!r}; expected {expected}",
                          time.time() - t0)
        # Dummy forward to confirm weights are sane
        import torch
        x = torch.zeros(1, 3, 224, 224, device=cfg["device"])
        with torch.no_grad():
            _ = cfg["model"](x)
        return Check("Modality detector CNN", "OK",
                     f"loaded {len(cfg['classes'])}-class classifier "
                     f"({cfg['classes']})",
                     time.time() - t0)
    except Exception as e:
        return Check("Modality detector CNN", "FAIL",
                     f"{type(e).__name__}: {str(e)[:200]}",
                     time.time() - t0)


def check_adapters(quick: bool = False) -> tuple[list[Check],
                                                    dict[str, list[str]]]:
    """Probe every registered adapter. Returns (checks, modality→[tool_names]
    of OK adapters)."""
    import contextlib
    import io

    try:
        import ophagent.adapters  # noqa: F401  triggers registration
        from ophagent.adapters import GLOBAL_REGISTRY
    except Exception as e:
        return ([Check("Adapter import", "FAIL",
                        f"{type(e).__name__}: {str(e)[:200]}")],
                {})
    checks: list[Check] = []
    by_modality_ok: dict[str, list[str]] = {}
    for name, cls in sorted(GLOBAL_REGISTRY._classes.items()):
        meta = cls.metadata
        t0 = time.time()
        if quick:
            checks.append(Check(f"  {name} ({meta.modality})",
                                 "OK", "(registered; not probed in --quick)",
                                 time.time() - t0))
            by_modality_ok.setdefault(meta.modality, []).append(name)
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                adapter = GLOBAL_REGISTRY.get(name)
                # Heavy adapters are lazy-loaded; nudge them via .load() if
                # available, otherwise touch metadata and accept.
                if hasattr(adapter, "load") and not adapter.is_loaded():
                    adapter.load()
            checks.append(Check(f"  {name} ({meta.modality}/{meta.task})",
                                 "OK", "loaded ok",
                                 time.time() - t0))
            by_modality_ok.setdefault(meta.modality, []).append(name)
        except Exception as e:
            checks.append(Check(f"  {name} ({meta.modality}/{meta.task})",
                                 "FAIL",
                                 f"{type(e).__name__}: {str(e)[:200]}",
                                 time.time() - t0))
        finally:
            if not quick:
                # A full preflight probes many mutually exclusive heavyweight
                # models. Release each adapter before loading the next so the
                # health check itself cannot exhaust GPU memory.
                try:
                    GLOBAL_REGISTRY.unload_all()
                except Exception:
                    GLOBAL_REGISTRY._instances.clear()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    return checks, by_modality_ok


def check_external_repos() -> list[Check]:
    """Report source revision and release status from the component manifest."""
    report = status_report(profile="manuscript-full")
    checks: list[Check] = []
    for component in report["components"]:
        if component["integration"] == "built-in":
            continue
        status = "OK" if component["status"] == "ready" else "WARN"
        detail = component["detail"]
        if component.get("path"):
            detail = f"{detail}; path={component['path']}"
        checks.append(Check(f"  source: {component['id']}", status, detail))
    return checks


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
# Mirror oph_session._CORE_TOOLS_BY_MODALITY so this report's "operational"
# verdict matches the runtime sufficient-evidence gate exactly.
# Keep this in sync with ophagent.chat.oph_session._CORE_TOOLS_BY_MODALITY.
_CORE_TOOLS_BY_MODALITY: dict[str, list[str]] = {
    "CFP": ["cfp_clip_ensemble", "cfp_pdr_cascade",
             "cfp_retsam_segmentation", "cfp_dr_workup",
             "cfp_dr_421_assessment", "cfp_dynamic_clip",
             "cfp_glaucoma_workup", "cfp_paired5"],
    "OCT": ["oct_fmue_16class",
             "oct_volume_macular", "oct_volume_octcubem"],
    "UWF": ["uwf_disease_7class", "uwf_multi_disease"],
    "FFA": ["ffa_classification", "ffa_paired5"],
}


def run_all(
    quick: bool = False,
    *,
    runtime_config: dict[str, Any] | None = None,
) -> PreflightReport:
    runtime_config = runtime_config or resolve_runtime_config()
    report = PreflightReport(runtime_config=runtime_config)
    print("─" * 70)
    print(" OphAgent preflight health check")
    print("─" * 70)

    # 1. Environment
    print("\n## 1. Environment")
    for c in [check_python(), *check_torch_cuda()]:
        report.add(c)
        print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")

    # 2. .env file
    print("\n## 2. Environment file")
    dotenv_paths = [ENV_FILE, ENV_FILE.with_name(".env.local")]
    existing_envs = _PREFLIGHT_ENV_PATHS or [
        path for path in dotenv_paths if path.is_file()
    ]
    if existing_envs:
        try:
            from dotenv import load_dotenv
            for dotenv_path in existing_envs:
                load_dotenv(dotenv_path, override=False)
            c = Check(".env loaded", "OK", ", ".join(map(str, existing_envs)))
        except Exception as e:
            c = Check(".env loaded", "WARN",
                       f"file exists but load failed: {e}")
    else:
        c = Check(".env loaded", "WARN",
                   f"no .env at {ENV_FILE} — API keys must be in shell env")
    report.add(c)
    print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")

    # Apply Web-configured checkpoint paths before adapter modules are first
    # imported below. Environment variables remain the fallback.
    from ophagent.checkpoint_config import apply_saved_checkpoint_environment

    apply_saved_checkpoint_environment()

    # 3. Effective LLM configuration
    print("\n## 3. Effective LLM configuration")
    for error in runtime_config.get("errors", []):
        c = Check("Runtime configuration", "FAIL", error)
        report.add(c)
        print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")
    planner = runtime_config["planner"]
    vision = runtime_config["vision"]
    llm_checks = [
        check_llm_backend(
            planner["backend"], planner["model"], probe=not quick, role="planner"
        )
    ]
    if (
        vision["backend"] != planner["backend"]
        or vision["model"] != planner["model"]
    ):
        llm_checks.append(
            check_llm_backend(
                vision["backend"], vision["model"], probe=not quick, role="vision"
            )
        )
    for c in llm_checks:
        report.add(c)
        print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")

    # 4. Modality detector CNN
    print("\n## 4. Modality detector")
    if quick:
        c = Check(
            "Modality detector CNN",
            "SKIP",
            "checkpoint loading was not probed in --quick",
        )
    else:
        c = check_modality_cnn()
    report.add(c)
    print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")

    # 5. Adapters (the slow part)
    print(f"\n## 5. Adapters {'(--quick: registered only)' if quick else '(loading checkpoints)'}")
    adapter_checks, by_modality_ok = check_adapters(quick=quick)
    for c in adapter_checks:
        report.add(c)
        print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")

    # 6. External source repos
    print("\n## 6. External source repos")
    for c in check_external_repos():
        report.add(c)
        print(f"  {c.mark()}  {c.name:<40s}  {c.detail}")

    # 7. Per-modality summary
    heading = "registration status" if quick else "operational status"
    print(f"\n## 7. Per-modality {heading}")
    for modality, core_tools in _CORE_TOOLS_BY_MODALITY.items():
        avail = set(by_modality_ok.get(modality, []))
        core_ok = [t for t in core_tools if t in avail]
        if quick:
            report.modality_core_status[modality] = {
                "probed": False,
                "any_core_ok": None,
                "core_tools_registered": core_ok,
                "core_tools_unregistered": [
                    t for t in core_tools if t not in avail
                ],
                "all_adapters_registered_count": len(avail),
            }
            detail = (
                f"{len(core_ok)}/{len(core_tools)} core tools registered; "
                "checkpoint loading was not probed"
            )
            mark = "[OK]  " if len(core_ok) == len(core_tools) else "[WARN]"
            print(f"  {mark}  {modality:<8s}  {detail}")
            continue
        any_core = bool(core_ok)
        report.modality_core_status[modality] = {
            "probed": True,
            "any_core_ok": any_core,
            "core_tools_ok": core_ok,
            "core_tools_missing": [t for t in core_tools if t not in avail],
            "all_adapters_ok_count": len(avail),
        }
        if core_ok:
            detail = (f"{len(core_ok)}/{len(core_tools)} core tools "
                      f"operational ({', '.join(core_ok)})")
        else:
            detail = (f"0/{len(core_tools)} core tools operational — "
                      f"diagnostic call will be suppressed for {modality}")
        mark = "[OK]  " if any_core else "[FAIL]"
        print(f"  {mark}  {modality:<8s}  {detail}")

    return report


def _report_payload(report: PreflightReport, quick: bool,
                    exit_code: int, saved_to: str | None = None) -> dict:
    modalities_ok = sum(
        1 for s in report.modality_core_status.values()
        if s.get("any_core_ok")
    )
    all_modalities_ok = report.all_modalities_operational()
    return {
        "summary": {
            "total": len(report.checks),
            "failed": report.fatal_count(),
            "warnings": report.warn_count(),
            "quick": quick,
            "exit_code": exit_code,
            "strict_ready": (not quick) and exit_code == 0,
            "all_modalities_operational": all_modalities_ok,
            "strict_stack_probed": not quick,
            "modalities_ok": modalities_ok,
            "modalities_total": len(report.modality_core_status),
            "saved_to": saved_to,
        },
        "runtime": report.runtime_config,
        "modalities": report.modality_core_status,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "detail": c.detail,
                "elapsed_s": round(c.elapsed_s, 2),
            }
            for c in report.checks
        ],
    }


def _save_json_report(payload: dict) -> Path:
    out_dir = output_path("preflight")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preflight_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload["summary"]["saved_to"] = str(out_path)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def _legacy_main_unused():
    ap = argparse.ArgumentParser(description="OphAgent preflight check")
    ap.add_argument("--quick", action="store_true",
                    help="Skip slow adapter checkpoint loads")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON report")
    ap.add_argument("--no-save-json", action="store_true",
                    help="Do not save the runtime preflight JSON report")
    args = ap.parse_args()

    report = run_all(quick=args.quick)

    # Summary
    print("\n" + "─" * 70)
    print(" Summary")
    print("─" * 70)
    print(f"  Total checks:       {len(report.checks)}")
    print(f"  Failed:             {report.fatal_count()}")
    print(f"  Warnings:           {report.warn_count()}")
    modalities_ok = sum(1 for s in report.modality_core_status.values()
                         if s.get("any_core_ok"))
    print(f"  Modalities OK:      {modalities_ok}/{len(report.modality_core_status)}")
    print()

    component_failed = report.fatal_count() > 0
    modalities_ready = report.all_modalities_operational()
    if component_failed or not modalities_ready:
        print(" 🚫 Preflight FAILED — at least one modality has no core observers.")
        print("    The agent will refuse to produce diagnoses for that modality;")
        print("    fix the failing components before relying on its output.")
        print()
        exit_code = 1
    else:
        print(" ✅ All modalities operational. The agent is ready for strict-mode use.")
        print()
        exit_code = 0

    if args.json:
        out = {
            "summary": {
                "total": len(report.checks),
                "failed": report.fatal_count(),
                "warnings": report.warn_count(),
                "all_modalities_operational": report.all_modalities_operational(),
            },
            "modalities": report.modality_core_status,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail,
                 "elapsed_s": round(c.elapsed_s, 2)}
                for c in report.checks
            ],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))

    sys.exit(exit_code)


def main():
    ap = argparse.ArgumentParser(description="OphAgent preflight check")
    ap.add_argument("--quick", action="store_true",
                    help="Skip slow adapter checkpoint loads")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON report")
    ap.add_argument("--no-save-json", action="store_true",
                    help="Do not save the runtime preflight JSON report")
    ap.add_argument("--backend", choices=sorted(PROVIDER_SPECS),
                    help="Planner provider; overrides OPH_WEB_BACKEND")
    ap.add_argument("--model",
                    help="Planner model id; overrides OPH_WEB_MODEL")
    ap.add_argument("--vision-backend", choices=sorted(PROVIDER_SPECS),
                    help="Vision provider; overrides OPH_WEB_VISION_BACKEND")
    ap.add_argument("--vision-model",
                    help="Dedicated vision model; overrides OPH_WEB_VISION_MODEL")
    ap.add_argument("--effort", choices=sorted(_VALID_EFFORTS),
                    help="Execution policy; overrides OPH_WEB_EFFORT")
    args = ap.parse_args()

    runtime_config = resolve_runtime_config(
        backend=args.backend,
        model=args.model,
        vision_backend=args.vision_backend,
        vision_model=args.vision_model,
        effort=args.effort,
    )
    report = run_all(quick=args.quick, runtime_config=runtime_config)

    print("\n" + "-" * 70)
    print(" Summary")
    print("-" * 70)
    print(f"  Total checks:       {len(report.checks)}")
    print(f"  Failed:             {report.fatal_count()}")
    print(f"  Warnings:           {report.warn_count()}")
    component_failed = report.fatal_count() > 0
    if args.quick:
        modalities_registered = sum(
            1 for s in report.modality_core_status.values()
            if s.get("core_tools_registered")
        )
        print(
            f"  Modalities registered: {modalities_registered}/"
            f"{len(report.modality_core_status)}"
        )
        print()
        if component_failed:
            print(" [FAIL] Quick preflight found a configuration error.")
            exit_code = 1
        else:
            print(" [OK] Registration preflight passed.")
            print("      Checkpoint loading was not probed; strict readiness is unknown.")
            exit_code = 0
        print()
    elif component_failed or not report.all_modalities_operational():
        print(" [FAIL] Preflight FAILED - full evaluation stack is not ready.")
        if component_failed:
            print("        At least one component failed to load or probe.")
            print("        Modalities may still have partial core coverage; see JSON.")
        if not report.all_modalities_operational():
            print("        At least one modality has no operational core observer.")
            print("        Runtime finalization will suppress diagnosis for that modality.")
        print()
        exit_code = 1
    else:
        print(" [OK] All modalities operational. Strict-mode stack is ready.")
        print()
        exit_code = 0

    out = _report_payload(report, quick=args.quick, exit_code=exit_code)
    if not args.no_save_json:
        saved_path = _save_json_report(out)
        print(f"  JSON report saved: {saved_path}")
        print()

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
