"""Install and audit pinned external source components.

This command manages source code only. It never downloads model weights,
datasets, credentials, or clinical examples.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import yaml

from ophagent.utils.paths import EXTERNAL_DIR


BUNDLED_ROOT = Path(__file__).resolve().parent


def load_component_manifest() -> dict[str, Any]:
    manifest = resources.files("ophagent.resources").joinpath("components.yaml")
    with manifest.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise RuntimeError("Unsupported or malformed component manifest")
    return data


def _components_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = manifest.get("components", [])
    if not isinstance(items, list):
        raise RuntimeError("Component manifest must contain a components list")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            raise RuntimeError("Every component must have an id")
        component_id = str(item["id"])
        if component_id in result:
            raise RuntimeError(f"Duplicate component id: {component_id}")
        result[component_id] = item
    return result


def _git_output(*args: str, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to install external components") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(detail) from exc
    return completed.stdout.strip()


def _target_path(component: dict[str, Any], external_root: Path) -> Path | None:
    target_dir = component.get("target_dir")
    if not target_dir:
        return None
    target = (external_root / str(target_dir)).resolve()
    root = external_root.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(f"Unsafe component target: {target}")
    return target


def _missing_markers(component: dict[str, Any], target: Path) -> list[str]:
    primary = list(component.get("markers", []))
    marker_sets = [primary, *component.get("compatible_marker_sets", [])]
    for markers in marker_sets:
        if markers and all((target / marker).exists() for marker in markers):
            return []
    return [marker for marker in primary if not (target / marker).exists()]


def component_status(
    component: dict[str, Any],
    external_root: Path,
) -> dict[str, Any]:
    integration = str(component.get("integration", ""))
    result: dict[str, Any] = {
        "id": component["id"],
        "name": component.get("name", component["id"]),
        "integration": integration,
        "license": component.get("license"),
        "license_status": component.get("license_status"),
        "required_for": component.get("required_for", []),
    }
    if integration == "built-in":
        result.update(status="ready", detail="inference code ships with OphAgent")
        return result
    if integration == "bundled":
        package_dir = str(component.get("package_dir") or "")
        if not package_dir:
            result.update(status="invalid", detail="bundled component has no package_dir")
            return result
        target = (BUNDLED_ROOT / package_dir).resolve()
        if target == BUNDLED_ROOT or BUNDLED_ROOT not in target.parents:
            result.update(status="invalid", detail="unsafe bundled package path")
            return result
        result["path"] = str(target)
        if not target.is_dir():
            result.update(status="incomplete", detail="bundled source directory is absent")
            return result
        missing = _missing_markers(component, target)
        if missing:
            result.update(
                status="incomplete",
                detail="missing marker(s): " + ", ".join(missing),
            )
            return result
        result.update(status="ready", detail="source ships with OphAgent")
        return result

    target = _target_path(component, external_root)
    result["path"] = str(target) if target else None
    if target is None:
        result.update(status="invalid", detail="external component has no target_dir")
        return result
    if not target.exists():
        status = "source-required"
        if integration == "private-source-pending":
            status = "release-blocked"
        result.update(status=status, detail="source directory is absent")
        return result

    missing = _missing_markers(component, target)
    if missing:
        result.update(
            status="incomplete",
            detail="missing marker(s): " + ", ".join(missing),
        )
        return result

    if integration == "private-source-pending":
        result.update(
            status="release-blocked",
            detail="local source exists; rights and provenance still require approval",
        )
        return result

    git_dir = target / ".git"
    if not git_dir.exists():
        result.update(
            status="present-unlocked",
            detail="source is present but has no Git revision metadata",
        )
        return result

    try:
        head = _git_output("-C", str(target), "rev-parse", "HEAD")
    except RuntimeError as exc:
        result.update(status="invalid", detail=str(exc))
        return result
    expected = str(component.get("revision") or "")
    result["revision"] = head
    if expected and head != expected:
        result.update(
            status="revision-mismatch",
            detail=f"expected {expected}, found {head}",
        )
    else:
        result.update(status="ready", detail=f"locked at {head}")
    return result


def status_report(
    *,
    external_root: Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    manifest = load_component_manifest()
    root = (external_root or EXTERNAL_DIR).resolve()
    components = list(_components_by_id(manifest).values())
    if profile:
        components = [
            item for item in components
            if profile in item.get("required_for", [])
        ]
    statuses = [component_status(item, root) for item in components]
    return {
        "schema_version": 1,
        "external_root": str(root),
        "profile": profile,
        "components": statuses,
        "ready": all(item["status"] == "ready" for item in statuses),
    }


def _safe_remove_partial(path: Path, external_root: Path) -> None:
    resolved = path.resolve()
    root = external_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove unsafe partial path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def install_component(
    component: dict[str, Any],
    *,
    external_root: Path,
    allow_unlicensed: bool = False,
) -> dict[str, Any]:
    integration = str(component.get("integration", ""))
    if integration in {"built-in", "bundled"}:
        return component_status(component, external_root)
    if integration != "git":
        raise RuntimeError(
            f"{component['id']} cannot be installed automatically: {integration}"
        )
    if (
        component.get("install_policy") == "manual-acknowledgement"
        and not allow_unlicensed
    ):
        raise RuntimeError(
            f"{component['id']} has no declared upstream license; "
            "rerun with --allow-unlicensed only after reviewing its terms"
        )

    target = _target_path(component, external_root)
    if target is None:
        raise RuntimeError(f"{component['id']} has no target directory")
    if target.exists():
        state = component_status(component, external_root)
        if state["status"] == "ready":
            return state
        raise RuntimeError(
            f"{target} already exists ({state['status']}); "
            "OphAgent will not overwrite source directories"
        )

    repository = str(component.get("repository") or "")
    revision = str(component.get("revision") or "")
    if not repository or not revision:
        raise RuntimeError(f"{component['id']} lacks a locked repository revision")

    external_root.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if partial.exists():
        _safe_remove_partial(partial, external_root)
    try:
        _git_output(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(partial),
        )
        _git_output(
            "-C",
            str(partial),
            "fetch",
            "--depth",
            "1",
            "origin",
            revision,
        )
        _git_output("-C", str(partial), "checkout", "--detach", revision)
        head = _git_output("-C", str(partial), "rev-parse", "HEAD")
        if head != revision:
            raise RuntimeError(
                f"Revision verification failed for {component['id']}: {head}"
            )
        missing = _missing_markers(component, partial)
        if missing:
            raise RuntimeError(
                f"Installed {component['id']} is missing: {', '.join(missing)}"
            )
        partial.replace(target)
    except Exception:
        _safe_remove_partial(partial, external_root)
        raise
    return component_status(component, external_root)


def _select_components(
    manifest: dict[str, Any],
    ids: Iterable[str],
    *,
    install_all: bool,
    include_unlicensed: bool,
) -> list[dict[str, Any]]:
    by_id = _components_by_id(manifest)
    requested = list(ids)
    if install_all:
        return [
            item for item in by_id.values()
            if item.get("integration") == "git"
            and (
                item.get("install_policy") != "manual-acknowledgement"
                or include_unlicensed
            )
        ]
    if not requested:
        raise RuntimeError("Specify one or more component IDs, or use --all")
    unknown = [component_id for component_id in requested if component_id not in by_id]
    if unknown:
        raise RuntimeError("Unknown component(s): " + ", ".join(unknown))
    return [by_id[component_id] for component_id in requested]


def _print_status(report: dict[str, Any]) -> None:
    print(f"External source root: {report['external_root']}")
    if report.get("profile"):
        print(f"Profile: {report['profile']}")
    for item in report["components"]:
        license_name = item.get("license") or "not declared"
        print(
            f"{item['id']:<16} {item['status']:<18} "
            f"license={license_name:<14} {item['detail']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ophagent-components",
        description="Audit and install pinned source components. Weights are never downloaded.",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=None,
        help="Override OPHAGENT_EXTERNAL_DIR for this command",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Report source-code readiness")
    status.add_argument("--profile", choices=["public-demo", "manuscript-full"])
    status.add_argument("--json", action="store_true", dest="as_json")

    install = subparsers.add_parser("install", help="Install locked public source")
    install.add_argument("components", nargs="*")
    install.add_argument("--all", action="store_true", dest="install_all")
    install.add_argument(
        "--allow-unlicensed",
        action="store_true",
        help="Allow cloning an upstream repository that declares no license",
    )
    install.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    external_root = (args.external_root or EXTERNAL_DIR).resolve()
    try:
        if args.command == "status":
            report = status_report(
                external_root=external_root,
                profile=args.profile,
            )
            if args.as_json:
                print(json.dumps(report, indent=2))
            else:
                _print_status(report)
            return 0 if report["ready"] else 1

        manifest = load_component_manifest()
        selected = _select_components(
            manifest,
            args.components,
            install_all=args.install_all,
            include_unlicensed=args.allow_unlicensed,
        )
        results = [
            install_component(
                item,
                external_root=external_root,
                allow_unlicensed=args.allow_unlicensed,
            )
            for item in selected
        ]
        if args.as_json:
            print(json.dumps({"components": results}, indent=2))
        else:
            _print_status(
                {
                    "external_root": str(external_root),
                    "profile": None,
                    "components": results,
                }
            )
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
