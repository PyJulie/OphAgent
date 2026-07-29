"""Inspect and verify separately distributed OphAgent model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml

from ophagent.utils.paths import CKPT_DIR


def load_model_asset_manifest() -> dict[str, Any]:
    manifest = resources.files("ophagent.resources").joinpath("model_assets.yaml")
    with manifest.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise RuntimeError("Unsupported or malformed model-asset manifest")
    assets = data.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("Model-asset manifest must contain an assets list")
    return data


def _portable_relative_path(raw_path: object) -> PurePosixPath:
    text = str(raw_path or "")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or ":" in text
        or ".." in path.parts
    ):
        raise RuntimeError(f"Unsafe model-asset path: {text!r}")
    return path


def assets_by_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_asset in manifest.get("assets", []):
        if not isinstance(raw_asset, dict) or not raw_asset.get("id"):
            raise RuntimeError("Every model asset must have an id")
        asset = dict(raw_asset)
        asset_id = str(asset["id"])
        if asset_id in result:
            raise RuntimeError(f"Duplicate model-asset id: {asset_id}")
        _portable_relative_path(asset.get("path"))
        result[asset_id] = asset
    return result


def model_asset_index(*, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Return the packaged checksum index keyed by portable relative path."""
    manifest = load_model_asset_manifest()
    index: dict[str, dict[str, Any]] = {}
    normalized_prefix = prefix.replace("\\", "/").strip("/")
    for asset in assets_by_id(manifest).values():
        path = _portable_relative_path(asset["path"]).as_posix()
        key = f"{normalized_prefix}/{path}" if normalized_prefix else path
        index[key] = {
            "size_bytes": asset.get("size_bytes"),
            "sha256": str(asset.get("sha256") or "").lower(),
        }
    return index


def select_assets(
    manifest: Mapping[str, Any],
    *,
    profile: str | None = None,
    asset_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    by_id = assets_by_id(manifest)
    requested = list(asset_ids)
    if requested:
        unknown = [asset_id for asset_id in requested if asset_id not in by_id]
        if unknown:
            raise RuntimeError("Unknown model asset(s): " + ", ".join(unknown))
        selected = [by_id[asset_id] for asset_id in requested]
        if profile:
            selected = [
                asset
                for asset in selected
                if profile in asset.get("required_for", [])
            ]
        return selected
    if profile:
        return [
            asset
            for asset in by_id.values()
            if profile in asset.get("required_for", [])
        ]
    return list(by_id.values())


def resolve_asset_path(
    asset: Mapping[str, Any],
    *,
    checkpoint_root: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, str]:
    env = environ or os.environ
    env_var = str(asset.get("env") or "")
    override = str(env.get(env_var, "")).strip() if env_var else ""
    if override:
        value = os.path.expandvars(os.path.expanduser(override))
        return Path(value).resolve(), f"environment:{env_var}"

    relative = _portable_relative_path(asset.get("path"))
    root = checkpoint_root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Model asset escapes checkpoint root: {relative}")
    return path, "checkpoint-root"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(
    asset: Mapping[str, Any],
    *,
    checkpoint_root: Path,
    verify_hash: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    path, source = resolve_asset_path(
        asset,
        checkpoint_root=checkpoint_root,
        environ=environ,
    )
    expected_size = asset.get("size_bytes")
    expected_hash = str(asset.get("sha256") or "").lower()
    result: dict[str, Any] = {
        "id": str(asset["id"]),
        "role": str(asset.get("role") or ""),
        "path": str(path),
        "source": source,
        "status": "ready",
        "size_bytes": None,
        "expected_size_bytes": expected_size,
        "checksum_available": bool(expected_hash),
        "message": "File exists and size matches",
    }
    if not path.exists():
        result.update(status="missing", message="File not found")
        return result
    if not path.is_file():
        result.update(status="mismatch", message="Expected a file")
        return result

    actual_size = path.stat().st_size
    result["size_bytes"] = actual_size
    if isinstance(expected_size, int) and actual_size != expected_size:
        result.update(status="mismatch", message="File size does not match")
        return result

    if verify_hash:
        if not expected_hash:
            result.update(
                status="unverified",
                message="No reference SHA-256 is available",
            )
            return result
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            result.update(status="mismatch", message="SHA-256 does not match")
            return result
        result.update(status="verified", message="SHA-256 verified")
    return result


def asset_report(
    *,
    checkpoint_root: Path | None = None,
    profile: str | None = None,
    asset_ids: Iterable[str] = (),
    verify_hash: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = load_model_asset_manifest()
    root = (checkpoint_root or CKPT_DIR).resolve()
    selected = select_assets(
        manifest,
        profile=profile,
        asset_ids=asset_ids,
    )
    assets = [
        inspect_asset(
            asset,
            checkpoint_root=root,
            verify_hash=verify_hash,
            environ=environ,
        )
        for asset in selected
    ]
    counts: dict[str, int] = {}
    for asset in assets:
        status = str(asset["status"])
        counts[status] = counts.get(status, 0) + 1
    accepted = {"verified"} if verify_hash else {"ready"}
    return {
        "schema_version": 1,
        "release_version": manifest.get("release_version"),
        "checkpoint_root": str(root),
        "profile": profile,
        "hashes_checked": verify_hash,
        "assets": assets,
        "summary": counts,
        "ready": bool(assets) and all(
            asset["status"] in accepted for asset in assets
        ),
    }


def _print_report(report: Mapping[str, Any]) -> None:
    print(f"Checkpoint root: {report['checkpoint_root']}")
    if report.get("profile"):
        print(f"Profile: {report['profile']}")
    for asset in report["assets"]:
        print(
            f"{asset['id']:<28} {asset['status']:<10} "
            f"{asset['message']} ({asset['path']})"
        )
    summary = ", ".join(
        f"{name}={count}"
        for name, count in sorted(report["summary"].items())
    )
    print(f"Summary: {summary or 'no assets selected'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ophagent-assets",
        description="Inspect or verify separately supplied OphAgent model assets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("status", "Check asset presence and expected file size"),
        ("verify", "Check asset presence, size, and SHA-256"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--checkpoint-root",
            type=Path,
            default=None,
            help="Override OPHAGENT_CKPT_DIR for this command",
        )
        subparser.add_argument(
            "--profile",
            choices=["public-demo", "manuscript-full"],
            default="manuscript-full",
        )
        subparser.add_argument(
            "--asset",
            action="append",
            default=[],
            dest="asset_ids",
            help="Limit the check to an asset id; repeat as needed",
        )
        subparser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = asset_report(
            checkpoint_root=args.checkpoint_root,
            profile=args.profile,
            asset_ids=args.asset_ids,
            verify_hash=args.command == "verify",
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
