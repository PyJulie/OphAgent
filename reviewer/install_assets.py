#!/usr/bin/env python3
"""Verify and install separately distributed OphAgent runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = Path(__file__).with_name("assets.json")
COPY_BUFFER_BYTES = 8 * 1024 * 1024


class AssetInstallError(RuntimeError):
    """The runtime asset archive cannot be installed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path = DEFAULT_METADATA) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetInstallError(f"Cannot read asset metadata: {exc}") from exc
    required = {
        "release_version",
        "profile",
        "archive",
        "archive_root",
        "archive_size_bytes",
        "archive_sha256",
        "asset_count",
        "payload_size_bytes",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise AssetInstallError(
            "Asset metadata is missing fields: " + ", ".join(missing)
        )
    return metadata


def _normalised_member(info: zipfile.ZipInfo, archive_root: str) -> PurePosixPath:
    raw = info.filename.replace("\\", "/")
    member = PurePosixPath(raw)
    parts = member.parts
    if (
        not parts
        or member.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
        or parts[0] != archive_root
    ):
        raise AssetInstallError(f"Unsafe archive member: {info.filename!r}")

    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise AssetInstallError(f"Symbolic links are not allowed: {info.filename!r}")
    return member


def inspect_archive(
    archive: Path,
    metadata: dict[str, Any],
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise AssetInstallError(f"Asset archive not found: {archive}")

    expected_size = int(metadata["archive_size_bytes"])
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise AssetInstallError(
            f"Archive size mismatch: expected {expected_size}, found {actual_size}"
        )

    actual_hash = None
    if verify_hash:
        print(
            f"Verifying {archive.name} ({actual_size / (1024 ** 3):.2f} GiB)...",
            file=sys.stderr,
        )
        actual_hash = _sha256(archive)
        expected_hash = str(metadata["archive_sha256"]).lower()
        if actual_hash.lower() != expected_hash:
            raise AssetInstallError(
                f"Archive SHA-256 mismatch: expected {expected_hash}, "
                f"found {actual_hash}"
            )

    archive_root = str(metadata["archive_root"])
    seen: set[str] = set()
    casefolded: set[str] = set()
    required_files = {
        f"{archive_root}/README.md",
        f"{archive_root}/ASSET_MANIFEST.json",
        f"{archive_root}/model_assets.yaml",
    }
    found_files: set[str] = set()
    checkpoint_files = 0
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                member = _normalised_member(info, archive_root)
                normalised = member.as_posix().rstrip("/")
                folded = normalised.casefold()
                if normalised in seen or folded in casefolded:
                    raise AssetInstallError(
                        f"Duplicate or case-colliding archive member: {info.filename!r}"
                    )
                seen.add(normalised)
                casefolded.add(folded)
                if not info.is_dir():
                    found_files.add(normalised)
                    total_uncompressed += info.file_size
                    if normalised.startswith(f"{archive_root}/checkpoints/"):
                        checkpoint_files += 1
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, AssetInstallError):
            raise
        raise AssetInstallError(f"Cannot inspect asset ZIP: {exc}") from exc

    missing_files = sorted(required_files - found_files)
    if missing_files:
        raise AssetInstallError(
            "Asset ZIP is missing required files: " + ", ".join(missing_files)
        )
    expected_assets = int(metadata["asset_count"])
    if checkpoint_files != expected_assets:
        raise AssetInstallError(
            f"Checkpoint count mismatch: expected {expected_assets}, "
            f"found {checkpoint_files}"
        )

    return {
        "archive": str(archive),
        "size_bytes": actual_size,
        "sha256": actual_hash,
        "archive_root": archive_root,
        "checkpoint_files": checkpoint_files,
        "uncompressed_size_bytes": total_uncompressed,
    }


def _assert_runtime_outside_repo(runtime_dir: Path, repo_root: Path) -> None:
    runtime_dir = runtime_dir.resolve()
    repo_root = repo_root.resolve()
    if runtime_dir == repo_root or repo_root in runtime_dir.parents:
        raise AssetInstallError(
            "The runtime directory must be outside the Git checkout"
        )


def _safe_cleanup(staging: Path, runtime_parent: Path) -> None:
    staging = staging.resolve()
    runtime_parent = runtime_parent.resolve()
    if (
        staging.parent != runtime_parent
        or not staging.name.startswith(".ophagent-assets-extract-")
    ):
        raise AssetInstallError(f"Refusing to remove unexpected path: {staging}")
    if staging.exists():
        shutil.rmtree(staging)


def _extract_archive(
    archive: Path,
    staging: Path,
    metadata: dict[str, Any],
) -> Path:
    archive_root = str(metadata["archive_root"])
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        total = sum(info.file_size for info in members if not info.is_dir())
        completed = 0
        next_percent = 10
        for info in members:
            member = _normalised_member(info, archive_root)
            destination = staging.joinpath(*member.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, COPY_BUFFER_BYTES)
            completed += info.file_size
            percent = 100 if total == 0 else int(completed * 100 / total)
            if percent >= next_percent:
                print(f"Extracting runtime assets: {percent}%", file=sys.stderr)
                next_percent = min(100, ((percent // 10) + 1) * 10)
    return staging / archive_root


def install_archive(
    archive: Path,
    runtime_dir: Path,
    metadata: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    archive = archive.expanduser().resolve()
    runtime_dir = runtime_dir.expanduser().resolve()
    _assert_runtime_outside_repo(runtime_dir, repo_root)
    if runtime_dir.exists():
        raise AssetInstallError(
            f"Runtime directory already exists; choose a new path: {runtime_dir}"
        )

    verification = inspect_archive(archive, metadata, verify_hash=True)
    runtime_parent = runtime_dir.parent
    runtime_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(runtime_parent).free
    required_bytes = int(verification["uncompressed_size_bytes"]) + 1024**3
    if free_bytes < required_bytes:
        raise AssetInstallError(
            f"Insufficient free space on {runtime_parent}: "
            f"need at least {required_bytes / (1024 ** 3):.2f} GiB"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".ophagent-assets-extract-",
            dir=runtime_parent,
        )
    ).resolve()
    try:
        extracted_root = _extract_archive(archive, staging, metadata)
        for relative in ("ASSET_MANIFEST.json", "model_assets.yaml", "checkpoints"):
            if not (extracted_root / relative).exists():
                raise AssetInstallError(
                    f"Extracted runtime is missing required path: {relative}"
                )

        env_target = extracted_root / ".env"
        if not env_target.exists():
            shutil.copy2(repo_root / ".env.example", env_target)
        installation = {
            "schema_version": 1,
            "project": "OphAgent",
            "release_version": metadata["release_version"],
            "profile": metadata["profile"],
            "archive": metadata["archive"],
            "archive_sha256": metadata["archive_sha256"],
            "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (extracted_root / "INSTALLATION.json").write_text(
            json.dumps(installation, indent=2) + "\n",
            encoding="utf-8",
        )
        extracted_root.replace(runtime_dir)
        try:
            staging.rmdir()
        except OSError:
            _safe_cleanup(staging, runtime_parent)
    except Exception:
        _safe_cleanup(staging, runtime_parent)
        raise

    return {
        **verification,
        "runtime_dir": str(runtime_dir),
        "env_file": str(runtime_dir / ".env"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and install the separately distributed OphAgent checkpoint archive "
            "outside the Git checkout."
        )
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="Downloaded OphAgent-runtime-assets-0.1.0.zip",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path.home() / "ophagent-runtime",
        help="Private extraction target (default: ~/ophagent-runtime)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify archive size, SHA-256, CRC, layout, and checkpoint count only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metadata = load_metadata()
        if args.verify_only:
            result = inspect_archive(args.archive, metadata, verify_hash=True)
        else:
            result = install_archive(
                args.archive,
                args.runtime_dir,
                metadata,
            )
    except AssetInstallError as exc:
        print(f"Reviewer asset setup failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **result}, indent=2))
    if not args.verify_only:
        runtime = Path(result["runtime_dir"])
        ps_runtime = str(runtime).replace("'", "''")
        sh_runtime = str(runtime).replace("'", "'\"'\"'")
        print("\nNext steps:")
        print(f"  PowerShell: $env:OPHAGENT_RUNTIME_DIR = '{ps_runtime}'")
        print(f"  bash/zsh:   export OPHAGENT_RUNTIME_DIR='{sh_runtime}'")
        print("  Add one provider API key to <runtime>/.env, then run:")
        print("    ophagent-assets verify --profile manuscript-full")
        print("    ophagent-components status --profile manuscript-full")
        print("    ophagent-preflight --json --no-save-json")
        print("    ophagent-web")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
