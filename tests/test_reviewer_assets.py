from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from reviewer.install_assets import (
    AssetInstallError,
    inspect_archive,
    install_archive,
)


def _build_archive(
    tmp_path: Path,
    *,
    unsafe_member: str | None = None,
) -> tuple[Path, dict]:
    archive_root = "OphAgent-runtime-assets-test"
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr(f"{archive_root}/README.md", "runtime assets\n")
        bundle.writestr(f"{archive_root}/ASSET_MANIFEST.json", "{}\n")
        bundle.writestr(f"{archive_root}/model_assets.yaml", "schema_version: 1\n")
        bundle.writestr(f"{archive_root}/checkpoints/cfp/a.pt", b"a")
        bundle.writestr(f"{archive_root}/checkpoints/oct/b.pt", b"bb")
        if unsafe_member:
            bundle.writestr(unsafe_member, b"unsafe")
    metadata = {
        "release_version": "test",
        "profile": "manuscript-full",
        "archive": archive.name,
        "archive_root": archive_root,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "asset_count": 2,
        "payload_size_bytes": 3,
    }
    return archive, metadata


def test_install_archive_outside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    archive, metadata = _build_archive(tmp_path)
    runtime = tmp_path / "runtime"

    result = install_archive(
        archive,
        runtime,
        metadata,
        repo_root=repo,
    )

    assert result["runtime_dir"] == str(runtime.resolve())
    assert (runtime / "checkpoints" / "cfp" / "a.pt").read_bytes() == b"a"
    assert (runtime / "checkpoints" / "oct" / "b.pt").read_bytes() == b"bb"
    assert (runtime / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=\n"
    installation = json.loads(
        (runtime / "INSTALLATION.json").read_text(encoding="utf-8")
    )
    assert installation["archive_sha256"] == metadata["archive_sha256"]


def test_install_refuses_runtime_inside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env.example").write_text("", encoding="utf-8")
    archive, metadata = _build_archive(tmp_path)

    with pytest.raises(AssetInstallError, match="outside the Git checkout"):
        install_archive(
            archive,
            repo / "runtime",
            metadata,
            repo_root=repo,
        )


def test_inspection_rejects_path_traversal(tmp_path: Path) -> None:
    archive_root = "OphAgent-runtime-assets-test"
    archive, metadata = _build_archive(
        tmp_path,
        unsafe_member=f"{archive_root}/../../escape.txt",
    )

    with pytest.raises(AssetInstallError, match="Unsafe archive member"):
        inspect_archive(archive, metadata)


def test_inspection_rejects_hash_mismatch(tmp_path: Path) -> None:
    archive, metadata = _build_archive(tmp_path)
    metadata["archive_sha256"] = "0" * 64

    with pytest.raises(AssetInstallError, match="SHA-256 mismatch"):
        inspect_archive(archive, metadata)
