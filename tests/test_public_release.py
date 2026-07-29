from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from ophagent.release import (
    MANIFEST_NAME,
    ReleaseError,
    _write_zip,
    audit_files,
    verify_release,
)


def test_release_audit_rejects_private_and_clinical_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "case.dcm").write_bytes(b"DICM")
    findings = audit_files(
        tmp_path,
        [PurePosixPath(".env"), PurePosixPath("case.dcm")],
    )
    assert {finding["kind"] for finding in findings} == {
        "credential-file",
        "blocked-file-type",
    }


def test_release_audit_rejects_provider_tokens(tmp_path: Path) -> None:
    token = "-".join(("sk", "abcdefghijklmnopqrstuvwxyz123456"))
    (tmp_path / "bad.txt").write_text(
        f"token={token}",
        encoding="utf-8",
    )
    findings = audit_files(tmp_path, [PurePosixPath("bad.txt")])
    assert findings[0]["kind"] == "secret"


def test_release_audit_rejects_generic_credentials_and_local_paths(
    tmp_path: Path,
) -> None:
    credential_value = "provider-specific-" + "secret-value"
    separator = chr(92)
    local_path = f"X:{separator}private{separator}models"
    (tmp_path / "settings.py").write_text(
        f'api_key = "{credential_value}"\n'
        f'model_root = "{local_path}"\n',
        encoding="utf-8",
    )
    findings = audit_files(tmp_path, [PurePosixPath("settings.py")])
    assert {finding["kind"] for finding in findings} == {
        "secret",
        "local-path",
    }


def test_release_audit_accepts_normal_source(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    assert audit_files(tmp_path, [PurePosixPath("module.py")]) == []


def test_release_manifest_detects_unlisted_file(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": "module.py",
                "size_bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert verify_release(tmp_path)["verified"] is True

    (tmp_path / "extra.txt").write_text("not listed\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="unlisted file"):
        verify_release(tmp_path)


def test_release_verification_ignores_generated_python_caches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": "module.py",
                "size_bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"generated cache")
    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "README.md").write_text("generated cache\n", encoding="utf-8")
    package_metadata = tmp_path / "ophagent.egg-info"
    package_metadata.mkdir()
    (package_metadata / "PKG-INFO").write_text(
        "Name: ophagent\n",
        encoding="utf-8",
    )

    assert verify_release(tmp_path)["verified"] is True


def test_release_zip_preserves_full_semantic_version(tmp_path: Path) -> None:
    release_root = tmp_path / "OphAgent-0.1.0"
    release_root.mkdir()
    (release_root / "README.md").write_text("release\n", encoding="utf-8")

    archive = _write_zip(release_root)

    assert archive.name == "OphAgent-0.1.0.zip"
    assert archive.is_file()
