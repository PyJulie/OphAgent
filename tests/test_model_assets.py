from __future__ import annotations

import hashlib
from pathlib import Path

from ophagent.model_assets import (
    asset_report,
    inspect_asset,
    load_model_asset_manifest,
    model_asset_index,
    select_assets,
)


def test_public_asset_index_matches_checkpoint_ui_keys() -> None:
    index = model_asset_index(prefix="checkpoints")
    assert "checkpoints/cfp/glaucoma.pth" in index
    assert len(index["checkpoints/cfp/glaucoma.pth"]["sha256"]) == 64


def test_profile_selection_is_explicit() -> None:
    manifest = load_model_asset_manifest()
    public_assets = select_assets(manifest, profile="public-demo")
    full_assets = select_assets(manifest, profile="manuscript-full")
    assert {asset["id"] for asset in public_assets} == {"modality_classifier"}
    assert len(full_assets) > len(public_assets)


def test_asset_verification_checks_size_and_sha256(tmp_path: Path) -> None:
    payload = b"approved-test-weight"
    expected = hashlib.sha256(payload).hexdigest()
    asset = {
        "id": "test",
        "path": "cfp/test.bin",
        "role": "Test asset",
        "size_bytes": len(payload),
        "sha256": expected,
    }
    path = tmp_path / "cfp" / "test.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    result = inspect_asset(
        asset,
        checkpoint_root=tmp_path,
        verify_hash=True,
        environ={},
    )
    assert result["status"] == "verified"

    path.write_bytes(payload + b"-changed")
    changed = inspect_asset(
        asset,
        checkpoint_root=tmp_path,
        verify_hash=True,
        environ={},
    )
    assert changed["status"] == "mismatch"


def test_missing_profile_assets_fail_readiness(tmp_path: Path) -> None:
    report = asset_report(
        checkpoint_root=tmp_path,
        profile="public-demo",
        verify_hash=False,
        environ={},
    )
    assert report["ready"] is False
    assert report["summary"] == {"missing": 1}
