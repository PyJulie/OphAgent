from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from ophagent import checkpoint_config
from ophagent.adapters.cfp.efiqa import DINO_MODEL_ID, DINO_REVISION
from ophagent.adapters.cfp.retsam import RETSAM_SRC
from ophagent.adapters.oct_volume.disc_analysis import G_DISC_ASSETS, G_DISC_ROOT
from ophagent.components.manager import (
    _select_components,
    load_component_manifest,
    status_report,
)
from ophagent.components.g_disc_oct4.src.octseg.runner import _resolve_python_cmd
from ophagent.utils.paths import DEFAULT_RUNTIME_DIR, RELEASE_ROOT


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _model_asset_manifest() -> dict:
    path = (
        Path(__file__).parents[1]
        / "ophagent"
        / "resources"
        / "model_assets.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_component_manifest_has_unique_locked_sources() -> None:
    manifest = load_component_manifest()
    components = manifest["components"]
    ids = [component["id"] for component in components]
    assert len(ids) == len(set(ids))

    for component in components:
        assert "\\" not in str(component.get("target_dir") or "")
        if component["integration"] == "git":
            assert component["repository"].startswith("https://github.com/")
            assert GIT_REVISION.fullmatch(component["revision"])
            assert component["markers"]
        if component["integration"] == "private-source-pending":
            assert component["install_policy"] == "unavailable"
            assert component["license_status"] == "rights-and-provenance-review-required"

    efiqa = next(component for component in components if component["id"] == "efiqa")
    assert efiqa["remote_model"]["id"] == DINO_MODEL_ID
    assert efiqa["remote_model"]["revision"] == DINO_REVISION


def test_default_runtime_directory_is_outside_the_source_checkout() -> None:
    assert DEFAULT_RUNTIME_DIR == (Path.home() / ".ophagent").resolve()
    assert DEFAULT_RUNTIME_DIR != RELEASE_ROOT
    assert RELEASE_ROOT not in DEFAULT_RUNTIME_DIR.parents


def test_default_install_excludes_unlicensed_components() -> None:
    manifest = load_component_manifest()
    default = _select_components(
        manifest,
        [],
        install_all=True,
        include_unlicensed=False,
    )
    acknowledged = _select_components(
        manifest,
        [],
        install_all=True,
        include_unlicensed=True,
    )
    assert {component["id"] for component in default} == {
        "chinese_clip",
        "flair",
        "octcubem",
    }
    assert {component["id"] for component in acknowledged} == {
        "chinese_clip",
        "flair",
        "retizero",
        "fmue",
        "octcubem",
    }


def test_source_status_distinguishes_builtin_and_release_blockers(
    tmp_path: Path,
) -> None:
    report = status_report(external_root=tmp_path, profile="manuscript-full")
    by_id = {component["id"]: component for component in report["components"]}
    assert by_id["efiqa"]["status"] == "ready"
    assert by_id["glaucoma"]["status"] == "ready"
    assert by_id["pdr_cascade"]["status"] == "ready"
    assert by_id["retsam"]["status"] == "ready"
    assert by_id["g_disc"]["status"] == "ready"
    assert report["ready"] is False


def test_public_demo_profile_excludes_uncleared_sources(tmp_path: Path) -> None:
    report = status_report(external_root=tmp_path, profile="public-demo")
    by_id = {component["id"]: component for component in report["components"]}
    assert set(by_id) == {
        "chinese_clip",
        "flair",
        "efiqa",
        "glaucoma",
        "pdr_cascade",
        "octcubem",
    }
    assert by_id["chinese_clip"]["status"] == "source-required"
    assert by_id["efiqa"]["status"] == "ready"
    assert "retizero" not in by_id
    assert "retsam" not in by_id


def test_legacy_chinese_clip_layout_is_recognised(tmp_path: Path) -> None:
    package = tmp_path / "CVL" / "CVL" / "clip"
    package.mkdir(parents=True)
    (package / "model.py").touch()
    (package / "utils.py").touch()
    report = status_report(external_root=tmp_path, profile="manuscript-full")
    by_id = {component["id"]: component for component in report["components"]}
    assert by_id["chinese_clip"]["status"] == "present-unlocked"


def test_checkpoint_ui_accepts_both_chinese_clip_layouts(tmp_path: Path) -> None:
    group = next(
        group
        for group in checkpoint_config.CHECKPOINT_GROUPS
        if group["id"] == "cfp_clip"
    )
    resource = next(
        resource
        for resource in group["resources"]
        if resource["id"] == "cvl_source"
    )
    for index, package_path in enumerate(("cn_clip/clip", "CVL/clip")):
        layout_root = tmp_path / f"layout-{index}"
        package = layout_root / package_path
        package.mkdir(parents=True)
        (package / "model.py").touch()
        status = checkpoint_config._quick_resource_status(resource, layout_root, {})
        assert status["status"] == "ready"


def test_model_asset_manifest_contains_only_portable_verified_paths() -> None:
    manifest = _model_asset_manifest()
    assets = manifest["assets"]
    ids = [asset["id"] for asset in assets]
    assert len(ids) == len(set(ids))
    for asset in assets:
        path = str(asset["path"])
        assert not Path(path).is_absolute()
        assert "\\" not in path
        assert ":" not in path
        assert SHA256.fullmatch(asset["sha256"])
        assert asset["size_bytes"] > 0
        assert "original_location" not in asset


def test_checkpoint_ui_uses_packaged_asset_manifest_without_runtime_copy() -> None:
    checkpoint_config._manifest_index.cache_clear()
    manifest = checkpoint_config._manifest_index()
    expected = manifest["checkpoints/cfp/glaucoma.pth"]
    assert expected["size_bytes"] == 1215259211
    assert expected["sha256"] == (
        "f8b3f0d4295369b9d07b4d283a3a2539ab738e9edadbad99f1c37cfa227cc7e8"
    )


def test_builtin_adapters_no_longer_require_private_source_directories() -> None:
    groups = {
        group["id"]: group
        for group in checkpoint_config.CHECKPOINT_GROUPS
    }
    for group_id in ("cfp_efiqa", "cfp_glaucoma", "cfp_pdr"):
        assert all(
            resource["kind"] == "file"
            for resource in groups[group_id]["resources"]
        )


def test_bundled_retsam_and_g_disc_are_default_source_paths() -> None:
    assert RETSAM_SRC.name == "retsam_2"
    assert (RETSAM_SRC / "scripts" / "infer.py").is_file()
    assert (RETSAM_SRC / "scripts" / "quantify.py").is_file()
    assert G_DISC_ROOT.name == "g_disc_oct4"
    assert (G_DISC_ROOT / "src" / "octseg" / "runner.py").is_file()
    assert (
        G_DISC_ROOT / "legacy_pipeline" / "seg1d2d_aspectfix.py"
    ).is_file()
    assert _resolve_python_cmd({}) == sys.executable
    assert _resolve_python_cmd({"python_cmd": "custom-python"}) == "custom-python"


def test_g_disc_resources_are_manifested_external_assets() -> None:
    manifest = _model_asset_manifest()
    by_id = {asset["id"]: asset for asset in manifest["assets"]}
    expected = {
        "g_disc_segmentation",
        "g_disc_wnet",
        "g_disc_histogram_3doct",
        "g_disc_histogram_triton",
    }
    assert expected <= set(by_id)
    assert all(
        by_id[asset_id]["path"].startswith("oct_volume/g_disc/")
        for asset_id in expected
    )
    assert {path.name for path in G_DISC_ASSETS.values()} == {
        "tohoku_full_slss_013_fold0_3.522%.t7",
        "wnet_disc512_best_UCL-new3-0.793.pth",
        "3DOCT_hist.cdf",
        "Triton_hist.cdf",
    }
