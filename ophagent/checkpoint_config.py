"""Deployment checkpoint configuration and integrity checks.

The Web UI stores only deployment state here. Checkpoint files and external
repositories remain outside the source checkout under ``OPHAGENT_RUNTIME_DIR``
or explicit environment paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from .model_assets import model_asset_index
from .utils.paths import (
    BUNDLED_COMPONENTS_DIR,
    CKPT_DIR,
    EXTERNAL_DIR,
    runtime_path,
)


CONFIG_PATH = runtime_path("config", "checkpoints.json")
_CONFIG_LOCK = threading.RLock()
_STARTUP_PATH_OVERRIDES: dict[str, str] = {}


def _file(
    resource_id: str,
    label: str,
    env_var: str,
    *default_parts: str,
    manifest_path: str,
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "label": label,
        "env_var": env_var,
        "kind": "file",
        "base": "checkpoint",
        "default_parts": default_parts,
        "manifest_path": manifest_path.replace("\\", "/"),
        "markers": (),
    }


def _directory(
    resource_id: str,
    label: str,
    env_var: str,
    *default_parts: str,
    markers: tuple[str, ...],
    compatible_marker_sets: tuple[tuple[str, ...], ...] = (),
    base: str = "external",
) -> dict[str, Any]:
    return {
        "id": resource_id,
        "label": label,
        "env_var": env_var,
        "kind": "directory",
        "base": base,
        "default_parts": default_parts,
        "manifest_path": None,
        "markers": markers,
        "compatible_marker_sets": compatible_marker_sets,
    }


CHECKPOINT_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "modality_detector",
        "label": "Modality detector",
        "modality": "Shared",
        "tools": (),
        "resources": (
            _file(
                "weights", "4-class detector", "OPHAGENT_MODALITY_CLASSIFIER_WEIGHTS",
                "_shared", "modality_classifier", "best.pt",
                manifest_path="checkpoints/_shared/modality_classifier/best.pt",
            ),
        ),
    },
    {
        "id": "cfp_clip",
        "label": "Retinal CLIP ensemble",
        "modality": "CFP",
        "tools": (
            "cfp_clip_ensemble", "cfp_clip_multi_disease", "cfp_flair",
            "cfp_retizero", "cfp_dynamic_clip", "cfp_openvocab_zeroshot",
            "cfp_dr_workup", "cross_cfp_oct", "cross_cfp_ffa",
        ),
        "resources": (
            _file(
                "cvl_weights", "CVL weights", "OPHAGENT_CFP_CLIP_WEIGHTS",
                "cfp", "cvl.pt", manifest_path="checkpoints/cfp/cvl.pt",
            ),
            _directory(
                "cvl_source", "CVL source", "OPHAGENT_CFP_CLIP_SRC", "CVL",
                markers=("cn_clip/clip/model.py",),
                compatible_marker_sets=(("CVL/clip/model.py",),),
            ),
            _file(
                "flair_weights", "FLAIR weights", "OPHAGENT_FLAIR_WEIGHTS",
                "cfp", "flair.pth", manifest_path="checkpoints/cfp/flair.pth",
            ),
            _directory(
                "flair_source", "FLAIR source", "OPHAGENT_FLAIR_ROOT", "flair",
                markers=("flair/__init__.py",),
            ),
            _file(
                "retizero_weights", "RetiZero weights", "OPHAGENT_RETIZERO_WEIGHTS",
                "cfp", "retizero.pth", manifest_path="checkpoints/cfp/retizero.pth",
            ),
            _directory(
                "retizero_source", "RetiZero source", "OPHAGENT_RETIZERO_ROOT",
                "retizero", markers=("zeroshot/__init__.py",),
            ),
        ),
    },
    {
        "id": "cfp_eyeq",
        "label": "EyeQ quality",
        "modality": "CFP",
        "tools": (
            "cfp_eyeq", "cfp_quality_robust", "cfp_dr_workup",
            "cross_cfp_oct",
        ),
        "resources": (
            _file(
                "weights", "EyeQ weights", "OPHAGENT_EYEQ_WEIGHTS",
                "cfp", "eyeq.pth", manifest_path="checkpoints/cfp/eyeq.pth",
            ),
        ),
    },
    {
        "id": "cfp_efiqa",
        "label": "EFIQA quality",
        "modality": "CFP",
        "tools": ("cfp_efiqa", "cfp_dr_workup", "cross_cfp_oct"),
        "resources": (
            _file(
                "weights", "EFIQA adapter weights", "OPHAGENT_EFIQA_WEIGHTS",
                "cfp", "efiqa_adapter.npz",
                manifest_path="checkpoints/cfp/efiqa_adapter.npz",
            ),
        ),
    },
    {
        "id": "cfp_od",
        "label": "Optic disc and fovea",
        "modality": "CFP",
        "tools": ("cfp_od_detection", "cfp_glaucoma", "cfp_glaucoma_workup"),
        "resources": (
            _file(
                "weights", "YOLO weights", "OPHAGENT_OD_WEIGHTS",
                "cfp", "od_fovea.pt", manifest_path="checkpoints/cfp/od_fovea.pt",
            ),
        ),
    },
    {
        "id": "cfp_glaucoma",
        "label": "Glaucoma classifier",
        "modality": "CFP",
        "tools": ("cfp_glaucoma", "cfp_glaucoma_workup"),
        "resources": (
            _file(
                "weights", "Glaucoma weights", "OPHAGENT_GLAUCOMA_WEIGHTS",
                "cfp", "glaucoma.pth", manifest_path="checkpoints/cfp/glaucoma.pth",
            ),
        ),
    },
    {
        "id": "cfp_pdr",
        "label": "PDR cascade",
        "modality": "CFP",
        "tools": ("cfp_pdr_cascade", "cfp_dr_workup", "cross_cfp_oct"),
        "resources": (
            _file(
                "main", "Main classifier", "OPHAGENT_PDR_MAIN_CKPT",
                "cfp", "pdr_main.pth", manifest_path="checkpoints/cfp/pdr_main.pth",
            ),
            _file(
                "active", "Active reasons", "OPHAGENT_PDR_ACTIVE_CKPT",
                "cfp", "pdr_active.pth", manifest_path="checkpoints/cfp/pdr_active.pth",
            ),
            _file(
                "active_thresholds", "Active thresholds", "OPHAGENT_PDR_ACTIVE_THR",
                "cfp", "pdr_active_thr.pth",
                manifest_path="checkpoints/cfp/pdr_active_thr.pth",
            ),
            _file(
                "inactive", "Inactive reasons", "OPHAGENT_PDR_INACTIVE_CKPT",
                "cfp", "pdr_inactive.pth",
                manifest_path="checkpoints/cfp/pdr_inactive.pth",
            ),
            _file(
                "inactive_thresholds", "Inactive thresholds",
                "OPHAGENT_PDR_INACTIVE_THR", "cfp", "pdr_inactive_thr.pth",
                manifest_path="checkpoints/cfp/pdr_inactive_thr.pth",
            ),
        ),
    },
    {
        "id": "cfp_retsam",
        "label": "ReT-SAM segmentation",
        "modality": "CFP",
        "tools": ("cfp_retsam_segmentation", "cfp_dr_421_assessment"),
        "resources": (
            _file(
                "weights", "ReT-SAM weights", "OPHAGENT_RETSAM_CKPT",
                "cfp", "retsam.ckpt", manifest_path="checkpoints/cfp/retsam.ckpt",
            ),
            _directory(
                "source", "ReT-SAM source", "OPHAGENT_RETSAM_SRC", "retsam_2",
                markers=("scripts/infer.py", "scripts/quantify.py"),
                base="bundled",
            ),
        ),
    },
    {
        "id": "oct_fmue",
        "label": "FMUE classifier",
        "modality": "OCT",
        "tools": ("oct_fmue_16class", "oct_volume_macular", "cross_cfp_oct"),
        "resources": (
            _file(
                "weights", "FMUE weights", "OPHAGENT_FMUE_WEIGHTS",
                "oct", "fmue.pth", manifest_path="checkpoints/oct/fmue.pth",
            ),
            _directory(
                "source", "FMUE source", "OPHAGENT_FMUE_SRC", "fmue",
                markers=("vit_model.py",),
            ),
        ),
    },
    {
        "id": "oct_bscan",
        "label": "OCT B-scan models",
        "modality": "OCT",
        "tools": ("oct_fluid_segmentation", "oct_layer_segmentation", "oct_quality"),
        "resources": (
            _file(
                "fluid", "Fluid segmentor", "OPHAGENT_OCT_FLUID_WEIGHTS",
                "oct", "fluid_segmentor", "best.pt",
                manifest_path="checkpoints/oct/fluid_segmentor/best.pt",
            ),
            _file(
                "layers", "Layer segmentor", "OPHAGENT_OCT_LAYER_WEIGHTS",
                "oct", "layer_segmentor", "best.pt",
                manifest_path="checkpoints/oct/layer_segmentor/best.pt",
            ),
            _file(
                "quality", "Quality assessor", "OPHAGENT_OCT_QUALITY_WEIGHTS",
                "oct", "quality_assessor", "best.pt",
                manifest_path="checkpoints/oct/quality_assessor/best.pt",
            ),
        ),
    },
    {
        "id": "oct_octcubem",
        "label": "OCTCubeM volume model",
        "modality": "OCT",
        "tools": ("oct_volume_octcubem",),
        "resources": (
            _file(
                "weights", "OCTCubeM weights", "OPHAGENT_OCTCUBEM_WEIGHTS",
                "oct_volume", "octcube.pth",
                manifest_path="checkpoints/oct_volume/octcube.pth",
            ),
            _directory(
                "source", "OCTCubeM source", "OPHAGENT_OCTCUBEM_ROOT", "OCTCubeM",
                markers=("inference_utils.py", "OCTCube"),
            ),
        ),
    },
    {
        "id": "oct_disc",
        "label": "OCT disc analysis",
        "modality": "OCT",
        "tools": ("oct_volume_disc",),
        "resources": (
            _file(
                "segmentation_weights", "Layer segmentation weights",
                "OPHAGENT_G_DISC_SEGMENTATION_WEIGHTS",
                "oct_volume", "g_disc", "tohoku_full_slss_013_fold0_3.522%.t7",
                manifest_path=(
                    "checkpoints/oct_volume/g_disc/"
                    "tohoku_full_slss_013_fold0_3.522%.t7"
                ),
            ),
            _file(
                "wnet_weights", "Disc localisation weights",
                "OPHAGENT_G_DISC_WNET_WEIGHTS",
                "oct_volume", "g_disc",
                "wnet_disc512_best_UCL-new3-0.793.pth",
                manifest_path=(
                    "checkpoints/oct_volume/g_disc/"
                    "wnet_disc512_best_UCL-new3-0.793.pth"
                ),
            ),
            _file(
                "histogram_3doct", "3D OCT histogram calibration",
                "OPHAGENT_G_DISC_3DOCT_HISTOGRAM",
                "oct_volume", "g_disc", "3DOCT_hist.cdf",
                manifest_path="checkpoints/oct_volume/g_disc/3DOCT_hist.cdf",
            ),
            _file(
                "histogram_triton", "Triton histogram calibration",
                "OPHAGENT_G_DISC_TRITON_HISTOGRAM",
                "oct_volume", "g_disc", "Triton_hist.cdf",
                manifest_path="checkpoints/oct_volume/g_disc/Triton_hist.cdf",
            ),
            _directory(
                "source", "G-DISC source", "OPHAGENT_G_DISC_ROOT",
                "g_disc_oct4",
                markers=(
                    "src/octseg/runner.py",
                    "legacy_pipeline/seg1d2d_aspectfix.py",
                ),
                base="bundled",
            ),
        ),
    },
    {
        "id": "uwf_classifiers",
        "label": "UWF classifiers",
        "modality": "UWF",
        "tools": ("uwf_multi_disease", "uwf_disease_7class"),
        "resources": (
            _file(
                "multilabel", "Multi-label weights", "OPHAGENT_UWF_WEIGHTS",
                "uwf", "multi_disease.pt",
                manifest_path="checkpoints/uwf/multi_disease.pt",
            ),
            _file(
                "disease7", "7-class weights", "OPHAGENT_UWF_DISEASE7_WEIGHTS",
                "uwf", "disease7", "best.pt",
                manifest_path="checkpoints/uwf/disease7/best.pt",
            ),
        ),
    },
    {
        "id": "uwf_vessels",
        "label": "UWF vessel segmentation",
        "modality": "UWF",
        "tools": ("uwf_vessel_segmentation",),
        "resources": (
            _file(
                "weights", "Vessel weights", "OPHAGENT_PRIME_FP20_WEIGHTS",
                "uwf", "vessel_seg.pth",
                manifest_path="checkpoints/uwf/vessel_seg.pth",
            ),
        ),
    },
    {
        "id": "ffa_core",
        "label": "FFA models",
        "modality": "FFA",
        "tools": (
            "ffa_classification", "ffa_lesion_detection", "cross_cfp_ffa",
        ),
        "resources": (
            _file(
                "classifier", "Classifier weights", "OPHAGENT_FFA_CLASSIFIER_WEIGHTS",
                "ffa", "classification.pt",
                manifest_path="checkpoints/ffa/classification.pt",
            ),
            _file(
                "detector", "Lesion detector", "OPHAGENT_FFA_DETECTOR_WEIGHTS",
                "ffa", "lesion_yolo11x.pt",
                manifest_path="checkpoints/ffa/lesion_yolo11x.pt",
            ),
            _file(
                "merge_map", "Class merge map", "OPHAGENT_FFA_MERGE_MAP",
                "ffa", "merge_map_ids.csv",
                manifest_path="checkpoints/ffa/merge_map_ids.csv",
            ),
        ),
    },
    {
        "id": "paired_cfp_ffa",
        "label": "Paired CFP + FFA",
        "modality": "Multi",
        "tools": (
            "cfp_paired5", "ffa_paired5", "cross_cfp_ffa_softvote",
            "cross_cfp_ffa_paired", "paired_bilingual_report",
        ),
        "resources": (
            _file(
                "cfp", "CFP branch", "OPHAGENT_PAIRED_CFP_WEIGHTS",
                "paired", "cfp", "best.pt",
                manifest_path="checkpoints/paired/cfp/best.pt",
            ),
            _file(
                "ffa", "FFA branch", "OPHAGENT_PAIRED_FFA_WEIGHTS",
                "paired", "ffa", "best.pt",
                manifest_path="checkpoints/paired/ffa/best.pt",
            ),
            _file(
                "joint", "Joint fusion", "OPHAGENT_PAIRED_JOINT_WEIGHTS",
                "paired", "joint", "best.pt",
                manifest_path="checkpoints/paired/joint/best.pt",
            ),
        ),
    },
)


GROUP_BY_ID = {group["id"]: group for group in CHECKPOINT_GROUPS}
RESOURCE_BY_ENV = {
    resource["env_var"]: resource
    for group in CHECKPOINT_GROUPS
    for resource in group["resources"]
}
_BASE_ENV_VALUES = {
    env_var: os.environ.get(env_var, "").strip()
    for env_var in RESOURCE_BY_ENV
}


def _default_path(resource: dict[str, Any]) -> Path:
    roots = {
        "checkpoint": CKPT_DIR,
        "external": EXTERNAL_DIR,
        "bundled": BUNDLED_COMPONENTS_DIR,
    }
    try:
        base = roots[resource["base"]]
    except KeyError as exc:
        raise ValueError(f"unknown resource base: {resource['base']}") from exc
    return (base / Path(*resource["default_parts"])).resolve()


def _expand_absolute_path(value: str) -> Path:
    if len(value) > 4096:
        raise ValueError("path is too long")
    expanded = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not expanded.is_absolute():
        raise ValueError("checkpoint paths must be absolute")
    return expanded.resolve()


def _fallback_path(resource: dict[str, Any]) -> tuple[Path, str]:
    env_value = _BASE_ENV_VALUES.get(resource["env_var"], "")
    if env_value:
        return _expand_absolute_path(env_value), "environment"
    return _default_path(resource), "default"


def _clean_config(raw: Any) -> dict[str, Any]:
    clean: dict[str, Any] = {"version": 1, "groups": {}}
    if not isinstance(raw, dict) or not isinstance(raw.get("groups"), dict):
        return clean
    for group_id, value in raw["groups"].items():
        group = GROUP_BY_ID.get(group_id)
        if not group or not isinstance(value, dict):
            continue
        entry: dict[str, Any] = {}
        if value.get("enabled") is False:
            entry["enabled"] = False
        valid_resources = {resource["id"] for resource in group["resources"]}
        paths: dict[str, str] = {}
        if isinstance(value.get("paths"), dict):
            for resource_id, path_value in value["paths"].items():
                if resource_id not in valid_resources or not isinstance(path_value, str):
                    continue
                try:
                    paths[resource_id] = str(_expand_absolute_path(path_value))
                except (OSError, ValueError):
                    continue
        if paths:
            entry["paths"] = paths
        if entry:
            clean["groups"][group_id] = entry
    return clean


def load_checkpoint_config() -> dict[str, Any]:
    with _CONFIG_LOCK:
        if not CONFIG_PATH.exists():
            return {"version": 1, "groups": {}}
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "groups": {}}
        return _clean_config(raw)


def _write_checkpoint_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not config.get("groups"):
        CONFIG_PATH.unlink(missing_ok=True)
        return
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(CONFIG_PATH)


def _path_overrides(config: dict[str, Any]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for group_id, entry in config.get("groups", {}).items():
        group = GROUP_BY_ID.get(group_id)
        if not group:
            continue
        by_id = {resource["id"]: resource for resource in group["resources"]}
        for resource_id, path in entry.get("paths", {}).items():
            resource = by_id.get(resource_id)
            if resource:
                overrides[resource["env_var"]] = path
    return overrides


def apply_saved_checkpoint_environment() -> dict[str, str]:
    """Apply saved path overrides before adapter modules are imported."""
    global _STARTUP_PATH_OVERRIDES
    overrides = _path_overrides(load_checkpoint_config())
    for env_var, path in overrides.items():
        os.environ[env_var] = path
    _STARTUP_PATH_OVERRIDES = dict(overrides)
    return overrides


def checkpoint_restart_required(config: dict[str, Any] | None = None) -> bool:
    current = _path_overrides(config or load_checkpoint_config())
    return current != _STARTUP_PATH_OVERRIDES


def group_is_enabled(group_id: str, config: dict[str, Any] | None = None) -> bool:
    cfg = config or load_checkpoint_config()
    return cfg.get("groups", {}).get(group_id, {}).get("enabled", True) is not False


def tool_is_enabled(tool_name: str, config: dict[str, Any] | None = None) -> bool:
    cfg = config or load_checkpoint_config()
    owners = [group for group in CHECKPOINT_GROUPS if tool_name in group["tools"]]
    return all(group_is_enabled(group["id"], cfg) for group in owners)


def disabled_tool_names(config: dict[str, Any] | None = None) -> set[str]:
    cfg = config or load_checkpoint_config()
    return {
        tool
        for group in CHECKPOINT_GROUPS
        if not group_is_enabled(group["id"], cfg)
        for tool in group["tools"]
    }


def update_checkpoint_group(
    group_id: str,
    *,
    enabled: bool | None = None,
    paths: dict[str, str] | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    group = GROUP_BY_ID.get(group_id)
    if not group:
        raise KeyError(group_id)
    with _CONFIG_LOCK:
        config = load_checkpoint_config()
        before = json.loads(json.dumps(config))
        entry = dict(config.get("groups", {}).get(group_id, {}))
        if enabled is not None:
            if enabled:
                entry.pop("enabled", None)
            else:
                entry["enabled"] = False
        if paths is not None:
            resources = {resource["id"]: resource for resource in group["resources"]}
            saved_paths = dict(entry.get("paths", {}))
            for resource_id, raw_path in paths.items():
                resource = resources.get(resource_id)
                if not resource:
                    raise ValueError(f"unknown resource: {resource_id}")
                raw_path = str(raw_path or "").strip()
                if not raw_path:
                    saved_paths.pop(resource_id, None)
                    continue
                path = _expand_absolute_path(raw_path)
                fallback, _ = _fallback_path(resource)
                if path == fallback:
                    saved_paths.pop(resource_id, None)
                else:
                    saved_paths[resource_id] = str(path)
            if saved_paths:
                entry["paths"] = saved_paths
            else:
                entry.pop("paths", None)
        groups = dict(config.get("groups", {}))
        if entry:
            groups[group_id] = entry
        else:
            groups.pop(group_id, None)
        config = {"version": 1, "groups": groups}
        _write_checkpoint_config(config)
        paths_changed = _path_overrides(before) != _path_overrides(config)
        enabled_changed = group_is_enabled(group_id, before) != group_is_enabled(
            group_id, config
        )
        return config, paths_changed, enabled_changed


@lru_cache(maxsize=1)
def _manifest_index() -> dict[str, dict[str, Any]]:
    # The package manifest is the public source of truth. A deployment may
    # provide a runtime MANIFEST.yaml to add or override authorised assets
    # without modifying the source checkout.
    index = model_asset_index(prefix="checkpoints")
    manifest_path = CKPT_DIR / "MANIFEST.yaml"
    if not manifest_path.is_file():
        return index
    try:
        import yaml

        text = manifest_path.read_text(encoding="utf-8")
        # Older manifests may contain a double-quoted Windows absolute path in
        # an informational ``original_location`` field. Its backslashes can be
        # invalid YAML escapes. The field is irrelevant to integrity checking,
        # so remove only that field before parsing the remaining manifest.
        text = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("original_location:")
        )
        payload = yaml.safe_load(text) or {}
    except Exception:
        return index
    for item in payload.get("checkpoints", []):
        if not isinstance(item, dict) or not item.get("path"):
            continue
        key = str(item["path"]).replace("\\", "/").lstrip("./")
        index[key] = {
            "size_bytes": item.get("size_bytes"),
            "sha256": str(item.get("sha256") or "").lower(),
        }
    return index


def _resource_path(
    group: dict[str, Any],
    resource: dict[str, Any],
    config: dict[str, Any],
    draft_paths: dict[str, str] | None = None,
) -> tuple[Path, str]:
    if draft_paths is not None and resource["id"] in draft_paths:
        value = str(draft_paths[resource["id"]] or "").strip()
        if value:
            return _expand_absolute_path(value), "draft"
        return _fallback_path(resource)
    entry = config.get("groups", {}).get(group["id"], {})
    saved = entry.get("paths", {}).get(resource["id"])
    if saved:
        return _expand_absolute_path(saved), "saved"
    return _fallback_path(resource)


def _quick_resource_status(
    resource: dict[str, Any],
    path: Path,
    manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = manifest.get(resource.get("manifest_path") or "", {})
    result: dict[str, Any] = {
        "status": "ready",
        "exists": path.exists(),
        "size_bytes": None,
        "expected_size_bytes": expected.get("size_bytes"),
        "checksum_available": bool(expected.get("sha256")),
        "message": "Ready",
    }
    if not path.exists():
        result.update(status="missing", message="Path not found")
        return result
    if resource["kind"] == "file":
        if not path.is_file():
            result.update(status="mismatch", message="Expected a file")
            return result
        size = path.stat().st_size
        result["size_bytes"] = size
        expected_size = expected.get("size_bytes")
        if isinstance(expected_size, int) and size != expected_size:
            result.update(status="mismatch", message="File size does not match")
    else:
        if not path.is_dir():
            result.update(status="mismatch", message="Expected a directory")
            return result
        primary_markers = tuple(resource.get("markers", ()))
        marker_sets = (
            primary_markers,
            *resource.get("compatible_marker_sets", ()),
        )
        marker_match = any(
            markers and all((path / marker).exists() for marker in markers)
            for markers in marker_sets
        )
        missing_markers = (
            []
            if marker_match
            else [
                marker for marker in primary_markers
                if not (path / marker).exists()
            ]
        )
        if missing_markers:
            result.update(
                status="mismatch",
                message=f"Missing marker: {missing_markers[0]}",
            )
    return result


def _group_status(resources: list[dict[str, Any]], enabled: bool) -> str:
    if not enabled:
        return "disabled"
    statuses = {resource["status"] for resource in resources}
    if "missing" in statuses:
        return "missing"
    if "mismatch" in statuses:
        return "mismatch"
    if statuses == {"verified"}:
        return "verified"
    return "ready"


def checkpoint_group_view(
    group_id: str,
    *,
    config: dict[str, Any] | None = None,
    draft_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    group = GROUP_BY_ID[group_id]
    cfg = config or load_checkpoint_config()
    manifest = _manifest_index()
    resources: list[dict[str, Any]] = []
    for resource in group["resources"]:
        path, source = _resource_path(group, resource, cfg, draft_paths)
        status = _quick_resource_status(resource, path, manifest)
        resources.append({
            "id": resource["id"],
            "label": resource["label"],
            "kind": resource["kind"],
            "path": str(path),
            "source": source,
            **status,
        })
    enabled = group_is_enabled(group_id, cfg)
    return {
        "id": group_id,
        "label": group["label"],
        "modality": group["modality"],
        "enabled": enabled,
        "status": _group_status(resources, enabled),
        "tool_count": len(group["tools"]),
        "resource_count": len(resources),
        "resources": resources,
    }


def checkpoint_settings_view() -> dict[str, Any]:
    config = load_checkpoint_config()
    groups = [
        checkpoint_group_view(group["id"], config=config)
        for group in CHECKPOINT_GROUPS
    ]
    counts: dict[str, int] = {}
    for group in groups:
        counts[group["status"]] = counts.get(group["status"], 0) + 1
    return {
        "groups": groups,
        "summary": counts,
        "restart_required": checkpoint_restart_required(config),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint_group(
    group_id: str,
    draft_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    group = GROUP_BY_ID.get(group_id)
    if not group:
        raise KeyError(group_id)
    config = load_checkpoint_config()
    manifest = _manifest_index()
    checked: list[dict[str, Any]] = []
    for resource in group["resources"]:
        path, source = _resource_path(group, resource, config, draft_paths)
        result = _quick_resource_status(resource, path, manifest)
        expected = manifest.get(resource.get("manifest_path") or "", {})
        if result["status"] == "ready":
            if resource["kind"] == "directory":
                result.update(status="verified", message="Source structure verified")
            elif expected.get("sha256"):
                actual = _sha256(path)
                if actual == expected["sha256"]:
                    result.update(status="verified", message="SHA-256 verified")
                else:
                    result.update(status="mismatch", message="SHA-256 does not match")
            else:
                result["message"] = "File exists; no reference checksum available"
        checked.append({
            "id": resource["id"],
            "label": resource["label"],
            "kind": resource["kind"],
            "path": str(path),
            "source": source,
            **result,
        })
    enabled = group_is_enabled(group_id, config)
    return {
        "id": group_id,
        "enabled": enabled,
        "status": _group_status(checked, enabled),
        "resources": checked,
    }
