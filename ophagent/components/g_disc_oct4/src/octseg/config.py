from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .legacy import LEGACY_FILES, project_path


CONFIG_ROOT = project_path("configs")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_pipeline_config(pipeline: str, override_path: Path | None = None) -> dict[str, Any]:
    defaults = load_json(CONFIG_ROOT / "defaults.json")
    profile = load_json(CONFIG_ROOT / "pipeline" / f"{pipeline}.json")
    merged = deep_merge(defaults, profile)
    if override_path is not None:
        merged = deep_merge(merged, load_json(override_path))
    merged.setdefault("paths", {}).update(
        {key: str(path) for key, path in LEGACY_FILES.items()}
    )
    histogram_key = "histogram_triton" if pipeline == "fda" else "histogram_3doct"
    segmentation_args = list(merged.get("segmentation", {}).get("extra_args", []))
    if "--hist_match" in segmentation_args:
        index = segmentation_args.index("--hist_match")
        if index + 1 < len(segmentation_args):
            segmentation_args[index + 1] = str(LEGACY_FILES[histogram_key])
        merged["segmentation"]["extra_args"] = segmentation_args
    return merged
