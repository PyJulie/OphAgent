"""
Top-level lesion analysis API.

Consumes per-class binary lesion masks plus an OD/OC mask (for disc
geometry + fundus denominator) and returns a nested JSON describing each
detected lesion class and group-level / global summaries.

Design
------
* One entry per lesion class across every provided group. Empty classes are
  still reported (with zeros) so downstream consumers don't have to special-
  case "this class wasn't detected".
* Size thresholds (small/medium/large) and ETDRS-like zone boundaries are
  DD-normalised so the same config applies across image sizes and FOVs.
* Spatial fields (quadrant counts, macula zone counts, distances) degrade
  gracefully: ``None`` when the required input (macula centre / eye side)
  isn't available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..disc_cup.geometry import (
    NoDiscError,
    compute_disc_geometry,
    from_od_oc_labelmap,
)
from .per_class import ClassMetrics, ComponentStats, compute_class_metrics
from .severity import (
    amd_severity_inputs,
    dr_severity_inputs,
    other_findings_inputs,
)
from .spatial import SpatialStats, compute_spatial_stats


# --------------------------------------------------------------------------- defaults

# These mirror postprocess/../inference/config.py::grouped_lesion_type_mappings,
# reproduced here so the lesion module does not import the inference config.
# Keep in sync with the saved mask filenames.
DEFAULT_GROUP_CLASS_MAP: Dict[str, Dict[int, str]] = {
    "lesion_dr": {
        1: "hemorrhage",
        2: "exudate",
        3: "cotton_wool_spot",
        4: "laser_spot",
    },
    "lesion_amd": {
        1: "drusen",
        2: "patch_hemorrhage",
    },
    "lesion_others": {
        1: "epiretinal_membrane",
        2: "artifact",
        3: "retinal_scar",
        4: "macular_hole",
    },
    "possible_lesions": {
        1: "other_lesions",
        2: "myelinated_nerve_fiber",
        3: "venous_tortuosity",
    },
}


@dataclass
class LesionAnalysisConfig:
    """All length/area thresholds are DD-normalised."""

    min_component_size_px: int = 15
    size_small_max_dd2: float = 0.005       # small-lesion cutoff in disc-diameter²
    size_medium_max_dd2: float = 0.05
    macula_zone_boundaries_dd: Sequence[float] = (1.0, 2.0)  # 0-1, 1-2, >2 DD
    heavy_hemorrhage_per_quadrant: int = 20


# --------------------------------------------------------------------------- helpers

def _px_to_um(value_px: float, spacing_um: Optional[float]) -> Optional[float]:
    return None if spacing_um is None else float(value_px) * float(spacing_um)


def _area_px_to_um2(area_px: int, spacing_um: Optional[float]) -> Optional[float]:
    if spacing_um is None:
        return None
    return float(area_px) * float(spacing_um) * float(spacing_um)


def _area_triple(area_px: int, DD: float, spacing_um: Optional[float]) -> Dict[str, Any]:
    area_dd2 = float(area_px) / (DD * DD) if DD > 0 else 0.0
    return {
        "px": int(area_px),
        "dd2": area_dd2,
        "um2": _area_px_to_um2(area_px, spacing_um),
    }


# --------------------------------------------------------------------------- loader

def load_lesion_masks_from_dir(
    masks_dir: str,
    group_class_map: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-class binary lesion masks saved by ``scripts/infer.py``.

    The on-disk convention is ``{group}_{class}.png`` (e.g.
    ``lesion_dr_hemorrhage.png``), values ``0`` or ``255``. Missing files
    are silently skipped (class simply absent from the returned dict, which
    the analyser treats as zero lesions of that class).
    """
    if group_class_map is None:
        group_class_map = DEFAULT_GROUP_CLASS_MAP
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for group, class_map in group_class_map.items():
        group_out: Dict[str, np.ndarray] = {}
        for class_name in class_map.values():
            path = os.path.join(masks_dir, f"{group}_{class_name}.png")
            if not os.path.isfile(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3:
                img = img[..., 0]
            group_out[class_name] = (img > 0).astype(np.uint8)
        out[group] = group_out
    return out


# --------------------------------------------------------------------------- output builder

def _component_to_dict(c: ComponentStats) -> Dict[str, Any]:
    return {
        "centroid_yx": [float(c.centroid_yx[0]), float(c.centroid_yx[1])],
        "area_px": int(c.area_px),
        "bbox_yxyx": list(c.bbox_yxyx),
        "circularity": float(c.circularity),
        "aspect_ratio": float(c.aspect_ratio),
    }


def _class_output(
    class_name: str,
    metrics: ClassMetrics,
    spatial: SpatialStats,
    DD: float,
    fundus_area_px: int,
    pixel_spacing_um: Optional[float],
    include_components: bool = True,
) -> Dict[str, Any]:
    coverage = (metrics.area_px / fundus_area_px) if fundus_area_px > 0 else 0.0
    return {
        "count": metrics.count,
        "area": _area_triple(metrics.area_px, DD, pixel_spacing_um),
        "area_dd2": float(metrics.area_px) / (DD * DD) if DD > 0 else 0.0,
        "coverage_ratio": float(coverage),
        "shape": {
            "mean_circularity": metrics.shape_mean_circularity,
            "mean_aspect_ratio": metrics.shape_mean_aspect_ratio,
        },
        "size_distribution": {
            "thresholds_dd2": list(metrics.size_thresholds_dd2),
            "thresholds_px": list(metrics.size_thresholds_px),
            "counts": dict(metrics.size_counts),
            "areas_px": dict(metrics.size_areas_px),
        },
        "spatial": {
            "quadrant_counts": spatial.quadrant_counts,
            "quadrants_with_lesions": spatial.quadrants_with_lesions,
            "distance_to_disc_dd": spatial.distance_to_disc_dd,
            "distance_to_macula_dd": spatial.distance_to_macula_dd,
            "macula_zone_counts": spatial.macula_zone_counts,
            "macula_zone_areas_px": spatial.macula_zone_areas_px,
            "zone_boundaries_dd": spatial.zone_boundaries_dd,
        },
        "components": [_component_to_dict(c) for c in metrics.components] if include_components else None,
    }


# --------------------------------------------------------------------------- public

def analyze_lesions(
    lesion_masks: Dict[str, Dict[str, np.ndarray]],
    od_oc_mask: np.ndarray,
    fundus_mask: Optional[np.ndarray] = None,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    eye_side: Optional[str] = None,
    pixel_spacing_um: Optional[float] = None,
    config: Optional[LesionAnalysisConfig] = None,
    group_class_map: Optional[Dict[str, Dict[int, str]]] = None,
    include_components: bool = True,
) -> Dict[str, Any]:
    """Compute per-class, per-group, and global lesion metrics."""
    if eye_side not in ("OS", "OD", None):
        raise ValueError(f"eye_side must be 'OS', 'OD' or None; got {eye_side!r}")

    cfg = config or LesionAnalysisConfig()
    gc_map = group_class_map or DEFAULT_GROUP_CLASS_MAP

    # --- disc geometry (required) -------------------------------------------
    disc_bin, _ = from_od_oc_labelmap(od_oc_mask)
    disc_geom = compute_disc_geometry(disc_bin)  # raises NoDiscError if empty
    DD = disc_geom.diameter_vertical_px

    # --- fundus denominator -------------------------------------------------
    if fundus_mask is not None:
        fundus_area_px = int((fundus_mask > 0).sum())
        fundus_source = "fundus_mask"
    else:
        fundus_area_px = int(disc_bin.size)
        fundus_source = "full_image"

    qc_messages: List[str] = []
    qc_passed = True
    if fundus_mask is None:
        qc_messages.append(
            "No fundus_mask provided; coverage ratios use full image area as denominator."
        )
    if macula_center_yx is None:
        qc_messages.append(
            "macula_center_yx not provided; quadrant counts and macula-distance "
            "metrics will be null."
        )
    if eye_side is None:
        qc_messages.append(
            "eye_side not provided; nasal/temporal quadrant split is unavailable "
            "and reported as null."
        )

    # --- per-class loop -----------------------------------------------------
    groups_out: Dict[str, Any] = {}
    any_detected = False
    global_total_area_px = 0

    for group_name, class_map in gc_map.items():
        group_masks = lesion_masks.get(group_name, {})
        class_outputs: Dict[str, Any] = {}
        group_total_count = 0
        group_total_area_px = 0
        for class_id, class_name in class_map.items():
            mask = group_masks.get(class_name)
            if mask is None:
                mask = np.zeros(disc_bin.shape, dtype=np.uint8)
            metrics = compute_class_metrics(
                mask,
                disc_diameter_px=DD,
                min_component_size_px=cfg.min_component_size_px,
                size_small_max_dd2=cfg.size_small_max_dd2,
                size_medium_max_dd2=cfg.size_medium_max_dd2,
            )
            spatial = compute_spatial_stats(
                metrics.components,
                disc_center_yx=disc_geom.center_yx,
                disc_diameter_px=DD,
                macula_center_yx=macula_center_yx,
                eye_side=eye_side,
                zone_boundaries_dd=cfg.macula_zone_boundaries_dd,
            )
            class_outputs[class_name] = _class_output(
                class_name, metrics, spatial, DD, fundus_area_px,
                pixel_spacing_um, include_components=include_components,
            )
            group_total_count += metrics.count
            group_total_area_px += metrics.area_px
            if metrics.count > 0:
                any_detected = True

        groups_out[group_name] = {
            "classes": class_outputs,
            "summary": {
                "total_count": group_total_count,
                "total_area": _area_triple(group_total_area_px, DD, pixel_spacing_um),
                "coverage_ratio": (group_total_area_px / fundus_area_px) if fundus_area_px > 0 else 0.0,
            },
        }
        global_total_area_px += group_total_area_px

    # --- severity inputs ----------------------------------------------------
    dr_inputs = dr_severity_inputs(groups_out, eye_side,
                                   heavy_hemorrhage_per_quadrant=cfg.heavy_hemorrhage_per_quadrant)
    amd_inputs = amd_severity_inputs(groups_out, macula_center_present=macula_center_yx is not None)
    others_inputs = other_findings_inputs(groups_out)

    # --- assemble -----------------------------------------------------------
    result: Dict[str, Any] = {
        "units": {
            "pixel_spacing_um": float(pixel_spacing_um) if pixel_spacing_um is not None else None,
            "disc_diameter_px": float(DD),
            "analysis_frame": "original_image",
            "length_fields_suffixes": ["_px", "_dd", "_um (if pixel_spacing_um given)"],
        },
        "qc": {
            "passed": bool(qc_passed),
            "messages": qc_messages,
            "fundus_area_source": fundus_source,
            "fundus_area_px": int(fundus_area_px),
            "any_lesion_detected": bool(any_detected),
            "eye_side_source": "user_provided" if eye_side in ("OS", "OD") else "unknown",
            "macula_center_present": macula_center_yx is not None,
        },
        "disc": {
            "center_yx": [float(disc_geom.center_yx[0]), float(disc_geom.center_yx[1])],
            "diameter_vertical_px": float(DD),
            "area_px": int(disc_geom.area_px),
        },
        "macula": (
            {"center_yx": [float(macula_center_yx[0]), float(macula_center_yx[1])]}
            if macula_center_yx is not None else None
        ),
        "groups": groups_out,
        "global_summary": {
            "total_area_px": int(global_total_area_px),
            "total_coverage_ratio": (global_total_area_px / fundus_area_px) if fundus_area_px > 0 else 0.0,
            "n_classes_detected": int(sum(
                1
                for grp in groups_out.values()
                for cls in grp["classes"].values()
                if cls["count"] > 0
            )),
        },
        "severity_inputs": {
            "dr": dr_inputs,
            "amd": amd_inputs,
            "other_findings": others_inputs,
        },
    }
    return result
