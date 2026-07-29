"""
Tessellation (豹纹样变) quantification.

Model outputs a single-foreground-class label map (``0=bg, 1=tessellation``).
Clinically, tessellation reflects visible choroidal vessels through a thin RPE,
associated with axial myopia. The report focuses on:

    * **coverage_ratio** — tessellation area / fundus area (the primary
      clinical signal)
    * **severity bucket** — coverage-based heuristic (minimal / mild / moderate
      / severe) with thresholds exposed in the config
    * **spatial** — quadrants and macula-zone distribution, using the same
      ETDRS-style conventions as the lesion module
    * **macula involvement** — tessellation touching within a configurable
      radius of the fovea (relevant to myopic maculopathy META-PM stage 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..disc_cup.geometry import (
    NoDiscError,
    compute_disc_geometry,
    from_od_oc_labelmap,
)
from ..lesions.per_class import compute_class_metrics
from ..lesions.spatial import compute_spatial_stats


@dataclass
class TessellationAnalysisConfig:
    min_component_size_px: int = 20     # tessellation patches are usually sizeable
    size_small_max_dd2: float = 0.05
    size_medium_max_dd2: float = 0.5
    macula_zone_boundaries_dd: Sequence[float] = (1.0, 2.0)
    macula_involvement_radius_dd: float = 1.0
    # Severity thresholds are coverage_ratio (fraction, not percent)
    severity_thresholds: Tuple[float, float, float] = (0.05, 0.15, 0.30)


SEVERITY_LABELS = ("minimal", "mild", "moderate", "severe")


def classify_severity(coverage_ratio: float,
                      thresholds: Tuple[float, float, float] =
                      (0.05, 0.15, 0.30)) -> str:
    if coverage_ratio < thresholds[0]:
        return SEVERITY_LABELS[0]
    if coverage_ratio < thresholds[1]:
        return SEVERITY_LABELS[1]
    if coverage_ratio < thresholds[2]:
        return SEVERITY_LABELS[2]
    return SEVERITY_LABELS[3]


def _area_triple(area_px: int, DD: float, spacing_um: Optional[float]) -> Dict[str, Any]:
    area_dd2 = float(area_px) / (DD * DD) if DD > 0 else 0.0
    area_um2 = None if spacing_um is None else float(area_px) * float(spacing_um) ** 2
    return {"px": int(area_px), "dd2": area_dd2, "um2": area_um2}


def _involves_macula(mask: np.ndarray,
                     macula_center_yx: Optional[Tuple[float, float]],
                     disc_diameter_px: float,
                     radius_dd: float) -> Optional[bool]:
    if macula_center_yx is None or disc_diameter_px <= 0:
        return None
    if (mask > 0).sum() == 0:
        return False
    ys, xs = np.nonzero(mask > 0)
    my, mx = macula_center_yx
    dist_px = np.sqrt((ys - my) ** 2 + (xs - mx) ** 2)
    return bool((dist_px <= radius_dd * disc_diameter_px).any())


def analyze_tessellation(tessellation_mask: np.ndarray,
                         od_oc_mask: np.ndarray,
                         fundus_mask: Optional[np.ndarray] = None,
                         macula_center_yx: Optional[Tuple[float, float]] = None,
                         eye_side: Optional[str] = None,
                         pixel_spacing_um: Optional[float] = None,
                         config: Optional[TessellationAnalysisConfig] = None
                         ) -> Dict[str, Any]:
    if eye_side not in ("OS", "OD", None):
        raise ValueError(f"eye_side must be 'OS'/'OD'/None; got {eye_side!r}")
    cfg = config or TessellationAnalysisConfig()

    if tessellation_mask is None or tessellation_mask.ndim != 2:
        raise ValueError("tessellation_mask must be 2D")

    disc_bin, _ = from_od_oc_labelmap(od_oc_mask)
    disc_geom = compute_disc_geometry(disc_bin)
    DD = disc_geom.diameter_vertical_px

    bin_mask = (tessellation_mask > 0).astype(np.uint8)
    if fundus_mask is not None:
        fundus_area_px = int((fundus_mask > 0).sum())
        fundus_source = "fundus_mask"
    else:
        fundus_area_px = int(bin_mask.size)
        fundus_source = "full_image"

    qc_messages: List[str] = []
    if fundus_mask is None:
        qc_messages.append("No fundus_mask provided; coverage uses full image area.")
    if macula_center_yx is None:
        qc_messages.append("macula_center_yx not provided; macula involvement and zone "
                           "stats will be null.")
    if eye_side is None:
        qc_messages.append("eye_side not provided; nasal/temporal quadrant split unavailable.")

    metrics = compute_class_metrics(
        bin_mask, disc_diameter_px=DD,
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
    coverage = metrics.area_px / fundus_area_px if fundus_area_px > 0 else 0.0
    severity = classify_severity(coverage, cfg.severity_thresholds)
    involves_macula = _involves_macula(
        bin_mask, macula_center_yx, DD, cfg.macula_involvement_radius_dd
    )

    return {
        "units": {
            "pixel_spacing_um": float(pixel_spacing_um) if pixel_spacing_um is not None else None,
            "disc_diameter_px": float(DD),
            "analysis_frame": "original_image",
            "length_fields_suffixes": ["_px", "_dd", "_um (if pixel_spacing_um given)"],
        },
        "qc": {
            "passed": True,
            "messages": qc_messages,
            "fundus_area_source": fundus_source,
            "fundus_area_px": int(fundus_area_px),
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
        "tessellation": {
            "count": metrics.count,
            "area": _area_triple(metrics.area_px, DD, pixel_spacing_um),
            "coverage_ratio": float(coverage),
            "severity": severity,
            "severity_thresholds_ratio": list(cfg.severity_thresholds),
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
            "involves_macula": involves_macula,
            "components": [
                {
                    "centroid_yx": [float(c.centroid_yx[0]), float(c.centroid_yx[1])],
                    "area_px": int(c.area_px),
                    "bbox_yxyx": list(c.bbox_yxyx),
                    "circularity": float(c.circularity),
                    "aspect_ratio": float(c.aspect_ratio),
                }
                for c in metrics.components
            ],
        },
        "severity_inputs": {
            "tessellation_coverage_ratio": float(coverage),
            "tessellation_severity_bucket": severity,
            "tessellation_present": metrics.count > 0,
            "tessellation_involves_macula": involves_macula,
            "notes": (
                "Tessellation coverage + macular involvement feed META-PM "
                "myopic maculopathy stage 1 (tessellated fundus). Severity "
                "bucket is a heuristic — clinical grading should integrate "
                "this with atrophy/disc/vessel findings."
            ),
        },
    }
