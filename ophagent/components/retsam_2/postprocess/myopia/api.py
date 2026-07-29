"""
Myopia-related atrophy quantification.

Consumes the myopia task label map produced by the segmentation model
(``0=bg, 1=arc_lesion, 2=diffuse_chorioretinal_atrophy, 3=patchy_chorioretinal_atrophy``)
and reports class-level metrics plus inputs relevant to myopic maculopathy
grading (META-PM / ATN style).

Scope
-----
* **arc_lesion (弧形斑)**: peripapillary crescent touching the disc.
  Reported as a disc-relative geometry:
    - angular coverage around the disc (in degrees)
    - which cardinal sectors (I/S/N/T) are involved
    - max radial extent from the disc margin (in DD)
* **diffuse_chorioretinal_atrophy (弥漫性脉络膜视网膜萎缩)**: continuous
  background atrophy; we report coverage, macula-involvement, and distance
  distribution.
* **patchy_chorioretinal_atrophy (斑片状萎缩)**: discrete atrophy patches;
  reported like a lesion class — count, size distribution, macula zones,
  per-quadrant counts.

We do NOT output a myopic maculopathy grade. Grading is clinical and the
inputs are exposed cleanly so a future disease layer can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..disc_cup.geometry import (
    NoDiscError,
    compute_disc_geometry,
    from_od_oc_labelmap,
)
from ..lesions.per_class import ClassMetrics, compute_class_metrics
from ..lesions.spatial import compute_spatial_stats


CLASS_MAP: Dict[int, str] = {
    1: "arc_lesion",
    2: "diffuse_chorioretinal_atrophy",
    3: "patchy_chorioretinal_atrophy",
}


@dataclass
class MyopiaAnalysisConfig:
    min_component_size_px: int = 15
    size_small_max_dd2: float = 0.01         # atrophy patches run larger than DR spots
    size_medium_max_dd2: float = 0.1
    arc_angular_resolution_deg: int = 360
    arc_max_ray_extent_dd: float = 2.0       # outer radius when tracing arc around disc, in DD
    macula_zone_boundaries_dd: Sequence[float] = (1.0, 2.0)
    macula_involvement_radius_dd: float = 0.5  # atrophy within this of macula = "involves macula"


# --------------------------------------------------------------------------- helpers

def _area_triple(area_px: int, DD: float, spacing_um: Optional[float]) -> Dict[str, Any]:
    area_dd2 = float(area_px) / (DD * DD) if DD > 0 else 0.0
    area_um2 = None if spacing_um is None else float(area_px) * float(spacing_um) ** 2
    return {"px": int(area_px), "dd2": area_dd2, "um2": area_um2}


def _px_to_um(value_px: float, spacing_um: Optional[float]) -> Optional[float]:
    return None if spacing_um is None else float(value_px) * float(spacing_um)


# --------------------------------------------------------------------------- arc-specific

def analyze_arc_lesion(arc_bin: np.ndarray,
                       disc_center_yx: Tuple[float, float],
                       disc_diameter_px: float,
                       eye_side: Optional[str],
                       angular_resolution_deg: int = 360,
                       max_ray_extent_dd: float = 2.0) -> Dict[str, Any]:
    """Geometry of the peripapillary crescent around the disc.

    For every arc pixel we compute (angle, radius_from_disc_center). Angular
    coverage = fraction of discrete angle bins that contain at least one arc
    pixel between the disc edge and ``max_ray_extent_dd``. Cardinal-sector
    involvement tests whether any pixel falls inside each 90° wedge around
    the cardinal direction.

    We also return ``angular_profile``: a 360-element list of
    ``[angle_deg, radial_extent_dd]``, where ``radial_extent_dd`` is the
    farthest arc pixel at that 1° bin measured beyond the disc edge in DD
    units (0 when no arc pixel at that angle). This is what
    ``render_arc_profile`` plots.
    """
    n_bins = int(max(4, angular_resolution_deg))

    arc = arc_bin > 0
    if not arc.any():
        return {
            "angular_coverage_deg": 0,
            "angular_coverage_ratio": 0.0,
            "sector_involvement": {"inferior": False, "superior": False,
                                   "nasal": False, "temporal": False,
                                   "east_raw": False, "west_raw": False},
            "max_radial_extent_dd": 0.0,
            "mean_radial_extent_dd": 0.0,
            "angular_profile": [[float(i * 360.0 / n_bins), 0.0] for i in range(n_bins)],
            "method": "pixel_angular_binning",
        }

    ys, xs = np.nonzero(arc)
    cy, cx = disc_center_yx
    dy = ys - cy
    dx = xs - cx
    dist_px = np.sqrt(dy * dy + dx * dx)
    # Convert to "distance beyond disc edge" by subtracting disc radius
    disc_radius_px = disc_diameter_px / 2.0
    radial_beyond_edge_dd = np.maximum(0.0, dist_px - disc_radius_px) / max(disc_diameter_px, 1.0)

    # Image-plane angles in [0, 360). 0°=east, 90°=south (inferior), 180°=west, 270°=north (superior).
    angles = np.degrees(np.arctan2(dy, dx)) % 360.0

    # Only count arc pixels that actually lie in the pericapillary search zone
    # (≤ max_ray_extent_dd from disc edge). Pixels farther out are more likely
    # noise or mislabel; excluding them keeps angular coverage clean.
    in_zone = radial_beyond_edge_dd <= max_ray_extent_dd
    if not in_zone.any():
        return {
            "angular_coverage_deg": 0,
            "angular_coverage_ratio": 0.0,
            "sector_involvement": {"inferior": False, "superior": False,
                                   "nasal": False, "temporal": False,
                                   "east_raw": False, "west_raw": False},
            "max_radial_extent_dd": float(radial_beyond_edge_dd.max()),
            "mean_radial_extent_dd": float(radial_beyond_edge_dd.mean()),
            "angular_profile": [[float(i * 360.0 / n_bins), 0.0] for i in range(n_bins)],
            "method": "pixel_angular_binning",
        }
    angles_in_zone = angles[in_zone]
    radii_in_zone = radial_beyond_edge_dd[in_zone]

    # Histogram angles into N bins; coverage = number of non-empty bins
    hist, _ = np.histogram(angles_in_zone, bins=n_bins, range=(0, 360))
    non_empty = int((hist > 0).sum())
    coverage_deg = int(round(non_empty * 360.0 / n_bins))
    coverage_ratio = non_empty / n_bins

    # Per-bin maximum radial extent (how far out the arc reaches at each degree)
    bin_idx = np.clip((angles_in_zone * n_bins / 360.0).astype(np.int64), 0, n_bins - 1)
    per_bin_max = np.zeros(n_bins, dtype=np.float64)
    np.maximum.at(per_bin_max, bin_idx, radii_in_zone)
    angular_profile = [[float(i * 360.0 / n_bins), float(per_bin_max[i])]
                       for i in range(n_bins)]

    # Sector involvement (eye-side-aware for nasal/temporal)
    def _wedge_has(a0: float, a1: float) -> bool:
        if a1 < a0:
            return bool(((angles_in_zone >= a0) | (angles_in_zone < a1)).any())
        return bool(((angles_in_zone >= a0) & (angles_in_zone < a1)).any())

    east_has = _wedge_has(315, 45)
    west_has = _wedge_has(135, 225)
    sector_involvement = {
        "inferior": _wedge_has(45, 135),
        "superior": _wedge_has(225, 315),
        "east_raw": east_has,
        "west_raw": west_has,
    }
    if eye_side == "OD":
        sector_involvement["temporal"] = east_has
        sector_involvement["nasal"] = west_has
    elif eye_side == "OS":
        sector_involvement["nasal"] = east_has
        sector_involvement["temporal"] = west_has
    else:
        sector_involvement["nasal"] = None
        sector_involvement["temporal"] = None

    return {
        "angular_coverage_deg": coverage_deg,
        "angular_coverage_ratio": float(coverage_ratio),
        "sector_involvement": sector_involvement,
        "max_radial_extent_dd": float(radial_beyond_edge_dd.max()),
        "mean_radial_extent_dd": float(radial_beyond_edge_dd[in_zone].mean()),
        "angular_profile": angular_profile,
        "method": "pixel_angular_binning",
    }


# --------------------------------------------------------------------------- macula involvement

def _involves_macula(mask: np.ndarray,
                     macula_center_yx: Optional[Tuple[float, float]],
                     disc_diameter_px: float,
                     involvement_radius_dd: float) -> Optional[bool]:
    """Any foreground pixel within ``involvement_radius_dd`` × DD of macula."""
    if macula_center_yx is None or disc_diameter_px <= 0:
        return None
    if (mask > 0).sum() == 0:
        return False
    ys, xs = np.nonzero(mask > 0)
    my, mx = macula_center_yx
    dy = ys - my
    dx = xs - mx
    dist_px = np.sqrt(dy * dy + dx * dx)
    radius_px = involvement_radius_dd * disc_diameter_px
    return bool((dist_px <= radius_px).any())


# --------------------------------------------------------------------------- main

def analyze_myopia(myopia_mask: np.ndarray,
                   od_oc_mask: np.ndarray,
                   fundus_mask: Optional[np.ndarray] = None,
                   macula_center_yx: Optional[Tuple[float, float]] = None,
                   eye_side: Optional[str] = None,
                   pixel_spacing_um: Optional[float] = None,
                   config: Optional[MyopiaAnalysisConfig] = None) -> Dict[str, Any]:
    """Run myopia-task quantification; see module docstring."""
    if eye_side not in ("OS", "OD", None):
        raise ValueError(f"eye_side must be 'OS'/'OD'/None, got {eye_side!r}")
    cfg = config or MyopiaAnalysisConfig()

    if myopia_mask is None or myopia_mask.ndim != 2:
        raise ValueError("myopia_mask must be a 2D array")

    disc_bin, _ = from_od_oc_labelmap(od_oc_mask)
    disc_geom = compute_disc_geometry(disc_bin)   # raises NoDiscError if empty
    DD = disc_geom.diameter_vertical_px

    if fundus_mask is not None:
        fundus_area_px = int((fundus_mask > 0).sum())
        fundus_source = "fundus_mask"
    else:
        fundus_area_px = int(myopia_mask.size)
        fundus_source = "full_image"

    qc_messages: List[str] = []
    qc_passed = True
    if fundus_mask is None:
        qc_messages.append("No fundus_mask provided; coverage uses full image area.")
    if macula_center_yx is None:
        qc_messages.append("macula_center_yx not provided; macula involvement and zone "
                           "stats will be null.")
    if eye_side is None:
        qc_messages.append("eye_side not provided; arc sector labels (nasal/temporal) "
                           "are reported as null, and quadrant splits for patchy atrophy "
                           "are unavailable.")

    # --- per-class metrics --------------------------------------------------
    classes_out: Dict[str, Any] = {}
    global_area_px = 0

    for cls_id, cls_name in CLASS_MAP.items():
        cls_mask = (myopia_mask == cls_id).astype(np.uint8)
        metrics = compute_class_metrics(
            cls_mask, disc_diameter_px=DD,
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

        involves_macula = _involves_macula(
            cls_mask, macula_center_yx, DD, cfg.macula_involvement_radius_dd
        )
        involves_disc = bool(((cls_mask > 0) & (disc_bin > 0)).any())

        coverage = metrics.area_px / fundus_area_px if fundus_area_px > 0 else 0.0

        entry: Dict[str, Any] = {
            "count": metrics.count,
            "area": _area_triple(metrics.area_px, DD, pixel_spacing_um),
            "area_dd2": metrics.area_px / (DD * DD) if DD > 0 else 0.0,
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
            "involves_macula": involves_macula,
            "involves_disc": involves_disc,
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
        }

        # arc_lesion carries an extra disc-relative geometry block
        if cls_name == "arc_lesion":
            entry["disc_relative"] = analyze_arc_lesion(
                cls_mask,
                disc_center_yx=disc_geom.center_yx,
                disc_diameter_px=DD,
                eye_side=eye_side,
                angular_resolution_deg=cfg.arc_angular_resolution_deg,
                max_ray_extent_dd=cfg.arc_max_ray_extent_dd,
            )

        classes_out[cls_name] = entry
        global_area_px += metrics.area_px

    # --- severity inputs (META-PM / ATN style, inputs only) ---------------------
    arc_entry = classes_out["arc_lesion"]
    diffuse_entry = classes_out["diffuse_chorioretinal_atrophy"]
    patchy_entry = classes_out["patchy_chorioretinal_atrophy"]

    patchy_within_1dd = None
    if macula_center_yx is not None:
        zones = patchy_entry["spatial"].get("macula_zone_counts")
        if isinstance(zones, dict):
            # Pick the innermost zone (everything up to 1 DD)
            # keys are like "0_1_dd" or "0p0_1p0_dd"
            for k, v in zones.items():
                if k.startswith("0_1_dd") or k.startswith("0p0_1p0_dd"):
                    patchy_within_1dd = int(v)
                    break

    severity_inputs = {
        "arc_lesion_present": arc_entry["count"] > 0,
        "arc_angular_coverage_deg": (
            arc_entry["disc_relative"]["angular_coverage_deg"] if arc_entry["count"] > 0 else 0
        ),
        "arc_sector_involvement": (
            arc_entry["disc_relative"]["sector_involvement"] if arc_entry["count"] > 0 else None
        ),
        "diffuse_atrophy_present": diffuse_entry["count"] > 0,
        "diffuse_coverage_ratio": diffuse_entry["coverage_ratio"],
        "diffuse_involves_macula": diffuse_entry["involves_macula"],
        "patchy_atrophy_present": patchy_entry["count"] > 0,
        "patchy_count": patchy_entry["count"],
        "patchy_involves_macula": patchy_entry["involves_macula"],
        "patchy_within_1dd_of_macula_count": patchy_within_1dd,
        "notes": (
            "Inputs for myopic maculopathy grading (META-PM / ATN). This module "
            "does NOT produce a grade. Tessellation contribution is reported by "
            "postprocess/tessellation separately."
        ),
    }

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
        "classes": classes_out,
        "global_summary": {
            "total_area_px": int(global_area_px),
            "total_coverage_ratio": (global_area_px / fundus_area_px) if fundus_area_px > 0 else 0.0,
            "n_classes_detected": int(sum(1 for c in classes_out.values() if c["count"] > 0)),
        },
        "severity_inputs": severity_inputs,
    }
    return result
