"""
Top-level vessel analysis API.

Inputs
------
- artery_mask, vein_mask : HxW binary (or any non-zero foreground)
- disc_mask              : HxW binary, from the OD/OC task (disc ∪ cup)
- fundus_mask            : HxW binary, optional; falls back to full image
- pixel_spacing_um       : float, optional; if provided, CRAE/CRVE/diameters
                           are reported in micrometres

All input masks must already be in a **shared coordinate frame** — i.e. the
same (H, W) spatial grid after the same black-edge crop + resize. The caller
is responsible for that alignment (see `scripts/infer.py` for how to produce
them from the model output).

Outputs
-------
A nested dict with meta (`units`, `qc`, `disc`, `measurement_zone`) and
per-metric subtrees. See postprocess/vessels/README.md for the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from skimage.morphology import skeletonize

from .crae_crve import K_ARTERIOLE, K_VENULE, DEFAULT_TOP_N, compute_equivalent
from .density import compute_density
from .diameter import (
    VesselSegment,
    extract_all_skeleton_segments,
    extract_vessel_segments,
)
from .fractal import box_counting_dimension
from .tortuosity import aggregate_tortuosity
from .zones import (
    DiscGeometry,
    NoDiscError,
    build_zone_b,
    compute_disc_geometry,
    zone_radii_px,
)


# --------------------------------------------------------------------------- config

@dataclass
class VesselAnalysisConfig:
    """Analysis runs in the **original image frame** where camera pixels are
    isotropic (the model internally trained on an anisotropically resized
    640×640 frame, but masks are NN-upsampled back to original shape before
    we see them). All length thresholds are expressed in **disc diameters (DD)**
    so they scale automatically with image size and imaging magnification.
    """
    zone_inner_dd: float = 0.5
    zone_outer_dd: float = 1.0
    knudtson_k_arteriole: float = K_ARTERIOLE
    knudtson_k_venule: float = K_VENULE
    top_n_vessels: int = DEFAULT_TOP_N
    # Minimum vessel-segment length for tortuosity, in disc diameters.
    # 0.5 DD is typically ~0.5 × 150-200 μm_DD ≈ 75-100 μm, a clinically
    # reasonable lower bound that drops pixel-quantization noise on small
    # images while still capturing mid-arborisation vessels.
    min_segment_length_dd: float = 0.5
    fractal_box_sizes: Optional[Sequence[int]] = None
    skeleton_qc_min_ratio: float = 0.005     # skeleton fg / fundus area; below this → warn
    r2_qc_min: float = 0.95                  # fractal fit R² threshold for QC


# --------------------------------------------------------------------------- helpers

def _as_binary(mask: np.ndarray, name: str) -> np.ndarray:
    if mask is None:
        raise ValueError(f"{name} is None")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def _px_to_um(value_px: float, spacing_um: Optional[float]) -> Optional[float]:
    """px → μm if a spacing is given, else None."""
    if spacing_um is None:
        return None
    return float(value_px) * float(spacing_um)


def _px_list_to_um(values_px: List[float], spacing_um: Optional[float]) -> Optional[List[float]]:
    if spacing_um is None:
        return None
    return [float(v) * float(spacing_um) for v in values_px]


def _px_to_dd(value_px: float, disc_diameter_px: float) -> float:
    if disc_diameter_px <= 0:
        return 0.0
    return float(value_px) / float(disc_diameter_px)


def _px_list_to_dd(values_px: List[float], disc_diameter_px: float) -> List[float]:
    return [_px_to_dd(v, disc_diameter_px) for v in values_px]


# --------------------------------------------------------------------------- public

def analyze_vessels(artery_mask: np.ndarray,
                    vein_mask: np.ndarray,
                    disc_mask: Optional[np.ndarray],
                    fundus_mask: Optional[np.ndarray] = None,
                    pixel_spacing_um: Optional[float] = None,
                    config: Optional[VesselAnalysisConfig] = None) -> Dict[str, Any]:
    """Run the full vessel quantification pipeline; see module docstring.

    All length fields in the output are reported in three units where
    meaningful:

      - ``_px`` : pixels in the input mask frame (original image frame)
      - ``_dd`` : disc diameters (= px / disc.diameter_px), makes values
                  comparable across images regardless of resolution or zoom
      - ``_um`` : micrometres if ``pixel_spacing_um`` is given, else ``None``

    Dimensionless metrics (AVR, fractal D, DF, density ratios) are reported
    without unit suffixes.
    """
    cfg = config or VesselAnalysisConfig()

    artery = _as_binary(artery_mask, "artery_mask")
    vein = _as_binary(vein_mask, "vein_mask")
    if artery.shape != vein.shape:
        raise ValueError("artery_mask and vein_mask must share shape")

    H, W = artery.shape
    qc_messages: List[str] = []
    qc_passed = True

    # --- Disc geometry ------------------------------------------------------
    if disc_mask is None:
        raise NoDiscError(
            "disc_mask is required for zone-based CRAE/CRVE. Pass the OD/OC "
            "task mask (disc ∪ cup)."
        )
    disc_bin = _as_binary(disc_mask, "disc_mask")
    disc_geom = compute_disc_geometry(disc_bin, source="od_oc_mask")
    DD = disc_geom.diameter_px  # shorthand

    zone = build_zone_b(artery.shape, disc_geom, cfg.zone_inner_dd, cfg.zone_outer_dd)
    inner_px, outer_px = zone_radii_px(disc_geom, cfg.zone_inner_dd, cfg.zone_outer_dd)

    if zone.sum() == 0:
        qc_messages.append("Zone B is empty — disc likely at image edge.")
        qc_passed = False

    # --- Per-task segments --------------------------------------------------
    artery_zone_segments = extract_vessel_segments(artery, disc_bin, zone)
    vein_zone_segments = extract_vessel_segments(vein, disc_bin, zone)

    # For fractal + tortuosity we use off-disc skeletons (no zone restriction).
    artery_all_segments = extract_all_skeleton_segments(artery, disc_bin)
    vein_all_segments = extract_all_skeleton_segments(vein, disc_bin)

    # --- Skeleton QC --------------------------------------------------------
    # Denominator for fg-rate: prefer fundus_mask (true retinal area); fall
    # back to full image with an explicit QC note.
    if fundus_mask is not None:
        fundus_area_px = int(fundus_mask.astype(bool).sum())
        fundus_source = "fundus_mask"
    else:
        fundus_area_px = int(artery.size)
        fundus_source = "full_image"
        qc_messages.append(
            "No fundus_mask provided; coverage ratios use full image area as denominator."
        )

    skel_union = skeletonize((artery | vein).astype(bool))
    skel_fg_rate = float(skel_union.sum()) / max(fundus_area_px, 1)
    if skel_fg_rate < cfg.skeleton_qc_min_ratio:
        qc_messages.append(
            f"Low vessel skeleton coverage: {skel_fg_rate:.4f} < {cfg.skeleton_qc_min_ratio}."
        )
        qc_passed = False

    # --- CRAE / CRVE --------------------------------------------------------
    artery_diams_px = [s.diameter_px for s in artery_zone_segments]
    vein_diams_px = [s.diameter_px for s in vein_zone_segments]

    crae_res = compute_equivalent(artery_diams_px, cfg.knudtson_k_arteriole, cfg.top_n_vessels)
    crve_res = compute_equivalent(vein_diams_px, cfg.knudtson_k_venule, cfg.top_n_vessels)

    if crae_res.n_vessels_used < cfg.top_n_vessels or crve_res.n_vessels_used < cfg.top_n_vessels:
        qc_messages.append(
            f"Reduced CRAE/CRVE sample (artery n={crae_res.n_vessels_used}, "
            f"vein n={crve_res.n_vessels_used}, target={cfg.top_n_vessels})."
        )

    # --- Fractal dimension --------------------------------------------------
    fd_artery = box_counting_dimension(
        skeletonize(artery.astype(bool)), box_sizes=cfg.fractal_box_sizes
    )
    fd_vein = box_counting_dimension(
        skeletonize(vein.astype(bool)), box_sizes=cfg.fractal_box_sizes
    )
    if fd_artery.r2 < cfg.r2_qc_min:
        qc_messages.append(f"Fractal(artery) poor fit R²={fd_artery.r2:.3f}")
    if fd_vein.r2 < cfg.r2_qc_min:
        qc_messages.append(f"Fractal(vein) poor fit R²={fd_vein.r2:.3f}")

    # --- Tortuosity ---------------------------------------------------------
    # Convert DD threshold to px for the per-image cut-off.
    min_seg_len_px = int(max(1, round(cfg.min_segment_length_dd * DD)))
    tort_artery = aggregate_tortuosity(artery_all_segments, min_seg_len_px)
    tort_vein = aggregate_tortuosity(vein_all_segments, min_seg_len_px)

    # --- Density ------------------------------------------------------------
    density = compute_density(artery, vein, fundus_mask)

    # --- Assemble output ----------------------------------------------------
    result: Dict[str, Any] = {
        "units": {
            "pixel_spacing_um": (float(pixel_spacing_um) if pixel_spacing_um is not None else None),
            "disc_diameter_px": float(DD),
            "analysis_frame": "original_image",
            "length_fields_suffixes": ["_px", "_dd", "_um (if pixel_spacing_um given)"],
        },
        "qc": {
            "passed": bool(qc_passed),
            "messages": qc_messages,
            "skeleton_fg_rate": skel_fg_rate,
            "fundus_area_source": fundus_source,
        },
        "disc": {
            "center_yx": [float(disc_geom.center_yx[0]), float(disc_geom.center_yx[1])],
            "radius_px": float(disc_geom.radius_px),
            "radius_dd": 0.5,  # definitional
            "radius_um": _px_to_um(disc_geom.radius_px, pixel_spacing_um),
            "area_px": int(disc_geom.area_px),
            "source": disc_geom.source,
        },
        "measurement_zone": {
            "method": "knudtson_zone_b",
            "inner_dd": cfg.zone_inner_dd,
            "outer_dd": cfg.zone_outer_dd,
            "inner_px": float(inner_px),
            "outer_px": float(outer_px),
            "inner_um": _px_to_um(inner_px, pixel_spacing_um),
            "outer_um": _px_to_um(outer_px, pixel_spacing_um),
            "area_px": int(zone.sum()),
        },
        "crae": {
            "value_px": float(crae_res.value),
            "value_dd": _px_to_dd(crae_res.value, DD),
            "value_um": _px_to_um(crae_res.value, pixel_spacing_um),
            "method": crae_res.method,
            "k": crae_res.k,
            "n_vessels_used": crae_res.n_vessels_used,
            "top_diameters_px": list(crae_res.top_diameters),
            "top_diameters_dd": _px_list_to_dd(crae_res.top_diameters, DD),
            "top_diameters_um": _px_list_to_um(crae_res.top_diameters, pixel_spacing_um),
        },
        "crve": {
            "value_px": float(crve_res.value),
            "value_dd": _px_to_dd(crve_res.value, DD),
            "value_um": _px_to_um(crve_res.value, pixel_spacing_um),
            "method": crve_res.method,
            "k": crve_res.k,
            "n_vessels_used": crve_res.n_vessels_used,
            "top_diameters_px": list(crve_res.top_diameters),
            "top_diameters_dd": _px_list_to_dd(crve_res.top_diameters, DD),
            "top_diameters_um": _px_list_to_um(crve_res.top_diameters, pixel_spacing_um),
        },
        "avr": {
            "value": (crae_res.value / crve_res.value) if crve_res.value > 0 else None,
            "definition": "CRAE / CRVE (Knudtson equivalents)",
        },
        "fractal_dimension": {
            "artery": {
                "value": fd_artery.value,
                "r2": fd_artery.r2,
                "method": fd_artery.method,
                "box_sizes_px": fd_artery.box_sizes_px,
                "box_counts": fd_artery.box_counts,
            },
            "vein": {
                "value": fd_vein.value,
                "r2": fd_vein.r2,
                "method": fd_vein.method,
                "box_sizes_px": fd_vein.box_sizes_px,
                "box_counts": fd_vein.box_counts,
            },
        },
        "tortuosity": {
            "artery": {
                "mean_df": tort_artery.mean_df,
                "median_df": tort_artery.median_df,
                "length_weighted_df": tort_artery.length_weighted_df,
                "n_segments": tort_artery.n_segments,
                "method": tort_artery.method,
                "min_segment_length_dd": cfg.min_segment_length_dd,
                "min_segment_length_px": min_seg_len_px,
            },
            "vein": {
                "mean_df": tort_vein.mean_df,
                "median_df": tort_vein.median_df,
                "length_weighted_df": tort_vein.length_weighted_df,
                "n_segments": tort_vein.n_segments,
                "method": tort_vein.method,
                "min_segment_length_dd": cfg.min_segment_length_dd,
                "min_segment_length_px": min_seg_len_px,
            },
        },
        "density": {
            "artery_ratio": density.artery_ratio,
            "vein_ratio": density.vein_ratio,
            "total_ratio": density.total_ratio,
            "fundus_area_px": density.fundus_area_px,
            "denominator_source": density.denominator_source,
        },
    }
    return result
