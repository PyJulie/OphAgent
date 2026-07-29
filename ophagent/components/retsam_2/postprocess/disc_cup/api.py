"""
Top-level disc/cup analysis API.

Consumes the OD/OC label map produced by the segmentation model (convention:
``0=background, 1=rim-only, 2=cup``) and returns a nested dict with every
metric we computed plus the supporting QC information. See
``postprocess/disc_cup/README.md`` for the full schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .ddls import (
    DEFAULT_DDLS_THRESHOLDS,
    DEFAULT_DISC_SIZE_CUTOFFS_MM2,
    classify_disc_size,
    compute_ddls,
)
from .geometry import (
    NoCupError,
    NoDiscError,
    compute_cup_geometry,
    compute_disc_geometry,
    from_od_oc_labelmap,
)
from .rim import (
    cardinal_rim_widths,
    isnt_compliance,
    rim_areas,
    rim_profile,
)


EYE_SIDE_VALUES = {"OS", "OD", None}


@dataclass
class DiscCupAnalysisConfig:
    """All measurements are done in the **original image frame** (isotropic
    pixels); length thresholds are **disc-diameter-normalised** so the same
    config applies across image sizes.
    """

    # Containment QC: cup pixels with at least one bg 4-neighbour are "touching
    # the disc outer edge" — a sign of segmentation weirdness since cup should
    # be enclosed by rim. We tolerate a handful of such pixels before flagging.
    cup_boundary_tolerance_px: int = 10
    sector_scan_n: int = 360
    # "rim absent" ≡ rim_width < x · disc_vertical_diameter.
    ddls_zero_rim_fraction_of_dd: float = 0.02
    ddls_disc_size_cutoffs_mm2: tuple = DEFAULT_DISC_SIZE_CUTOFFS_MM2
    ddls_thresholds: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.ddls_thresholds is None:
            # dataclass defaults that are dicts need this dance
            object.__setattr__(self, "ddls_thresholds", dict(DEFAULT_DDLS_THRESHOLDS))


# --------------------------------------------------------------------------- helpers

def _px_to_dd(value_px: float, disc_diameter_px: float) -> float:
    if disc_diameter_px <= 0:
        return 0.0
    return float(value_px) / float(disc_diameter_px)


def _px_to_um(value_px: float, pixel_spacing_um: Optional[float]) -> Optional[float]:
    if pixel_spacing_um is None:
        return None
    return float(value_px) * float(pixel_spacing_um)


def _area_px_to_mm2(area_px: int, pixel_spacing_um: Optional[float]) -> Optional[float]:
    if pixel_spacing_um is None:
        return None
    spacing_mm = pixel_spacing_um / 1000.0
    return float(area_px) * (spacing_mm ** 2)


def _triple(value_px: float, disc_diameter_px: float,
            pixel_spacing_um: Optional[float]) -> Dict[str, Optional[float]]:
    """Return ``{_px, _dd, _um}`` keys for a single length measurement."""
    return {
        "px": float(value_px),
        "dd": _px_to_dd(value_px, disc_diameter_px),
        "um": _px_to_um(value_px, pixel_spacing_um),
    }


# --------------------------------------------------------------------------- public

def analyze_disc_cup(od_oc_mask: np.ndarray,
                     eye_side: Optional[str] = None,
                     pixel_spacing_um: Optional[float] = None,
                     config: Optional[DiscCupAnalysisConfig] = None) -> Dict[str, Any]:
    """Run the full disc/cup quantification pipeline.

    Parameters
    ----------
    od_oc_mask : (H, W) uint8
        Segmentation label map. ``0=background``, ``1=rim-only``, ``2=cup``.
        Disc = ``mask > 0`` ; cup = ``mask == 2``.
    eye_side : {"OS", "OD", None}
        Left / right / unknown. We do NOT auto-derive this. When ``None``,
        nasal/temporal rim widths and ISNT compliance are reported as null.
    pixel_spacing_um : float or None
        μm per pixel in the original image. Required for millimetric outputs
        (disc_area_mm2) and DDLS grading.
    """
    if eye_side not in EYE_SIDE_VALUES:
        raise ValueError(f"eye_side must be one of {EYE_SIDE_VALUES}, got {eye_side!r}")
    cfg = config or DiscCupAnalysisConfig()

    disc_bin, cup_bin = from_od_oc_labelmap(od_oc_mask)
    disc_geom = compute_disc_geometry(disc_bin)  # raises NoDiscError if empty
    try:
        cup_geom = compute_cup_geometry(cup_bin)
    except NoCupError:
        cup_geom = None

    qc_messages: List[str] = []
    qc_passed = True

    # Containment QC: in the OD/OC labelmap convention, cup is always in the
    # "disc region" (disc = mask > 0 ⊇ cup = mask == 2) by construction, so
    # checking "cup outside disc" is vacuous. The meaningful check is:
    # **any cup pixel with a bg 4-neighbour** — i.e. cup pixels touching the
    # disc outer edge (or detached from the disc entirely). In a valid
    # segmentation the cup is fully surrounded by rim, so this count is 0.
    if cup_geom is not None:
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        bg_neighbour_count = cv2.filter2D((disc_bin == 0).astype(np.uint8), -1, kernel)
        cup_on_boundary = int(((cup_bin > 0) & (bg_neighbour_count > 0)).sum())
    else:
        cup_on_boundary = 0
    cup_enclosed_by_rim = cup_on_boundary <= cfg.cup_boundary_tolerance_px
    if cup_geom is not None and not cup_enclosed_by_rim:
        qc_messages.append(
            f"{cup_on_boundary} cup pixels touch the disc outer boundary "
            f"(tolerance {cfg.cup_boundary_tolerance_px}). Cup should be "
            "enclosed by rim; this points to a segmentation inconsistency."
        )
        qc_passed = False
    if cup_geom is None:
        qc_messages.append(
            "Cup mask is empty; cup-derived metrics (CDR, ISNT widths, DDLS) "
            "are reported as zero/null."
        )

    if eye_side is None:
        qc_messages.append(
            "eye_side not provided; nasal/temporal rim widths and ISNT "
            "compliance are reported as null."
        )

    DD = disc_geom.diameter_vertical_px
    spacing = pixel_spacing_um

    disc_area_mm2 = _area_px_to_mm2(disc_geom.area_px, spacing)
    cup_area_mm2 = _area_px_to_mm2(cup_geom.area_px, spacing) if cup_geom is not None else None

    # --- CDR -----------------------------------------------------------------
    if cup_geom is not None and DD > 0:
        vcdr = cup_geom.diameter_vertical_px / disc_geom.diameter_vertical_px
        hcdr = (cup_geom.diameter_horizontal_px / disc_geom.diameter_horizontal_px
                if disc_geom.diameter_horizontal_px > 0 else 0.0)
        acdr = (cup_geom.area_px / disc_geom.area_px) if disc_geom.area_px > 0 else 0.0
    else:
        vcdr = hcdr = acdr = 0.0

    # --- Rim -----------------------------------------------------------------
    if cup_geom is not None:
        cardinal = cardinal_rim_widths(disc_bin, cup_bin, disc_geom.center_yx, eye_side)
        isnt_follows, isnt_violations = isnt_compliance(cardinal)
        areas = rim_areas(disc_bin, cup_bin, disc_geom.center_yx, eye_side)
        profile = rim_profile(
            disc_bin, cup_bin, disc_geom.center_yx,
            disc_diameter_px=DD,
            eye_side=eye_side,
            n_sectors=cfg.sector_scan_n,
            zero_rim_fraction_of_dd=cfg.ddls_zero_rim_fraction_of_dd,
        )
        rim_area_px = int((disc_bin > 0).sum() - (cup_bin > 0).sum())
        rim_area_mm2 = _area_px_to_mm2(rim_area_px, spacing)
        rim_disc_ratio = rim_area_px / disc_geom.area_px if disc_geom.area_px > 0 else 0.0
        min_rim_disc_ratio = profile.min_width_px / DD if DD > 0 else 0.0
    else:
        cardinal = None
        isnt_follows, isnt_violations = None, []
        areas = None
        profile = None
        rim_area_px = int((disc_bin > 0).sum())
        rim_area_mm2 = _area_px_to_mm2(rim_area_px, spacing)
        rim_disc_ratio = 1.0 if disc_geom.area_px > 0 else 0.0
        min_rim_disc_ratio = 1.0   # no cup → rim is everywhere

    # --- DDLS -----------------------------------------------------------------
    if profile is not None:
        ddls = compute_ddls(
            min_rim_disc_ratio=min_rim_disc_ratio,
            zero_run_deg=profile.zero_run_deg,
            disc_area_mm2=disc_area_mm2,
            thresholds=cfg.ddls_thresholds,
            disc_size_cutoffs_mm2=cfg.ddls_disc_size_cutoffs_mm2,
        )
    else:
        ddls = None

    # --- Schema assembly -----------------------------------------------------
    def _width_triple(v: Optional[float]) -> Optional[Dict[str, Optional[float]]]:
        if v is None:
            return None
        return _triple(v, DD, spacing)

    result: Dict[str, Any] = {
        "units": {
            "pixel_spacing_um": float(spacing) if spacing is not None else None,
            "disc_diameter_px": float(DD),
            "analysis_frame": "original_image",
            "length_fields_suffixes": ["_px", "_dd", "_um (if pixel_spacing_um given)"],
        },
        "qc": {
            "passed": bool(qc_passed),
            "messages": qc_messages,
            "cup_enclosed_by_rim": bool(cup_enclosed_by_rim),
            "cup_on_boundary_px": int(cup_on_boundary),
            "eye_side_source": "user_provided" if eye_side in ("OS", "OD") else "unknown",
            "cup_present": cup_geom is not None,
        },
        "disc": {
            "center_yx": [float(disc_geom.center_yx[0]), float(disc_geom.center_yx[1])],
            "area_px": int(disc_geom.area_px),
            "area_mm2": disc_area_mm2,
            "diameter_vertical": _triple(disc_geom.diameter_vertical_px, DD, spacing),
            "diameter_horizontal": _triple(disc_geom.diameter_horizontal_px, DD, spacing),
            "equivalent_diameter": _triple(disc_geom.equivalent_diameter_px, DD, spacing),
            "ovality": float(disc_geom.ovality),
            "tilt_angle_deg": float(disc_geom.tilt_angle_deg),
        },
        "cup": None if cup_geom is None else {
            "center_yx": [float(cup_geom.center_yx[0]), float(cup_geom.center_yx[1])],
            "area_px": int(cup_geom.area_px),
            "area_mm2": cup_area_mm2,
            "diameter_vertical": _triple(cup_geom.diameter_vertical_px, DD, spacing),
            "diameter_horizontal": _triple(cup_geom.diameter_horizontal_px, DD, spacing),
            "aspect_ratio": float(cup_geom.aspect_ratio),
        },
        "cdr": {
            "vertical": float(vcdr),
            "horizontal": float(hcdr),
            "area": float(acdr),
            "definition": {
                "vertical": "cup_vertical_diameter / disc_vertical_diameter (bbox y-span)",
                "horizontal": "cup_horizontal_diameter / disc_horizontal_diameter (bbox x-span)",
                "area": "cup_area / disc_area",
            },
        },
        "rim": {
            "area_px": int(rim_area_px),
            "area_mm2": rim_area_mm2,
            "rim_disc_area_ratio": float(rim_disc_ratio),
            "cardinal_widths": None if cardinal is None else {
                "inferior": _width_triple(cardinal.inferior_px),
                "superior": _width_triple(cardinal.superior_px),
                "east_raw": _width_triple(cardinal.east_px),
                "west_raw": _width_triple(cardinal.west_px),
                "nasal": _width_triple(cardinal.nasal_px),
                "temporal": _width_triple(cardinal.temporal_px),
            },
            "isnt": {
                "eye_side": eye_side,
                "follows_rule": isnt_follows,
                "violations": list(isnt_violations),
                "ordered_values_px": (
                    None if cardinal is None or cardinal.nasal_px is None
                    else [cardinal.inferior_px, cardinal.superior_px,
                          cardinal.nasal_px, cardinal.temporal_px]
                ),
            },
            "sector_scan": None if profile is None else {
                "method": "angular_scan",
                "n_sectors": int(cfg.sector_scan_n),
                "min_rim_width": _triple(profile.min_width_px, DD, spacing),
                "min_rim_angle_deg": float(profile.min_angle_deg),
                "min_rim_sector_label": profile.min_sector_label,
                "zero_rim_threshold": _triple(profile.zero_threshold_px, DD, spacing),
                "zero_run_deg": int(profile.zero_run_deg),
                "profile_deg_to_width_px": [
                    [float(a), float(w)]
                    for a, w in zip(profile.angles_deg, profile.widths_px)
                ],
            },
            "quadrant_areas_px": None if areas is None else dict(areas.quadrant_px),
        },
        "ddls": None if ddls is None else {
            "grade": ddls.grade,
            "method": ddls.method,
            "disc_size": ddls.disc_size,
            "disc_area_mm2": ddls.disc_area_mm2,
            "min_rim_disc_ratio": ddls.min_rim_disc_ratio,
            "zero_run_deg": ddls.zero_run_deg,
            "reasoning": ddls.reasoning,
        },
    }
    return result
