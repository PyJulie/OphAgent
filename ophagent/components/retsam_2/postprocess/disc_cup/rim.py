"""
Rim measurements: area, cardinal widths, angular profile, quadrant areas.

All measurements are done directly on the raw disc/cup masks — **no convex
hulls, no morphological smoothing**. That matches what the model actually
segmented; downstream callers can interpret segmentation irregularities
correctly rather than having them quietly wallpapered.

Geometry conventions
--------------------
Image coordinates use ``(y, x)`` with y growing downward. Angles are
measured in degrees in the image plane with:

    θ = 0°   → pointing right  (+x)
    θ = 90°  → pointing down   (+y, **inferior** in the image)
    θ = 180° → pointing left   (-x)
    θ = 270° → pointing up     (-y, **superior**)

Nasal/temporal depend on eye side (see ``label_sector``):

    OD (right eye): nasal is to the **image left**  (≈180°), temporal to the right (≈0°)
    OS (left eye):  nasal is to the **image right** (≈0°),   temporal to the left (≈180°)

Each 90°-wedge centered on a cardinal direction defines a sector:

    inferior = (45°, 135°]       superior = (225°, 315°]
    eye-side-dependent nasal/temporal for the other two wedges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SectorLabel = str  # "inferior" | "superior" | "nasal" | "temporal" | "unknown"


# --------------------------------------------------------------------------- ray walking

def _width_along_ray(disc_bin: np.ndarray, cup_bin: np.ndarray,
                     center_yx: Tuple[float, float], angle_deg: float) -> float:
    """Rim width along a ray leaving ``center_yx`` at ``angle_deg``.

    We define rim width as **the number of pixels along the ray that lie
    inside the disc but outside the cup**, measured continuously from the
    centre outward. This definition is robust to whether the ray starts
    inside the cup or not:

        * Ray going through the centre of a concentric cup → steps 1..cup_r
          are cup (skipped), steps cup_r+1..disc_r are rim, so width = disc_r - cup_r.
        * Ray in a direction where the cup is entirely on the "wrong side" →
          cup is never entered, every disc step counts, width = full disc radius.
        * Cup detached from disc (segmentation weirdness) → cup is never
          visited while inside the disc, rim width still reports the disc
          half-diameter along that axis.

    We terminate the walk the first time the ray leaves the disc. If the
    centre itself is outside the disc (shouldn't happen with our centroid
    input), we return 0.
    """
    h, w = disc_bin.shape[:2]
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)  # y grows down; sin(90°) = 1 → pointing down = inferior
    norm = float(np.hypot(dy, dx))
    if norm == 0:
        return 0.0
    dy_n, dx_n = dy / norm, dx / norm

    cy, cx = center_yx
    rim_px = 0
    t = 0.0
    while True:
        y = cy + dy_n * t
        x = cx + dx_n * t
        yi, xi = int(round(y)), int(round(x))
        if yi < 0 or yi >= h or xi < 0 or xi >= w:
            break
        in_disc = disc_bin[yi, xi] > 0
        if t > 0 and not in_disc:
            break  # left the disc → ray finished
        in_cup = cup_bin[yi, xi] > 0
        if t > 0 and in_disc and not in_cup:
            rim_px += 1
        t += 1.0
        # Safety guard against runaway loops
        if t > max(h, w) * 2:
            break
    return float(rim_px)


# --------------------------------------------------------------------------- sector labeling

def label_sector(angle_deg: float, eye_side: Optional[str]) -> SectorLabel:
    """Classify an angle in [0, 360) into I / S / N / T (or ``unknown`` when
    eye_side is not given and the wedge is nasal/temporal)."""
    a = float(angle_deg) % 360.0
    # inferior and superior are eye-side-independent
    if 45.0 < a <= 135.0:
        return "inferior"
    if 225.0 < a <= 315.0:
        return "superior"
    # The remaining wedges are [315, 360) ∪ [0, 45] (east) and (135, 225] (west)
    is_east = (a > 315.0 or a <= 45.0)
    if eye_side == "OD":
        return "temporal" if is_east else "nasal"
    if eye_side == "OS":
        return "nasal" if is_east else "temporal"
    return "unknown"


def cardinal_angles() -> Dict[str, float]:
    """Canonical cardinal-direction angle labels (independent of eye side)."""
    return {"inferior": 90.0, "superior": 270.0, "east": 0.0, "west": 180.0}


# --------------------------------------------------------------------------- public: metrics

@dataclass(frozen=True)
class CardinalRimWidths:
    """Rim widths measured along the 4 cardinal image axes.

    ``nasal`` and ``temporal`` are ``None`` when eye_side is unknown (the
    raw east/west widths are still exposed so callers can look at them).
    """

    inferior_px: float
    superior_px: float
    east_px: float             # always present, raw image-right width
    west_px: float             # always present, raw image-left width
    nasal_px: Optional[float]  # resolved via eye_side
    temporal_px: Optional[float]


def cardinal_rim_widths(disc_bin: np.ndarray, cup_bin: np.ndarray,
                        disc_center_yx: Tuple[float, float],
                        eye_side: Optional[str]) -> CardinalRimWidths:
    inf = _width_along_ray(disc_bin, cup_bin, disc_center_yx, 90.0)
    sup = _width_along_ray(disc_bin, cup_bin, disc_center_yx, 270.0)
    east = _width_along_ray(disc_bin, cup_bin, disc_center_yx, 0.0)
    west = _width_along_ray(disc_bin, cup_bin, disc_center_yx, 180.0)

    if eye_side == "OD":
        nasal, temporal = west, east
    elif eye_side == "OS":
        nasal, temporal = east, west
    else:
        nasal, temporal = None, None

    return CardinalRimWidths(
        inferior_px=inf, superior_px=sup,
        east_px=east, west_px=west,
        nasal_px=nasal, temporal_px=temporal,
    )


@dataclass(frozen=True)
class RimProfile:
    """Full 360° scan of rim widths.

    Attributes
    ----------
    angles_deg : (N,) array
    widths_px  : (N,) array     — rim width at each angle
    min_width_px : float
    min_angle_deg : float
    min_sector_label : str      — "inferior" / "superior" / nasal/temporal (if eye side known) / "unknown"
    zero_run_deg : int          — longest run of consecutive "rim-absent" sectors (in degrees)
    zero_threshold_px : float   — width below this is treated as "absent rim"
    """

    angles_deg: np.ndarray
    widths_px: np.ndarray
    min_width_px: float
    min_angle_deg: float
    min_sector_label: str
    zero_run_deg: int
    zero_threshold_px: float


def rim_profile(disc_bin: np.ndarray, cup_bin: np.ndarray,
                disc_center_yx: Tuple[float, float],
                disc_diameter_px: float,
                eye_side: Optional[str],
                n_sectors: int = 360,
                zero_rim_fraction_of_dd: float = 0.02) -> RimProfile:
    """Sweep 360° sampling ``n_sectors`` directions, producing a rim-width profile.

    A rim width below ``zero_rim_fraction_of_dd × disc_diameter_px`` is
    classified as "rim absent" for the purpose of computing the longest
    contiguous rim-absent arc. That quantity feeds the DDLS grading.
    """
    angles = np.linspace(0.0, 360.0, n_sectors, endpoint=False)
    widths = np.array([
        _width_along_ray(disc_bin, cup_bin, disc_center_yx, float(a))
        for a in angles
    ], dtype=np.float64)

    min_idx = int(np.argmin(widths))
    min_w = float(widths[min_idx])
    min_angle = float(angles[min_idx])
    min_label = label_sector(min_angle, eye_side)

    zero_thr = float(zero_rim_fraction_of_dd * disc_diameter_px)
    zero_thr = max(zero_thr, 1.0)  # never less than 1 px
    absent = widths < zero_thr
    zero_run = _longest_wraparound_run(absent)
    # Convert run length (in sector count) to degrees
    zero_deg = int(round(zero_run * 360.0 / n_sectors))

    return RimProfile(
        angles_deg=angles,
        widths_px=widths,
        min_width_px=min_w,
        min_angle_deg=min_angle,
        min_sector_label=min_label,
        zero_run_deg=zero_deg,
        zero_threshold_px=zero_thr,
    )


def _longest_wraparound_run(bool_arr: np.ndarray) -> int:
    """Longest run of True in a *circular* boolean array."""
    n = len(bool_arr)
    if n == 0 or not bool_arr.any():
        return 0
    if bool_arr.all():
        return n
    # Walk twice to handle wraparound, but stop counting once we've seen n items
    # so we never report more than the actual array length.
    best = 0
    run = 0
    for i in range(2 * n):
        if bool_arr[i % n]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return min(best, n)


# --------------------------------------------------------------------------- public: areas

@dataclass(frozen=True)
class RimAreas:
    total_px: int
    quadrant_px: Dict[str, int]  # keys: inferior/superior, and nasal/temporal (or east/west if unknown)


def rim_areas(disc_bin: np.ndarray, cup_bin: np.ndarray,
              disc_center_yx: Tuple[float, float],
              eye_side: Optional[str]) -> RimAreas:
    """Compute total rim area and per-sector breakdown.

    Quadrant assignment follows the same 90°-wedge convention as
    ``label_sector``: superior = wedge centered at 270°, inferior at 90°,
    east at 0° (temporal for OD, nasal for OS), west at 180°.
    """
    disc = disc_bin > 0
    cup = cup_bin > 0
    rim = disc & ~cup
    total = int(rim.sum())
    if total == 0:
        empty = {"inferior": 0, "superior": 0}
        if eye_side in ("OS", "OD"):
            empty.update({"nasal": 0, "temporal": 0})
        else:
            empty.update({"east": 0, "west": 0})
        return RimAreas(total_px=0, quadrant_px=empty)

    ys, xs = np.nonzero(rim)
    cy, cx = disc_center_yx
    dy = ys - cy
    dx = xs - cx
    # Image-plane angle: atan2(y, x), in (-180, 180], y grows down so positive angle = below.
    angles = np.degrees(np.arctan2(dy, dx)) % 360.0

    inferior = int(((angles > 45.0) & (angles <= 135.0)).sum())
    superior = int(((angles > 225.0) & (angles <= 315.0)).sum())
    east = int(((angles > 315.0) | (angles <= 45.0)).sum())
    west = int(((angles > 135.0) & (angles <= 225.0)).sum())

    out: Dict[str, int] = {"inferior": inferior, "superior": superior}
    if eye_side == "OD":
        out["nasal"] = west
        out["temporal"] = east
    elif eye_side == "OS":
        out["nasal"] = east
        out["temporal"] = west
    else:
        out["east"] = east
        out["west"] = west
    return RimAreas(total_px=total, quadrant_px=out)


def isnt_compliance(cardinal: CardinalRimWidths) -> Tuple[Optional[bool], List[str]]:
    """Evaluate ISNT rule: inferior ≥ superior ≥ nasal ≥ temporal.

    Returns ``(follows, violations)``. When the eye side wasn't provided
    (nasal/temporal unknown), returns ``(None, [])``.
    """
    if cardinal.nasal_px is None or cardinal.temporal_px is None:
        return None, []
    i = cardinal.inferior_px
    s = cardinal.superior_px
    n = cardinal.nasal_px
    t = cardinal.temporal_px
    violations: List[str] = []
    if s > i:
        violations.append("S > I")
    if n > s:
        violations.append("N > S")
    if t > n:
        violations.append("T > N")
    return (len(violations) == 0), violations
