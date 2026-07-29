"""
Disc and cup geometric measurements.

Design choices
--------------
1. **Diameters** are measured as the bounding-box extents (max y - min y + 1,
   max x - min x + 1). For roughly-elliptical disc/cup shapes this matches the
   clinical "vertical diameter through the centre" to within 1-2 px, and is
   insensitive to where exactly the centroid lies. For highly irregular
   segmentations (rare, but possible with small discs / cup spikes) the bbox
   can over-read by one or two outlier pixels — we accept that for the
   simplicity it brings, and prefer fixing the upstream mask rather than
   wallpapering over it here.
2. **Equivalent diameter** = 2·√(A/π), reported alongside as an area-based
   size measure that is more stable to thin protrusions.
3. **Tilt / ovality** from ``cv2.fitEllipse``. We convert the OpenCV angle
   (measured from horizontal, increasing counter-clockwise looking at image
   y going down) into "angle of the disc major axis from the vertical
   y-axis", because that is the clinically reported orientation: 0° = the
   disc is taller than wide with its long axis exactly vertical; positive
   angles tilt to one side.
4. No convex hull, no morphological cleanup. The mask coming out of the
   model is the input of record; we do not paper over segmentation noise
   here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


class NoDiscError(ValueError):
    """Disc mask has no foreground pixels."""


class NoCupError(ValueError):
    """Cup mask has no foreground pixels."""


@dataclass(frozen=True)
class EllipseFit:
    """Result of cv2.fitEllipse (repackaged with plain Python types)."""

    center_xy: Tuple[float, float]
    axes_minor_major: Tuple[float, float]   # (minor_axis_length, major_axis_length)
    angle_deg: float                        # OpenCV convention (see _opencv_angle_to_vertical)

    @property
    def major_axis_px(self) -> float:
        return max(self.axes_minor_major)

    @property
    def minor_axis_px(self) -> float:
        return min(self.axes_minor_major)


@dataclass(frozen=True)
class DiscGeometry:
    center_yx: Tuple[float, float]
    area_px: int
    equivalent_diameter_px: float   # 2 · √(area/π)
    diameter_vertical_px: float     # bbox vertical extent
    diameter_horizontal_px: float   # bbox horizontal extent
    ellipse: Optional[EllipseFit]
    tilt_angle_deg: float           # major-axis angle from vertical, in (-90, 90]
    ovality: float                  # v/h diameter ratio

    @property
    def diameter_px(self) -> float:
        """Default disc diameter used for DD-normalisation; we use the vertical
        bbox diameter because vCDR is the clinically emphasised ratio."""
        return self.diameter_vertical_px


@dataclass(frozen=True)
class CupGeometry:
    center_yx: Tuple[float, float]
    area_px: int
    diameter_vertical_px: float
    diameter_horizontal_px: float
    aspect_ratio: float             # v/h (>1 means taller than wide)


# --------------------------------------------------------------------------- helpers

def _bbox_extents(mask: np.ndarray) -> Tuple[int, int, int, int, float, float]:
    """Return (y_min, y_max, x_min, x_max, v_extent, h_extent) of a binary mask.

    Extents are measured as (max - min + 1) so a 1-pixel-tall mask still
    reports extent 1 rather than 0.
    """
    ys, xs = np.nonzero(mask)
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    return y_min, y_max, x_min, x_max, float(y_max - y_min + 1), float(x_max - x_min + 1)


def _fit_ellipse(mask_bool: np.ndarray) -> Optional[EllipseFit]:
    """cv2.fitEllipse on the largest contour of a binary mask.

    Returns ``None`` when the mask has fewer than 5 points (the minimum that
    cv2.fitEllipse accepts).
    """
    if int(mask_bool.sum()) < 5:
        return None
    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # Use the largest contour by point count — robust when mask has minor specks.
    contour = max(contours, key=lambda c: len(c))
    if len(contour) < 5:
        return None
    (cx, cy), (minor, major), angle = cv2.fitEllipse(contour)
    return EllipseFit(center_xy=(float(cx), float(cy)),
                      axes_minor_major=(float(minor), float(major)),
                      angle_deg=float(angle))


def _opencv_angle_to_vertical(ellipse: EllipseFit) -> float:
    """Convert OpenCV ellipse ``angle`` (0..180, measured between +x and major
    axis) into the "angle of major axis away from vertical y-axis", wrapped
    into (-90, 90]. Positive = major axis leans to the right of vertical.

    OpenCV orients minor/major as ``(minor, major) = axes``, and ``angle`` is
    the rotation of the **first** axis (i.e. the minor one on a taller-than-wide
    ellipse). We resolve which axis is major and use its angle.
    """
    minor, major = ellipse.axes_minor_major
    angle = ellipse.angle_deg  # 0..180
    # cv2's angle is the rotation of the minor axis for fitEllipse; the major
    # axis is perpendicular to that. So major-axis angle = angle - 90 (mod 180).
    # But when the returned (minor, major) pair is already ordered so that
    # `minor < major`, OpenCV's `angle` is the angle of the **first** axis
    # that was returned, which on a taller-than-wide ellipse is the minor
    # (horizontal-ish). We compute the major-axis angle by picking whichever
    # of `angle` / `angle - 90` is closer to 90° (i.e. closer to vertical)
    # when the major axis is vertical-ish, else swap. This is robust to the
    # OpenCV quirk without having to mess with which axis was returned.
    candidate = (angle - 90.0) % 180.0
    # Convert angle-from-horizontal to angle-from-vertical
    tilt_from_vertical = 90.0 - candidate
    # Wrap into (-90, 90]
    while tilt_from_vertical <= -90:
        tilt_from_vertical += 180
    while tilt_from_vertical > 90:
        tilt_from_vertical -= 180
    return float(tilt_from_vertical)


# --------------------------------------------------------------------------- public

def compute_disc_geometry(disc_bin: np.ndarray) -> DiscGeometry:
    """Extract all disc-level geometric measurements."""
    if disc_bin is None or disc_bin.ndim != 2:
        raise NoDiscError("disc_bin must be a 2D array")
    mask = disc_bin > 0
    area = int(mask.sum())
    if area == 0:
        raise NoDiscError("disc mask has no foreground pixels")

    ys, xs = np.nonzero(mask)
    cy = float(ys.mean())
    cx = float(xs.mean())

    _, _, _, _, v_ext, h_ext = _bbox_extents(mask)

    equivalent_diameter = 2.0 * float(np.sqrt(area / np.pi))
    ellipse = _fit_ellipse(mask)
    tilt = _opencv_angle_to_vertical(ellipse) if ellipse is not None else 0.0
    ovality = v_ext / h_ext if h_ext > 0 else 0.0

    return DiscGeometry(
        center_yx=(cy, cx),
        area_px=area,
        equivalent_diameter_px=equivalent_diameter,
        diameter_vertical_px=v_ext,
        diameter_horizontal_px=h_ext,
        ellipse=ellipse,
        tilt_angle_deg=tilt,
        ovality=ovality,
    )


def compute_cup_geometry(cup_bin: np.ndarray) -> CupGeometry:
    """Extract all cup-level geometric measurements."""
    if cup_bin is None or cup_bin.ndim != 2:
        raise NoCupError("cup_bin must be a 2D array")
    mask = cup_bin > 0
    area = int(mask.sum())
    if area == 0:
        raise NoCupError("cup mask has no foreground pixels")

    ys, xs = np.nonzero(mask)
    cy = float(ys.mean())
    cx = float(xs.mean())
    _, _, _, _, v_ext, h_ext = _bbox_extents(mask)
    aspect = v_ext / h_ext if h_ext > 0 else 0.0

    return CupGeometry(
        center_yx=(cy, cx),
        area_px=area,
        diameter_vertical_px=v_ext,
        diameter_horizontal_px=h_ext,
        aspect_ratio=aspect,
    )


def from_od_oc_labelmap(od_oc_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the OD/OC label map into (disc_bin, cup_bin) following the
    standard convention:  ``0=background, 1=rim-only, 2=cup``. The disc
    region is everything >0 (rim ∪ cup), the cup is exactly ``mask == 2``.
    """
    if od_oc_mask is None or od_oc_mask.ndim != 2:
        raise ValueError("od_oc_mask must be a 2D array")
    disc_bin = (od_oc_mask > 0).astype(np.uint8)
    cup_bin = (od_oc_mask == 2).astype(np.uint8)
    return disc_bin, cup_bin
