"""
Disc geometry and measurement-zone construction.

Knudtson et al. (2003) defines measurement **zone B** as the annular region
between 0.5 and 1.0 disc diameters from the optic disc margin, i.e.
[1·R, 2·R] from the disc centre where R is the disc radius. Some later work
re-expresses this as [0.5 DD, 1.0 DD] "from the disc margin" which is the
identical annulus. We parameterise the inner/outer factors in disc diameters
from the centre so zone selection stays explicit.

References
----------
Knudtson MD, Lee KE, Hubbard LD, Wong TY, Klein R, Klein BEK.
"Revised formulas for summarizing retinal vessel diameters."
Curr Eye Res. 2003;27(3):143-9. doi:10.1076/ceyr.27.3.143.16049
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class NoDiscError(ValueError):
    """Raised when the disc mask is empty or invalid."""


@dataclass(frozen=True)
class DiscGeometry:
    """Geometry of the optic disc extracted from a binary mask.

    Attributes
    ----------
    center_yx : (float, float)
        (row, col) centroid of the disc mask, in pixel coordinates.
    radius_px : float
        Equivalent-area radius: sqrt(area / pi). We do **not** use bounding-box
        or convex-hull radii because they inflate the disc for non-circular
        masks and shift the measurement annulus outward.
    area_px : int
        Number of foreground pixels in the disc mask.
    source : str
        Provenance string, e.g. "od_oc_mask".
    """

    center_yx: Tuple[float, float]
    radius_px: float
    area_px: int
    source: str

    @property
    def diameter_px(self) -> float:
        return 2.0 * self.radius_px


def compute_disc_geometry(disc_mask: np.ndarray,
                          source: str = "od_oc_mask") -> DiscGeometry:
    """Return disc centroid, equivalent-area radius, and area from a binary mask.

    Parameters
    ----------
    disc_mask : (H, W) array
        Non-zero where the disc is present. Any foreground value is accepted;
        we binarise internally.
    source : str
        Provenance tag to store on the returned geometry.

    Raises
    ------
    NoDiscError
        If the mask is empty (no foreground pixels).
    """
    if disc_mask is None or disc_mask.ndim != 2:
        raise NoDiscError("disc_mask must be a 2D array")
    bin_mask = disc_mask > 0
    area_px = int(bin_mask.sum())
    if area_px == 0:
        raise NoDiscError("disc_mask has no foreground pixels")

    ys, xs = np.nonzero(bin_mask)
    cy = float(ys.mean())
    cx = float(xs.mean())
    radius = float(np.sqrt(area_px / np.pi))
    return DiscGeometry(center_yx=(cy, cx), radius_px=radius, area_px=area_px, source=source)


def build_zone_b(shape: Tuple[int, int],
                 disc: DiscGeometry,
                 inner_dd: float = 0.5,
                 outer_dd: float = 1.0) -> np.ndarray:
    """Build a boolean annular mask for CRAE/CRVE diameter measurement.

    The annulus is defined in disc-diameters (DD) *from the disc centre*:

        inner_px = inner_dd * (2 * R)
        outer_px = outer_dd * (2 * R)

    With the Knudtson defaults (inner_dd=0.5, outer_dd=1.0), this corresponds
    to the [1·R, 2·R] annulus around disc centre, i.e. "0.5-1.0 DD from the
    disc margin".

    Parameters
    ----------
    shape : (H, W)
        Output mask shape.
    disc : DiscGeometry
        Disc centre and radius.
    inner_dd, outer_dd : float
        Annulus inner/outer radii expressed in disc-diameters.

    Returns
    -------
    (H, W) bool array, True inside zone B.
    """
    if outer_dd <= inner_dd:
        raise ValueError("outer_dd must exceed inner_dd")

    h, w = shape
    cy, cx = disc.center_yx
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    inner_px = inner_dd * disc.diameter_px
    outer_px = outer_dd * disc.diameter_px
    return (dist >= inner_px) & (dist <= outer_px)


def zone_radii_px(disc: DiscGeometry, inner_dd: float, outer_dd: float) -> Tuple[float, float]:
    """Helper: return (inner_px, outer_px) of the zone for reporting in JSON."""
    return inner_dd * disc.diameter_px, outer_dd * disc.diameter_px


def disc_binary_from_labelmap(od_oc_mask: np.ndarray) -> np.ndarray:
    """Convert an OD/OC label map ({0=bg, 1=disc, 2=cup}) into a disc-region
    binary mask (disc ∪ cup).

    The OD/OC head labels the cup as class 2 *nested inside* the disc (rim is
    class 1). For disc geometry we need the whole optic disc region, i.e.
    ``mask > 0``.
    """
    if od_oc_mask is None or od_oc_mask.ndim != 2:
        raise NoDiscError("od_oc_mask must be a 2D array")
    return (od_oc_mask > 0).astype(np.uint8)
