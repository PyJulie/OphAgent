"""
Per-vessel diameter extraction.

Pipeline
--------
1. Skeletonize the binary vessel mask (Zhang-Suen via skimage).
2. Remove skeleton pixels that fall inside the disc region — these represent
   vessels crossing through the disc, which are not measured.
3. Break the skeleton at **bifurcation points** (pixels with ≥ 3 skeleton
   neighbours in the 8-connected neighbourhood). The resulting connected
   components are single-arc "segments".
4. Keep only segments that intersect the measurement zone (zone B).
5. For each kept segment, estimate local width at every skeleton pixel that
   lies *inside* zone B using the Euclidean distance transform of the vessel
   mask (diameter = 2 × distance at that skeleton pixel). The segment's
   representative diameter is the **median** of those per-pixel samples —
   robust to the distance-transform dip near skeleton termini while still
   tracking the true local width.

The segment length (in pixels along the skeleton) is measured as the sum of
Euclidean steps between consecutive skeleton pixels along its arc.

This module is used both for CRAE/CRVE (needs representative diameters) and
for tortuosity (needs the arc geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# 8-connected neighbour offsets (dy, dx).
_NB8 = np.array(
    [(-1, -1), (-1, 0), (-1, 1),
     (0, -1),           (0, 1),
     (1, -1),  (1, 0),  (1, 1)],
    dtype=np.int32,
)


@dataclass
class VesselSegment:
    """A single skeletal arc between two bifurcations / endpoints."""

    points_yx: np.ndarray          # (N, 2) int skeleton pixel coordinates, ordered along the arc
    diameter_px: float             # median per-pixel diameter from distance transform
    diameter_samples: np.ndarray   # raw per-pixel diameters inside zone B (for debugging / QC)
    length_px: float               # polyline length in pixels
    in_zone_point_count: int = 0   # how many skeleton pixels lie inside zone B
    raw_component_id: int = -1

    def __repr__(self) -> str:
        return (
            f"VesselSegment(n={len(self.points_yx)}, d={self.diameter_px:.2f}px, "
            f"len={self.length_px:.1f}px, zoneB_pts={self.in_zone_point_count})"
        )


# --------------------------------------------------------------------------- helpers

def _count_skeleton_neighbours(skel: np.ndarray) -> np.ndarray:
    """For every skeleton pixel, count how many of its 8 neighbours are also
    skeleton pixels. Non-skeleton pixels get 0.
    """
    s = skel.astype(np.uint8)
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    nb = convolve(s, kernel, mode="constant", cval=0)
    return nb * s  # zero-out non-skeleton pixels


def _break_at_bifurcations(skel: np.ndarray) -> np.ndarray:
    """Remove bifurcation pixels so each connected component is a single arc.

    A bifurcation is any skeleton pixel with ≥ 3 skeleton neighbours in 8-conn.
    (Zhang-Suen can produce 2×2 crossings; removing them here is the simplest
    robust break strategy.)
    """
    nb = _count_skeleton_neighbours(skel)
    bifurc = nb >= 3
    arcs = skel & ~bifurc
    return arcs


def _order_arc_points(arc_pts: np.ndarray) -> np.ndarray:
    """Given unordered skeleton pixels of a single arc, return them ordered
    from one endpoint to the other via a greedy 8-connected walk.

    We start from the pixel with the fewest skeleton neighbours (an endpoint);
    if the arc is a loop, we start from any pixel.
    """
    if len(arc_pts) <= 1:
        return arc_pts

    pts_set = {tuple(p) for p in arc_pts}

    # Find neighbour counts within this arc to locate endpoints.
    def nb_count(p):
        y, x = p
        return sum(1 for dy, dx in _NB8 if (y + dy, x + dx) in pts_set)

    endpoints = [p for p in pts_set if nb_count(p) == 1]
    start = endpoints[0] if endpoints else next(iter(pts_set))

    ordered = [start]
    visited = {start}
    current = start
    while True:
        y, x = current
        found_next = None
        for dy, dx in _NB8:
            cand = (y + dy, x + dx)
            if cand in pts_set and cand not in visited:
                found_next = cand
                break
        if found_next is None:
            break
        ordered.append(found_next)
        visited.add(found_next)
        current = found_next

    return np.array(ordered, dtype=np.int32)


def _polyline_length(points_yx: np.ndarray) -> float:
    if len(points_yx) < 2:
        return 0.0
    d = np.diff(points_yx.astype(np.float64), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


# --------------------------------------------------------------------------- public

def extract_vessel_segments(vessel_mask: np.ndarray,
                            disc_mask_bin: np.ndarray,
                            zone_mask: np.ndarray) -> List[VesselSegment]:
    """Extract single-arc skeleton segments crossing the measurement zone.

    Parameters
    ----------
    vessel_mask : (H, W) array
        Non-zero = vessel foreground. We binarise internally.
    disc_mask_bin : (H, W) array or None
        Non-zero = disc region to exclude from the skeleton. If ``None`` we
        skip this step (no disc-based exclusion).
    zone_mask : (H, W) bool array
        True inside zone B. Only segments containing at least one zone-B pixel
        are returned, and only zone-B pixels contribute to width sampling.

    Returns
    -------
    list of VesselSegment, sorted by diameter descending.
    """
    if vessel_mask.ndim != 2 or zone_mask.ndim != 2:
        raise ValueError("vessel_mask and zone_mask must be 2D")
    if vessel_mask.shape != zone_mask.shape:
        raise ValueError("shape mismatch between vessel_mask and zone_mask")

    vmask = vessel_mask > 0
    skel = skeletonize(vmask)
    if disc_mask_bin is not None:
        if disc_mask_bin.shape != vmask.shape:
            raise ValueError("shape mismatch between disc_mask_bin and vessel_mask")
        skel = skel & ~(disc_mask_bin > 0)

    # Distance transform on full vessel mask (not skeleton!) — the value at a
    # skeleton pixel is the radius to the nearest vessel boundary.
    dt = distance_transform_edt(vmask)

    arcs = _break_at_bifurcations(skel)

    # Label arcs — each connected component in 8-conn is a single arc.
    from scipy.ndimage import label
    struct = np.ones((3, 3), dtype=np.uint8)
    labeled, n_labels = label(arcs, structure=struct)

    segments: List[VesselSegment] = []
    for comp_id in range(1, n_labels + 1):
        ys, xs = np.nonzero(labeled == comp_id)
        if len(ys) == 0:
            continue
        pts = np.stack([ys, xs], axis=1)

        in_zone_mask = zone_mask[ys, xs]
        if not in_zone_mask.any():
            continue

        ordered = _order_arc_points(pts)
        length_px = _polyline_length(ordered)

        # Per-pixel diameters only over zone-B subset
        zy, zx = ys[in_zone_mask], xs[in_zone_mask]
        per_pixel_d = 2.0 * dt[zy, zx]
        per_pixel_d = per_pixel_d[per_pixel_d > 0]
        if per_pixel_d.size == 0:
            continue

        diameter = float(np.median(per_pixel_d))

        segments.append(
            VesselSegment(
                points_yx=ordered,
                diameter_px=diameter,
                diameter_samples=per_pixel_d.astype(np.float32),
                length_px=length_px,
                in_zone_point_count=int(in_zone_mask.sum()),
                raw_component_id=comp_id,
            )
        )

    segments.sort(key=lambda s: s.diameter_px, reverse=True)
    return segments


def extract_all_skeleton_segments(vessel_mask: np.ndarray,
                                  disc_mask_bin: np.ndarray | None = None) -> List[VesselSegment]:
    """Like `extract_vessel_segments` but without zone filtering — used for
    tortuosity, where we want long off-disc arcs regardless of zone B.

    Diameter is still reported (median of per-pixel diameters along the whole
    arc) so callers can use it if they wish, but tortuosity does not depend on
    width.
    """
    if vessel_mask.ndim != 2:
        raise ValueError("vessel_mask must be 2D")
    vmask = vessel_mask > 0
    skel = skeletonize(vmask)
    if disc_mask_bin is not None:
        skel = skel & ~(disc_mask_bin > 0)

    dt = distance_transform_edt(vmask)
    arcs = _break_at_bifurcations(skel)

    from scipy.ndimage import label
    struct = np.ones((3, 3), dtype=np.uint8)
    labeled, n_labels = label(arcs, structure=struct)

    segments: List[VesselSegment] = []
    for comp_id in range(1, n_labels + 1):
        ys, xs = np.nonzero(labeled == comp_id)
        if len(ys) == 0:
            continue
        pts = np.stack([ys, xs], axis=1)
        ordered = _order_arc_points(pts)
        length_px = _polyline_length(ordered)

        per_pixel_d = 2.0 * dt[ys, xs]
        per_pixel_d = per_pixel_d[per_pixel_d > 0]
        diameter = float(np.median(per_pixel_d)) if per_pixel_d.size else 0.0

        segments.append(
            VesselSegment(
                points_yx=ordered,
                diameter_px=diameter,
                diameter_samples=per_pixel_d.astype(np.float32),
                length_px=length_px,
                in_zone_point_count=0,
                raw_component_id=comp_id,
            )
        )
    return segments
