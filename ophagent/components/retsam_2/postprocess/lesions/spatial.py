"""
Spatial distribution of lesions: quadrants, distances, ETDRS-like zones.

Coordinate conventions match the rest of retsam-2.0:

    * image-plane angles: θ = 0° → +x (east), 90° → +y (south/inferior),
      180° → -x (west), 270° → -y (north/superior)
    * 90°-wedges centred on cardinal directions define I / S / N / T
    * Nasal vs temporal depends on eye side:
        - OD (right eye) → nasal is image-west (180°), temporal is image-east (0°)
        - OS (left eye)  → nasal is image-east (0°),  temporal is image-west (180°)
    * Quadrants are centred on the **macula** (not the disc) because most
      lesion clinical thinking is macula-referenced (ETDRS)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _angle_from_center(yx: Tuple[float, float],
                       center_yx: Tuple[float, float]) -> float:
    """Image-plane angle in [0, 360)."""
    dy = yx[0] - center_yx[0]
    dx = yx[1] - center_yx[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)


def _classify_quadrant(angle_deg: float, eye_side: Optional[str]) -> Optional[str]:
    """Return one of the four macula-centred quadrant labels, or None when
    eye_side is unknown (we refuse to guess nasal/temporal without it)."""
    if eye_side not in ("OS", "OD"):
        return None
    a = angle_deg % 360.0
    # Vertical classification (macula-relative)
    if a > 180.0:
        vertical = "superior"   # upper half of image plane
    else:
        vertical = "inferior"   # lower half
    # Horizontal classification: east (a<90 or a>270) vs west (90<a<270)
    is_east = (a < 90.0) or (a > 270.0)
    if eye_side == "OD":
        horizontal = "temporal" if is_east else "nasal"
    else:  # OS
        horizontal = "nasal" if is_east else "temporal"
    return f"{vertical}_{horizontal}"


@dataclass
class SpatialStats:
    quadrant_counts: Optional[Dict[str, int]]       # None when eye_side unknown
    quadrants_with_lesions: Optional[int]
    distance_to_disc_dd: Optional[Dict[str, float]]  # min/mean/median/max
    distance_to_macula_dd: Optional[Dict[str, float]]
    macula_zone_counts: Optional[Dict[str, int]]     # keys like "0_1_dd", "1_2_dd", "gt_2_dd"
    macula_zone_areas_px: Optional[Dict[str, int]]
    zone_boundaries_dd: Optional[List[float]]


def _distance_stats_dd(distances_px: Sequence[float],
                       disc_diameter_px: float) -> Dict[str, float]:
    if not distances_px:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    arr = np.asarray(distances_px, dtype=np.float64)
    DD = float(disc_diameter_px) if disc_diameter_px > 0 else 1.0
    arr_dd = arr / DD
    return {
        "min": float(arr_dd.min()),
        "mean": float(arr_dd.mean()),
        "median": float(np.median(arr_dd)),
        "max": float(arr_dd.max()),
    }


def compute_spatial_stats(
    components,                                     # list[ComponentStats]
    disc_center_yx: Tuple[float, float],
    disc_diameter_px: float,
    macula_center_yx: Optional[Tuple[float, float]] = None,
    eye_side: Optional[str] = None,
    zone_boundaries_dd: Sequence[float] = (1.0, 2.0),
) -> SpatialStats:
    """Compute spatial distribution metrics for one class's components.

    All distances are reported in disc-diameter (DD) units so they are
    image-resolution invariant.
    """
    if not components:
        return SpatialStats(
            quadrant_counts=None,
            quadrants_with_lesions=None,
            distance_to_disc_dd=None,
            distance_to_macula_dd=None,
            macula_zone_counts=None,
            macula_zone_areas_px=None,
            zone_boundaries_dd=list(zone_boundaries_dd),
        )

    # --- distance to disc / macula ------------------------------------------
    disc_distances = []
    macula_distances = []
    for comp in components:
        dy = comp.centroid_yx[0] - disc_center_yx[0]
        dx = comp.centroid_yx[1] - disc_center_yx[1]
        disc_distances.append(float(np.hypot(dy, dx)))
        if macula_center_yx is not None:
            dym = comp.centroid_yx[0] - macula_center_yx[0]
            dxm = comp.centroid_yx[1] - macula_center_yx[1]
            macula_distances.append(float(np.hypot(dym, dxm)))

    dist_disc_dd = _distance_stats_dd(disc_distances, disc_diameter_px)
    dist_mac_dd = (_distance_stats_dd(macula_distances, disc_diameter_px)
                   if macula_center_yx is not None else None)

    # --- quadrant classification --------------------------------------------
    q_counts: Optional[Dict[str, int]] = None
    q_with_lesions: Optional[int] = None
    if macula_center_yx is not None and eye_side in ("OS", "OD"):
        q_counts = {
            "superior_temporal": 0,
            "superior_nasal": 0,
            "inferior_temporal": 0,
            "inferior_nasal": 0,
        }
        for comp in components:
            angle = _angle_from_center(comp.centroid_yx, macula_center_yx)
            label = _classify_quadrant(angle, eye_side)
            if label is not None and label in q_counts:
                q_counts[label] += 1
        q_with_lesions = sum(1 for v in q_counts.values() if v > 0)

    # --- ETDRS-like zones around macula -------------------------------------
    zone_counts: Optional[Dict[str, int]] = None
    zone_areas: Optional[Dict[str, int]] = None
    if macula_center_yx is not None and disc_diameter_px > 0:
        boundaries = [float(b) for b in sorted(zone_boundaries_dd)]
        zone_labels = _zone_labels(boundaries)
        zone_counts = {lab: 0 for lab in zone_labels}
        zone_areas = {lab: 0 for lab in zone_labels}
        for comp, mac_d_px in zip(components, macula_distances):
            mac_d_dd = mac_d_px / disc_diameter_px
            label = _zone_for_distance(mac_d_dd, boundaries, zone_labels)
            zone_counts[label] += 1
            zone_areas[label] += comp.area_px

    return SpatialStats(
        quadrant_counts=q_counts,
        quadrants_with_lesions=q_with_lesions,
        distance_to_disc_dd=dist_disc_dd,
        distance_to_macula_dd=dist_mac_dd,
        macula_zone_counts=zone_counts,
        macula_zone_areas_px=zone_areas,
        zone_boundaries_dd=list(zone_boundaries_dd),
    )


def _zone_labels(boundaries: Sequence[float]) -> List[str]:
    """Human-friendly labels for the concentric zones defined by
    ``boundaries``: e.g. boundaries=(1, 2) → ["0_1_dd", "1_2_dd", "gt_2_dd"].
    """
    labels: List[str] = []
    prev = 0.0
    for b in boundaries:
        labels.append(f"{prev:.2g}_{b:.2g}_dd".replace(".", "p"))
        prev = b
    labels.append(f"gt_{prev:.2g}_dd".replace(".", "p"))
    return labels


def _zone_for_distance(distance_dd: float,
                       boundaries: Sequence[float],
                       zone_labels: Sequence[str]) -> str:
    for boundary, label in zip(boundaries, zone_labels[:-1]):
        if distance_dd <= boundary:
            return label
    return zone_labels[-1]
