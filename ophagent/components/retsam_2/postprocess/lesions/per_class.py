"""
Per-class lesion metrics.

Every lesion class (hemorrhage, exudate, drusen, …) independently reports:

    * **count** — number of connected components above the noise-filter threshold
    * **area** — in px, DD² (disc-diameter²), and μm²
    * **coverage_ratio** — area / fundus area
    * **shape** — mean circularity and aspect ratio across components
    * **size distribution** — components bucketed into small / medium / large
      using DD²-normalised thresholds, so the buckets mean the same thing
      across images of different resolution
    * **per-component** — the raw (centroid, area) list; downstream spatial
      analysis (quadrant, distance-to-disc/macula, ETDRS zones) consumes it

No convex hulls, no morphology. The mask is the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ComponentStats:
    """Per-component summary used by spatial analysis downstream."""

    centroid_yx: Tuple[float, float]
    area_px: int
    bbox_yxyx: Tuple[int, int, int, int]
    circularity: float
    aspect_ratio: float


@dataclass
class ClassMetrics:
    """Aggregated metrics for a single lesion class (e.g. hemorrhage)."""

    count: int
    area_px: int
    shape_mean_circularity: float
    shape_mean_aspect_ratio: float
    size_thresholds_dd2: Tuple[float, float]   # (small_max, medium_max)
    size_thresholds_px: Tuple[int, int]
    size_counts: Dict[str, int]                # small / medium / large
    size_areas_px: Dict[str, int]
    components: List[ComponentStats] = field(default_factory=list)


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / (perimeter * perimeter)))


def _aspect_ratio(contour: np.ndarray) -> float:
    if len(contour) < 3:
        return 1.0
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w <= 0 or h <= 0:
        return 1.0
    return float(max(w, h) / min(w, h))


def compute_class_metrics(
    mask: np.ndarray,
    disc_diameter_px: float,
    min_component_size_px: int = 15,
    size_small_max_dd2: float = 0.005,
    size_medium_max_dd2: float = 0.05,
) -> ClassMetrics:
    """Compute the per-class metric bundle for one binary lesion mask.

    Parameters
    ----------
    mask : (H, W) array
        Non-zero = lesion foreground.
    disc_diameter_px : float
        Used to compute per-image size thresholds in pixels.
    min_component_size_px : int
        Components smaller than this are treated as segmentation noise and
        dropped. Count of surviving components is reported.
    size_small_max_dd2, size_medium_max_dd2 : float
        Lesion size bucket cutoffs **in DD²**, so a "small" lesion means the
        same clinical thing across images.
    """
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    DD = float(disc_diameter_px) if disc_diameter_px else 1.0
    small_thr_px = int(round(size_small_max_dd2 * DD * DD))
    medium_thr_px = int(round(size_medium_max_dd2 * DD * DD))
    # Guard against degenerate images where the thresholds collapse
    small_thr_px = max(small_thr_px, min_component_size_px)
    medium_thr_px = max(medium_thr_px, small_thr_px + 1)

    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return ClassMetrics(
            count=0,
            area_px=0,
            shape_mean_circularity=0.0,
            shape_mean_aspect_ratio=0.0,
            size_thresholds_dd2=(size_small_max_dd2, size_medium_max_dd2),
            size_thresholds_px=(small_thr_px, medium_thr_px),
            size_counts={"small": 0, "medium": 0, "large": 0},
            size_areas_px={"small": 0, "medium": 0, "large": 0},
            components=[],
        )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    components: List[ComponentStats] = []
    total_area = 0
    circularities: List[float] = []
    aspect_ratios: List[float] = []
    size_counts = {"small": 0, "medium": 0, "large": 0}
    size_areas = {"small": 0, "medium": 0, "large": 0}

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_size_px:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        bbox = (y, y + h - 1, x, x + w - 1)

        comp_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            perim = float(cv2.arcLength(cnt, True))
            circ = _circularity(area, perim)
            aspect = _aspect_ratio(cnt)
        else:
            circ = 0.0
            aspect = 1.0

        # Component centroid (from the connected-component stats, not the
        # bounding box, so asymmetric lesions aren't biased).
        ys, xs = np.nonzero(comp_mask)
        centroid = (float(ys.mean()), float(xs.mean()))

        components.append(ComponentStats(
            centroid_yx=centroid,
            area_px=area,
            bbox_yxyx=bbox,
            circularity=circ,
            aspect_ratio=aspect,
        ))

        total_area += area
        circularities.append(circ)
        aspect_ratios.append(aspect)

        if area <= small_thr_px:
            size_counts["small"] += 1
            size_areas["small"] += area
        elif area <= medium_thr_px:
            size_counts["medium"] += 1
            size_areas["medium"] += area
        else:
            size_counts["large"] += 1
            size_areas["large"] += area

    mean_circ = float(np.mean(circularities)) if circularities else 0.0
    mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0

    return ClassMetrics(
        count=len(components),
        area_px=total_area,
        shape_mean_circularity=mean_circ,
        shape_mean_aspect_ratio=mean_aspect,
        size_thresholds_dd2=(size_small_max_dd2, size_medium_max_dd2),
        size_thresholds_px=(small_thr_px, medium_thr_px),
        size_counts=size_counts,
        size_areas_px=size_areas,
        components=components,
    )
