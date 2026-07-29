"""
Tortuosity via the Distance Factor (DF).

For a single vessel segment represented as an ordered skeleton polyline with
arc length L and chord length C (Euclidean distance between endpoints),

    DF = L / C

DF = 1 for a straight segment and grows with bendiness. We report the mean,
median, and length-weighted mean across all segments longer than a minimum
length threshold (default 40 px at 640² analysis resolution); very short
segments are dominated by pixel grid quantisation and inflate the statistic.

Reference
---------
Hart WE, Goldbaum M, Côté B, Kube P, Nelson MR.
"Measurement and classification of retinal vascular tortuosity."
Int J Med Inform. 1999;53(2-3):239-52.

More sophisticated curvature-based measures (Grisan 2008, Trucco 2010) exist;
they require spline fitting of the centreline and are out of scope for a
first-pass, reproducible implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class TortuosityResult:
    mean_df: float
    median_df: float
    length_weighted_df: float
    n_segments: int
    min_segment_length_px: int
    method: str = "distance_factor_per_segment"


def distance_factor_from_polyline(points_yx: np.ndarray) -> float:
    """DF = arc_length / chord_length for a single polyline.

    Returns 1.0 when arc and chord are equal (straight), 0.0 when the polyline
    is degenerate (zero or one point).
    """
    if points_yx is None or len(points_yx) < 2:
        return 0.0
    pts = points_yx.astype(np.float64)
    diffs = np.diff(pts, axis=0)
    arc = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    if chord <= 0:
        # Closed loop or coincident endpoints; DF is undefined here.
        return 0.0
    return arc / chord


def aggregate_tortuosity(segments: Sequence,
                         min_segment_length_px: int = 40) -> TortuosityResult:
    """Aggregate DF across segments, filtering by minimum arc length.

    Each item in `segments` must have `.points_yx` (ordered polyline) and
    `.length_px` — the shape produced by ``diameter.extract_*_segments``.
    """
    df_values: List[float] = []
    lengths: List[float] = []

    for seg in segments:
        length_px = float(getattr(seg, "length_px", 0.0))
        if length_px < min_segment_length_px:
            continue
        df = distance_factor_from_polyline(seg.points_yx)
        if df <= 0:
            continue
        df_values.append(df)
        lengths.append(length_px)

    if not df_values:
        return TortuosityResult(
            mean_df=0.0,
            median_df=0.0,
            length_weighted_df=0.0,
            n_segments=0,
            min_segment_length_px=min_segment_length_px,
        )

    df_arr = np.asarray(df_values, dtype=np.float64)
    len_arr = np.asarray(lengths, dtype=np.float64)
    total_len = float(len_arr.sum())
    weighted = float((df_arr * len_arr).sum() / total_len) if total_len > 0 else 0.0

    return TortuosityResult(
        mean_df=float(df_arr.mean()),
        median_df=float(np.median(df_arr)),
        length_weighted_df=weighted,
        n_segments=len(df_values),
        min_segment_length_px=min_segment_length_px,
    )
