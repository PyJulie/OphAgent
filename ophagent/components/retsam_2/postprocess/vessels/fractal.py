"""
Fractal dimension via box counting (Hausdorff-Besicovitch estimator).

We tile the image with non-overlapping boxes of side ``s`` and count boxes
that contain at least one skeleton pixel (N(s)). The box-counting dimension
is the slope of

    log N(s)   vs   log (1 / s)

For a well-behaved fractal in the chosen scale window, this is linear, and
the slope approximates D. We also report the R² of the log-log fit so
downstream consumers can flag unreliable estimates (e.g. very sparse or
clipped skeletons).

Box sizes
---------
We default to a log-spaced set of integer sizes from 2 up to ``min(H, W) // 4``.
This avoids both the grid-finite-size regime (s too close to 1 pixel) and the
image-boundary regime (s too large to produce multiple boxes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class FractalResult:
    value: float              # box-counting dimension
    r2: float                 # goodness of log-log fit, 0..1
    box_sizes_px: List[int]
    box_counts: List[int]
    method: str = "box_counting_hausdorff"


def _default_box_sizes(shape, n_sizes: int = 12) -> List[int]:
    h, w = shape
    upper = max(4, min(h, w) // 4)
    if upper <= 2:
        return [2]
    # log-spaced, dedup, integer, ascending
    raw = np.geomspace(2, upper, num=n_sizes)
    sizes = sorted({int(round(s)) for s in raw if s >= 2})
    return sizes


def _count_boxes(mask: np.ndarray, box: int) -> int:
    """Count non-overlapping boxes containing at least one True pixel."""
    h, w = mask.shape
    # Pad to be divisible so we don't lose edge content.
    pad_h = (box - h % box) % box
    pad_w = (box - w % box) % box
    if pad_h or pad_w:
        padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    else:
        padded = mask

    H, W = padded.shape
    reshaped = padded.reshape(H // box, box, W // box, box)
    # reduce each box to any()
    any_present = reshaped.any(axis=(1, 3))
    return int(any_present.sum())


def box_counting_dimension(skeleton_bool: np.ndarray,
                           box_sizes: Sequence[int] | None = None) -> FractalResult:
    """Estimate the box-counting dimension of a binary skeleton."""
    if skeleton_bool.ndim != 2:
        raise ValueError("skeleton_bool must be a 2D array")
    mask = skeleton_bool.astype(bool)
    if not mask.any():
        return FractalResult(value=0.0, r2=0.0, box_sizes_px=[], box_counts=[])

    sizes = list(box_sizes) if box_sizes is not None else _default_box_sizes(mask.shape)
    counts = [_count_boxes(mask, s) for s in sizes]

    # Keep only (size, count) pairs where count > 0 (log(0) is undefined).
    pairs = [(s, c) for s, c in zip(sizes, counts) if c > 0]
    if len(pairs) < 2:
        return FractalResult(value=0.0, r2=0.0,
                             box_sizes_px=sizes, box_counts=counts)

    s_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    c_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    x = np.log(1.0 / s_arr)
    y = np.log(c_arr)

    # slope, intercept via least squares
    slope, intercept = np.polyfit(x, y, 1)

    # coefficient of determination
    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FractalResult(
        value=float(slope),
        r2=float(max(0.0, min(1.0, r2))),
        box_sizes_px=sizes,
        box_counts=counts,
    )
