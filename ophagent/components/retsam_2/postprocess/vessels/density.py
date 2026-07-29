"""
Vessel density (coverage ratios).

We report fractions in [0, 1]. The denominator is the fundus area — the
area of the illuminated retina in the image — so the ratio does not get
deflated by black borders. If no fundus mask is supplied, we fall back to
the full image area and flag qc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DensityResult:
    artery_ratio: float
    vein_ratio: float
    total_ratio: float
    fundus_area_px: int
    denominator_source: str   # "fundus_mask" | "full_image"


def _fundus_from_image(image_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Simple fallback fundus mask: grayscale > threshold + morphological cleanup.

    This is NOT a high-quality fundus segmenter; prefer passing a proper
    `fundus_mask`. Provided so callers without one still get a sane denominator.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    mask = (gray > threshold).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask.astype(bool)


def compute_density(artery_mask: np.ndarray,
                    vein_mask: np.ndarray,
                    fundus_mask: Optional[np.ndarray] = None) -> DensityResult:
    """Return vessel coverage ratios against the fundus area."""
    if artery_mask.shape != vein_mask.shape:
        raise ValueError("artery_mask and vein_mask must share shape")
    a = artery_mask > 0
    v = vein_mask > 0
    union = a | v

    if fundus_mask is not None:
        if fundus_mask.shape != a.shape:
            raise ValueError("fundus_mask shape must match artery_mask")
        f = fundus_mask > 0
        denom = int(f.sum())
        denom_source = "fundus_mask"
    else:
        denom = int(a.size)
        denom_source = "full_image"

    if denom <= 0:
        return DensityResult(0.0, 0.0, 0.0, 0, denom_source)

    return DensityResult(
        artery_ratio=float(a.sum()) / denom,
        vein_ratio=float(v.sum()) / denom,
        total_ratio=float(union.sum()) / denom,
        fundus_area_px=denom,
        denominator_source=denom_source,
    )


def fundus_mask_from_image(image_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Publicly-exposed helper mirroring `_fundus_from_image` for callers."""
    return _fundus_from_image(image_bgr, threshold=threshold)
