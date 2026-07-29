"""
Disc Damage Likelihood Scale (DDLS) — simplified grading.

Reference
---------
Bayer AU, Nicolela MT. "Disc damage likelihood scale (DDLS)."
British Journal of Ophthalmology 2006; 90: 1045. (Original concept 2003.)

Our implementation follows the published 0b–10 structure, simplified to a
reproducible numerical grade:

    - Grades 1–5 are driven by the **minimum rim-to-disc-diameter ratio**,
      with thresholds that shift depending on the disc size bucket
      (small / average / large). A smaller disc tolerates a smaller rim
      because the retinal nerve fibre density is higher at baseline, so the
      thresholds are tighter for small discs and looser for large ones.
    - Grades 6–10 kick in once there is a **contiguous rim-absent arc**:
      - >0°   → grade 6
      - ≥45°  → grade 7
      - ≥90°  → grade 8
      - ≥180° → grade 9
      - ≥270° → grade 10
    - If the disc is not calibrated (no ``pixel_spacing_um``), we cannot
      bucket it and return ``grade = None`` with an explanatory input block.

The thresholds are exposed via the API config so callers can swap them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


DiscSize = str  # "small" | "average" | "large" | None

# Ratio thresholds per disc size for grades 1..5. A grade k means:
#   THRESHOLDS[size][k-1] > min_rim_disc_ratio >= THRESHOLDS[size][k]
# Grade 1 is assigned when min_ratio >= THRESHOLDS[size][0].
# Grade 5 when min_ratio < THRESHOLDS[size][3] but > 0 and no absent arc.
DEFAULT_DDLS_THRESHOLDS: Dict[str, Tuple[float, float, float, float, float]] = {
    # grade 1     2     3     4     5  (five thresholds, grade 5 is "below last")
    "small":   (0.50, 0.40, 0.30, 0.20, 0.10),
    "average": (0.40, 0.30, 0.20, 0.10, 0.05),
    "large":   (0.30, 0.20, 0.10, 0.05, 0.025),
}

# Disc-size bucket cutoffs in mm². Inclusive on the lower bound.
DEFAULT_DISC_SIZE_CUTOFFS_MM2: Tuple[float, float] = (1.70, 2.50)


@dataclass(frozen=True)
class DDLSResult:
    grade: Optional[int]                     # 1..10 or None if undetermined
    method: str
    disc_size: Optional[DiscSize]            # "small" | "average" | "large" | None
    disc_area_mm2: Optional[float]
    min_rim_disc_ratio: float
    zero_run_deg: int
    reasoning: str                           # human-readable explanation


def classify_disc_size(disc_area_mm2: Optional[float],
                       cutoffs: Tuple[float, float] = DEFAULT_DISC_SIZE_CUTOFFS_MM2
                       ) -> Optional[DiscSize]:
    if disc_area_mm2 is None:
        return None
    small_max, avg_max = cutoffs
    if disc_area_mm2 < small_max:
        return "small"
    if disc_area_mm2 < avg_max:
        return "average"
    return "large"


def compute_ddls(min_rim_disc_ratio: float,
                 zero_run_deg: int,
                 disc_area_mm2: Optional[float],
                 thresholds: Dict[str, Tuple[float, ...]] = DEFAULT_DDLS_THRESHOLDS,
                 disc_size_cutoffs_mm2: Tuple[float, float] = DEFAULT_DISC_SIZE_CUTOFFS_MM2
                 ) -> DDLSResult:
    """Return a DDLS grade in 1–10 (or None if the disc is uncalibrated).

    Parameters
    ----------
    min_rim_disc_ratio : float
        Smallest rim-width / disc-diameter ratio across the 360° sweep.
    zero_run_deg : int
        Longest contiguous arc where rim is effectively absent, in degrees.
    disc_area_mm2 : float or None
        Disc area in square millimetres. If None, the grade is undetermined
        and we return ``grade=None`` with a reason.
    """
    # Grades 6–10 from rim-absent arc length take precedence: once a sector
    # has no rim, the ratio-based grading no longer applies.
    if zero_run_deg >= 270:
        return DDLSResult(grade=10, method="bayer_nicolela_simplified",
                          disc_size=classify_disc_size(disc_area_mm2, disc_size_cutoffs_mm2),
                          disc_area_mm2=disc_area_mm2,
                          min_rim_disc_ratio=min_rim_disc_ratio,
                          zero_run_deg=zero_run_deg,
                          reasoning=f"rim absent over ≥270° (measured {zero_run_deg}°)")
    if zero_run_deg >= 180:
        return DDLSResult(grade=9, method="bayer_nicolela_simplified",
                          disc_size=classify_disc_size(disc_area_mm2, disc_size_cutoffs_mm2),
                          disc_area_mm2=disc_area_mm2,
                          min_rim_disc_ratio=min_rim_disc_ratio,
                          zero_run_deg=zero_run_deg,
                          reasoning=f"rim absent over ≥180° (measured {zero_run_deg}°)")
    if zero_run_deg >= 90:
        return DDLSResult(grade=8, method="bayer_nicolela_simplified",
                          disc_size=classify_disc_size(disc_area_mm2, disc_size_cutoffs_mm2),
                          disc_area_mm2=disc_area_mm2,
                          min_rim_disc_ratio=min_rim_disc_ratio,
                          zero_run_deg=zero_run_deg,
                          reasoning=f"rim absent over ≥90° (measured {zero_run_deg}°)")
    if zero_run_deg >= 45:
        return DDLSResult(grade=7, method="bayer_nicolela_simplified",
                          disc_size=classify_disc_size(disc_area_mm2, disc_size_cutoffs_mm2),
                          disc_area_mm2=disc_area_mm2,
                          min_rim_disc_ratio=min_rim_disc_ratio,
                          zero_run_deg=zero_run_deg,
                          reasoning=f"rim absent over ≥45° (measured {zero_run_deg}°)")
    if zero_run_deg > 0:
        return DDLSResult(grade=6, method="bayer_nicolela_simplified",
                          disc_size=classify_disc_size(disc_area_mm2, disc_size_cutoffs_mm2),
                          disc_area_mm2=disc_area_mm2,
                          min_rim_disc_ratio=min_rim_disc_ratio,
                          zero_run_deg=zero_run_deg,
                          reasoning=f"focal rim loss over {zero_run_deg}° (< 45°)")

    # Grades 1–5 require disc size
    disc_size = classify_disc_size(disc_area_mm2, disc_size_cutoffs_mm2)
    if disc_size is None:
        return DDLSResult(grade=None, method="bayer_nicolela_simplified",
                          disc_size=None,
                          disc_area_mm2=disc_area_mm2,
                          min_rim_disc_ratio=min_rim_disc_ratio,
                          zero_run_deg=zero_run_deg,
                          reasoning="disc_area_mm2 not available (no pixel_spacing_um); "
                                    "cannot choose ratio thresholds.")

    t = thresholds[disc_size]
    if min_rim_disc_ratio >= t[0]:
        grade = 1
    elif min_rim_disc_ratio >= t[1]:
        grade = 2
    elif min_rim_disc_ratio >= t[2]:
        grade = 3
    elif min_rim_disc_ratio >= t[3]:
        grade = 4
    else:
        grade = 5
    return DDLSResult(
        grade=grade,
        method="bayer_nicolela_simplified",
        disc_size=disc_size,
        disc_area_mm2=disc_area_mm2,
        min_rim_disc_ratio=min_rim_disc_ratio,
        zero_run_deg=zero_run_deg,
        reasoning=(
            f"disc_size={disc_size} (area={disc_area_mm2:.2f} mm²), "
            f"min_rim/disc={min_rim_disc_ratio:.3f} → grade {grade}."
        ),
    )
