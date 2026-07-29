"""
Lesion-derived severity **inputs**.

This module does NOT produce diagnostic grades. It exposes the clinical
inputs that downstream grading schemes (ETDRS, AREDS simplified, etc.)
would consume, so that:

    1. a grading-naive tool still has meaningful summary numbers, and
    2. a future disease-classification layer has a clean contract to read
       from, without needing to re-aggregate per-class metrics.

Each helper returns either a dict of the relevant inputs or ``None`` when
the required inputs (e.g. eye_side for DR quadrant analysis) aren't
present — we stay faithful to the project's fail-loud principle.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ETDRS "4:2:1" rule (severe NPDR) trigger counts at the clinical threshold;
# see Early Treatment Diabetic Retinopathy Study Report Number 12. We use a
# simplified intraretinal-hemorrhage-per-quadrant count as the proxy input;
# venous-beading and IRMA detection are out of scope for a mask-based tool.
DEFAULT_HEAVY_HEMORRHAGE_PER_QUADRANT = 20


def dr_severity_inputs(groups_summary: Dict[str, Any],
                       eye_side: Optional[str],
                       heavy_hemorrhage_per_quadrant: int = DEFAULT_HEAVY_HEMORRHAGE_PER_QUADRANT
                       ) -> Optional[Dict[str, Any]]:
    """Inputs that downstream DR grading schemes would consume.

    Returns ``None`` when DR inputs can't be computed (e.g. eye_side
    unknown). Does NOT return a grade.
    """
    dr = groups_summary.get("lesion_dr", {}).get("classes", {})
    if not dr or eye_side not in ("OS", "OD"):
        return None

    hemorrhage = dr.get("hemorrhage", {})
    cw = dr.get("cotton_wool_spot", {})
    exudate = dr.get("exudate", {})
    laser = dr.get("laser_spot", {})

    q_counts = None
    heavy_q = None
    if hemorrhage and isinstance(hemorrhage.get("spatial"), dict):
        q_counts = hemorrhage["spatial"].get("quadrant_counts")
    if q_counts is not None:
        heavy_q = sum(1 for n in q_counts.values() if n >= heavy_hemorrhage_per_quadrant)

    return {
        "hemorrhage_total_count": int(hemorrhage.get("count", 0)),
        "hemorrhage_per_quadrant": q_counts,
        "heavy_hemorrhage_quadrants_count": heavy_q,
        "heavy_hemorrhage_threshold_per_quadrant": heavy_hemorrhage_per_quadrant,
        "four_quadrants_heavy": (heavy_q == 4) if heavy_q is not None else None,
        "cotton_wool_spot_count": int(cw.get("count", 0)),
        "exudate_count": int(exudate.get("count", 0)),
        "laser_spot_count": int(laser.get("count", 0)),
        "notes": (
            "Hemorrhage counts per macula-centred quadrant. ETDRS 4:2:1 rule "
            "uses H in all four quadrants as one marker for severe NPDR — "
            "flagged as ``four_quadrants_heavy``. This module reports the "
            "inputs only; clinical grading is out of scope."
        ),
    }


def amd_severity_inputs(groups_summary: Dict[str, Any],
                        macula_center_present: bool) -> Optional[Dict[str, Any]]:
    """Inputs for AMD-oriented grading."""
    amd = groups_summary.get("lesion_amd", {}).get("classes", {})
    if not amd:
        return None

    drusen = amd.get("drusen", {})
    patch_hemorrhage = amd.get("patch_hemorrhage", {})

    drusen_count_in_1dd = None
    if drusen and macula_center_present:
        zones = drusen.get("spatial", {}).get("macula_zone_counts")
        if isinstance(zones, dict):
            # Count drusen whose centroid falls within 0–1 DD of macula
            for key, val in zones.items():
                if key.startswith("0p0_1p0_dd") or key.startswith("0_1_dd"):
                    drusen_count_in_1dd = int(val)
                    break

    return {
        "drusen_total_count": int(drusen.get("count", 0)),
        "drusen_total_area_dd2": float(drusen.get("area_dd2", 0.0)),
        "drusen_count_within_1dd_of_macula": drusen_count_in_1dd,
        "patch_hemorrhage_present": int(patch_hemorrhage.get("count", 0)) > 0,
        "patch_hemorrhage_count": int(patch_hemorrhage.get("count", 0)),
        "notes": (
            "Drusen count + macula-1DD count are standard AMD grading inputs "
            "(AREDS simplified). Patch hemorrhage presence is a red flag "
            "for exudative AMD. This module reports inputs only."
        ),
    }


def other_findings_inputs(groups_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Miscellaneous per-class presence/absence and gross counts."""
    others = groups_summary.get("lesion_others", {}).get("classes", {})
    possible = groups_summary.get("possible_lesions", {}).get("classes", {})
    if not others and not possible:
        return None

    def _present(node: Dict[str, Any]) -> bool:
        return int(node.get("count", 0)) > 0

    out: Dict[str, Any] = {
        "macular_hole_present": _present(others.get("macular_hole", {})),
        "epiretinal_membrane_present": _present(others.get("epiretinal_membrane", {})),
        "retinal_scar_present": _present(others.get("retinal_scar", {})),
        "artifact_present": _present(others.get("artifact", {})),
        "myelinated_nerve_fiber_present": _present(possible.get("myelinated_nerve_fiber", {})),
        "venous_tortuosity_present": _present(possible.get("venous_tortuosity", {})),
        "other_unspecified_lesions_count": int(possible.get("other_lesions", {}).get("count", 0)),
    }
    return out
