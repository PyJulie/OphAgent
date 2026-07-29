"""Structured DR 4-2-1 proxy assessment from ReT-SAM evidence.

This adapter does not claim to measure a strict clinical 4-2-1 rule. It makes
the computable part explicit: hemorrhage burden and quadrant distribution from
ReT-SAM lesion masks/components. Venous beading and IRMA are currently reported
as unavailable unless a future detector is added. Because confluent hemorrhages
can be segmented as one large connected component, the adapter also reports an
area-weighted hemorrhage proxy in addition to component counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, GLOBAL_REGISTRY, register


_QUADRANTS = ("superior_left", "superior_right", "inferior_left", "inferior_right")
_AREA_WEIGHTED_QUADRANT_THRESHOLD_PX = 75
_AREA_WEIGHTED_TOTAL_HEMORRHAGE_THRESHOLD_PX = 350


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _class_summary(classes: dict[str, Any], name: str) -> dict[str, Any]:
    return classes.get(name) or {}


def _count(classes: dict[str, Any], name: str) -> int:
    return int((_class_summary(classes, name).get("count") or 0) or 0)


def _area_px(classes: dict[str, Any], name: str) -> int:
    area = _class_summary(classes, name).get("area") or {}
    return int((area.get("px") or 0) or 0)


def _coverage(classes: dict[str, Any], name: str) -> float:
    return float((_class_summary(classes, name).get("coverage_ratio") or 0.0) or 0.0)


def _components(classes: dict[str, Any], name: str) -> list[dict[str, Any]]:
    comps = _class_summary(classes, name).get("components") or []
    return comps if isinstance(comps, list) else []


def _image_center_from_mask(mask_path: str | None, image_path: str) -> tuple[float, float, str]:
    """Return center_yx and provenance.

    Prefer the fundus-mask bounding-box center; fall back to image midpoint.
    """

    try:
        if mask_path and Path(mask_path).exists():
            import numpy as np
            from PIL import Image

            arr = np.array(Image.open(mask_path).convert("L"))
            ys, xs = np.nonzero(arr > 0)
            if len(xs) and len(ys):
                return (
                    (float(ys.min()) + float(ys.max())) / 2.0,
                    (float(xs.min()) + float(xs.max())) / 2.0,
                    "fundus_mask_bbox",
                )
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        return float(height) / 2.0, float(width) / 2.0, "image_midpoint"
    except Exception:
        return 0.0, 0.0, "unknown"


def _hemorrhage_overlap_guard(mask_files: dict[str, Any]) -> dict[str, Any]:
    """Detect when DR and AMD heads describe the same hemorrhagic region."""
    dr_path = mask_files.get("lesion_dr_hemorrhage")
    amd_path = mask_files.get("lesion_amd_patch_hemorrhage")
    if not (dr_path and amd_path):
        return {"status": "not_assessed"}
    try:
        import numpy as np
        from PIL import Image

        dr = np.asarray(Image.open(dr_path)) > 0
        amd = np.asarray(Image.open(amd_path)) > 0
        dr_px = int(dr.sum())
        amd_px = int(amd.sum())
        if not dr_px or not amd_px:
            return {"status": "separable"}
        overlap_px = int((dr & amd).sum())
        overlap_of_dr = overlap_px / dr_px
        overlap_of_amd = overlap_px / amd_px
        overlap_of_smaller = overlap_px / min(dr_px, amd_px)
        ambiguous = overlap_px >= 20 and overlap_of_smaller >= 0.5
        return {
            "status": "ambiguous" if ambiguous else "separable",
            "dr_head_area_px": dr_px,
            "amd_head_area_px": amd_px,
            "overlap_area_px": overlap_px,
            "overlap_fraction_of_dr_head": round(overlap_of_dr, 4),
            "overlap_fraction_of_amd_head": round(overlap_of_amd, 4),
            "interpretation": (
                "The disease-labelled heads substantially overlap. The "
                "hemorrhage cannot be assigned to DR from this segmentation "
                "alone; consider macular neovascular hemorrhage."
                if ambiguous else
                "The hemorrhage heads are spatially separable."
            ),
        }
    except Exception as exc:
        return {
            "status": "not_assessed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _quadrant_for(centroid_yx: Any, center_yx: tuple[float, float]) -> str | None:
    if not isinstance(centroid_yx, (list, tuple)) or len(centroid_yx) < 2:
        return None
    try:
        y = float(centroid_yx[0])
        x = float(centroid_yx[1])
    except (TypeError, ValueError):
        return None
    cy, cx = center_yx
    if y < cy and x < cx:
        return "superior_left"
    if y < cy and x >= cx:
        return "superior_right"
    if y >= cy and x < cx:
        return "inferior_left"
    return "inferior_right"


def _hemorrhage_quadrants(classes: dict[str, Any], center_yx: tuple[float, float]) -> dict[str, Any]:
    counts = {q: 0 for q in _QUADRANTS}
    areas = {q: 0 for q in _QUADRANTS}
    max_areas = {q: 0 for q in _QUADRANTS}
    for comp in _components(classes, "hemorrhage"):
        q = _quadrant_for(comp.get("centroid_yx"), center_yx)
        if not q:
            continue
        area_px = int((comp.get("area_px") or 0) or 0)
        counts[q] += 1
        areas[q] += area_px
        max_areas[q] = max(max_areas[q], area_px)

    positive = [q for q in _QUADRANTS if counts[q] > 0]
    return {
        "center_yx": [round(center_yx[0], 2), round(center_yx[1], 2)],
        "count_by_quadrant": counts,
        "area_px_by_quadrant": areas,
        "max_component_area_px_by_quadrant": max_areas,
        "positive_quadrants": positive,
        "n_positive_quadrants": len(positive),
        "min_count_per_quadrant": min(counts.values()) if counts else 0,
        "max_count_per_quadrant": max(counts.values()) if counts else 0,
    }


@register
class CFPDR421AssessmentAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_dr_421_assessment",
        modality="CFP",
        task="classification",
        description=(
            "Structured diabetic-retinopathy severity proxy for the 4-2-1 "
            "rule. Computes hemorrhage burden and quadrant distribution from "
            "ReT-SAM lesion components and areas, reports whether strict "
            "hemorrhage 4-quadrant evidence is available, adds an area-weighted "
            "proxy for confluent hemorrhages, and separately marks venous "
            "beading / IRMA as not directly measured. Use only when DR is "
            "already independently supported or the user explicitly requests "
            "DR grading; this is not a general CFP differential classifier."
        ),
        labels=["no_apparent_dr", "mild_npdr_proxy", "moderate_npdr_proxy", "severe_npdr_proxy"],
        confidence_threshold=0.0,
        limitations=[
            "Venous beading and IRMA are not directly measured by current tools.",
            "Quadrants are image/fundus-mask quadrants, not clinician-drawn fields.",
            "Hemorrhage mask includes dot/blot-like red lesions; microaneurysm separation is unavailable.",
            "Area-weighted proxy compensates for confluent hemorrhages that may be counted as one component.",
            "Abstains when DR-hemorrhage and AMD-patch-hemorrhage masks substantially overlap.",
        ],
        requires_tools=["cfp_retsam_segmentation"],
        cost_class="slow",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        retsam = GLOBAL_REGISTRY.predict(
            "cfp_retsam_segmentation",
            image_path,
            quantify_modules=["lesions"],
        )
        if not retsam.success:
            return AdapterResult(
                success=False,
                tool=self.metadata.name,
                modality="CFP",
                task="classification",
                error=f"cfp_retsam_segmentation failed: {retsam.error}",
                raw_output={"retsam": retsam.to_jsonable()},
            )

        classes = (
            _nested(
                retsam.predictions,
                "quantitative", "lesions", "groups", "lesion_dr", "classes",
            )
            or {}
        )
        mask_files = retsam.predictions.get("mask_files") or {}
        etiology_guard = _hemorrhage_overlap_guard(mask_files)
        center_yx = _image_center_from_mask(mask_files.get("fundus_mask"), image_path)
        quadrants = _hemorrhage_quadrants(classes, (center_yx[0], center_yx[1]))
        quadrants["center_source"] = center_yx[2]

        hem = _count(classes, "hemorrhage")
        hem_area_px = _area_px(classes, "hemorrhage") or sum(
            (quadrants.get("area_px_by_quadrant") or {}).values()
        )
        exu = _count(classes, "exudate")
        cws = _count(classes, "cotton_wool_spot")
        laser = _count(classes, "laser_spot")
        total = hem + exu + cws
        coverage = (
            _coverage(classes, "hemorrhage")
            + _coverage(classes, "exudate")
            + _coverage(classes, "cotton_wool_spot")
        )

        # Strict 4-2-1 computable part: the classic "4" criterion requires
        # roughly >20 hemorrhages/microaneurysms in each of four quadrants.
        hemorrhage_4_strict = all(
            quadrants["count_by_quadrant"].get(q, 0) >= 20 for q in _QUADRANTS
        )
        hemorrhage_4_loose = (
            quadrants["n_positive_quadrants"] == 4 and hem >= 20
        )
        area_positive_quadrants = [
            q for q in _QUADRANTS
            if quadrants["area_px_by_quadrant"].get(q, 0) >= _AREA_WEIGHTED_QUADRANT_THRESHOLD_PX
        ]
        hemorrhage_4_area_weighted = (
            len(area_positive_quadrants) >= 3
            and hem_area_px >= _AREA_WEIGHTED_TOTAL_HEMORRHAGE_THRESHOLD_PX
        )
        heavy_lesion_burden = (
            hem >= 20 or total >= 25 or (exu >= 50 and total >= 50)
        )
        cws_with_hemorrhage = cws >= 2 and hem >= 5
        strong_severe_npdr_proxy = bool(
            hemorrhage_4_strict
            or hemorrhage_4_loose
            or hemorrhage_4_area_weighted
            or heavy_lesion_burden
        )

        severe_npdr_proxy = bool(
            strong_severe_npdr_proxy
            or cws_with_hemorrhage
        )
        if strong_severe_npdr_proxy:
            severity_proxy = 3
            severity_label = "severe_npdr_proxy"
            confidence = 0.78 if hemorrhage_4_strict else 0.72
        elif exu > 0 or cws > 0 or hem >= 5:
            severity_proxy = 2
            severity_label = "moderate_npdr_proxy"
            confidence = 0.66
        elif hem > 0:
            severity_proxy = 1
            severity_label = "mild_npdr_proxy"
            confidence = 0.60
        else:
            severity_proxy = 0
            severity_label = "no_apparent_dr"
            confidence = 0.80

        unadjusted_proxy = {
            "severity_proxy": severity_proxy,
            "severity_label": severity_label,
            "confidence": confidence,
            "rule_4_hemorrhage_all_quadrants_strict": hemorrhage_4_strict,
            "rule_4_hemorrhage_all_quadrants_loose_proxy": hemorrhage_4_loose,
            "rule_4_hemorrhage_area_weighted_proxy": hemorrhage_4_area_weighted,
            "heavy_lesion_burden_proxy": heavy_lesion_burden,
            "strong_severe_npdr_proxy": strong_severe_npdr_proxy,
            "severe_npdr_proxy": severe_npdr_proxy,
        }
        attribution_ambiguous = etiology_guard.get("status") == "ambiguous"
        if attribution_ambiguous:
            hemorrhage_4_strict = False
            hemorrhage_4_loose = False
            hemorrhage_4_area_weighted = False
            heavy_lesion_burden = False
            strong_severe_npdr_proxy = False
            severe_npdr_proxy = False
            severity_proxy = None
            severity_label = "indeterminate_hemorrhagic_macular_lesion"
            confidence = 0.0

        evidence = {
            "hemorrhage_count": hem,
            "hemorrhage_area_px": hem_area_px,
            "exudate_count": exu,
            "cotton_wool_spot_count": cws,
            "laser_spot_count": laser,
            "dr_lesion_count": total,
            "dr_lesion_coverage": coverage,
            "hemorrhage_quadrants": quadrants,
            "rule_4_hemorrhage_all_quadrants_strict": hemorrhage_4_strict,
            "rule_4_hemorrhage_all_quadrants_loose_proxy": hemorrhage_4_loose,
            "rule_4_hemorrhage_area_weighted_proxy": hemorrhage_4_area_weighted,
            "area_weighted_positive_quadrants": area_positive_quadrants,
            "area_weighted_quadrant_threshold_px": _AREA_WEIGHTED_QUADRANT_THRESHOLD_PX,
            "area_weighted_total_hemorrhage_threshold_px": _AREA_WEIGHTED_TOTAL_HEMORRHAGE_THRESHOLD_PX,
            "rule_2_venous_beading": "not_available",
            "rule_1_irma": "not_available",
            "heavy_lesion_burden_proxy": heavy_lesion_burden,
            "cws_with_hemorrhage_proxy": cws_with_hemorrhage,
            "strong_severe_npdr_proxy": strong_severe_npdr_proxy,
            "severe_npdr_proxy": severe_npdr_proxy,
            "severity_proxy": severity_proxy,
            "severity_label": severity_label,
            "eligible_for_dr_grading": not attribution_ambiguous,
            "etiology_guard": etiology_guard,
            "unadjusted_proxy_before_etiology_guard": unadjusted_proxy,
            "interpretation": (
                "DR severity is indeterminate because the DR and AMD "
                "hemorrhage heads substantially overlap. Do not report "
                "severe NPDR/PDR from these component counts; prioritise a "
                "macular neovascular differential and obtain OCT/angiography."
                if attribution_ambiguous else
                "The DR proxy is eligible for cautious clinical integration."
            ),
            "limitations": [
                "This is a severe-NPDR proxy, not a strict clinical 4-2-1 measurement.",
                "Venous beading and IRMA are not directly measured.",
                "Area-weighted proxy compensates for confluent hemorrhages that may be counted as one component.",
                "CWS with hemorrhage is reported as supportive evidence but does not by itself force severe NPDR.",
                "PDR signs must be assessed by cfp_pdr_cascade / cfp_dr_workup.",
            ],
        }

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions=evidence,
            confidence=confidence,
            undetermined=attribution_ambiguous,
            raw_output={"retsam": retsam.to_jsonable()},
            metadata={"method": "retsam_hemorrhage_quadrant_area_proxy_v3"},
        )
