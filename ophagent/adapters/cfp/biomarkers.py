"""
Cross-modal / derived-biomarker tools.

These tools build on top of one or more base adapters to expose higher-level,
clinically meaningful biomarkers — without the LLM having to chain low-level
detections itself.

  - cfp_disc_metrics : optic disc + cup geometry + CDR (uses retsam segmentation
                       or OD-detection as fallback)
  - cfp_dr_workup    : runs quality → PDR cascade → optional retsam → composite
  - cfp_glaucoma_workup : OD detection → disc crop → glaucoma signs → CDR
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, register, GLOBAL_REGISTRY


@register
class CFPDRWorkupAdapter(AdapterBase):
    """Convenience composite tool: full DR workup on a CFP image.

    Pipeline:
      1. cfp_eyeq → quality check (skip workup if Reject + low confidence)
      2. cfp_pdr_cascade → 4-class PDR + reasons
      3. (optional) cfp_clip_multi_disease → broader differential
    """

    metadata = ToolMetadata(
        name="cfp_dr_workup",
        modality="CFP",
        task="classification",
        description=(
            "Convenience composite tool — runs the full diabetic retinopathy "
            "workup on a CFP in one call: image quality → PDR cascade (no/inactive/"
            "active PDR + reasons) → broader CLIP-based differential. Returns a "
            "single structured summary with all findings + a top-line impression. "
            "Use this when the user asks 'evaluate this for DR' or when DR is "
            "independently supported. Do not use it as the primary classifier "
            "for an open-ended CFP differential."
        ),
        input_size=None,
        labels=[],
        confidence_threshold=0.5,
        limitations=[
            "Skips downstream tools if quality is Rejected with high confidence",
            "The PDR cascade can be confounded by non-DR hemorrhagic or atrophic "
            "pathology; high-confound outputs are suppressed rather than reported "
            "as PDR.",
        ],
        cost_class="medium",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        summary: dict[str, Any] = {}
        steps: list[dict] = []

        # Prefer EFIQA (lesion-safe, MIDL 2026) over bare EyeQ — EyeQ flags
        # heavy real pathology as "artefact" and we'd then skip the workup.
        efiqa = GLOBAL_REGISTRY.predict("cfp_efiqa", image_path)
        steps.append({"tool": "cfp_efiqa", "result": efiqa.to_jsonable()})
        eyeq = GLOBAL_REGISTRY.predict("cfp_eyeq", image_path)
        steps.append({"tool": "cfp_eyeq", "result": eyeq.to_jsonable()})
        summary["quality"] = (
            efiqa.predictions.get("quality") if efiqa.success
            else (eyeq.predictions.get("quality") if eyeq.success else "unknown")
        )
        summary["quality_confidence"] = efiqa.confidence or eyeq.confidence
        if eyeq.success and efiqa.success and \
           eyeq.predictions.get("is_rejected") and \
           efiqa.predictions.get("quality") != "Reject":
            summary["quality_note"] = (
                "EyeQ flagged 'Reject' but EFIQA (anatomy-aware) disagrees. "
                "Continuing analysis — EyeQ tends to misread heavy pathology "
                "as artefact."
            )

        # Surface quality status but DO NOT short-circuit — heavily-pathologic
        # images often get flagged as "Reject" by EyeQ while still containing
        # diagnosable findings. The downstream tools' confidences already
        # account for borderline quality; the LLM will combine quality + findings.
        both_reject = (
            eyeq.success and eyeq.predictions.get("is_rejected")
            and (eyeq.confidence or 0) > 0.7
            and (not efiqa.success or efiqa.predictions.get("quality") == "Reject")
        )
        if both_reject:
            summary["quality_warning"] = (
                "Both EyeQ and EFIQA flagged poor quality — findings below "
                "should be reviewed manually before clinical reliance."
            )

        pdr = GLOBAL_REGISTRY.predict("cfp_pdr_cascade", image_path)
        steps.append({"tool": "cfp_pdr_cascade", "result": pdr.to_jsonable()})
        if pdr.success:
            summary["pdr_category"] = pdr.predictions.get("category")
            summary["pdr_confidence"] = pdr.confidence
            summary["pdr_signs"] = pdr.predictions.get("predicted_reasons", [])

        # Always run the broader differential — needed for the confound check.
        clip = GLOBAL_REGISTRY.predict("cfp_clip_multi_disease", image_path)
        steps.append({"tool": "cfp_clip_multi_disease", "result": clip.to_jsonable()})
        if clip.success:
            summary["clip_top3"] = clip.predictions.get("top_3", [])

        # ── PDR-confound cross-check ───────────────────────────────────────
        # The PDR cascade was trained without 'other-pathology' negatives, so
        # large chorioretinal atrophy patches (pathological myopia), retinal
        # detachment, and similar can be misread as "inactive PDR with laser
        # scars". When the CLIP differential strongly suggests a non-DR
        # primary disease, we surface a `pdr_confounded_by` flag and downgrade
        # the PDR claim's perceived reliability — the LLM is instructed to
        # trust CLIP in that case.
        # RVO and DR commonly coexist (hypertensive diabetics often have both)
        # — we removed RVO from this list because PDR + RVO is real and the
        # PDR cascade is usually right in that scenario. Keep only the
        # **truly non-DR-related** classes that the PDR cascade can't
        # plausibly co-occur with.
        NON_DR_LABELS = {
            "Pathological myopia", "Myopic", "Pathologic Myopia",
            "Retinal detachment", "RD",
            "Cataract", "Suspected cataract",
            "Age-related macular degeneration", "AMD",
            "Retinitis pigmentosa", "RP",
        }
        confound_threshold = 0.30     # raised — RVO/RP co-occurring with DR
                                       # gave false positives at 0.25

        def _hit_label(h: dict) -> str:
            # CLIP returns top_3 entries as {"label_en", "label_zh", "probability"}
            return h.get("label_en") or h.get("label") or h.get("name") or ""

        pdr_confounded_by: list[dict] = []
        if clip.success and pdr.success:
            for hit in (clip.predictions.get("top_3") or []):
                label = _hit_label(hit)
                prob  = float(hit.get("probability", hit.get("prob", 0)) or 0)
                if label in NON_DR_LABELS and prob >= confound_threshold:
                    pdr_confounded_by.append({"label": label, "probability": prob})
            # Also flag if PDR returned "inactive PDR" but DR is NOT in the CLIP
            # top-3 at all — strong signal of misfire.
            dr_in_top3 = any(
                _hit_label(h).lower().startswith("dr")
                or "diabetic" in _hit_label(h).lower()
                for h in (clip.predictions.get("top_3") or [])
            )
            pdr_says_pdr = "PDR" in (pdr.predictions.get("category") or "")
            if pdr_says_pdr and pdr_confounded_by and not dr_in_top3:
                summary["pdr_confound_severity"] = "high"
            elif pdr_says_pdr and pdr_confounded_by:
                summary["pdr_confound_severity"] = "moderate"

        if pdr_confounded_by:
            summary["pdr_confounded_by"] = pdr_confounded_by
            summary["pdr_confound_note"] = (
                "PDR cascade was trained without non-DR-pathology negatives, "
                "so chorioretinal atrophy / staphyloma / RD can be misread "
                "as 'inactive PDR with laser scars'. The CLIP differential "
                "suggests another primary disease — trust CLIP top-1 over "
                "the PDR cascade unless the user explicitly has a diabetic "
                "history with prior PRP."
            )

        # A high-confound PDR result is outside the cascade's reliable
        # attribution domain. Preserve the raw output for audit, but suppress it
        # from the clinical summary rather than presenting a diluted PDR claim.
        high_pdr_confound = summary.get("pdr_confound_severity") == "high"
        if high_pdr_confound:
            summary["raw_pdr_category_before_confound_guard"] = summary.get(
                "pdr_category"
            )
            summary["raw_pdr_signs_before_confound_guard"] = summary.get(
                "pdr_signs", []
            )
            summary["pdr_category"] = "indeterminate_non_dr_confound"
            summary["pdr_signs"] = []
            summary["pdr_eligible_for_reporting"] = False
            summary["do_not_report_as_pdr"] = True
            summary["interpretation"] = (
                "The PDR cascade is strongly confounded by a non-DR primary "
                "pathology and cannot support a positive PDR diagnosis in this "
                "case. Retain the raw category for audit only and adjudicate the "
                "macular hemorrhagic differential with OCT and/or angiography."
            )
        else:
            summary["pdr_eligible_for_reporting"] = bool(pdr.success)
            summary["do_not_report_as_pdr"] = False

        # Composite confidence: penalise moderate confounding and abstain for
        # high confounding.
        pdr_conf_effective = pdr.confidence or 0
        if high_pdr_confound:
            pdr_conf_effective = 0.0
        elif pdr_confounded_by and pdr_conf_effective:
            pdr_conf_effective *= 0.5    # halve it so the planner notices
        confs = [c for c in [eyeq.confidence, pdr_conf_effective] if c is not None]
        composite_conf = min(confs) if confs else None

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions=summary,
            confidence=composite_conf,
            undetermined=high_pdr_confound,
            raw_output={"steps": steps},
        )


@register
class CFPCLIPEnsembleAdapter(AdapterBase):
    """Three-CLIP ensemble — ViLReF (Chinese, multi-template) + RetiZero
    (English LoRA + dictionary multi-template) + FLAIR (English ResNet).
    Each runs independently; their canonical-11-class projections are
    averaged into a fused probability vector.

    Use case: tiebreak for the PDR-confound check. When two or more CLIPs
    agree on a non-DR class, the verifier should treat the PDR cascade's
    'inactive PDR' verdict as a likely false positive."""

    metadata = ToolMetadata(
        name="cfp_clip_ensemble",
        modality="CFP",
        task="classification",
        description=(
            "Run THREE independent CFP CLIPs (ViLReF Chinese, RetiZero "
            "English LoRA, FLAIR English ResNet) and fuse their predictions "
            "in the canonical 11-class space (Normal / DR / AMD / Glaucoma / "
            "Pathological myopia / MH / ERM / Hypertensive / RVO / RD / "
            "Cataract). Returns per-CLIP top-3, the fused softmax, and an "
            "`agreement_level` flag (high / moderate / low). Call this when "
            "any single classifier returns moderate confidence (~0.3-0.7) — "
            "ensemble agreement is much more reliable than any one model."
        ),
        input_size=(224, 224),
        labels=[],
        confidence_threshold=0.4,
        limitations=[
            "Three-model load is heavier (~3 GB GPU memory total).",
            "RetiZero / FLAIR may be unavailable on systems where the FLAIR "
            "bundled FLAIR/RetiZero source trees are "
            "missing — the ensemble degrades gracefully (returns whichever "
            "models loaded).",
        ],
        cost_class="slow",
        source_dir="(composite)",
    )

    # 5-class space used by paired_600 + retizero canon mapping. ViLReF
    # already returns canonical names; RetiZero/FLAIR have predicted_class_canon.
    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        # ── V2 (OPH_DECISION_V2): native-vocab ensemble. Scores the canonical
        # conditions via each open-vocab CLIP's NATIVE labels + calibrated fusion.
        import os as _os
        if _os.environ.get("OPH_DECISION_V2", "0") == "1":
            try:
                return self._predict_native_v2(image_path)
            except Exception as _e:
                # never break the tool — fall back to the legacy ensemble
                pass
        votes: dict[str, dict[str, float]] = {}    # tool -> {class: prob}
        per_tool_top3: dict[str, list[dict]] = {}
        failed: list[str] = []
        for tool_name in ("cfp_clip_multi_disease", "cfp_retizero", "cfp_flair"):
            try:
                r = GLOBAL_REGISTRY.predict(tool_name, image_path)
            except Exception as e:
                failed.append(f"{tool_name}: {type(e).__name__}: {e}")
                continue
            if not r.success:
                failed.append(f"{tool_name}: {r.error}")
                continue
            # ViLReF uses `top_3` with label_en; RetiZero/FLAIR add `canon_top3`.
            if tool_name == "cfp_clip_multi_disease":
                top3 = r.predictions.get("top_3") or []
                probs = {h.get("label_en", ""): float(h.get("probability", 0))
                         for h in top3}
            else:
                top3 = r.predictions.get("canon_top3") or []
                probs = {h.get("label_en", ""): float(h.get("probability", 0))
                         for h in top3}
            votes[tool_name] = probs
            per_tool_top3[tool_name] = top3

        # Fuse: average across the canonical labels that ANY voter scored.
        all_labels = set()
        for d in votes.values():
            all_labels.update(d.keys())
        fused = {}
        if votes:
            for lab in all_labels:
                vals = [d.get(lab, 0.0) for d in votes.values()]
                fused[lab] = sum(vals) / len(votes)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        # Agreement level: high if top-1 of all three voters agrees
        agreement = "low"
        if votes:
            top1_per_tool = [
                max(d.items(), key=lambda kv: kv[1])[0]
                for d in votes.values() if d
            ]
            if top1_per_tool:
                most = max(set(top1_per_tool), key=top1_per_tool.count)
                consensus = top1_per_tool.count(most)
                if consensus == len(votes):
                    agreement = "high"
                elif consensus >= 2:
                    agreement = "moderate"

        return AdapterResult(
            success=bool(votes),
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            error=None if votes else "all CLIPs failed: " + "; ".join(failed),
            predictions={
                "fused_top3": [
                    {"label_en": lab, "probability": round(p, 4)}
                    for lab, p in ranked[:3]
                ],
                "fused_top1": ranked[0][0] if ranked else None,
                "fused_top1_probability": round(ranked[0][1], 4) if ranked else None,
                "per_tool_top3": per_tool_top3,
                "agreement_level": agreement,
                "voters": list(votes.keys()),
                "failed": failed,
            },
            confidence=round(ranked[0][1], 4) if ranked else 0.0,
        )

    def _predict_native_v2(self, image_path: str) -> AdapterResult:
        """Native-vocab ensemble (FLAIR + RetiZero open-vocab) over the native
        task canons + Normal, with per-disease calibrated fusion weights/thresholds.
        Returns the SAME schema (fused_top3/top1/agreement_level) + a calibrated
        `present` list."""
        from ._clip_native_score import native_ensemble
        from ._clip_native_vocab import TASK7, TASK7_TO_CANON
        targets = [TASK7_TO_CANON[c] for c in TASK7] + ["Normal"]
        ranked, present, fused, per_model_top1 = native_ensemble(image_path, targets)
        if not ranked:
            return AdapterResult(success=False, tool=self.metadata.name, modality="CFP",
                                 task="classification", error="native ensemble: no CLIP loaded")
        tops = [t for t in per_model_top1.values() if t]
        agreement = "low"
        if tops:
            most = max(set(tops), key=tops.count)
            n = tops.count(most)
            agreement = "high" if n == len(tops) and len(tops) > 1 else ("moderate" if n >= 2 else "low")
        return AdapterResult(
            success=True, tool=self.metadata.name, modality="CFP", task="classification",
            predictions={
                "fused_top3": [{"label_en": lab, "probability": round(p, 4)} for lab, p in ranked[:3]],
                "fused_top1": ranked[0][0], "fused_top1_probability": round(ranked[0][1], 4),
                "present_conditions": present,
                "all_scores": {lab: round(p, 4) for lab, p in ranked},
                "agreement_level": agreement,
                "voters": list(per_model_top1.keys()),
                "native_vocab": True,
            },
            confidence=round(ranked[0][1], 4),
        )


@register
class CFPGlaucomaWorkupAdapter(AdapterBase):
    """Convenience composite: full glaucoma workup on a CFP.

    Pipeline:
      1. cfp_od_detection → OD bounding box (required upstream)
      2. cfp_glaucoma → RG + 10 signs on disc crop
    """

    metadata = ToolMetadata(
        name="cfp_glaucoma_workup",
        modality="CFP",
        task="classification",
        description=(
            "Convenience composite — full glaucoma workup on a CFP: optic disc "
            "localisation followed by referable-glaucoma classification with "
            "10 morphological signs (RNFL defects, disc haemorrhage, notching, "
            "large cup, laminar dots). Returns a single summary; use this when "
            "the user asks about glaucoma."
        ),
        input_size=None,
        labels=[],
        confidence_threshold=0.5,
        limitations=[
            "Fails if optic disc cannot be localised",
        ],
        requires_tools=["cfp_od_detection", "cfp_glaucoma"],
        cost_class="medium",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        steps: list[dict] = []
        od = GLOBAL_REGISTRY.predict("cfp_od_detection", image_path)
        steps.append({"tool": "cfp_od_detection", "result": od.to_jsonable()})
        if not od.success or not od.predictions.get("has_od"):
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="CFP",
                task="classification",
                error="Optic disc not localised; glaucoma assessment skipped",
                raw_output={"steps": steps},
            )

        gla = GLOBAL_REGISTRY.predict("cfp_glaucoma", image_path)
        steps.append({"tool": "cfp_glaucoma", "result": gla.to_jsonable()})

        summary: dict[str, Any] = {
            "optic_disc_localised": True,
            "od_confidence": od.predictions.get("best", {}).get("OD", {}).get("confidence"),
        }
        if gla.success:
            summary["referable_glaucoma_probability"] = gla.predictions.get(
                "referable_glaucoma_probability"
            )
            summary["referable_glaucoma"] = gla.predictions.get("referable_glaucoma")
            summary["predicted_signs"] = gla.predictions.get("predicted_signs", [])
            summary["aux_signs"] = gla.predictions.get("auxiliary_signs", {})

        # ── Morphology fallback ────────────────────────────────────────────
        # `cfp_glaucoma` is calibrated on referable-glaucoma datasets and
        # **misses extreme cases** (e.g. CDR > 0.9) — the classifier sees an
        # OOD-looking disc and returns near-zero. To catch those, ask
        # retsam for the cup/disc segmentation and compute CDR directly. If
        # the morphometric CDR is large, override with "suspected glaucoma"
        # regardless of what the bespoke classifier said.
        morphology_cdr = None
        try:
            rs = GLOBAL_REGISTRY.predict("cfp_retsam_segmentation", image_path)
            steps.append({"tool": "cfp_retsam_segmentation", "result": rs.to_jsonable()})
            if rs.success:
                q = rs.predictions.get("quantitative") or {}
                # retsam exposes vCDR in TWO places (we accept either):
                #   q.top_line.vCDR              ← flat headline, current schema
                #   q.disc_cup.cdr.vertical      ← nested module summary
                # Older schemas also used q.disc_cup.{vCDR,vertical_CDR}.
                top_line = q.get("top_line") or {}
                dc = q.get("disc_cup") or {}
                cdr_nested = (dc.get("cdr") or {})

                candidates = [
                    top_line.get("vCDR"),
                    cdr_nested.get("vertical"),
                    dc.get("vCDR"), dc.get("vertical_CDR"),
                    dc.get("vcdr"), dc.get("vertical_cup_disc_ratio"),
                ]
                # Some older schemas also stash CDR under disc_cup.summary
                if isinstance(dc.get("summary"), dict):
                    s = dc["summary"]
                    candidates += [s.get("vCDR"), s.get("vertical_CDR"),
                                   s.get("vertical_cup_disc_ratio")]
                for v in candidates:
                    if v is not None:
                        try:
                            morphology_cdr = float(v)
                            break
                        except (TypeError, ValueError):
                            continue
        except Exception as _e:
            pass
        summary["morphology_vCDR"] = morphology_cdr
        # Two-tier morphology override:
        #   vCDR >= 0.55  → "suspected glaucoma" (referable, mid confidence)
        #   vCDR >= 0.70  → "highly suspicious"  (referable, high confidence)
        # The 0.55 lower bound catches cases like TRAIN000146 where the
        # actual retsam vCDR was 0.587 — the referable-glaucoma classifier
        # said 0.06% but the morphology clearly showed glaucomatous cupping
        # and the LLM was missing this signal entirely.
        if morphology_cdr is not None and morphology_cdr >= 0.55:
            summary["morphology_override"] = True
            tier = "highly suspicious" if morphology_cdr >= 0.70 else "suspected"
            summary["morphology_note"] = (
                f"vCDR={morphology_cdr:.2f} from retsam segmentation is "
                f"{tier} for glaucomatous cupping (>=0.55 threshold), even if "
                f"the referable-glaucoma classifier returned a low probability "
                f"(it can fail on extreme/atypical discs that fall outside its "
                f"training distribution). Treat as 'suspected glaucoma' until "
                f"clinically excluded (RNFL OCT + visual fields)."
            )
            # Boost composite confidence so the planner notices.
            # Confidence ramps from 0.55 at vCDR=0.55 up to 0.95 at vCDR>=0.82.
            summary["referable_glaucoma"] = True
            summary["composite_confidence"] = max(
                summary.get("referable_glaucoma_probability") or 0,
                min(0.95, 0.5 + (morphology_cdr - 0.55) * 1.5),
            )

        return AdapterResult(
            success=True, tool=self.metadata.name,
            modality="CFP", task="classification",
            predictions=summary,
            confidence=summary.get("composite_confidence") or gla.confidence,
            raw_output={"steps": steps},
        )


@register
class CFPQualityRobustAdapter(AdapterBase):
    """Lesion-aware CFP quality check.

    Strategy:
      1. Run `cfp_eyeq` (fast ResNet18) for first-pass verdict.
      2. If EyeQ says Reject OR low confidence, query a vision LLM with a
         lesion-aware prompt: "Is this image actually unusable due to artefact,
         OR are the dark/bright patches lesions that the model confused with
         artefact?" The LLM's verdict overrides EyeQ when they disagree.
      3. Returns the final verdict + both opinions for transparency.

    The LLM client is passed in via kwargs (the chat session injects it).
    """

    metadata = ToolMetadata(
        name="cfp_quality_robust",
        modality="CFP",
        task="quality",
        description=(
            "Lesion-aware colour fundus quality assessment. Combines the fast "
            "EyeQ classifier with a vision-LLM second opinion that distinguishes "
            "true artefact (motion, defocus, shadow) from large lesions the "
            "first model often confuses for artefact. PREFER this over plain "
            "`cfp_eyeq` whenever heavy pathology is plausible."
        ),
        input_size=None,
        labels=["Good", "Usable", "Reject"],
        confidence_threshold=0.5,
        limitations=[
            "Calls a vision LLM only when EyeQ rejects — costs ~$0.001/call",
            "If no LLM client is available, falls back to EyeQ verdict",
        ],
        cost_class="medium",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str,
                      llm_client=None, llm_model: str | None = None,
                      **_) -> AdapterResult:
        eyeq = GLOBAL_REGISTRY.predict("cfp_eyeq", image_path)
        if not eyeq.success:
            return eyeq  # propagate failure

        eyeq_quality = eyeq.predictions.get("quality", "unknown")
        eyeq_probs = eyeq.predictions.get("probabilities", {})

        # Cheap path: EyeQ confident NOT-Reject → trust it
        if eyeq_quality != "Reject":
            return AdapterResult(
                success=True,
                tool=self.metadata.name, modality="CFP", task="quality",
                predictions={
                    "quality": eyeq_quality,
                    "verdict_source": "eyeq",
                    "eyeq_probabilities": eyeq_probs,
                    "vision_llm_reviewed": False,
                    "quality_review_status": "not_required",
                    "is_usable": eyeq_quality != "Reject",
                },
                confidence=eyeq.confidence,
            )

        # EyeQ rejected — ask vision LLM if this is really artefact or lesions
        if llm_client is None:
            # No LLM available — return EyeQ but flag low confidence in verdict
            return AdapterResult(
                success=True,
                tool=self.metadata.name, modality="CFP", task="quality",
                predictions={
                    "quality": eyeq_quality,
                    "verdict_source": "eyeq_only",
                    "eyeq_probabilities": eyeq_probs,
                    "vision_llm_reviewed": False,
                    "quality_review_status": "unadjudicated",
                    "warning": "EyeQ rejected the image but no LLM was "
                               "available to verify whether the rejection is "
                               "due to true artefact or lesions misclassified "
                               "as artefact. Verdict may be unreliable.",
                    "is_usable": False,
                },
                confidence=eyeq.confidence,
            )

        # Ask vision LLM
        import base64
        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        prompt = (
            "A fast quality classifier flagged this colour fundus photograph "
            "as 'Reject' (unusable). However, that model is known to confuse "
            "large lesions (haemorrhages, exudates, neovascular complexes, "
            "drusen, geographic atrophy) with imaging artefact. "
            "Look at this image and answer in EXACTLY this format:\n"
            "VERDICT: <one of: USABLE, REJECT>\n"
            "REASON: <one sentence>\n"
            "Use USABLE if the dark/bright patches look like clinical lesions "
            "and the optic disc + vessels are still visible. Use REJECT only "
            "if the image really is unreadable due to motion blur, severe "
            "defocus, large dark shadow covering the macula, or extreme "
            "over/under-exposure."
        )

        try:
            resp = llm_client.chat.completions.create(
                model=llm_model,
                max_tokens=160,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
            )
            txt = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return AdapterResult(
                success=True,
                tool=self.metadata.name, modality="CFP", task="quality",
                predictions={
                    "quality": eyeq_quality,
                    "verdict_source": "eyeq_llm_failed",
                    "eyeq_probabilities": eyeq_probs,
                    "vision_llm_reviewed": False,
                    "quality_review_status": "failed",
                    "llm_error": str(e),
                    "is_usable": False,
                },
                confidence=eyeq.confidence,
            )

        # A text-only or misrouted endpoint may accept the multimodal payload
        # yet explicitly say that no image was received. Never parse that as a
        # clinical REJECT confirmation.
        missing_image_markers = (
            "no image was provided",
            "image was not provided",
            "cannot see the image",
            "can't see the image",
            "unable to view the image",
            "please include the image",
            "please upload the image",
        )
        if any(marker in txt.lower() for marker in missing_image_markers):
            return AdapterResult(
                success=True,
                tool=self.metadata.name, modality="CFP", task="quality",
                predictions={
                    "quality": eyeq_quality,
                    "verdict_source": "eyeq_vision_unavailable",
                    "eyeq_probabilities": eyeq_probs,
                    "vision_llm_reviewed": False,
                    "quality_review_status": "unadjudicated",
                    "vision_llm_reason": txt,
                    "warning": (
                        "The configured LLM did not receive the image; its "
                        "response was excluded from quality adjudication."
                    ),
                    "is_usable": False,
                },
                confidence=eyeq.confidence,
            )

        # Parse the LLM's reply
        verdict: str | None = None
        reason = txt
        for line in txt.splitlines():
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                if "USABLE" in v:
                    verdict = "USABLE"
                elif "REJECT" in v:
                    verdict = "REJECT"
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        if verdict is None:
            return AdapterResult(
                success=True,
                tool=self.metadata.name, modality="CFP", task="quality",
                predictions={
                    "quality": eyeq_quality,
                    "verdict_source": "eyeq_llm_unparsed",
                    "eyeq_probabilities": eyeq_probs,
                    "vision_llm_reviewed": False,
                    "quality_review_status": "unadjudicated",
                    "vision_llm_reason": txt,
                    "warning": (
                        "The visual quality response did not contain a valid "
                        "VERDICT field and was excluded."
                    ),
                    "is_usable": False,
                },
                confidence=eyeq.confidence,
            )

        final_quality = "Usable" if verdict == "USABLE" else "Reject"
        return AdapterResult(
            success=True,
            tool=self.metadata.name, modality="CFP", task="quality",
            predictions={
                "quality": final_quality,
                "verdict_source": "vision_llm_override" if verdict == "USABLE" else "vision_llm_confirmed",
                "eyeq_quality": eyeq_quality,
                "eyeq_probabilities": eyeq_probs,
                "vision_llm_reviewed": True,
                "quality_review_status": "completed",
                "vision_llm_verdict": verdict,
                "vision_llm_reason": reason,
                "is_usable": (verdict == "USABLE"),
                "note": (
                    "EyeQ rejected the image, but the vision LLM identified "
                    "the dark/bright regions as clinical lesions, not artefact. "
                    "Treating as usable for downstream analysis."
                    if verdict == "USABLE" else
                    "Both EyeQ and the vision LLM agree the image is unusable."
                ),
            },
            confidence=eyeq.confidence,
        )


@register
class JointCFPOCTAdapter(AdapterBase):
    """Cross-modal adapter: takes a CFP and an OCT, runs both pipelines,
    and produces a joint interpretation."""

    metadata = ToolMetadata(
        name="cross_cfp_oct",
        modality="multi",
        task="classification",
        description=(
            "Joint analysis of paired CFP + OCT images of the same eye. Runs "
            "the CFP DR workup on the fundus photo and the FMUE 16-class "
            "classifier on the OCT B-scan, then reports both findings with "
            "an internal consistency check (e.g. CFP says DR & OCT says DME → "
            "consistent; CFP says normal & OCT says nAMD → flag conflict)."
        ),
        input_size=None,
        labels=[],
        confidence_threshold=0.5,
        cost_class="slow",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, oct_path: str = "", **_) -> AdapterResult:
        if not oct_path:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="classification",
                error="cross_cfp_oct requires both image_path (CFP) and oct_path",
            )

        cfp_r = GLOBAL_REGISTRY.predict("cfp_dr_workup", image_path)
        oct_r = GLOBAL_REGISTRY.predict("oct_fmue_16class", oct_path)

        # consistency analysis
        cfp_dx = cfp_r.predictions.get("pdr_category", "")
        oct_dx = oct_r.predictions.get("predicted_class", "")
        conflict = False
        notes: list[str] = []
        if cfp_dx and oct_dx:
            cfp_is_dr = "PDR" in cfp_dx or "糖尿病" in cfp_dx
            oct_dr_family = oct_dx in {"DME", "DR_without_ME"}
            oct_amd_family = oct_dx in {"dAMD", "nAMD", "PCV", "mCNV"}
            if cfp_is_dr and oct_amd_family:
                conflict = True
                notes.append(
                    f"CFP suggests DR/PDR but OCT predicts {oct_dx} (AMD-family). "
                    "Co-existence is possible but warrants careful review."
                )
            elif not cfp_is_dr and oct_dr_family:
                conflict = True
                notes.append(
                    f"OCT predicts {oct_dx} but CFP did not flag DR features. "
                    "Consider re-checking CFP grader, or that DR is mild."
                )

        return AdapterResult(
            success=True, tool=self.metadata.name, modality="multi",
            task="classification",
            predictions={
                "cfp_findings": cfp_r.predictions if cfp_r.success else {"error": cfp_r.error},
                "oct_findings": oct_r.predictions if oct_r.success else {"error": oct_r.error},
                "conflict_flag": conflict,
                "consistency_notes": notes,
            },
            confidence=min([c for c in [cfp_r.confidence, oct_r.confidence] if c is not None] + [1.0]),
            raw_output={
                "cfp": cfp_r.to_jsonable(),
                "oct": oct_r.to_jsonable(),
            },
        )


# ── CFP + FFA paired analysis (for Paired_600_CFPFFA_MPOS) ───────────────────
# Maps between the FFA classifier's 9 merged groups and the paired-dataset
# 5-way clinical labels.
_PAIRED_5CLASS = ["Normal", "DR", "RVO", "AMD", "CSC"]
_FFA_MERGED_TO_PAIRED = {
    # FFA classifier label  →  Paired_600 label
    "DR": "DR", "RVO": "RVO", "AMD": "AMD", "PCV": "AMD",
    "CSC": "CSC", "Pathologic Myopia": "AMD",
    "Macular Disorders": "Normal", "Uveitis": "Normal", "Other": "Normal",
}
_CFP_CLIP_TO_PAIRED = {
    # CFP CLIP label → Paired_600 label   (best-effort coarsening)
    "Normal": "Normal", "DR": "DR", "AMD": "AMD", "RVO": "RVO",
    "Glaucoma": "Normal", "ERM": "Normal", "MH": "Normal",
    "RP": "Normal", "Myopic": "AMD", "CSC": "CSC", "Other": "Normal",
}


@register
class JointCFPFFAAdapter(AdapterBase):
    """Cross-modal CFP + FFA — fuse the two single-modality classifiers
    into one 5-class verdict matching the Paired_600 benchmark schema."""

    metadata = ToolMetadata(
        name="cross_cfp_ffa",
        modality="multi",
        task="classification",
        description=(
            "Joint analysis of paired CFP + FFA of the same eye. Runs the CFP "
            "CLIP differential and the FFA multi-task classifier independently, "
            "then fuses them into a single verdict over the 5 clinical groups "
            "used in the Paired_600 benchmark (Normal/DR/RVO/AMD/CSC). FFA is "
            "weighted higher (the angiographic findings are usually more "
            "diagnostic than colour photography for these conditions). Returns "
            "per-modality findings, the fused probability vector, and a "
            "consistency flag when the two modalities disagree."
        ),
        input_size=None,
        labels=_PAIRED_5CLASS,
        confidence_threshold=0.5,
        cost_class="slow",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, ffa_path: str = "", **_) -> AdapterResult:
        if not ffa_path:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="classification",
                error="cross_cfp_ffa requires both image_path (CFP) and ffa_path",
            )

        cfp_r = GLOBAL_REGISTRY.predict("cfp_clip_multi_disease", image_path)
        ffa_r = GLOBAL_REGISTRY.predict("ffa_classification", ffa_path)

        # Project both into the 5-class paired-dataset label space.
        def _to_5class(probs_dict: dict, mapping: dict) -> dict:
            out = {c: 0.0 for c in _PAIRED_5CLASS}
            for src_label, p in probs_dict.items():
                dst = mapping.get(src_label)
                if dst is None:
                    continue
                # multi-label semantics: keep the max contribution per dst label
                out[dst] = max(out[dst], float(p))
            return out

        cfp_5 = _to_5class(
            cfp_r.predictions.get("probabilities", {}) if cfp_r.success else {},
            _CFP_CLIP_TO_PAIRED,
        )
        ffa_5 = _to_5class(
            ffa_r.predictions.get("merged_probabilities", {}) if ffa_r.success else {},
            _FFA_MERGED_TO_PAIRED,
        )

        # Weighted fuse: FFA is more diagnostic for these specific conditions.
        W_FFA, W_CFP = 0.65, 0.35
        fused = {
            c: round(W_FFA * ffa_5[c] + W_CFP * cfp_5[c], 4)
            for c in _PAIRED_5CLASS
        }
        # Renormalise so it reads like a probability vector
        s = sum(fused.values()) or 1.0
        fused = {c: round(v / s, 4) for c, v in fused.items()}

        primary = max(fused.items(), key=lambda kv: kv[1])
        cfp_top = max(cfp_5.items(), key=lambda kv: kv[1])[0] if cfp_5 else None
        ffa_top = max(ffa_5.items(), key=lambda kv: kv[1])[0] if ffa_5 else None
        conflict = (cfp_top is not None and ffa_top is not None and cfp_top != ffa_top)

        notes: list[str] = []
        if conflict:
            notes.append(
                f"CFP top-1 = {cfp_top}, FFA top-1 = {ffa_top} — disagreement. "
                "FFA evidence weighted higher; check both images manually."
            )
        if ffa_r.success and ffa_r.confidence and ffa_r.confidence < 0.5:
            notes.append("FFA confidence < 0.5 — verdict is tentative.")

        return AdapterResult(
            success=True, tool=self.metadata.name, modality="multi",
            task="classification",
            predictions={
                "primary_diagnosis": primary[0],
                "primary_probability": primary[1],
                "fused_probabilities": fused,
                "cfp_5class": cfp_5,
                "ffa_5class": ffa_5,
                "cfp_top1": cfp_top,
                "ffa_top1": ffa_top,
                "conflict_flag": conflict,
                "notes": notes,
            },
            confidence=primary[1],
            raw_output={
                "cfp": cfp_r.to_jsonable(),
                "ffa": ffa_r.to_jsonable(),
            },
        )
