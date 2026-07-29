"""Opt-in prompt profiles for bounded evaluation workflows.

The default OphAgent profile remains unchanged. ``compact-mac`` is a
diagnostic-only profile for the fixed six-label MAC CFP benchmark, while
``task-focused-dr`` is a single-image ICDR grading profile used for matched
Messidor2 architecture and cost analyses. Both keep the existing tool
implementations and verifier logic intact; only unrelated global prose,
schemas, and oversized result payloads are removed from hosted LLM requests.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


STANDARD_PROFILE = "standard"
COMPACT_MAC_PROFILE = "compact-mac"
TASK_FOCUSED_DR_PROFILE = "task-focused-dr"
SUPPORTED_PROFILES = (
    STANDARD_PROFILE,
    COMPACT_MAC_PROFILE,
    TASK_FOCUSED_DR_PROFILE,
)


COMPACT_MAC_SYSTEM_PROMPT = """\
You are OphAgent running the fixed MAC colour-fundus multi-label screening
protocol. Use the callable tools as a Planner-Executor-Verifier system.

Rules:
1. Ground every condition in returned tool evidence. Never invent findings,
   fields, probabilities, measurements, or successful tool execution.
2. Respect each tool's confidence, undetermined flag, limitations, image
   quality, and explicit conflicts. A quality result alone is not a diagnosis.
3. Treat requested labels independently; zero, one, or several may be present.
4. Batch independent calls. Use targeted escalation only when it can resolve a
   genuine conflict or fill missing evidence. Do not repeat a completed tool.
5. Call verify_findings after collecting sufficient evidence. If evidence
   remains inadequate or conflicting, preserve uncertainty instead of forcing
   a positive or negative label.
6. Follow the user's exact machine-readable output contract and output no
   unrequested disease labels.

The callable schemas are the complete tool list for this profile. Do not infer
capabilities that are not represented there.
"""


TASK_FOCUSED_DR_SYSTEM_PROMPT = """\
You are OphAgent running a fixed single-image colour-fundus diabetic
retinopathy grading protocol on the ICDR 0--4 scale. Use the callable tools as
a Planner-Executor-Verifier system when those roles are available.

Rules:
1. Ground the final grade in returned tool evidence. Never invent lesions,
   measurements, probabilities, signs, or successful tool execution.
2. The benchmark images have already passed dataset-level image curation; no
   separate quality tool is exposed in this task-focused profile.
3. Gather the four complementary evidence streams in one parallel batch:
   cfp_dr_421_assessment, cfp_pdr_cascade, and cfp_dynamic_clip. Do not repeat
   any of them; also use cfp_clip_ensemble for lesion-name support and
   non-DR confound checks.
4. Use the DR 4-2-1 assessment as the primary non-proliferative lesion-burden
   signal and the PDR cascade as the primary proliferative-disease signal.
   An active- or inactive-PDR category with returned PDR signs supports grade
   4 even when the tool marks the result for review; that flag lowers
   confidence but does not turn positive PDR evidence into absence.
5. Dynamic CLIP provides an independent severity prior. ReT-SAM does not
   directly separate microaneurysms, so a zero lesion proxy must not veto a
   specific mild-NPDR signal. When the 4-2-1 proxy is grade 0 and Dynamic CLIP
   favours grade 1, or the ensemble specifically identifies microaneurysms,
   grade 1 is supported. For grades 1--3, use a specific Dynamic CLIP
   adjacent-grade signal to examine a one-grade conflict, but do not let a
   weak or distant prior erase strong lesion evidence.
6. Grade 4 requires PDR or treated-PDR evidence. Otherwise map lesion burden
   to grades 0--3 and preserve uncertainty where key signs are not measured.
7. Call verify_findings exactly once after the evidence batch when that tool
   is available. Do not repeat the verifier or a completed evidence tool.
8. Follow the user's exact machine-readable output contract.

The callable schemas are the complete tool list for this profile. Do not infer
capabilities that are not represented there.
"""


# This set covers every diagnostic tool used by the completed 724-case MEDIUM
# and HIGH MAC runs, plus the disease-specific DR fallback.  Composite tools
# are preferred over exposing their redundant internal components.
_COMPACT_MAC_TOOLS = frozenset(
    {
        "detect_modality",
        "cfp_efiqa",
        "cfp_eyeq",
        "cfp_quality_robust",
        "cfp_clip_multi_disease",
        "cfp_clip_ensemble",
        "cfp_retsam_segmentation",
        "cfp_dr_workup",
        "cfp_glaucoma_workup",
        "cfp_od_detection",
        "cfp_paired5",
        "vision_impression",
        "verify_findings",
    }
)

_REQUIRED_MAC_TOOLS = frozenset(
    {
        "cfp_efiqa",
        "cfp_clip_multi_disease",
        "cfp_retsam_segmentation",
        "verify_findings",
    }
)


_TASK_FOCUSED_DR_TOOLS = frozenset(
    {
        "cfp_dr_421_assessment",
        "cfp_pdr_cascade",
        "cfp_dynamic_clip",
        "cfp_clip_ensemble",
        "verify_findings",
    }
)

_REQUIRED_DR_TOOLS = frozenset(
    {
        "cfp_dr_421_assessment",
        "cfp_pdr_cascade",
        "cfp_dynamic_clip",
        "verify_findings",
    }
)


def normalize_prompt_profile(value: str | None) -> str:
    """Return a canonical prompt-profile name or raise on unknown input."""
    profile = (value or STANDARD_PROFILE).strip().lower().replace("_", "-")
    if profile == "compact":
        profile = COMPACT_MAC_PROFILE
    if profile in {"dr", "dr-grading", "task-focused-dr-grading"}:
        profile = TASK_FOCUSED_DR_PROFILE
    if profile not in SUPPORTED_PROFILES:
        choices = ", ".join(SUPPORTED_PROFILES)
        raise ValueError(f"unknown prompt profile {value!r}; choose one of: {choices}")
    return profile


def system_prompt_for_profile(profile: str) -> str:
    """Return a focused base prompt; standard is owned by oph_session.py."""
    profile = normalize_prompt_profile(profile)
    prompts = {
        COMPACT_MAC_PROFILE: COMPACT_MAC_SYSTEM_PROMPT,
        TASK_FOCUSED_DR_PROFILE: TASK_FOCUSED_DR_SYSTEM_PROMPT,
    }
    try:
        return prompts[profile]
    except KeyError as exc:
        raise ValueError(
            "the standard system prompt is defined in oph_session.py"
        ) from exc


def tool_schemas_for_profile(
    profile: str,
    schemas: Iterable[dict[str, Any]],
    *,
    attached_modalities: Iterable[str],
    multimodal: bool,
) -> list[dict[str, Any]]:
    """Select schemas without rewriting any selected schema's semantics.

    Focused profiles deliberately fail closed outside a single CFP workflow.
    This prevents a cost profile from silently removing capabilities in OCT,
    UWF, FFA, or multimodal clinical use.
    """
    profile = normalize_prompt_profile(profile)
    schema_list = list(schemas)
    if profile == STANDARD_PROFILE:
        return schema_list

    modalities = {
        str(modality).upper().split(":", 1)[0]
        for modality in attached_modalities
        if modality
    }
    if multimodal or modalities != {"CFP"}:
        raise ValueError(
            "compact-mac requires exactly one attached modality: CFP; "
            f"received {sorted(modalities) or ['unknown']}"
        )

    selected_names = (
        _COMPACT_MAC_TOOLS
        if profile == COMPACT_MAC_PROFILE
        else _TASK_FOCUSED_DR_TOOLS
    )
    required_names = (
        _REQUIRED_MAC_TOOLS
        if profile == COMPACT_MAC_PROFILE
        else _REQUIRED_DR_TOOLS
    )
    selected = [
        schema
        for schema in schema_list
        if ((schema.get("function") or {}).get("name") in selected_names)
    ]
    available = {
        (schema.get("function") or {}).get("name")
        for schema in selected
    }
    missing = sorted(required_names - available)
    if missing:
        raise RuntimeError(
            f"{profile} cannot start because required tools are unavailable: "
            + ", ".join(missing)
        )
    return selected


def compact_mac_tool_names() -> tuple[str, ...]:
    """Expose the profile contract for tests and reproducibility reports."""
    return tuple(sorted(_COMPACT_MAC_TOOLS))


def task_focused_dr_tool_names() -> tuple[str, ...]:
    """Expose the ICDR task-focused profile contract for reproducibility."""
    return tuple(sorted(_TASK_FOCUSED_DR_TOOLS))


def tool_result_for_profile(
    profile: str,
    tool_name: str,
    result: Any,
) -> Any:
    """Project a full tool result into the evidence needed by the LLM.

    Full results remain in ``OphContext.analyses`` and UI events.  The compact
    projection only controls what is re-sent to the hosted LLM on later turns.
    This removes file paths, masks, overlays, rendered markdown duplicates,
    model metadata, and debate transcripts without changing diagnostic values.
    """
    profile = normalize_prompt_profile(profile)
    if profile == STANDARD_PROFILE or not isinstance(result, dict):
        return result

    common_keys = (
        "success",
        "confidence",
        "undetermined",
        "error",
        "policy_skipped",
        "reason",
        "hint",
    )
    compact = {key: result[key] for key in common_keys if key in result}

    if tool_name == "detect_modality":
        for key in ("modality", "filename_hint", "cnn_hint", "llm_verdict", "llm_used"):
            if key in result:
                compact[key] = result[key]
        return compact

    if tool_name == "cfp_retsam_segmentation":
        if "llm_headline" in result:
            compact["llm_headline"] = result["llm_headline"]
        return compact

    if tool_name == "cfp_dr_421_assessment":
        predictions = result.get("predictions") or {}
        keep = (
            "hemorrhage_count",
            "hemorrhage_area_px",
            "exudate_count",
            "cotton_wool_spot_count",
            "laser_spot_count",
            "dr_lesion_count",
            "rule_4_hemorrhage_all_quadrants_strict",
            "rule_4_hemorrhage_all_quadrants_loose_proxy",
            "rule_4_hemorrhage_area_weighted_proxy",
            "heavy_lesion_burden_proxy",
            "strong_severe_npdr_proxy",
            "severe_npdr_proxy",
            "severity_proxy",
            "severity_label",
            "limitations",
        )
        compact["predictions"] = {
            key: predictions[key] for key in keep if key in predictions
        }
        return compact

    if tool_name == "cfp_pdr_cascade":
        predictions = result.get("predictions") or {}
        keep = (
            "category_en",
            "probabilities_en",
            "needs_review",
            "active_reasons",
            "predicted_reasons",
            "has_active_signs",
            "has_inactive_signs",
            "mixed_pattern",
        )
        compact["predictions"] = {
            key: predictions[key] for key in keep if key in predictions
        }
        return compact

    if tool_name == "cfp_dynamic_clip":
        predictions = result.get("predictions") or {}
        topk = []
        for item in predictions.get("fused_topk") or []:
            if not isinstance(item, dict):
                continue
            topk.append({
                key: item[key]
                for key in ("label", "score", "model_scores")
                if key in item
            })
        compact["predictions"] = {
            "task_hint": predictions.get("task_hint"),
            "models_used": predictions.get("models_used"),
            "fused_topk": topk,
        }
        return compact

    if tool_name == "vision_impression":
        for key in (
            "reliability",
            "stage1_morphology",
            "stage2_differential",
            "stage1_validation",
        ):
            if key in result:
                compact[key] = result[key]
        return compact

    if tool_name == "verify_findings":
        for key in (
            "status",
            "n_tools_run",
            "issues",
            "warnings",
            "evidence",
            "verify_passed",
            "next_actions",
            "recommendation",
        ):
            if key in result:
                compact[key] = result[key]
        for review_key in ("independent_review", "debate_review"):
            review = result.get(review_key)
            if not isinstance(review, dict):
                continue
            compact[review_key] = {
                key: review[key]
                for key in ("final_diagnosis", "resolved", "reason", "request_tool")
                if key in review
            }
        return compact

    predictions = result.get("predictions")
    if predictions is not None:
        compact["predictions"] = predictions
    if "llm_headline" in result:
        compact["llm_headline"] = result["llm_headline"]
    return compact
