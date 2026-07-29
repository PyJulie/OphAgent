"""
Vision-only stage1 prompts for OPHTHALMOLOGIC_OTHER modalities.

When the modality detector returns `OPHTHALMOLOGIC_OTHER:<name>`, the chat
session enters a degraded "vision-only" mode: no trained classifiers,
no segmentation, no verifier loop — only a single structured impression
from the vision LLM, formatted with a modality-tailored JSON schema.

The output is wrapped in `===VISION_ONLY_IMPRESSION===` (NOT `===FINAL===`)
so it is structurally distinct from a normal in-scope diagnosis and the
downstream UI / parser cannot confuse it with a full-pipeline result.

Public entry points:
    SCHEMAS                       # dict[str, dict]
    SYSTEM_PROMPTS                # dict[str, str]
    get_schema(sub_label) -> dict # falls back to generic_ophth
    get_system_prompt(sub_label) -> str
    build_user_prompt(sub_label, optional_user_question) -> str

Hand-curated entries (best fidelity):
    visual_field, octa, faf, icga, asoct, slit_lamp, bscan_us, topography

Fallback for unrecognised eye imagery:
    unknown_ophth
"""
from __future__ import annotations

import json
from typing import Any


# ─── Shared header used by every vision-only prompt ───────────────────────
_SHARED_PREAMBLE = (
    "You are running in VISION-ONLY MODE. No trained classifier or "
    "segmentation model is available for this imaging modality — your "
    "output is the LLM's gestalt impression based solely on what is "
    "visible in the image.\n\n"
    "IMPORTANT constraints:\n"
    "  - You MUST NOT emit a single confident top-1 diagnosis. Instead, "
    "list 2-4 differential considerations ranked by likelihood.\n"
    "  - You MUST mark image-quality concerns honestly. If the image is "
    "uninterpretable, set image_quality='uninterpretable' and refuse to "
    "speculate.\n"
    "  - Output ONLY the JSON object specified by the schema below — no "
    "free-text outside the JSON.\n"
    "  - This output will be wrapped in a `===VISION_ONLY_IMPRESSION===` "
    "block and is structurally distinct from a full-pipeline diagnostic "
    "call. The downstream system relies on this distinction for clinical "
    "safety. Do not output `===FINAL===` or a top1/confidence field.\n\n"
)


# ─── Per-modality schemas ─────────────────────────────────────────────────

VISUAL_FIELD_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["visual_field", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "test_type": {"enum": ["24_2", "30_2", "10_2", "60_4",
                            "esterman", "unknown_perimetry_type"]},
    "laterality": {"enum": ["OD", "OS", "both", "unclear"]},
    "reliability_concerns": {
        "type": "list",
        "items": {"enum": ["high_false_positive", "high_false_negative",
                            "high_fixation_loss", "none_visible"]},
    },
    "defect_pattern": {
        "type": "list",
        "items": {"enum": [
            "unremarkable", "isolated_paracentral_scotoma",
            "arcuate_or_Bjerrum", "nasal_step", "altitudinal",
            "centrocecal", "hemianopic_homonymous",
            "hemianopic_bitemporal", "generalized_constriction",
            "tunnel_vision", "scattered_unreliable"]},
    },
    "severity_estimate": {"enum": ["unremarkable", "mild", "moderate",
                                     "advanced", "cannot_assess"]},
    "hemisphere_involvement": {"enum": ["superior", "inferior", "both",
                                          "temporal", "nasal", "none"]},
    "MD_value_visible": {"type": "string",
                          "note": "Mean Deviation in dB if printed on image, else 'not_visible'"},
    "PSD_value_visible": {"type": "string",
                          "note": "Pattern Standard Deviation in dB if printed, else 'not_visible'"},
    "differential_considerations": {
        # Multi-label differential — at most 4 entries ordered by likelihood.
        "type": "list",
        "items": {"enum": [
            "glaucoma", "anterior_ischemic_optic_neuropathy",
            "compressive_optic_neuropathy", "optic_neuritis",
            "retinitis_pigmentosa", "branch_retinal_artery_occlusion",
            "branch_retinal_vein_occlusion", "retinal_detachment",
            "chiasmal_lesion", "post_chiasmal_lesion",
            "functional_or_unreliable", "no_specific_pattern"]},
    },
    "narrative": {"type": "string",
                   "note": "1-2 sentence clinical impression in plain English"},
}

OCTA_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["oct_angiography", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "scan_layer": {"enum": ["superficial_capillary_plexus",
                              "deep_capillary_plexus", "choriocapillaris",
                              "outer_retina", "vitreoretinal_interface",
                              "en_face_full_thickness", "unknown_layer"]},
    "FAZ_appearance": {"enum": ["normal", "enlarged", "distorted",
                                  "absent_or_filled_in", "not_clearly_visible"]},
    "vessel_density_visual": {"enum": ["normal", "reduced_focal",
                                         "reduced_widespread", "increased",
                                         "cannot_assess"]},
    "non_perfusion_areas": {"enum": ["none", "focal_small", "focal_large",
                                       "widespread", "cannot_assess"]},
    "neovascular_complex": {"enum": ["none", "intra_retinal_IRMA_like",
                                      "preretinal_or_NV", "subretinal_or_CNV",
                                      "cannot_assess"]},
    "artifact_concerns": {
        "type": "list",
        "items": {"enum": ["projection_artifact_into_deeper_layer",
                            "motion_artifact", "segmentation_error",
                            "shadowing", "none_visible"]},
    },
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["diabetic_retinopathy",
                            "retinal_vein_occlusion",
                            "neovascular_AMD_with_CNV",
                            "myopic_CNV", "vascular_anomaly",
                            "macular_telangiectasia",
                            "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

FAF_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["fundus_autofluorescence", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "background_AF_pattern": {"enum": ["normal", "diffusely_increased",
                                         "diffusely_decreased", "mottled",
                                         "cannot_assess"]},
    "hyperAF_pattern": {"enum": ["absent", "focal", "ring_around_atrophy",
                                   "multifocal_speckled", "banded_at_macula",
                                   "cannot_assess"]},
    "hypoAF_pattern": {"enum": ["absent", "focal", "geographic_atrophy_like",
                                  "patchy_RPE_loss", "macular_dropout",
                                  "cannot_assess"]},
    "macular_involvement": {"enum": ["yes", "no", "partial", "cannot_assess"]},
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["dry_AMD_with_geographic_atrophy",
                            "early_intermediate_AMD",
                            "Stargardt_disease",
                            "retinitis_pigmentosa",
                            "pattern_dystrophy",
                            "central_serous_residual",
                            "hydroxychloroquine_toxicity",
                            "white_dot_syndrome",
                            "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

ICGA_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["icga", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "phase_visible": {"enum": ["early", "mid", "late", "unknown_phase"]},
    "polypoidal_lesions": {"enum": ["present", "suspected", "absent"]},
    "branching_vascular_network": {"enum": ["present", "suspected", "absent"]},
    "hyperfluorescent_plaque": {"enum": ["present", "absent"]},
    "hypofluorescent_dark_dots": {"enum": ["present", "absent"]},
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["polypoidal_choroidal_vasculopathy",
                            "neovascular_AMD",
                            "central_serous_chorioretinopathy",
                            "choroidal_haemangioma",
                            "vogt_koyanagi_harada",
                            "uveal_effusion_syndrome",
                            "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

ASOCT_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["anterior_segment_oct", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "primary_structures_visible": {
        "type": "list",
        "items": {"enum": ["cornea", "anterior_chamber",
                            "iridocorneal_angle", "iris", "lens", "ciliary_body"]},
    },
    "corneal_findings": {"enum": ["normal_thickness_and_curvature",
                                    "thinned", "edematous_thickened",
                                    "scarred", "ectatic_keratoconus_like",
                                    "post_surgical", "cannot_assess"]},
    "anterior_chamber_depth": {"enum": ["normal", "shallow", "deep", "cannot_assess"]},
    "angle_status": {"enum": ["open", "narrow", "closed", "cannot_assess"]},
    "lens_findings": {"enum": ["clear", "cataract_visible",
                                 "pseudophakic_IOL", "aphakic", "cannot_assess"]},
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["primary_angle_closure_glaucoma",
                            "keratoconus", "corneal_oedema_or_decompensation",
                            "iridocorneal_endothelial_syndrome",
                            "post_cataract_surgery_normal",
                            "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

SLIT_LAMP_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["slit_lamp_photograph", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "laterality": {"enum": ["OD", "OS", "unclear"]},
    "illumination_technique": {"enum": ["diffuse", "slit_beam", "retro_illumination",
                                          "cobalt_blue", "unknown"]},
    "conjunctiva_findings": {"enum": ["normal", "injected_hyperaemic",
                                        "chemotic", "subconjunctival_haemorrhage",
                                        "pterygium", "cannot_assess"]},
    "cornea_findings": {"enum": ["clear", "epithelial_defect_or_abrasion",
                                   "infiltrate_or_ulcer", "scar_or_opacity",
                                   "oedema", "neovascularisation",
                                   "post_keratoplasty", "cannot_assess"]},
    "anterior_chamber_findings": {"enum": ["quiet", "cells_or_flare_uveitis_like",
                                             "hyphaema", "hypopyon",
                                             "shallow_or_flat", "cannot_assess"]},
    "iris_findings": {"enum": ["normal", "atrophy_or_synechiae",
                                 "neovascularisation_NVI", "transillumination_defects",
                                 "post_iridotomy", "cannot_assess"]},
    "lens_findings": {"enum": ["clear", "nuclear_cataract", "cortical_cataract",
                                 "posterior_subcapsular", "pseudophakic_IOL",
                                 "aphakic", "cannot_assess"]},
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["microbial_keratitis", "viral_keratitis_HSV_VZV",
                            "dry_eye_syndrome", "anterior_uveitis",
                            "acute_angle_closure_glaucoma",
                            "traumatic_anterior_segment",
                            "cataract", "pterygium",
                            "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

BSCAN_US_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["bscan_ultrasound", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "vitreous_findings": {"enum": ["clear", "vitreous_haemorrhage",
                                     "vitreous_opacities",
                                     "posterior_vitreous_detachment_PVD",
                                     "vitritis_inflammatory", "cannot_assess"]},
    "retinal_findings": {"enum": ["attached", "retinal_detachment_flat",
                                    "retinal_detachment_funnel",
                                    "macular_thickening", "cannot_assess"]},
    "choroidal_findings": {"enum": ["normal_contour",
                                      "choroidal_mass_or_melanoma_like",
                                      "choroidal_detachment_serous",
                                      "choroidal_detachment_haemorrhagic",
                                      "thickening_diffuse", "cannot_assess"]},
    "optic_nerve_findings": {"enum": ["normal", "cupped", "enlarged_or_swollen",
                                        "mass_or_drusen", "cannot_assess"]},
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["vitreous_haemorrhage_dense",
                            "rhegmatogenous_retinal_detachment",
                            "tractional_retinal_detachment",
                            "choroidal_melanoma_or_metastasis",
                            "choroidal_haemorrhage",
                            "posterior_scleritis",
                            "endophthalmitis", "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

TOPOGRAPHY_SCHEMA: dict[str, Any] = {
    "modality_confirm": {"enum": ["corneal_topography", "uncertain"]},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "map_type": {"enum": ["axial_curvature", "tangential", "elevation",
                           "pachymetry", "abcd_keratoconus_grading",
                           "multiple_maps_composite", "unknown"]},
    "central_corneal_power_pattern": {"enum": ["regular_symmetric",
                                                  "with_the_rule_astigmatism",
                                                  "against_the_rule_astigmatism",
                                                  "oblique_or_irregular_astigmatism",
                                                  "asymmetric_bowtie",
                                                  "inferior_steepening",
                                                  "central_steepening",
                                                  "cannot_assess"]},
    "irregular_features": {
        "type": "list",
        "items": {"enum": ["inferior_steepening_keratoconus_like",
                            "skewed_radial_axis", "thinning_inferior_paracentral",
                            "post_LASIK_oblate_pattern", "none_visible"]},
    },
    "differential_considerations": {
        "type": "list",
        "items": {"enum": ["keratoconus", "pellucid_marginal_degeneration",
                            "post_refractive_ectasia", "post_LASIK_normal",
                            "regular_astigmatism", "irregular_astigmatism_other",
                            "no_specific_pattern"]},
    },
    "narrative": {"type": "string"},
}

# ─── Generic fallback for unrecognised eye images ─────────────────────────
GENERIC_OPHTH_SCHEMA: dict[str, Any] = {
    "modality_best_guess": {"type": "string",
                              "note": "1-3 words describing the modality"},
    "image_quality": {"enum": ["good", "usable", "poor", "uninterpretable"]},
    "primary_findings": {"type": "list", "items": {"type": "string"},
                          "note": "bullet list of what is visible"},
    "differential_considerations": {"type": "list", "items": {"type": "string"},
                                     "note": "ranked plausibility list"},
    "narrative": {"type": "string"},
}


# ─── Registry ─────────────────────────────────────────────────────────────
SCHEMAS: dict[str, dict] = {
    "visual_field": VISUAL_FIELD_SCHEMA,
    "vf":           VISUAL_FIELD_SCHEMA,      # alias
    "perimetry":    VISUAL_FIELD_SCHEMA,      # alias
    "octa":         OCTA_SCHEMA,
    "oct_a":        OCTA_SCHEMA,
    "octangiography": OCTA_SCHEMA,
    "faf":          FAF_SCHEMA,
    "autofluorescence": FAF_SCHEMA,
    "icga":         ICGA_SCHEMA,
    "asoct":        ASOCT_SCHEMA,
    "anterior_segment_oct": ASOCT_SCHEMA,
    "slit_lamp":    SLIT_LAMP_SCHEMA,
    "slitlamp":     SLIT_LAMP_SCHEMA,
    "bscan_us":     BSCAN_US_SCHEMA,
    "bscan":        BSCAN_US_SCHEMA,
    "ultrasound":   BSCAN_US_SCHEMA,
    "topography":   TOPOGRAPHY_SCHEMA,
    "pentacam":     TOPOGRAPHY_SCHEMA,
    "unknown_ophth": GENERIC_OPHTH_SCHEMA,
}

_DESCRIPTIONS: dict[str, str] = {
    "visual_field":   "Visual field / perimetry (Humphrey, Octopus). Focus on defect pattern (arcuate / nasal step / altitudinal / hemianopic), reliability indices, and consistency with optic-nerve vs post-chiasmal disease.",
    "octa":           "OCT-Angiography en-face. Focus on FAZ size/shape, capillary perfusion, non-perfusion areas, and any neovascular complex.",
    "faf":            "Fundus autofluorescence. Focus on hyper/hypo-AF patterns (ring around atrophy, multifocal, macular dropout) and their diagnostic specificity.",
    "icga":           "Indocyanine green angiography. Focus on choroidal vasculature, polypoidal lesions, branching vascular networks, and plaque hyperfluorescence.",
    "asoct":          "Anterior-segment OCT. Focus on corneal thickness, anterior chamber depth, iridocorneal angle status, and lens findings.",
    "slit_lamp":      "Slit-lamp photograph of the anterior segment. Describe by structure (conjunctiva, cornea, anterior chamber, iris, lens) and note the illumination technique.",
    "bscan_us":       "B-scan ophthalmic ultrasound. Focus on vitreous opacities, retinal attachment, choroidal contour, and optic nerve cupping/mass.",
    "topography":     "Corneal topography / Pentacam. Identify the map type and read the central pattern (regular astigmatism vs keratoconus-like inferior steepening).",
    "unknown_ophth":  "Unrecognised ophthalmologic imaging modality. Give the best modality guess and a generic structural description.",
}


def _normalise(sub_label: str) -> str:
    """Map LLM's free-form sub_label to a registry key."""
    if not sub_label:
        return "unknown_ophth"
    key = sub_label.strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in SCHEMAS else "unknown_ophth"


def get_schema(sub_label: str) -> dict:
    return SCHEMAS[_normalise(sub_label)]


def get_description(sub_label: str) -> str:
    return _DESCRIPTIONS.get(_normalise(sub_label),
                              _DESCRIPTIONS["unknown_ophth"])


def get_system_prompt(sub_label: str) -> str:
    key = _normalise(sub_label)
    description = _DESCRIPTIONS.get(key, _DESCRIPTIONS["unknown_ophth"])
    schema = SCHEMAS[key]
    return (
        _SHARED_PREAMBLE
        + f"Modality: {key}\n"
        + f"Focus: {description}\n\n"
        + "Output schema (strict JSON, no extra fields):\n"
        + json.dumps(schema, indent=2)
    )


def build_user_prompt(sub_label: str, user_question: str | None = None) -> str:
    key = _normalise(sub_label)
    if user_question:
        return (
            "Below is a single ophthalmologic image of modality "
            f"`{key}`. The user asked: \"{user_question}\".\n\n"
            "Produce ONLY the JSON object specified in the system prompt. "
            "Do NOT pick a single top diagnosis — list 2-4 differential "
            "considerations and add a narrative."
        )
    return (
        f"Modality: {key}. Produce the JSON impression per the schema in "
        "the system prompt. Do NOT pick a single top diagnosis — list "
        "2-4 differential considerations and add a narrative."
    )
