"""
UWF (ultra-wide-field fundus) — two-stage vision prompts, mirroring cfp.py / oct.py.

Stage 1: structured wide-field MORPHOLOGY (gestalt; no diagnosis, no class commitment —
         class identity belongs to the trained UWF tools). Reads disc, vessels, macula
         (INCLUDING the macular surface, for ERM), posterior-pole myopic changes, and the
         PERIPHERY (UWF's advantage over a 45° CFP).
Stage 2: a per-condition assessment over the 6 screened diseases (DR / AMD / Glaucoma /
         RVO / PM / ERM), grounded in stage-1 + the two UWF classifiers + a UWF rubric.

Design parallels CFP/OCT, with two UWF-specific facts baked into the rubric (both are
STRUCTURAL — from the tools' design, NOT from any test data, so no leakage):
  • Tool COVERAGE map: uwf_multi_disease covers DR/AMD/RVO/Glaucoma (no PM, no ERM);
    uwf_disease_7class covers DR/AMD/RVO/PM (no Glaucoma, no ERM). So Glaucoma is only
    seen by the multi-label tool, PM only by the 7-class tool, and ERM by NEITHER.
  • ERM is therefore an OPEN-SET / VISION-ONLY call — it rests entirely on the stage-1
    macular_surface read; the agent must not wait for a tool flag that cannot exist.
The 7-class tool is single-label softmax: a class can LEAD without exceeding 0.5 (mass
split 7 ways) and it can only name ONE disease — both are properties of softmax, stated
so the orchestrator reads it correctly, again not test-set calibration.
"""
from __future__ import annotations

import json
from .cfp import _walk_schema_validate, _get_deep   # generic schema-walk reused


# ───────────────────────────────────────────────────────────────────────────
# STAGE 1 — wide-field morphology (gestalt). FLAT schema (+ nested self-assess).
# ───────────────────────────────────────────────────────────────────────────
STAGE1_SCHEMA: dict = {
    "image_quality": {"enum": ["good", "usable", "poor"]},
    "image_quality_reason": {"type": "string"},
    "field_of_view": {
        # UWF pseudo-colour montage vs a cropped posterior view.
        "enum": ["ultrawide_peripheral", "posterior_pole_only", "cannot_assess"]},
    # ─── optic disc (for Glaucoma) ──────────────────────────────────────────
    "cup_disc_ratio_estimate": {
        "enum": ["normal_le_0.5", "borderline_0.5_0.6", "enlarged_ge_0.6",
                 "cannot_assess"]},
    "neuroretinal_rim": {
        "enum": ["intact_ISNT_respected", "focal_thinning_or_notch",
                 "diffuse_thinning", "cannot_assess"]},
    "rnfl_or_disc_haemorrhage": {
        "enum": ["absent", "rnfl_wedge_defect", "disc_haemorrhage",
                 "cannot_assess"]},
    "optic_disc_colour": {
        # OPEN-SET (Optic pallor): an abnormally pale / chalky-white / atrophic disc
        # — distinct from a large physiological cup, which keeps a pink rim.
        "enum": ["normal_pink", "pale_or_atrophic", "cannot_assess"]},
    # ─── vessels (for DR vs RVO; arterial whitening for RAO) ────────────────
    "vascular_findings": {
        "enum": ["normal", "microaneurysms_dot_blot", "flame_haemorrhages",
                 "venous_tortuosity_or_dilation", "arterial_attenuation_whitening",
                 "neovascularisation", "vascular_sheathing", "cannot_assess"]},
    "retinal_whitening_or_cherry_red": {
        # OPEN-SET (RAO): ischaemic retinal WHITENING (sectoral along an artery, or
        # diffuse with a foveal cherry-red spot) — the signature of artery occlusion.
        "enum": ["absent", "present_sectoral", "present_diffuse_with_cherry_red",
                 "cannot_assess"]},
    "haemorrhage_distribution": {
        # KEY DR vs RVO discriminator.
        "enum": ["none", "scattered_diffuse_posterior_and_periphery",
                 "sectoral_along_a_vein", "cannot_assess"]},
    # ─── macula ─────────────────────────────────────────────────────────────
    "macular_surface": {
        # KEY for ERM — the ONLY signal either UWF tool lacks a head for.
        "enum": ["normal", "cellophane_sheen_or_striae",
                 "frank_membrane_with_wrinkling", "cannot_assess"]},
    "macular_exudates_or_drusen": {
        "enum": ["none", "hard_exudates", "drusen", "both", "cannot_assess"]},
    "macular_oedema_or_thickening": {
        "enum": ["absent", "suspected", "cannot_assess"]},
    # ─── posterior pole / myopia (for PM) ───────────────────────────────────
    "myopic_changes": {
        # tessellation ALONE / a crescent ALONE is NOT PM; atrophy IS.
        "enum": ["none", "tessellation_only", "peripapillary_atrophy_crescent_only",
                 "chorioretinal_atrophy_patches", "lacquer_cracks_or_staphyloma",
                 "cannot_assess"]},
    # ─── periphery (UWF advantage) ──────────────────────────────────────────
    "peripheral_findings": {
        "enum": ["none", "peripheral_haemorrhages_or_exudates", "laser_scars",
                 "lattice_or_retinal_break", "retinal_detachment",
                 "pigmentary_changes", "cannot_assess"]},
    "location_of_main_finding": {
        # a clean macula does NOT exclude a peripheral / disc / vascular disease.
        "enum": ["macular", "peripapillary_disc", "vascular_arcades",
                 "peripheral", "diffuse", "none_looks_clean", "cannot_assess"]},
    "one_phrase_impression": {"type": "string"},
    "model_self_assessment": {
        "confidence_overall": {"enum": ["high", "moderate", "low"]},
        "what_is_hard_to_see": {"type": "string"},
        "image_artifacts_present": {"enum": ["none", "eyelash_or_lid", "peripheral_distortion",
                                             "pseudocolour_artifact", "blur", "shadow",
                                             "cropped", "other"]},
    },
}


STAGE1_SYSTEM = """\
You are a senior retinal specialist reading an ULTRA-WIDE-FIELD (UWF) fundus
image (~200° field, often a pseudo-colour montage that captures the far
periphery, not just the posterior pole). Your job in this stage is a GESTALT
morphology read. You do NOT commit to a final disease class — that is done by
the trained UWF classifiers:
  • uwf_multi_disease — multi-label (independent 0-1 per disease), covers
    DR / AMD / RVO / Glaucoma (and MH / RP / RD);
  • uwf_disease_7class — single-label softmax (Healthy / DR / AMD / PM / RVO /
    Uveitis / RD), the only tool with a Pathologic-Myopia head.

Your role is the QUALITATIVE GESTALT that gates the differential, plus the
pattern discriminators that separate the 6 screened conditions — and, crucially,
the ONE read NEITHER classifier can make: the MACULAR SURFACE (for epiretinal
membrane).

What you should DO:
  • Optic disc: estimate cup-disc ratio, judge the neuroretinal rim (ISNT),
    look for an RNFL wedge defect or a disc haemorrhage (→ Glaucoma).
  • Vessels: microaneurysms / dot-blot haemorrhages / hard exudates SCATTERED
    across pole + periphery (→ DR) vs SECTORAL flame haemorrhages following a
    single vein with venous tortuosity/dilation (→ RVO). Set
    haemorrhage_distribution honestly — it is the key DR-vs-RVO discriminator.
  • Macula: read the SURFACE for a cellophane sheen / retinal striae / frank
    membrane wrinkling (→ ERM); note drusen / hard exudates; note suspected
    oedema or thickening.
  • Posterior pole: distinguish mere TESSELLATION or an isolated peripapillary
    crescent (NOT pathologic myopia) from true myopic CHORIORETINAL ATROPHY /
    lacquer cracks / staphyloma (→ PM).
  • Periphery: haemorrhages, laser scars, lattice / breaks, detachment,
    pigmentary change — the UWF advantage.
  • Optic disc COLOUR: a chalky-pale / atrophic disc (optic_disc_colour=
    pale_or_atrophic) — distinct from a large but PINK physiological cup.
  • Ischaemic RETINAL WHITENING: cloudy-white inner-retinal opacification along an
    artery / sector, or diffuse whitening with a foveal cherry-red spot
    (retinal_whitening_or_cherry_red) — the signature of artery occlusion.
  • Set location_of_main_finding HONESTLY. If everything looks clean, say
    "none_looks_clean" — but remember UWF montages can distort or miss areas.

What you should NOT do:
  • Do NOT output a disease name or probability (the classifiers do that).
  • Do NOT call ERM from a peripapillary/peri-vascular sheen or isolated vessel
    straightening — only a macular surface membrane / striae counts.
  • Do NOT call PM from tessellation alone or a crescent alone — atrophy is
    required.
  • Do NOT invent. If you cannot tell, use "cannot_assess".

Strict output rules:
  1. Fill EVERY field. Use the exact enum values shown.
  2. ONE JSON object. No prose outside it. No markdown fences.
"""


def stage1_user_prompt(focus_hint: str = "") -> str:
    schema_block = json.dumps(STAGE1_SCHEMA, indent=2, ensure_ascii=False)
    extra = ""
    if focus_hint:
        extra = (f"\n\nFocus hint from the user / agent: {focus_hint}\n"
                 f"Pay extra attention to this, but still fill EVERY field.\n")
    return (
        "Describe the morphology of this ULTRA-WIDE-FIELD fundus image using "
        "exactly the schema below. Output ONE JSON object matching the schema's "
        "nested shape; for each leaf output the indicated value (a string from "
        "the given enum, or a free string). No commentary, no code fences, no "
        "extra fields.\n"
        + extra +
        f"\nSCHEMA:\n```json\n{schema_block}\n```"
    )


# ───────────────────────────────────────────────────────────────────────────
# STAGE 2 — per-condition assessment grounded in stage-1 + the UWF tools
# ───────────────────────────────────────────────────────────────────────────
STAGE2_SCHEMA: dict = {
    "per_condition": "list of EXACTLY 8 objects, one per condition in "
                     "[DR, AMD, Glaucoma, RVO, PM, ERM, RAO, OpticPallor], each = "
                     "{condition, assessment (present|absent|uncertain), likelihood "
                     "(high|moderate|low), supporting_observations (list of "
                     "schema-path strings you cite from stage1 / the UWF tools), "
                     "against_observations (list of strings)}. The last THREE "
                     "(ERM, RAO, OpticPallor) are OPEN-SET: no classifier covers them, "
                     "decide from the stage-1 morphology alone.",
    "coexisting_note": "string — name any conditions that co-occur (UWF eyes "
                       "often carry >1); the single-label 7-class tool can only "
                       "report ONE, so do not let it suppress a second disease",
    "open_set_flag": "string — ERM is covered by NEITHER UWF classifier; its "
                     "call rests on stage1.macular_surface alone. State explicitly "
                     "whether the ERM call is vision-only and how confident",
    "needs_oct_correlation": "bool — true if ERM or macular oedema is suspected "
                             "(OCT confirms a surface membrane / oedema far better "
                             "than UWF)",
    "image_quality_caveat": "string (uses stage1.image_quality + artifacts)",
    "recommended_followup": "list of strings (e.g. OCT for ERM/oedema, OCT-A, "
                            "dilated exam, IOP/visual-field for glaucoma)",
    "single_line_impression": "ONE-sentence clinical impression",
}

UWF_RUBRIC = """\
UWF MORPHOLOGY → DISEASE RUBRIC (ultra-wide-field). For EACH of the 6 screened
conditions: the UWF signature it MUST / MUST-NOT show, and WHICH trained tool
(if any) can score it.

TOOL COVERAGE (STRUCTURAL — from the tools' design, NOT from any test data):
  • uwf_multi_disease (multi-label, independent 0-1): covers DR, AMD, RVO,
    Glaucoma. It has NO Pathologic-Myopia head and NO ERM head.
  • uwf_disease_7class (single-label softmax): covers DR, AMD, RVO, PM. It has
    NO Glaucoma head and NO ERM head.
  ⇒ Glaucoma can be scored ONLY by uwf_multi_disease.
  ⇒ PM can be scored ONLY by uwf_disease_7class.
  ⇒ ERM is scored by NEITHER — it is an OPEN-SET, VISION-ONLY call resting on
    stage1.macular_surface. Do NOT wait for a tool flag that cannot exist; do
    NOT read "no tool fired ERM" as evidence of absence.

HOW TO READ THE TOOLS (properties of the models, not calibration):
  • uwf_disease_7class is single-label softmax over 7 classes: a class can be the
    LEADING / top class WITHOUT its probability exceeding 0.5 (the mass is split
    7 ways) — use the RELATIVE / top-ranked signal, not an absolute 0.5 cut. And
    it can name only ONE disease, so for co-existing diseases cross-check the
    multi-label tool + the stage-1 read.
  • When BOTH tools cover a disease (DR / AMD / RVO) and they disagree, do NOT
    blindly average. Weigh each tool against the stage-1 morphology: a tool flag
    with NO morphological support (e.g. AMD flagged but no drusen/exudate on
    stage1) is discounted; a tool flag WITH a matching sign (RVO flagged AND
    sectoral flame haemorrhages along a vein) is trusted.

PER-CONDITION:
• DR — microaneurysms / dot-blot haemorrhages / hard exudates SCATTERED across
  the posterior pole and periphery (haemorrhage_distribution=
  scattered_diffuse_posterior_and_periphery), ± neovascularisation / laser scars.
  Covered by BOTH tools. MUST-NOT: a single isolated speck → not DR.
• AMD — macular DRUSEN ± RPE change (macular_exudates_or_drusen=drusen/both);
  exudative AMD adds macular haemorrhage/exudate. Covered by BOTH tools.
  MUST-NOT: peripheral drusen-like dots or peripapillary atrophy alone; do not
  confuse with myopic atrophy or hard exudates of DR.
• Glaucoma — enlarged cup (cup_disc_ratio_estimate=enlarged_ge_0.6) WITH a
  neuroretinal-rim defect (neuroretinal_rim=focal_thinning_or_notch /
  diffuse_thinning) and/or RNFL wedge defect / disc haemorrhage. Covered ONLY by
  uwf_multi_disease. MUST-NOT: a large cup with an intact, ISNT-respecting rim →
  physiological cupping, NOT glaucoma.
• RVO — SECTORAL flame haemorrhages following a single vein, with venous
  tortuosity / dilation (haemorrhage_distribution=sectoral_along_a_vein,
  vascular_findings=flame_haemorrhages / venous_tortuosity_or_dilation). Covered
  by BOTH tools. MUST-NOT: scattered diffuse dot-blots WITHOUT a venous sector →
  that pattern is DR, not RVO.
• PM (pathologic myopia) — TRUE myopic chorioretinal ATROPHY: chorioretinal
  atrophy patches exposing choroid/sclera, lacquer cracks, or a staphyloma
  (myopic_changes=chorioretinal_atrophy_patches / lacquer_cracks_or_staphyloma).
  Covered ONLY by uwf_disease_7class. MUST-NOT: tessellation_only or a
  peripapillary crescent alone → NOT PM (the atrophy, not mere tessellation, is
  required). Does NOT require macular-centre involvement.
• ERM — macular SURFACE membrane: cellophane sheen / retinal striae / frank
  wrinkling (macular_surface=cellophane_sheen_or_striae /
  frank_membrane_with_wrinkling). Covered by NEITHER tool → VISION-ONLY. MUST-NOT:
  peripapillary or peri-vascular sheen, or isolated vessel straightening → not
  ERM. OCT confirms ERM far better than UWF → if present/suspected set
  needs_oct_correlation=true.

OPEN-SET (NO classifier head — decide from stage-1 morphology ALONE; a tool's
silence is NOT evidence of absence):
• RAO (retinal artery occlusion) — ischaemic retinal WHITENING along an artery /
  sector, or diffuse whitening with a foveal CHERRY-RED spot
  (retinal_whitening_or_cherry_red=present_*), ± attenuated/boxcar arteries
  (vascular_findings=arterial_attenuation_whitening). MUST-NOT: confuse with the
  RED haemorrhages of DR/RVO — RAO is pale ISCHAEMIC whitening, not blood.
• OpticPallor (optic atrophy) — an abnormally PALE / chalky-white / atrophic optic
  disc (optic_disc_colour=pale_or_atrophic). MUST-NOT: a large but PINK
  physiological cup (that is cup enlargement, not pallor) → not OpticPallor.
"""

STAGE2_SYSTEM_TEMPLATE = """\
You are a senior retinal specialist. Independent observers have examined the
same ULTRA-WIDE-FIELD fundus image and reported:

  • STAGE 1 GESTALT (you, in stage 1) — qualitative wide-field morphology +
    discriminators (haemorrhage_distribution, macular_surface, myopic_changes,
    disc/rim) + location_of_main_finding.
  • uwf_multi_disease — a multi-label classifier (independent probability per
    disease) covering DR / AMD / RVO / Glaucoma; trust it for those, but it has
    NO PM and NO ERM head.
  • uwf_disease_7class — a single-label softmax (Healthy / DR / AMD / PM / RVO /
    Uveitis / RD), the ONLY tool with a PM head; it has NO Glaucoma and NO ERM
    head, and can name only ONE disease.

Your job: produce a per-condition assessment over the 6 screened diseases (DR,
AMD, Glaucoma, RVO, PM, ERM) with explicit evidence chains, using the UWF rubric
below. ERM and (partly) PM/Glaucoma cannot be settled by tools alone — reason
from morphology.

═══════════════════════════════════════════════════════════════════════
{rubric}
═══════════════════════════════════════════════════════════════════════

Strict rules:
  1. Output EXACTLY 8 per_condition entries, one for each of
     [DR, AMD, Glaucoma, RVO, PM, ERM, RAO, OpticPallor].
  2. EVERY supporting / against observation MUST cite a path, e.g.
     "stage1.haemorrhage_distribution=sectoral_along_a_vein",
     "uwf_multi_disease.per_label.RVO.probability=0.71",
     "uwf_disease_7class.top_class=PM".
  3. OPEN-SET (ERM, RAO, OpticPallor): decide from the stage-1 morphology ALONE
     (macular_surface for ERM; retinal_whitening_or_cherry_red / arterial_attenuation
     for RAO; optic_disc_colour for OpticPallor). If the morphology is normal/
     cannot_assess, mark absent/uncertain and SAY it is a vision-only judgement in
     open_set_flag. NEVER write "absent because no tool flagged it" for these three.
  4. Co-existing diseases are common on UWF — do NOT let the single-label 7-class
     tool's one winner suppress a second disease the multi-label tool or stage1
     supports. List co-occurrences in coexisting_note.
  5. Do NOT blindly average the two tools on DR/AMD/RVO — arbitrate against the
     stage-1 morphology (see rubric "HOW TO READ THE TOOLS").
  6. PM only from true myopic atrophy; Glaucoma only from cup+rim/RNFL defect.
  7. Output ONE JSON object. No prose outside it. No markdown fences.

OBSERVER WEIGHTING:
  • DR / AMD / RVO: weigh BOTH classifiers, each checked against stage-1 signs.
  • Glaucoma: uwf_multi_disease + the stage-1 disc/rim read (the 7-class tool is
    blind here).
  • PM: uwf_disease_7class + the stage-1 myopic_changes read (the multi-label
    tool is blind here).
  • ERM: stage-1 macular_surface ONLY.

Output schema:
{schema}
"""


def stage2_system_prompt() -> str:
    schema = json.dumps(STAGE2_SCHEMA, indent=2, ensure_ascii=False)
    return STAGE2_SYSTEM_TEMPLATE.format(rubric=UWF_RUBRIC, schema=schema)


def stage2_user_prompt(stage1_json: dict, tool_context: str = "") -> str:
    s1 = json.dumps(stage1_json, indent=2, ensure_ascii=False)
    ctx = (f"\n\nUWF TOOL OUTPUTS (uwf_multi_disease / uwf_disease_7class) "
           f"already in context:\n{tool_context}" if tool_context else "")
    return (
        "Stage-1 UWF morphology output is below. Produce the stage-2 "
        "per-condition assessment JSON now, grounded in stage-1 and any UWF tool "
        "outputs.\n\n"
        f"STAGE 1 OBSERVATIONS:\n```json\n{s1}\n```{ctx}"
    )


# ───────────────────────────────────────────────────────────────────────────
# Cross-tool checks + self-consistency validators
# ───────────────────────────────────────────────────────────────────────────
def _dr_match(v, c):
    """stage1 scattered haemorrhages vs uwf_multi_disease DR predicted."""
    if v is None or c is None:
        return None
    vis_dr = str(v) == "scattered_diffuse_posterior_and_periphery"
    tool_dr = bool(c)
    if vis_dr and not tool_dr:
        return ("stage1 sees scattered diffuse haemorrhages (DR-like) but "
                "uwf_multi_disease did not flag DR — re-weigh")
    if tool_dr and not vis_dr:
        return ("uwf_multi_disease flagged DR but stage1 sees no scattered "
                "haemorrhage pattern — confirm it is not isolated specks")
    return None


def _rvo_match(v, c):
    """stage1 sectoral-vein haemorrhages vs uwf_multi_disease RVO predicted."""
    if v is None or c is None:
        return None
    vis_rvo = str(v) == "sectoral_along_a_vein"
    tool_rvo = bool(c)
    if vis_rvo and not tool_rvo:
        return ("stage1 sees sectoral flame haemorrhages along a vein (RVO-like) "
                "but uwf_multi_disease did not flag RVO — re-weigh")
    return None


CROSS_TOOL_FIELDS = [
    {
        "vision_path": "haemorrhage_distribution",
        "tool": "uwf_multi_disease",
        "tool_path": "predictions.per_label.DR.predicted",
        "compare": _dr_match,
    },
    {
        "vision_path": "haemorrhage_distribution",
        "tool": "uwf_multi_disease",
        "tool_path": "predictions.per_label.RVO.predicted",
        "compare": _rvo_match,
    },
]


def _consistency(s1: dict) -> list[str]:
    def g(k):
        x = s1.get(k)
        return x.get("value") if isinstance(x, dict) else x
    issues = []
    # sectoral-vein pattern should co-occur with a flame/venous vascular finding
    if g("haemorrhage_distribution") == "sectoral_along_a_vein" and \
            g("vascular_findings") not in ("flame_haemorrhages",
                                           "venous_tortuosity_or_dilation",
                                           "cannot_assess", None):
        issues.append("haemorrhage_distribution=sectoral_along_a_vein but "
                      "vascular_findings shows neither flame haemorrhage nor "
                      "venous tortuosity")
    # a frank membrane on the surface should not coexist with location 'clean'
    if g("macular_surface") == "frank_membrane_with_wrinkling" and \
            g("location_of_main_finding") == "none_looks_clean":
        issues.append("macular_surface=frank_membrane_with_wrinkling but "
                      "location_of_main_finding=none_looks_clean")
    # PM atrophy claimed but myopic_changes only tessellation is contradictory
    if g("myopic_changes") == "tessellation_only" and \
            g("location_of_main_finding") == "peripapillary_disc":
        pass  # not contradictory; tessellation can be peripapillary — no issue
    return issues


def schema_validators(stage1: dict) -> list[str]:
    return _walk_schema_validate(stage1, STAGE1_SCHEMA)


VALIDATORS = [schema_validators, _consistency]
