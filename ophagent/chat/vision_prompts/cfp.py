"""
CFP (colour fundus photography) — two-stage vision prompts.

Stage 1: structured MORPHOLOGY description (no diagnosis).
Stage 2: differential diagnosis cited against stage 1 + literature rubric.

Both stages return strict JSON; the user prompt enforces the JSON schema
shape and the system prompt explains *how* to look.
"""
from __future__ import annotations

import json

from ._evidence_rubric import compile_rubric, ENTITY_REGISTRY


# ───────────────────────────────────────────────────────────────────────────
# STAGE 1 — Morphology description
# ───────────────────────────────────────────────────────────────────────────

# JSON-schema-like dict (used by validators); also serialised into the prompt.
STAGE1_SCHEMA: dict = {
    # ─── GESTALT v2 ────────────────────────────────────────────────────
    # Stage 1 is now the GESTALT observer ONLY. It does NOT count lesions,
    # estimate ratios, or judge subtle presence/absence of small findings.
    # Those jobs belong to retsam (pixel-level segmentation + quantification)
    # and the CLIP fleet (semantic class identity). Stage 1's only job is
    # to provide the BIG-PICTURE qualitative read that gates which
    # workups to spawn and which discriminators to consult.
    #
    # 12 enum fields. Vision LLMs are reliable at this granularity; they
    # are NOT reliable at numeric estimation or counting small lesions.
    "laterality_guess": {"enum": ["OD", "OS", "unclear"]},
    "image_quality": {"enum": ["good", "usable", "poor"]},
    "image_quality_reason": {"type": "string"},
    "overall_pattern": {
        # Big-picture appearance — what genre of image is this?
        "enum": ["normal-appearing", "hemorrhagic", "atrophic",
                 "inflammatory", "detached", "opaque-media",
                 "surgical-altered", "cannot_assess"]},
    "disc_appearance": {
        # Gestalt judgment about the optic disc. Numeric CDR is retsam's
        # job; this is just "does it look normal / cupped / pale / swollen?".
        "enum": ["normal", "cupped", "pale", "swollen",
                 "tilted_or_oblique", "cannot_assess"]},
    "macula_appearance": {
        # "Is there SOMETHING at the macula?" — lesion identification is
        # retsam/CLIP's job.
        "enum": ["normal", "lesion-present", "obscured", "cannot_assess"]},
    "peripheral_appearance": {
        # Big-picture peripheral retina state.
        "enum": ["normal", "atrophic", "detached", "pigmented",
                 "hemorrhagic", "cannot_assess"]},
    "dominant_color_outside_disc": {
        # Useful gestalt for AMD/DR (yellow-white) vs RVO/PDR (red_orange)
        # vs RD (grayish) vs Cataract/FFA (washed out / grayscale).
        "enum": ["red_orange", "yellow_white_dominant", "mixed",
                 "pale_grey", "grayscale_or_washed"]},
    # ─── HR / RVO / PDR discriminators (gestalt-level) ────────────────
    # These three fields are NOT for finding lesions — they're for
    # PATTERN judgment when retsam reports lesions are present.
    "hemorrhage_predominant_shape": {
        # Discriminates DR (dot_blot) vs HR (flame_NFL) vs RVO
        # (sector_along_vein) vs PDR (preretinal_or_subhyaloid).
        # 'mixed' = multiple shapes present; 'none' = no hemorrhages at all.
        "enum": ["none", "dot_blot", "flame_NFL", "sector_along_vein",
                 "preretinal_or_subhyaloid", "mixed", "cannot_assess"]},
    "macular_star_present": {
        # HR pathognomonic — radiating hard exudates around the fovea.
        "enum": ["absent", "suspected", "present"]},
    "prominent_AV_nicking_or_arteriolar_narrowing": {
        # HR-typical vascular change. Gestalt-level call; if the vessels
        # look uneven at crossings, mark suspected/present.
        "enum": ["absent", "suspected", "present"]},
    "one_phrase_impression": {"type": "string"},
    "model_self_assessment": {
        "confidence_overall": {"enum": ["high", "moderate", "low"]},
        "what_is_hard_to_see": {"type": "string"},
        "image_artifacts_present": {"enum": ["none", "blur", "glare",
                                              "shadow", "off_center", "other"]},
    },
}


STAGE1_SYSTEM = """\
You are a senior retinal specialist. Your job in this stage is to give a
GESTALT (big-picture) read of the colour fundus photograph (CFP). You do
NOT count lesions. You do NOT estimate cup-to-disc ratios. You do NOT
commit to the presence or absence of small individual findings.

Those detailed jobs are done by:
  • a segmentation model (retsam-2.0) that gives objective lesion counts
    and the numeric vCDR, and
  • a CLIP ensemble (3 models) that votes on the disease class.

Your role is to provide the QUALITATIVE GESTALT that gates which workups
the agent runs next, and the three pattern discriminators that
distinguish HR / RVO / PDR / DR when lesions ARE present.

What you should DO:
  • Look at the image as a whole. Decide its overall pattern (normal,
    hemorrhagic, atrophic, detached, opaque-media, ...).
  • Judge the optic disc as a whole shape (normal, cupped, pale, swollen).
    Do NOT report a numeric CDR — retsam handles that.
  • Judge the macula at a glance (normal vs lesion-present).
  • Judge the peripheral retina at a glance (normal, atrophic, detached, ...).
  • For the THREE pattern discriminators:
    - hemorrhage_predominant_shape — if hemorrhages are visible, decide
      their PREDOMINANT SHAPE: dot_blot (small, round, deep — DR-typical),
      flame_NFL (feather-shaped, superficial — HR-typical), sector_along_vein
      (wedge along one vein — RVO), preretinal_or_subhyaloid (boat-shaped —
      PDR), or mixed. If no hemorrhages, "none".
    - macular_star_present — radiating hard exudates around the fovea is
      HR pathognomonic. If you see this distinctive star pattern, mark
      "present"; if uncertain, "suspected".
    - prominent_AV_nicking_or_arteriolar_narrowing — HR sign. Vessels look
      uneven at crossings, arteries appear thinner than normal? Mark
      "suspected" or "present".

What you should NOT do:
  • Do not count small things (microaneurysms, drusen, hemorrhages).
  • Do not estimate numeric CDR or vessel calibre.
  • Do not commit to "macular hole absent" — vision LLMs miss subtle central
    defects. Use macula_appearance = "normal" only when the macula looks
    truly uneventful; if there is ANYTHING focal at the macula, use
    "lesion-present" and let retsam/CLIP say what it is.
  • Do not invent. If you cannot tell, use "cannot_assess".

Strict output rules:
  1. Fill EVERY field. Use the exact enum values shown.
  2. ONE JSON object. No prose outside it. No markdown fences.

Anatomical orientation reminders:
  • OD (right eye) → optic disc appears LEFT-of-centre in the image
    (because the fundus image is taken as the examiner sees it, which is
    a mirror image of patient anatomy in some systems — defer to vascular
    arcade direction as the tiebreaker).
  • Inferotemporal vascular arcade descends to the temporal-inferior side.
  • Macula is approximately 2 disc-diameters temporal to the optic disc.
"""


def stage1_user_prompt(focus_hint: str = "") -> str:
    schema_block = json.dumps(STAGE1_SCHEMA, indent=2, ensure_ascii=False)
    extra = ""
    if focus_hint:
        extra = (f"\n\nFocus hint from the user / agent: {focus_hint}\n"
                 f"Pay extra attention to this when describing, but still "
                 f"fill EVERY field of the schema below.\n")
    return (
        "Describe the morphology of this CFP using exactly the schema "
        "below. Output ONE JSON object matching the schema's nested shape; "
        "for each leaf, output the value type indicated (a string from the "
        "given enum, an integer, a float, etc.). No commentary, no code "
        "fences, no extra fields.\n"
        + extra +
        f"\nSCHEMA:\n```json\n{schema_block}\n```"
    )


# ───────────────────────────────────────────────────────────────────────────
# STAGE 2 — Differential diagnosis with citations to stage 1
# ───────────────────────────────────────────────────────────────────────────

STAGE2_SCHEMA: dict = {
    "top3_differential": "list of {diagnosis, likelihood (high|moderate|low), "
                         "supporting_observations (list of schema-path strings "
                         "from stage1 you cite), still_uncertain_because "
                         "(list of strings), suggested_severity (optional string)}",
    "ruled_out": "list of {diagnosis, reason (string explicitly citing "
                 "stage1 observation paths)}",
    "image_quality_caveat": "string describing how the image quality "
                            "(from stage1.image_quality + .model_self_assessment "
                            ".image_artifacts_present) limits the interpretation",
    "recommended_followup_imaging": "list of strings (e.g. FFA, OCT macula, "
                                    "ICGA, OCT-A, axial length)",
    "single_line_impression": "ONE-sentence clinical impression",
}

STAGE2_SYSTEM_TEMPLATE = """\
You are a senior retinal specialist. Three independent observers have
already examined the same colour fundus photograph and reported their
findings:

  • STAGE 1 GESTALT (you, in stage 1) — qualitative big-picture read:
    overall pattern, disc appearance, macula appearance, peripheral
    appearance, and the three discriminators (hemorrhage_predominant_shape,
    macular_star_present, prominent_AV_nicking).

  • RETSAM SEGMENTATION (`cfp_retsam_segmentation.llm_headline`) — pixel-
    level lesion counts (hemorrhage_count, exudate_count, drusen_count,
    macular_hole_count, etc.), the conservative `dr_signal_confidence`
    tier (high/low/ambiguous/absent), the
    `hemorrhage_etiology` overlap guard, and the numeric
    `optic_disc.vCDR`. Retsam masks support lesion PRESENCE, but a
    disease-labelled head is not proof of disease ETIOLOGY. Vision LLMs
    may miss subtle defects, while overlapping segmentation heads may
    assign the same lesion to more than one disease family.

  • CLIP ENSEMBLE (`cfp_clip_ensemble`) — three trained CLIP models
    (ViLReF/RetiZero/FLAIR) voting on the 11-class fundus taxonomy.
    Trust CLIP top-1 when probability ≥ 0.85, especially for classes
    retsam cannot detect (Retinal detachment, Pathological myopia,
    Cataract). When CLIP top-3 is split (no clear winner), it's a
    differential, not a verdict.

Your job: SYNTHESISE these three observers into a rank-ordered top-3
differential, with explicit evidence chains. Use the canonical clinical
rubric below as the reference for what features each disease must / must
not show. The rubric is distilled from AAO PPP, ETDRS, ISGEO, AREDS, and
Wong-Mitchell criteria.

═══════════════════════════════════════════════════════════════════════
{rubric}
═══════════════════════════════════════════════════════════════════════

Strict rules:
  1. EVERY supporting observation in `top3_differential` MUST cite a path
     in one of the three observers, e.g.:
       "stage1.hemorrhage_predominant_shape=dot_blot",
       "retsam.diabetic_retinopathy_signs.dr_signal_confidence=high",
       "clip_ensemble.fused_top3[0]=(label:DR, prob:0.71)".
     If a key feature was NOT reported, list under `still_uncertain_because`.
  2. The `ruled_out` list must cite observer evidence — e.g.
     "ruled out RVO because stage1.hemorrhage_predominant_shape was
     dot_blot, not sector_along_vein".
  3. Diagnoses that often coexist (e.g. PDR + RVO; AMD + DR) — DO NOT put
     one of them in `ruled_out` solely because the other is the top
     differential. List both as separate top-3 entries if features support
     both.
  4. If `retsam.hemorrhage_etiology.status=ambiguous`, the DR-hemorrhage
     and AMD-patch-hemorrhage masks substantially overlap. Count them as
     ONE hemorrhagic lesion. Do not infer DR from component counts alone.
     Use lesion shape/distribution and independent DR signs. For a confluent
     macula-centred hemorrhagic-exudative lesion, explicitly consider
     nAMD/PCV and myopic CNV where clinically appropriate.

OBSERVER WEIGHTING (read carefully):
  • For LESION PRESENCE (drusen, hemorrhage, macular hole, ERM, vCDR
    numeric): TRUST RETSAM > vision LLM. For DISEASE ATTRIBUTION, honour
    `hemorrhage_etiology` and never treat overlapping disease-head masks as
    independent lesions.
  • For CLASS IDENTITY when CLIP top-1 prob ≥ 0.85: TRUST CLIP over an
    "absent" stage1 reading, especially for classes retsam cannot detect
    (Retinal detachment, Pathological myopia, Cataract).
  • For LESION SHAPE / PATTERN (dot_blot vs flame_NFL vs sector_along_vein,
    macular_star, AV_nicking): TRUST STAGE1 GESTALT — these are
    discrimination calls retsam doesn't make and CLIP only does
    implicitly. The three stage1 discriminator fields are designed
    specifically for this purpose.

THE HR/DR DISCRIMINATOR (memorise this):
  When retsam reports `dr_signal_confidence = high` AND the candidates
  include both DR and HR:
    • If stage1.macular_star_present is suspected/present → call HR
    • elif stage1.hemorrhage_predominant_shape == flame_NFL → call HR
    • elif stage1.prominent_AV_nicking == present AND no dot_blot
           hemorrhages → call HR
    • else → call DR (default)

  5. Output ONE JSON object. No prose outside it. No markdown fences.

Output schema:
{schema}
"""


def stage2_system_prompt() -> str:
    rubric = compile_rubric()
    schema = json.dumps(STAGE2_SCHEMA, indent=2, ensure_ascii=False)
    return STAGE2_SYSTEM_TEMPLATE.format(rubric=rubric, schema=schema)


def stage2_user_prompt(stage1_json: dict) -> str:
    s1 = json.dumps(stage1_json, indent=2, ensure_ascii=False)
    return (
        "Stage-1 morphology output for this image is below. Produce the "
        "stage-2 differential JSON now.\n\n"
        f"STAGE 1 OBSERVATIONS:\n```json\n{s1}\n```"
    )


# ───────────────────────────────────────────────────────────────────────────
# Validator hooks — used by `validators.py::run_validators`
# ───────────────────────────────────────────────────────────────────────────

# (vision_field_path, classifier_tool_name, classifier_field_path, comparator)
# `comparator(vision_value, classifier_value) -> Optional[str]` returns a
# disagreement message or None if consistent.
def _quality_match(v, c):
    """Vision says quality X; EFIQA/EyeQ classifier say Y. Both have 3-bucket
    schemas; we accept fuzzy match (good~Good, usable~Usable, poor~Reject)."""
    if not v or not c:
        return None
    vmap = {"good": "Good", "usable": "Usable", "poor": "Reject"}
    expected = vmap.get(str(v).lower())
    if expected and str(c).strip().lower() != expected.lower():
        return (f"vision quality='{v}' but classifier quality='{c}'")
    return None


# Note (v2 schema): the old per-field cross-tool comparators (NVD, NVE,
# laser_scars, cup_appearance vs vCDR, hemorrhage pattern vs retsam) are
# RETIRED. Stage 1 no longer commits to those fields — the new gestalt
# schema deliberately offloads lesion / count claims to retsam and
# class identity to CLIP. The stage 2 prompt now does the synthesis
# directly, reading both observers from the conversation context.
#
# Only the IMAGE-QUALITY cross-checks remain, because image_quality IS
# a gestalt judgment the vision LLM should still make and it has direct
# classifier comparisons (efiqa / eyeq).
CROSS_TOOL_FIELDS = [
    {
        "vision_path": "image_quality",
        "tool": "cfp_efiqa",
        "tool_path": "predictions.quality",
        "compare": _quality_match,
    },
    {
        "vision_path": "image_quality",
        "tool": "cfp_eyeq",
        "tool_path": "predictions.quality",
        "compare": _quality_match,
    },
]


# ── Self-consistency validators ────────────────────────────────────────────
# v2 schema is gestalt-only — the old count-based self_consistency_probe is
# gone, so the old confabulation-detection validators are retired with it.
# We keep a single sanity check: hemorrhage pattern says 'none' but
# overall_pattern says 'hemorrhagic'.
def _hemorrhage_gestalt_consistency(s1: dict) -> list[str]:
    pat = s1.get("hemorrhage_predominant_shape")
    overall = s1.get("overall_pattern")
    # Some vision models return these gestalt fields as nested objects; coerce
    # to a scalar so the membership test below can't raise on an unhashable type.
    if isinstance(pat, dict):
        pat = pat.get("value") or pat.get("finding") or pat.get("label")
    if isinstance(overall, dict):
        overall = overall.get("value") or overall.get("finding") or overall.get("label")
    if pat == "none" and overall == "hemorrhagic":
        return ["self-inconsistent: overall_pattern='hemorrhagic' but "
                "hemorrhage_predominant_shape='none' — re-examine"]
    if pat in {"sector_along_vein", "preretinal_or_subhyaloid"} and \
            overall == "normal-appearing":
        return [f"self-inconsistent: hemorrhage_predominant_shape='{pat}' "
                f"but overall_pattern='normal-appearing'"]
    return []


SELF_CONSISTENCY_VALIDATORS = [_hemorrhage_gestalt_consistency]


# ── Schema enum validator (auto-generated from STAGE1_SCHEMA) ───────────────
def _walk_schema_validate(stage1: dict, schema: dict, prefix: str = "") -> list[str]:
    issues = []
    for key, spec in schema.items():
        path = f"{prefix}.{key}" if prefix else key
        if not isinstance(spec, dict):
            continue
        # Leaf with type/enum
        if "enum" in spec or "type" in spec:
            value = _get_deep(stage1, path)
            if value is None:
                issues.append(f"missing field: {path}")
                continue
            if "enum" in spec and value not in spec["enum"]:
                issues.append(
                    f"invalid enum at {path}: got {value!r}, expected one "
                    f"of {spec['enum']}")
            if spec.get("type") == "integer" and not isinstance(value, int):
                issues.append(f"type mismatch at {path}: expected int, got {type(value).__name__}")
            if spec.get("type") == "number" and not isinstance(value, (int, float)):
                issues.append(f"type mismatch at {path}: expected number, got {type(value).__name__}")
            if "min" in spec and isinstance(value, (int, float)) and value < spec["min"]:
                issues.append(f"out of range at {path}: {value} < min {spec['min']}")
            if "max" in spec and isinstance(value, (int, float)) and value > spec["max"]:
                issues.append(f"out of range at {path}: {value} > max {spec['max']}")
        else:
            # Nested dict — recurse
            issues.extend(_walk_schema_validate(stage1, spec, prefix=path))
    return issues


def _get_deep(d: dict, dotted: str):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
        if cur is None:
            return None
    return cur


def schema_validators(stage1: dict) -> list[str]:
    return _walk_schema_validate(stage1, STAGE1_SCHEMA)


VALIDATORS = [schema_validators] + SELF_CONSISTENCY_VALIDATORS


# ── Public summary string (for docs / debugging) ────────────────────────────
def STAGE1_SCHEMA_SUMMARY() -> str:
    return json.dumps(STAGE1_SCHEMA, indent=2, ensure_ascii=False)
