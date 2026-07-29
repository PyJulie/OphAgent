"""
OCT (macular B-scan / volume montage) — two-stage vision prompts, mirroring cfp.py.

Stage 1: structured MORPHOLOGY of the B-scan (gestalt; no diagnosis, no fluid-area
         numbers, no class commitment — those belong to the trained OCT tools).
Stage 2: differential grounded in stage-1 + the OCT tool outputs + an OCT rubric.

Design parallels CFP: the VLM is the QUALITATIVE GESTALT + pattern discriminators
(edema pattern, RPE/PED pattern, foveal-defect type). It defers (a) class identity to
FMUE (16-class), (b) objective fluid area to oct_fluid_segmentation, (c) layer integrity
to oct_layer_segmentation. CRITICAL OCT-specific rule: DR / RVO are FUNDUS diagnoses —
OCT shows their macular EDEMA, not the cause — so vascular calls must be flagged
"needs fundus correlation", never asserted from OCT alone.
"""
from __future__ import annotations

import json
from .cfp import _walk_schema_validate, _get_deep   # generic schema-walk reused


# ───────────────────────────────────────────────────────────────────────────
# STAGE 1 — B-scan morphology (gestalt)
# ───────────────────────────────────────────────────────────────────────────
STAGE1_SCHEMA: dict = {
    "scan_quality": {"enum": ["good", "usable", "poor"]},
    "scan_quality_reason": {"type": "string"},
    "view_note": {
        # The image may be a MONTAGE of representative B-scans (lesion slice +
        # max-fluid slice). Say whether the fovea is captured.
        "enum": ["single_bscan", "montage_multi_slice", "cannot_assess"]},
    "foveal_contour": {
        "enum": ["normal_depression", "blunted_or_flattened", "elevated",
                 "full_thickness_defect", "partial_or_lamellar_defect",
                 "not_in_view", "cannot_assess"]},
    "retinal_thickness": {
        "enum": ["normal", "increased_thickened", "decreased_atrophic",
                 "cannot_assess"]},
    "intraretinal_fluid": {
        # cystoid = round hyporeflective spaces WITHIN the retina
        "enum": ["absent", "cystoid_present", "cannot_assess"]},
    "subretinal_fluid": {
        # neurosensory detachment — hyporeflective space BELOW the retina, above RPE
        "enum": ["absent", "present", "cannot_assess"]},
    "rpe_sub_rpe": {
        "enum": ["normal", "drusen_or_irregular", "serous_PED",
                 "fibrovascular_or_double_layer_PED", "atrophy_with_transmission",
                 "cannot_assess"]},
    "vitreoretinal_interface": {
        "enum": ["normal", "epiretinal_membrane", "vitreomacular_traction",
                 "posterior_hyaloid_detached", "cannot_assess"]},
    "outer_retina_ez": {
        "enum": ["intact", "focally_disrupted", "diffusely_lost", "cannot_assess"]},
    "hyperreflective_foci_or_exudates": {
        "enum": ["absent", "present", "cannot_assess"]},
    "choroid_appearance": {
        "enum": ["normal", "thinned", "thickened_pachychoroid", "cannot_assess"]},
    "location_of_main_finding": {
        # KEY: a clean central macula does NOT exclude a peripheral / vascular
        # disease that this slice does not capture.
        "enum": ["foveal", "parafoveal", "extrafoveal_or_edge", "diffuse",
                 "none_macula_looks_clean", "cannot_assess"]},
    # ─── 3 pattern discriminators (gestalt, like CFP's HR/RVO/PDR) ──────────
    "edema_pattern": {
        # cystoid -> macular edema (DME / RVO / post-op; cause needs FUNDUS);
        # subretinal_serous -> CSC / wet-AMD spectrum.
        "enum": ["none", "intraretinal_cystoid", "subretinal_serous",
                 "mixed", "cannot_assess"]},
    "pigment_epithelium_pattern": {
        # drusenoid -> AMD; serous_dome (+pachychoroid) -> CSC;
        # double_layer / sharp-peaked PED -> PCV (definitive needs ICGA);
        # atrophic -> geographic atrophy / late dry AMD.
        "enum": ["normal", "drusenoid", "serous_dome",
                 "double_layer_or_polypoidal", "atrophic", "cannot_assess"]},
    "foveal_defect_type": {
        # MH discriminator.
        "enum": ["none", "full_thickness_hole", "lamellar_or_pseudohole",
                 "cannot_assess"]},
    "one_phrase_impression": {"type": "string"},
    "model_self_assessment": {
        "confidence_overall": {"enum": ["high", "moderate", "low"]},
        "what_is_hard_to_see": {"type": "string"},
        "image_artifacts_present": {"enum": ["none", "blur", "shadow",
                                             "cropped", "low_signal", "other"]},
    },
}


STAGE1_SYSTEM = """\
You are a senior retinal specialist reading a macular OCT B-scan (it may be a
MONTAGE of two representative slices — the most lesion-bearing slice and the
max-fluid slice). Your job in this stage is a GESTALT morphology read. You do
NOT measure fluid area, you do NOT commit to a final disease class.

Those detailed jobs are done by:
  • FMUE — a 16-class OCT B-scan classifier (votes the disease class),
  • oct_fluid_segmentation — objective IRF/SRF/PED fluid area, and
  • oct_layer_segmentation — objective layer integrity.

Your role is the QUALITATIVE GESTALT that gates the differential, plus three
pattern discriminators that separate the OCT-confirmable entities (macular
edema vs CSC vs AMD-PED vs PCV double-layer vs macular hole vs ERM).

What you should DO:
  • Read the B-scan layer by layer: foveal contour, retinal thickness,
    intraretinal (cystoid) fluid, subretinal fluid, RPE / sub-RPE, the
    vitreoretinal interface (membrane / traction), outer-retina / ellipsoid
    zone, and the choroid.
  • Set the THREE discriminators:
    - edema_pattern: intraretinal_cystoid (round intraretinal spaces →
      macular EDEMA, e.g. DME/RVO/post-op) vs subretinal_serous (clear space
      under a smooth detached neurosensory retina → CSC / wet-AMD spectrum).
    - pigment_epithelium_pattern: drusenoid (→AMD), serous_dome (smooth RPE
      dome, often pachychoroid →CSC), double_layer_or_polypoidal (irregular
      shallow RPE elevation / sharp-peaked PED →PCV), atrophic (→GA).
    - foveal_defect_type: full_thickness_hole (→ macular hole) vs
      lamellar/pseudohole vs none.
  • Set location_of_main_finding HONESTLY. If the central macula looks clean,
    say "none_macula_looks_clean" — this does NOT exclude a vascular disease
    that lives on the fundus / in slices not shown.

What you should NOT do:
  • Do NOT quantify fluid in pixels or measure thickness in microns
    (fluid-seg / layer-seg do that).
  • Do NOT commit to a final class label (FMUE does that).
  • Do NOT diagnose DR or RVO from the OCT — OCT shows macular EDEMA, not its
    cause; the diabetic-vs-venous distinction is a FUNDUS call. Report the
    edema; defer the cause.
  • Do NOT claim "macular hole absent" lightly — if the fovea is not clearly
    in view, use foveal_contour="not_in_view" / foveal_defect_type="cannot_assess".
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
        "Describe the morphology of this OCT B-scan using exactly the schema "
        "below. Output ONE JSON object matching the schema's nested shape; for "
        "each leaf output the indicated value (a string from the given enum, "
        "or a free string). No commentary, no code fences, no extra fields.\n"
        + extra +
        f"\nSCHEMA:\n```json\n{schema_block}\n```"
    )


# ───────────────────────────────────────────────────────────────────────────
# STAGE 2 — Differential grounded in stage-1 + the OCT tools
# ───────────────────────────────────────────────────────────────────────────
STAGE2_SCHEMA: dict = {
    "top3_differential": "list of {diagnosis, likelihood (high|moderate|low), "
                         "supporting_observations (list of schema-path strings you "
                         "cite from stage1 / fmue / fluid_seg), still_uncertain_because "
                         "(list of strings)}",
    "needs_fundus_correlation": "bool — true if the leading pattern is (a) macular "
                                "edema / any vascular pattern whose CAUSE (DR vs RVO "
                                "vs other) cannot be settled on OCT alone, OR (b) a "
                                "SUBRETINAL SEROUS detachment / serous-or-shallow PED "
                                "whose cause (CSC vs central-serous-like DR vs PCV vs "
                                "early wet-AMD) cannot be settled on OCT alone",
    "ruled_out": "list of {diagnosis, reason citing an observation path}",
    "image_quality_caveat": "string (uses stage1.scan_quality + artifacts)",
    "recommended_followup": "list of strings (e.g. colour fundus / CFP, FFA, "
                            "ICGA for suspected PCV, OCT-A, dilated exam)",
    "single_line_impression": "ONE-sentence clinical impression",
}

OCT_RUBRIC = """\
OCT MORPHOLOGY → ENTITY RUBRIC (macula). Each entity lists the OCT signature it
MUST / MUST-NOT show. OCT is the gold standard for the STRUCTURAL entities; the
VASCULAR entities (DR, RVO) are NOT OCT-decidable — see the caveat.

• Normal: normal foveal depression, no intraretinal/subretinal fluid, intact
  ellipsoid zone, no RPE elevation, no epiretinal membrane.
• Macular edema: intraretinal cystoid spaces ± increased central thickness.
  — WITH hard exudates / hyperreflective foci (lipid extravasation, Bolz 2009)
    and no SRF/PED/drusen → DIABETIC MACULAR EDEMA (DME); name DME, favor DR
    (RVO secondary). Do NOT defer (see rule 4b).
  — WITHOUT hard exudates (truly nonspecific cystoid oedema) → the CAUSE
    (diabetic / venous-occlusive / post-op / uveitic) is a fundus call →
    needs_fundus_correlation=true (rule 4c).
• Central serous chorioretinopathy (CSC): SUBRETINAL serous fluid under a
  smooth, shallow neurosensory detachment; thickened (pachy) choroid; little
  intraretinal cystoid fluid; NO drusen. Acute CSC has a clean dome.
• Dry AMD: drusen / drusenoid PED, RPE irregularity or geographic atrophy with
  choroidal hypertransmission; NO sub/intraretinal fluid.
• Wet (neovascular) AMD: sub- and/or intraretinal fluid + fibrovascular PED or
  irregular RPE elevation; hyperreflective material; ± subretinal hyperreflective.
• Polypoidal choroidal vasculopathy (PCV): sharp-peaked PED and/or the
  DOUBLE-LAYER SIGN (shallow irregular RPE elevation over Bruch's), ± subretinal
  fluid; on the wet-AMD spectrum. DEFINITIVE diagnosis needs ICGA → at most
  "PCV (suggested), confirm with ICGA".
• Macular hole: FULL-THICKNESS foveal defect (vs lamellar/pseudohole which
  spares the outer retina). ERM often coexists.
• Epiretinal membrane (ERM): hyperreflective band on the inner surface, retinal
  wrinkling / inner-surface contour change ± increased thickness. May coexist
  with edema or a (pseudo)hole.

CAVEAT — VASCULAR DISEASES ON OCT:
  Hard-exudative cystoid macular edema IS a diagnosable DME pattern on OCT
  (hyperreflective foci = lipid, Bolz 2009; cystoid IRF = DME morphology, Otani
  1999 / ESASO 2020) — name DME and favor DR (rule 4b). What OCT cannot fully
  settle is only the VASCULAR ETIOLOGY (DR vs RVO), since RVO-related macular
  edema shares the same exudative pattern — so DR-vs-RVO is a mild fundus caveat,
  NOT a reason to withhold the DME diagnosis. Only TRULY NONSPECIFIC cystoid
  oedema (NO hard exudates, NO structural anchor) reads as "macular edema — cause
  needs fundus" (rule 4c). A clean central macula does NOT exclude DR/RVO.

CAVEAT — SEROUS DETACHMENTS ON OCT:
  A subretinal serous detachment / serous PED is morphologically SHARED by CSC,
  central-serous-like vascular maculopathy, PCV, and early wet-AMD. OCT alone
  CANNOT settle which — do NOT commit "high" to CSC (or any one) on the serous
  appearance; keep them as a moderate-likelihood serous DIFFERENTIAL, set
  needs_fundus_correlation=true, and recommend CFP + ICGA.
"""

STAGE2_SYSTEM_TEMPLATE = """\
You are a senior retinal specialist. Independent observers have examined the
same macular OCT and reported:

  • STAGE 1 GESTALT (you, in stage 1) — qualitative B-scan morphology +
    discriminators (edema_pattern, pigment_epithelium_pattern,
    foveal_defect_type) + location_of_main_finding.
  • FMUE (`oct_fmue_16class`) — a trained 16-class OCT B-scan classifier
    (Normal, dAMD, nAMD, PCV, DME, DR_without_ME, iERM, iMH, MTM, mCNV, RD,
    acute_CSC, acute_RAO, acute_RVO, acute_VKH, RP), MAX-pooled over the volume.
    Trust FMUE for class identity, but it is a 2D per-slice model and can be
    swamped by peripheral / off-fovea slices.
  • oct_fluid_segmentation — objective IRF / SRF / PED fluid presence + area;
    GROUND TRUTH for "is there fluid?".
  • (if present) oct_layer_segmentation — objective layer integrity.

Your job: SYNTHESISE these into a rank-ordered top-3 differential with explicit
evidence chains, using the OCT rubric below. Then set needs_fundus_correlation.

═══════════════════════════════════════════════════════════════════════
{rubric}
═══════════════════════════════════════════════════════════════════════

Strict rules:
  1. EVERY supporting observation MUST cite a path, e.g.
     "stage1.edema_pattern=subretinal_serous",
     "fmue.top_class=acute_CSC (0.89)",
     "fluid_seg.subretinal_fluid_present=true".
  2. Missing features → list under still_uncertain_because.
  3. Coexisting entities (ERM + edema; MH + ERM; wet-AMD + PCV) — list both;
     do not rule one out merely because the other leads.
  4. STRUCTURAL-ANCHOR PRIORITY — decide the TOP-1 in this order. OCT is the
     gold standard for the STRUCTURAL entities, so when one is present it is the
     top-1 and needs_fundus_correlation=FALSE (the OCT has settled it):
     (a) inner-surface hyperreflective band / retinal wrinkling / inner-contour
         change → Epiretinal membrane (ERM).
     (b) FULL-THICKNESS foveal defect → Macular hole (± coexisting ERM).
     (c) drusen / drusenoid PED / RPE atrophy with hypertransmission, no fluid
         → Dry AMD.
     (d) sub-/intraretinal fluid WITH fibrovascular PED / irregular RPE
         elevation / subretinal hyperreflective material → Wet (neovascular)
         AMD — NAME it (ICGA only to SUBTYPE PCV vs wAMD; do not defer).
     (e) double-layer sign / sharp-peaked PED → PCV (suggested, confirm ICGA).
     Edema or fluid that COEXISTS with any anchor above is SECONDARY — keep the
     structural entity as top-1; do NOT downgrade it to "macular edema needs
     fundus". Only when NO structural anchor is present do you fall to rule 4b/4c.
  4b. EXUDATIVE-DME anchor (literature-grounded — Bolz 2009; Otani 1999; ESASO/
     Panozzo 2020): if intraretinal CYSTOID edema is accompanied by HARD EXUDATES /
     hyperreflective foci (stage1.hyperreflective_foci_or_exudates=present) and there
     is NO drusen and NO fibrovascular PED, this IS DIABETIC MACULAR EDEMA (DME) —
     EVEN IF a shallow subretinal-fluid component coexists (serous retinal detachment
     is a recognised DME subtype, Otani 1999); the cystoid IRF + hard exudate is the
     decisive signal, NOT the SRF. (This is what separates DME-with-serous from pure
     CSC: CSC has serous SRF WITHOUT cystoid IRF and WITHOUT hard exudates — rule 4d.)
     The hyperreflective foci ARE lipid extravasation / hard exudate
     (Bolz, Ophthalmology 2009) and cystoid intraretinal fluid is a DME morphologic
     pattern (Otani 1999; ESASO 2020). REPORT "Diabetic macular edema (DME)" as the
     top-1 OCT entity and FAVOR diabetic etiology (put DR first), list RVO-related
     macular edema as the secondary vascular cause. needs_fundus_correlation may be
     set ONLY as a MILD etiology caveat (DR-vs-RVO needs fundus laterality / vascular
     distribution) — it does NOT withhold the DME diagnosis. Corroborate with
     fmue.top_class (DME / DR_without_ME) when present. Do NOT downgrade this to
     "nonspecific oedema, cause needs fundus".
  4c. PURE-EDEMA rule (no anchor AND no exudate): ONLY if cystoid intraretinal edema
     is present WITHOUT hard exudates / hyperreflective foci AND none of 4(a–e)/4b —
     i.e. truly NONSPECIFIC oedema — the top entry is "macular edema — cause needs
     fundus", set needs_fundus_correlation=true, and put DR / RVO in
     recommended_followup (CFP / FFA). NEVER assert DR or RVO as a settled OCT
     diagnosis, and never guess a specific entity (CSC / AMD) onto nonspecific oedema.
  4d. PURE-SEROUS rule (no structural anchor): a subretinal SEROUS detachment /
     serous PED that is the DOMINANT finding WITH essentially NO cystoid intraretinal
     fluid AND NO hard exudates/hyperreflective foci, NO drusen, NO fibrovascular/RPE
     disease — this serous pattern is shared by CSC / central-serous-like vascular
     cause / PCV / early wAMD — cap any single one at "moderate", list the
     alternatives, set needs_fundus_correlation=true, recommend CFP + ICGA. (If cystoid
     IRF + hard exudates accompany the SRF → that is DME-with-serous, rule 4b, NAME DME.
     If drusen / RPE disease / fibrovascular PED accompany → AMD, rule 4a, NAME AMD.)
  5. PCV may only be "suggested, confirm with ICGA".
  6. Output ONE JSON object. No prose outside it. No markdown fences.

OBSERVER WEIGHTING:
  • FLUID PRESENCE (IRF/SRF/PED): trust oct_fluid_segmentation > vision.
  • CLASS IDENTITY: weigh FMUE, but discount it when its top class conflicts
    with a clear stage-1 structural sign (e.g. FMUE=RP but stage1 shows a
    clean macula with a focal PED → trust the structural read).
  • EDEMA / PED / DEFECT PATTERN: trust the stage-1 discriminators — that is
    what they are for.

Output schema:
{schema}
"""


def stage2_system_prompt() -> str:
    schema = json.dumps(STAGE2_SCHEMA, indent=2, ensure_ascii=False)
    return STAGE2_SYSTEM_TEMPLATE.format(rubric=OCT_RUBRIC, schema=schema)


def stage2_user_prompt(stage1_json: dict, tool_context: str = "") -> str:
    s1 = json.dumps(stage1_json, indent=2, ensure_ascii=False)
    ctx = f"\n\nOCT TOOL OUTPUTS (FMUE / fluid-seg / layer-seg) already in context:\n{tool_context}" if tool_context else ""
    return (
        "Stage-1 OCT morphology output is below. Produce the stage-2 "
        "differential JSON now, grounded in stage-1 and any OCT tool outputs.\n\n"
        f"STAGE 1 OBSERVATIONS:\n```json\n{s1}\n```{ctx}"
    )


# ───────────────────────────────────────────────────────────────────────────
# Cross-tool checks + self-consistency validators
# ───────────────────────────────────────────────────────────────────────────
def _fluid_match(v, c):
    """vision intraretinal/subretinal fluid vs fluid-seg fluid_present."""
    if v is None or c is None:
        return None
    vis_fluid = str(v).lower() in ("cystoid_present", "present", "subretinal_serous",
                                   "intraretinal_cystoid", "mixed")
    seg_fluid = bool(c)
    if vis_fluid and not seg_fluid:
        return "vision sees fluid but fluid-seg found none (peripheral/subtle? or vision over-read)"
    if seg_fluid and not vis_fluid:
        return "fluid-seg found fluid but vision read none — re-examine the shown slice"
    return None


CROSS_TOOL_FIELDS = [
    {
        "vision_path": "intraretinal_fluid",
        "tool": "oct_fluid_segmentation",
        "tool_path": "predictions.fluid_present",
        "compare": _fluid_match,
    },
]


def _consistency(s1: dict) -> list[str]:
    def g(k):
        x = s1.get(k)
        return x.get("value") if isinstance(x, dict) else x
    issues = []
    if g("foveal_defect_type") == "full_thickness_hole" and \
            g("foveal_contour") not in ("full_thickness_defect", "cannot_assess", None):
        issues.append("foveal_defect_type=full_thickness_hole but foveal_contour "
                      "does not reflect a full-thickness defect")
    if g("edema_pattern") == "none" and g("intraretinal_fluid") == "cystoid_present":
        issues.append("edema_pattern=none but intraretinal_fluid=cystoid_present")
    if g("location_of_main_finding") == "none_macula_looks_clean" and \
            g("intraretinal_fluid") == "cystoid_present":
        issues.append("macula called clean but cystoid fluid reported")
    return issues


def schema_validators(stage1: dict) -> list[str]:
    return _walk_schema_validate(stage1, STAGE1_SCHEMA)


VALIDATORS = [schema_validators, _consistency]
