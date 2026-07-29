"""
Clinical-evidence rubric for CFP differential diagnosis.

Each entry encodes the *canonical* morphological signature of one disease
as it appears in published guidance — AAO Preferred Practice Patterns,
ETDRS severity scale, ISGEO glaucoma consensus, AREDS drusen classification,
Wong-Mitchell hypertensive retinopathy scheme, etc.

The rubric is BAKED INTO the stage-2 prompt verbatim so the vision LLM
reasons against a fixed, citable knowledge frame rather than its own
fuzzy training memory. Each disease lists:

    must_see   : features that should be visible if the disease is present
    cannot_see : features that, if present, EXCLUDE the disease
    can_coexist: list of other diseases that often co-present (e.g. PDR + RVO
                 in hypertensive diabetics — these are NOT confounders)

The structure is also queryable by the validators, so the agent can
deterministically derive "ruled-out because feature X was absent" without
re-asking the LLM.
"""

from __future__ import annotations


# ── Diabetic retinopathy ────────────────────────────────────────────────────
NPDR = {
    "name": "Diabetic retinopathy — non-proliferative (NPDR)",
    "must_see": [
        "scattered dot-blot intraretinal hemorrhages in multiple quadrants",
        "microaneurysms (small red dots)",
        "hard exudates (yellow, sharp-edged) — common but not required",
    ],
    "supportive": [
        "venous beading (severe NPDR)",
        "IRMA (intraretinal microvascular abnormalities, severe NPDR)",
        "cotton-wool spots (small, few)",
    ],
    "cannot_see": [
        "neovascularization at disc (NVD) — that would be PDR",
        "neovascularization elsewhere (NVE) — that would be PDR",
        "preretinal/vitreous hemorrhage — that would be active PDR",
        "purely sector-shaped hemorrhage along a single venous drainage — "
        "that's RVO, not DR",
    ],
    "can_coexist": ["Hypertensive retinopathy", "RVO", "Macular edema"],
    "severity_grading": (
        "ETDRS: mild = ≥1 microaneurysm, no other; moderate = more than mild "
        "but less than severe; severe = '4-2-1' rule (>20 intraretinal "
        "hemorrhages in each of 4 quadrants, OR definite venous beading in "
        "≥2 quadrants, OR prominent IRMA in ≥1 quadrant)."
    ),
}

PDR_ACTIVE = {
    "name": "Proliferative diabetic retinopathy — active",
    "must_see": [
        "neovascularization at disc (NVD) OR elsewhere (NVE) — fine, "
        "irregular vessel networks on the disc surface or along arcades",
    ],
    "supportive": [
        "preretinal (boat-shaped) or vitreous hemorrhage",
        "fibrovascular proliferation membranes",
        "background NPDR features (dot-blot hem, exudates)",
    ],
    "cannot_see": [
        "exclusively sector-shaped hemorrhage along one vein "
        "(suggests RVO instead)",
    ],
    "can_coexist": ["PRP laser scars (in treated cases)", "RVO", "Macular edema"],
}

PDR_TREATED = {
    "name": "Proliferative diabetic retinopathy — treated (post-PRP)",
    "must_see": [
        "scattered, regular, round chorioretinal scars in the periphery "
        "(panretinal photocoagulation pattern) — typically 500-1000 µm "
        "white-yellow circular spots in even distribution",
    ],
    "supportive": [
        "fibrotic membranes (scarred fibrovascular proliferation)",
        "regression of NV (but residual or recurrent NV is COMMON)",
    ],
    "important_note": (
        "'Treated' does NOT imply 'inactive' — a treated eye can still "
        "have ongoing active neovascularization. Laser scars and active "
        "NV CAN coexist. Do not let the presence of laser scars suppress "
        "the search for active NV."
    ),
    "cannot_see": [],
    "can_coexist": ["Active PDR (NV recurrence)", "RVO"],
}

# ── Retinal vein occlusion ──────────────────────────────────────────────────
RVO = {
    "name": "Retinal vein occlusion (RVO)",
    "must_see": [
        "DENSE hemorrhage in a SECTORAL distribution along ONE venous "
        "drainage territory (BRVO) — typically a wedge-shaped or quadrantic "
        "pattern radiating from the affected vein toward the periphery",
        "OR (in CRVO): hemorrhage in ALL FOUR quadrants with severe venous "
        "tortuosity and dilation",
    ],
    "supportive": [
        "marked venous tortuosity and dilation",
        "cotton-wool spots clustered in the affected territory",
        "macular edema (in the affected sector)",
        "optic disc swelling (CRVO)",
        "Flame-shaped hemorrhages along the nerve fiber layer",
    ],
    "cannot_see": [
        "diffuse symmetric dot-blot hemorrhages spread across all quadrants "
        "without sectoral predilection — that's DR not RVO",
        "scattered PRP laser scars without venous obstruction signs — "
        "that's treated DR, not RVO",
    ],
    "can_coexist": ["Diabetic retinopathy", "Hypertensive retinopathy"],
    "critical_discriminator": (
        "The pattern of hemorrhage is the single most discriminative "
        "feature. Sectoral / quadrantic = RVO. Diffuse symmetric = NOT RVO. "
        "If hemorrhages are scattered evenly across the fundus, RVO is "
        "essentially ruled out."
    ),
}

# ── Age-related macular degeneration ────────────────────────────────────────
AMD = {
    "name": "Age-related macular degeneration",
    "must_see": [
        "drusen (hard, soft, or reticular) in the macula — yellow round "
        "deposits beneath the RPE; soft drusen are larger (>125 µm) with "
        "indistinct borders",
        "OR (advanced dry): geographic atrophy (well-circumscribed "
        "depigmentation with visible choroidal vessels)",
        "OR (wet): subretinal/sub-RPE hemorrhage, exudation, or "
        "fibrovascular scar centered on the macula",
    ],
    "cannot_see": [
        "dot-blot hemorrhages spread to mid-periphery (more consistent "
        "with DR)",
        "purely sectoral hemorrhage (suggests RVO)",
    ],
    "can_coexist": ["NPDR", "Hypertensive retinopathy"],
}

# ── Pathological myopia ─────────────────────────────────────────────────────
PATHOLOGICAL_MYOPIA = {
    "name": "Pathological myopia",
    "must_see": [
        "striking TESSELLATED fundus background (visible choroidal vessels "
        "throughout, giving a 'tiger-striped' appearance)",
        "peripapillary chorioretinal atrophy (whitish crescent or ring "
        "surrounding the optic disc)",
    ],
    "supportive": [
        "lacquer cracks (linear yellow-white breaks in Bruch's membrane)",
        "myopic CNV (subretinal hemorrhage at the macula, in pathological "
        "myopia rather than AMD)",
        "tilted optic disc",
        "staphyloma (posterior bulge — hard to see on 2D CFP)",
        "Fuchs spot (round pigmented scar from old myopic CNV)",
    ],
    "cannot_see": [
        "normal homogeneous reddish fundus background "
        "(rules out pathological myopia)",
    ],
    "can_coexist": ["Glaucoma (myopic optic neuropathy)", "Myopic CNV"],
}

# ── Hypertensive retinopathy ────────────────────────────────────────────────
HYPERTENSIVE_RETINOPATHY = {
    "name": "Hypertensive retinopathy",
    "wong_mitchell_grading": (
        "Mild: generalized/focal arteriolar narrowing, AV nicking, opacity. "
        "Moderate: + flame hemorrhages, cotton-wool spots, hard exudates, "
        "microaneurysms. Malignant: + optic disc swelling."
    ),
    "must_see": [
        "ANY ONE OF the three HR-specific gestalt discriminators is "
        "positive: stage1.macular_star_present ∈ {suspected, present}, "
        "OR stage1.hemorrhage_predominant_shape == flame_NFL, "
        "OR stage1.prominent_AV_nicking_or_arteriolar_narrowing == present.",
    ],
    "supportive": [
        "cotton-wool spots from retsam (typically scattered, fewer than RVO)",
        "retsam exudate_count moderate with star-like distribution around "
        "the fovea (macular star)",
        "optic disc swelling (malignant hypertension; rare)",
    ],
    "cannot_see": [
        "stage1.hemorrhage_predominant_shape == sector_along_vein "
        "(that's RVO)",
        "stage1.hemorrhage_predominant_shape == dot_blot AND "
        "macular_star_present == absent AND prominent_AV_nicking == absent "
        "(that's DR — HR vs DR discriminators must be positive to call HR)",
    ],
    "can_coexist": ["DR", "RVO (common co-occurrence)"],
    "critical_discriminator": (
        "HR vs DR — the two share lesion types (hemorrhages, exudates, "
        "CWS). The DISCRIMINATORS are HR-specific signs:\n"
        "  HR pathognomonic: macular_star_present (radiating hard exudates "
        "around fovea) — single most reliable HR sign.\n"
        "  HR-typical: hemorrhage shape = flame_NFL (superficial, "
        "feather/flame-shaped along NFL), AV nicking, arteriolar narrowing.\n"
        "  DR-typical: hemorrhage shape = dot_blot (deeper, round, "
        "small), microaneurysm-like, circinate exudate rings.\n"
        "When retsam dr_signal_confidence == high AND both DR + HR are in "
        "the differential, classify HR ONLY if at least one HR-specific "
        "discriminator is positive. Otherwise default to DR."
    ),
}

# ── Glaucoma (advanced cupping) ─────────────────────────────────────────────
GLAUCOMA = {
    "name": "Glaucoma (advanced)",
    "must_see": [
        "increased cup-to-disc ratio (vertical CDR ≥ 0.7 is suspicious; "
        "≥ 0.9 is severe)",
        "rim thinning, especially inferior or superior (violation of "
        "ISNT rule)",
    ],
    "supportive": [
        "RNFL defects (wedge-shaped dark bands radiating from the disc "
        "into the macula)",
        "disc hemorrhage (Drance hemorrhage, splinter at disc margin)",
        "peripapillary atrophy (zone beta)",
        "bayonet sign (sharp turn of vessels at the disc rim)",
    ],
    "cannot_see": [
        "florid hemorrhages or exudates suggesting a non-glaucoma primary",
    ],
    "can_coexist": ["Pathological myopia (myopic optic neuropathy)", "DR"],
}

# ── Retinitis pigmentosa ────────────────────────────────────────────────────
RP = {
    "name": "Retinitis pigmentosa",
    "must_see": [
        "BONE-SPICULE pigmentation in the mid/far periphery — branching "
        "black star-shaped figures along venules",
        "attenuated (thin) retinal arterioles",
        "waxy pallor of the optic disc",
    ],
    "supportive": [
        "atrophy and depigmentation of peripheral RPE",
        "macula may be relatively spared until late stage",
    ],
    "cannot_see": [
        "hemorrhages, exudates, or NV (these point away from RP)",
    ],
    "can_coexist": ["Cystoid macular edema", "Posterior subcapsular cataract"],
}

# ── Macular hole / ERM / CSC — short blocks ────────────────────────────────
MACULAR_HOLE = {
    "name": "Macular hole (idiopathic)",
    "must_see": [
        "ANY ONE OF: retsam llm_headline.other_findings.macular_hole_count "
        "≥ 1 (segmentation-confirmed defect); "
        "OR stage1.macula_appearance == lesion-present AND CLIP ensemble "
        "top-1 == Macular hole; "
        "OR CLIP ensemble probability for Macular hole ≥ 0.85 (very "
        "high-confidence CLIP vote, trust it).",
    ],
    "supportive": [
        "round, sharply demarcated red/dark defect in the foveal center, "
        "typically 200-500 µm — gestalt may register this as "
        "macula_appearance == lesion-present",
        "cuff of detached retina surrounding the hole",
        "yellow deposits at the base",
    ],
    "important_note": (
        "Vision LLMs systematically miss subtle central retinal defects — "
        "DO NOT override a positive retsam macular_hole_count or a "
        "high-confidence CLIP MH vote with a 'macula appears normal' "
        "stage1 reading. Segmentation + CLIP are the trusted sources here."
    ),
}

ERM = {
    "name": "Epiretinal membrane",
    "must_see": [
        "translucent or whitish sheen on the macular surface",
        "fine retinal striae radiating from the macula (cellophane "
        "maculopathy) or distorted vessels pulled toward a central focus "
        "(macular pucker)",
    ],
}

CSC = {
    "name": "Central serous chorioretinopathy",
    "must_see": [
        "well-circumscribed serous detachment of the neurosensory retina "
        "at the macula — appears as a round darker disc-shaped halo",
    ],
    "supportive": ["yellowish subretinal deposits", "RPE changes"],
    "cannot_see": ["hemorrhages or exudates outside the macula"],
}

# ── Retinal detachment ──────────────────────────────────────────────────────
RETINAL_DETACHMENT = {
    "name": "Retinal detachment",
    "must_see": [
        "ANY ONE OF: stage1.overall_pattern == detached; "
        "OR stage1.peripheral_appearance == detached; "
        "OR cfp_clip_ensemble top-1 == Retinal detachment with "
        "probability ≥ 0.6 (CLIP is well-trained on RD — trust it).",
    ],
    "supportive": [
        "billowing, elevated, corrugated retina visible on gestalt "
        "inspection (overall_pattern shifts to detached)",
        "demarcation line between attached and detached retina",
        "pale-grey appearance of detached retina vs orange-red of "
        "attached retina (color asymmetry)",
        "loss of choroidal pattern in the detached area",
    ],
    "cannot_see": [
        "stage1.overall_pattern == normal-appearing AND CLIP RD "
        "probability < 0.2 (RD is essentially excluded)",
    ],
    "can_coexist": ["Severe PDR (tractional RD)", "Pathological myopia "
                    "(myopic RD)"],
    "important_note": (
        "Retsam does NOT have a dedicated RD lesion class, so retsam "
        "headline counts will be silent on RD. Rely on stage1 gestalt "
        "(overall_pattern, peripheral_appearance) + CLIP ensemble score. "
        "Vision LLM sometimes misses early/shallow RD — when CLIP votes "
        "RD ≥ 0.6, trust CLIP over an 'absent' gestalt call."
    ),
}

# ── Cataract (lens opacity affecting fundus view) ───────────────────────────
CATARACT = {
    "name": "Cataract (lens opacity affecting fundus view)",
    "must_see": [
        "stage1.overall_pattern == opaque-media OR stage1.image_quality "
        "== poor with reason citing media opacity / hazy view (not motion "
        "blur or off-center), AND the retina itself appears largely normal "
        "behind the haze.",
    ],
    "supportive": [
        "CLIP ensemble top-1 == Cataract with probability ≥ 0.4 "
        "(CLIP detects the global haze signature reliably)",
        "EFIQA quality classifier reports Usable/Reject AND vision_impression "
        "explanation cites media opacity",
        "Diffuse loss of contrast across the entire retina (uniform fog "
        "rather than focal pathology)",
    ],
    "cannot_see": [
        "Focal retinal lesions (hemorrhages, exudates, drusen) are clearly "
        "visible — that means the view is good enough that lens opacity "
        "isn't the primary finding.",
    ],
    "can_coexist": ["Any retinal pathology (cataract is a lens issue, "
                    "not retinal; co-existence is the norm)"],
    "important_note": (
        "Cataract is fundamentally a LENS issue, not a retinal disease. "
        "The retina behind the cataract is often normal. Classify as "
        "Cataract when (a) the image is fogged in a uniform diffuse way "
        "AND (b) no specific retinal lesion is identifiable. If retinal "
        "lesions ARE identifiable, the cataract is incidental and a "
        "retinal diagnosis takes precedence."
    ),
}

# ── Healthy / Normal ────────────────────────────────────────────────────────
NORMAL = {
    "name": "Normal fundus",
    "must_see": [
        "clear optic disc with visible cup (CDR < 0.5), pink rim",
        "normal arteriovenous ratio (~2:3)",
        "smooth macula with visible foveal reflex",
        "no hemorrhages, exudates, drusen, or scars",
    ],
}

# ── Compiled rubric for prompt embedding ───────────────────────────────────
# We compile a concise text version baked into the stage-2 prompt. The
# Python dicts above remain queryable by validators.

ENTITY_REGISTRY = {
    "NPDR": NPDR,
    "PDR_active": PDR_ACTIVE,
    "PDR_treated": PDR_TREATED,
    "RVO": RVO,
    "AMD": AMD,
    "Pathological_myopia": PATHOLOGICAL_MYOPIA,
    "Hypertensive_retinopathy": HYPERTENSIVE_RETINOPATHY,
    "Glaucoma": GLAUCOMA,
    "RP": RP,
    "Macular_hole": MACULAR_HOLE,
    "ERM": ERM,
    "CSC": CSC,
    "Retinal_detachment": RETINAL_DETACHMENT,
    "Cataract": CATARACT,
    "Normal": NORMAL,
}


def _fmt_entity(eid: str, entity: dict) -> str:
    """Compile one entity into a compact text block for prompt embedding."""
    lines = [f"### {entity['name']}"]
    if entity.get("must_see"):
        lines.append("MUST see (any one is usually required):")
        for f in entity["must_see"]:
            lines.append(f"  • {f}")
    if entity.get("supportive"):
        lines.append("Supportive features:")
        for f in entity["supportive"]:
            lines.append(f"  • {f}")
    if entity.get("cannot_see"):
        lines.append("RULE OUT if present (negative evidence):")
        for f in entity["cannot_see"]:
            lines.append(f"  • {f}")
    if entity.get("can_coexist"):
        lines.append(f"Often co-occurs with: {', '.join(entity['can_coexist'])}")
    if entity.get("critical_discriminator"):
        lines.append(f"⚠ Critical discriminator: {entity['critical_discriminator']}")
    if entity.get("important_note"):
        lines.append(f"⚠ {entity['important_note']}")
    if entity.get("severity_grading"):
        lines.append(f"Severity: {entity['severity_grading']}")
    if entity.get("wong_mitchell_grading"):
        lines.append(f"Grading: {entity['wong_mitchell_grading']}")
    return "\n".join(lines)


def compile_rubric() -> str:
    """Return the full rubric as a single text block, in stable order."""
    blocks = [_fmt_entity(eid, e) for eid, e in ENTITY_REGISTRY.items()]
    return "\n\n".join(blocks)
