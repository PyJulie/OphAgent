"""
OphSession — multi-modal ophthalmology chat agent with Planner-Executor-
Verifier loop and persistent context.

Sits on top of OphToolKit (which wraps the cross-modality adapter registry).

Design highlights
-----------------
* Model-supported **Planner**, **Executor repair**, and **Verifier** roles have
  distinct contracts. The Planner proposes registered tools, the Executor can
  repair a failed malformed invocation under deterministic schema and safety
  gates, and the Verifier checks evidence before finalisation.
* Per-session context tracks the current image, current volume, all tool
  results (cached by image path), and the last visual report.
* Streams events (`thinking`, `tool_call`, `tool_result`, `text`, `done`) so
  the UI can show a live trace.
* Designed to coexist with the older OCT-only ChatSession in `session.py`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse, unquote

from .oph_tools import OphToolKit
from .api_config import create_provider_client, PROVIDER_SPECS
from .executor_role import (
    executor_repair_tool_schema,
    parse_executor_repair,
    repairable_invocation_failure,
    schema_for_tool,
)
from .run_policy import EffortPolicy, get_effort_policy
from ..utils.paths import RELEASE_ROOT, output_path


log = logging.getLogger(__name__)
_PROJECT_ROOT = RELEASE_ROOT


def _default_temperature() -> float | None:
    """Default LLM decoding temperature (paper config = 0.4), overridable via the
    EVAL_TEMPERATURE env var so the temperature ablation can sweep it. An empty /
    'none' / 'default' value means "don't send temperature" (use backend default)."""
    v = os.environ.get("EVAL_TEMPERATURE", "0.4")
    if str(v).strip().lower() in ("", "none", "default", "unset"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.4


def _looks_like_reasoning_fragment(content: str, finish_reason: str | None) -> bool:
    """Heuristic: did the LLM dump its internal CoT instead of a real answer?

    Detection strategy (any one of these → fragment):
      1. `finish_reason == 'length'` — model was truncated mid-answer.
      2. **No markdown structure** AND high density of self-talk markers
         ("need", "maybe", "let's", "use", "call", "run", "verify", "?", etc.,
         case-insensitive). A real clinical answer uses bullets / headers /
         bold; a reasoning monologue uses imperative fragments and questions
         to itself.

    Tuned against actual leakage samples — the empirical cutoff is ≥3
    self-talk markers in a no-markdown reply, OR ≥6 anywhere, OR density
    ≥ 1 marker per 80 chars.
    """
    if not content:
        return False
    text = content.strip()
    if finish_reason == "length":
        return True
    # RETRY FALSE-POSITIVE FIX (OPH_RETRY_FIX=1, default OFF): a terse but
    # COMPLETE final answer — the required "===FINAL===\n{json}" format — ends in
    # '}'/']' with no period and is short, which the truncation-tell below
    # mis-flags as a fragment, forcing a wasteful ~88s re-ask that reproduces the
    # same answer. If the model finished cleanly (not 'length') AND the text looks
    # like a complete answer, it is NOT a CoT fragment.
    if os.environ.get("OPH_RETRY_FIX") == "1":
        if "===FINAL===" in text or text.endswith(("}", "]")):
            return False

    # Markdown structure → almost certainly a real answer.
    markdown_tokens = ("**", "## ", "### ", "\n- ", "\n* ", "\n1.", "`", " | ",
                       "---\n", "1) ", "2) ", "•")
    has_markdown = any(tok in text for tok in markdown_tokens)
    if has_markdown:
        return False

    # Case-insensitive self-talk vocabulary. These are the verbs / hedges /
    # questions that the reasoning model uses to address itself, almost
    # never present (with this density) in a real clinical answer.
    self_talk_words = (
        "need ", "needs ", "maybe ", "could ", "should ", "let's ", "let me ",
        "will run", " perhaps ", "perhaps,", "try ", "wait", "hmm", "actually",
        "plan ", "step ", "next:", "todo", "consider ",
        "use ", "call ", "run ", "verify", "already ", "single task",
        "user asks", "user wants", "does that",
    )
    lowered = text.lower()
    count = sum(lowered.count(tok) for tok in self_talk_words)
    # Question marks (the model literally asking itself "?") count too.
    count += text.count("?")

    density_trigger = count / max(1, len(text)) > 1.0 / 80
    if count >= 6:
        return True
    if count >= 3 and (density_trigger or len(text) < 400):
        return True

    # Truncation tell: ends mid-clause (no terminal punctuation in last 40
    # chars), AND short, AND no markdown — likely ran out mid-sentence.
    tail = text[-40:]
    if not any(c in tail for c in (".", "!", "?", "。", "！", "？")):
        if len(text) < 600:
            return True
    return False


def _model_is_text_only(model: str) -> bool:
    """Best-effort check: does this model lack image input?

    Used to decide whether `vision_impression` (and LLM modality detection)
    can safely be handed an image. The danger we guard against is a gateway
    that SILENTLY drops the image for a text-only model — the model then
    hallucinates a "visual" impression from the prompt template alone.

    Policy: assume vision-capable unless the name clearly belongs to a
    known text-only family. Vision-capable family tokens always win, so a
    name like 'qwen2-vl' is never mis-flagged by the 'qwen' text variants.
    """
    m = (model or "").lower()
    _VISION_TOKENS = (
        "-vl", "vision", "gpt-4o", "gpt-4.1", "gpt-5", "claude",
        "gemini", "glm-4v", "glm-4.1v", "pixtral", "llava", "qwen-vl",
        "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "omni", "qvq",
        "internvl", "minicpm-v", "molmo",
    )
    if any(v in m for v in _VISION_TOKENS):
        return False
    _TEXT_ONLY_TOKENS = (
        "deepseek",                         # deepseek-chat / v3 / r1 / reasoner
        "o1-mini", "o1-preview", "o3-mini",  # OpenAI text-only reasoners
        "qwq",                              # Qwen QwQ reasoning — text only
        "ernie-speed", "ernie-lite",
        "yi-large", "yi-medium",
    )
    if any(t in m for t in _TEXT_ONLY_TOKENS):
        return True
    # Qwen text flagships (qwen-max, qwen3.7-max, qwen3.6-plus, qwen-flash,
    # qwen-turbo, qwen-long …) — text only. The VL/omni vision variants were
    # already caught above, so any remaining qwen here is a text model.
    if "qwen" in m and any(s in m for s in ("max", "-plus", "plus", "flash", "turbo", "-long")):
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Effort directives — modality-agnostic BASE + modality-specific ADDENDUM
# ──────────────────────────────────────────────────────────────────────
# Design rationale (see #review on planner-not-generalizing-to-UWF):
#   Previous medium/high directives were 100% CFP-centric. They hardcoded
#   tool names like "ALWAYS call cfp_retsam_segmentation", so when the
#   attached image was UWF/OCT/FFA the LLM had no triage script and just
#   ran the closest-name-matched tool.
#
#   New layout:
#     MEDIUM_BASE / HIGH_BASE — generic 3-observer flow that references
#     ONLY the per-session feasibility list (see _modality_feasibility_hint)
#     to pick tool names. No tool names hardcoded.
#
#     MEDIUM_ADDENDUMS[m] / HIGH_ADDENDUMS[m] — purely CLINICAL rules
#     specific to modality m (vCDR thresholds, DR signal tiers, HR
#     discriminators, OCT EDL uncertainty hints, etc.). These describe
#     domain-knowledge invariants — they don't go stale when a new
#     adapter ships. Tool field-names are OK to mention here because
#     they are part of each adapter's stable output contract.
#
#   What goes WHERE:
#     • "vCDR ≥ 0.55 → referable glaucoma" → addendum (clinical fact)
#     • "ALWAYS run cfp_clip_ensemble"      → BASE pulls "calibrated
#                                              classifier" from the
#                                              feasibility list instead
#     • "UWF has no disc-cup tool"          → NOWHERE; the feasibility
#                                              list shows what's there
#                                              and the LLM infers gaps

_MEDIUM_BASE_NOVISION = """\
# Thinking effort: MEDIUM (text-only mode)

**Do NOT call `vision_impression`** — this chat model cannot see images.
Rely entirely on the specialised classifiers + segmentation tools from
the Available Tools list above.

Standard chain:
  1. `detect_modality(path)`.
  2. Run the modality's CALIBRATED CLASSIFIER from the list (highest-
     coverage one).
  3. Run the modality's QUANTITATIVE TOOL from the list.
  4. Apply modality-specific clinical rules from the addendum below.
  5. `verify_findings`, up to 2 rounds.

Final answer integrates all classifier verdicts + numeric measurements.
You have no visual opinion — say so when uncertainty is high.
"""

_HIGH_BASE_VISION_HEADER = """\
# Thinking effort: HIGH - broader targeted cross-checking

Use the same objective-first decision rules as MEDIUM, with one additional
round for overlapping classifiers or an indicated specialist workup. Do not
equate higher effort with indiscriminate tool use.

Mandatory pipeline:
  1. In the first batch, confirm modality and run the strongest calibrated
     classifier, the relevant quantitative tool, and a dedicated quality
     adapter only when one is listed for the attached modality. Do not invent
     a generic quality call when no such adapter is available.
  2. After seeing those results, run overlapping broad classifiers and only
     the specialist workups indicated by the question or observed evidence.
     A single targeted `vision_impression` is allowed here when it resolves a
     concrete ambiguity; never run it before objective evidence is available.
  3. Finish all diagnostic tools before `verify_findings`. The controller
     calls the verifier as a separate terminal stage and permits only a
     verifier-authorised re-check afterwards.
"""

_HIGH_BASE_NOVISION_HEADER = """\
# Thinking effort: HIGH (text-only mode)

Skip `vision_impression`. Start with the strongest calibrated classifier, the
relevant quantitative tool, and a dedicated quality adapter only when one is
listed for the attached modality. Do not invent a generic quality call when no
such adapter is available. Then use the additional planning round for
overlapping broad classifiers or an evidence-triggered specialist workup. You
cannot see the image, so every finding must be grounded in tool output
numerics.

Mandatory pipeline:
  1. Run objective tools in a first batched round.
  2. Run targeted cross-checks in a second batched round.
  3. Finish all diagnostic tools before the controller's terminal
     `verify_findings` stage.
"""

# ── Per-modality clinical-rule addendums (stable medical knowledge)

_MEDIUM_ADDENDUM_CFP = """\
## CFP-specific clinical rules

  Read these retsam (`cfp_retsam_segmentation`) headline fields:
    - `diabetic_retinopathy_signs.dr_signal_confidence`
        (high / low / absent) — conservative tier:
        • `high` → flip Normal→DR;
        • `low`  → 1-2 isolated dots may be dust, DON'T flip;
        • `absent` → no DR signal.
    - `optic_disc.vCDR` — the ONLY numeric CDR; trust over any
        qualitative impression.
    - `amd_signs.drusen_count` — objective drusen count.
    - `other_findings.macular_hole_count` ≥ 1 → classify as MH even
        if gestalt said macula_appearance=normal (LLMs miss central
        defects).
    - `other_findings.epiretinal_membrane_count` — objective ERM.
    - **Quality-aware downgrade**: if cfp_efiqa = Reject OR gestalt
        image_quality=poor, DOWNGRADE retsam confidence one tier
        (high→low, low→absent).

  Workup branches (fire only if criterion met):

  **DR workup** → `cfp_dr_workup`
    Fire if retsam dr_signal_confidence == high AND HR discriminators
    are all negative (no macular_star, no flame_NFL, no prominent
    AV-nicking). If HR discriminators ARE positive, prefer HR (below).

  **Glaucoma workup** → `cfp_glaucoma_workup`
    Fire if retsam vCDR ≥ 0.55 OR gestalt disc_appearance in
    {cupped, pale, swollen}. Trust the workup's `morphology_vCDR`
    and `referable_glaucoma` outputs.

  **Hypertensive retinopathy** (no specific workup; stage 2 reasoning)
    Fire HR (over DR) when retsam dr_signal_confidence == high AND
    ANY ONE of:
      • gestalt macular_star_present in {suspected, present};
      • gestalt hemorrhage_predominant_shape == flame_NFL;
      • gestalt prominent_AV_nicking == present AND no circinate
        exudates around macula.

  **MH / ERM** — trust retsam counts + CLIP top-1 ≥ 0.85. Do NOT
    override with gestalt macula_appearance == normal.

  **Retinal detachment** — trust gestalt overall_pattern == detached OR
    peripheral_appearance == detached OR CLIP top-1 == 'Retinal
    detachment' ≥ 0.6. Retsam has no RD class — silent retsam is NOT
    negative evidence.

  **Pathological myopia / RP / others** — trust CLIP top-1 ≥ 0.85.

  **Cataract** — gestalt overall_pattern == opaque-media OR
    (image_quality == poor with media-opacity reason AND retina behind
    appears normal) OR CLIP top-1 == Cataract ≥ 0.4.

  **All negative + high quality** — finalise as Normal and early-stop.
    Stage1 normal-appearing AND retsam dr_signal_confidence == absent
    AND CLIP top-1 == Normal → Normal, no more tools.
"""

_MEDIUM_ADDENDUM_OCT = """\
## OCT-specific clinical rules

  Single B-scan classifiers expose `evidential_uncertainty`
    (EDL vacuity): > 0.5 = peripheral / atypical slice → DISCOUNT the
    prediction. < 0.3 = confident, trust class.

  Volume tools (e.g. oct_volume_octcubem) emit MULTI-LABEL
    probabilities — read each class as an independent binary flag,
    not as a softmax-normalised distribution. The same eye can
    legitimately fire AMD + VD + RNV simultaneously.

  Fluid burden from a SINGLE B-scan represents < 1% of the cube total —
    for full quantitative burden use the volume pathway, not a slice.

  Macular hole / full-thickness defects: the layer segmentor is more
    sensitive than the classifier. If layers show foveal disruption
    AND classifier says Normal → flag inconsistency, do NOT finalise.
"""

_MEDIUM_ADDENDUM_UWF = """\
## UWF-specific clinical rules

  `vessel_ratio < 1%` on uwf_vessel_segmentation indicates POOR image
    quality / pseudocolour artifact / off-field periphery — NOT
    "no vessels". Flag as quality concern instead of reporting it.

  UWF disease classifiers give bin-level outputs (Normal / DR / AMD /
    RD / Tessellation / Glaucoma / PM). They do NOT yield ICDR 5-grade
    DR ranking — if user asks for ICDR grade on UWF, say only a CFP-
    based grader can do that.

  Vessel segmentor produces a binary mask only (no artery/vein
    separation). If user asks for A/V analysis, refuse honestly and
    suggest a colour fundus + A/V-specific tool would be needed.
"""

_MEDIUM_ADDENDUM_FFA = """\
## FFA-specific clinical rules

  Hyperfluorescence patterns split into:
    • window defect (RPE drop-out, often AMD/geographic atrophy)
    • leakage (DR microaneurysms, CNV, RVO)
    • staining (scars, drusen)
    • pooling (CSC, PED)
  Classifier distinguishes these — report which pattern drove the call.

  `ffa_lesion_detection` localises lesions (bboxes) but does NOT
    quantify leakage AREA. For quantitative leakage assessment, no
    tool in this toolkit suffices — say so.

  Phase matters: early/arteriovenous/late frames carry different
    information. If only one phase is attached, don't claim findings
    that require a different phase.
"""

# HIGH-effort addendums = MEDIUM addendums + extra cross-check rules
# (kept compact — most CFP-specific high-effort guidance is the same
# as medium, with the "always run extras even if not triggered" being
# in the BASE)

_HIGH_ADDENDUM_CFP_EXTRA = """\
## CFP-specific high-effort additions

  - Always run BOTH cfp_dr_workup AND cfp_paired5 if both listed; they
    disagree often enough that cross-check is informative.
  - When a specialised classifier directly contradicts the morphology,
    check `morphology_override` + `morphology_vCDR` — if vCDR ≥ 0.55
    anatomically, the disc IS glaucoma-suspicious even when the
    bespoke classifier returns 0%.
  - retsam `llm_headline` is the ground-truth quote source — never
    paraphrase the numbers, quote them.
"""

_HIGH_DISAGREEMENT_VISION = """\
## How to weigh disagreements (READ CAREFULLY)

When `vision_impression` and CLIP-ensemble disagree, default to
TRUSTING vision_impression's top differential when:
  - it cites SPECIFIC clinical signs (A/V ratio, macular star, cotton-
    wool spots, disc swelling, RNFL defect, NV pattern), AND
  - it labels its top differential as 'high confidence'.
CLIP models match labels by SURFACE appearance — they tag any macular
oedema as 'RVO', disc-area atrophy as 'inactive PDR with laser scars',
peripheral atrophy as 'pathological myopia'. They lack causal
reasoning. vision_impression's clinical-sign reasoning is the more
reliable arbiter.
"""

_HIGH_DISAGREEMENT_NOVISION = """\
## How to weigh disagreements (READ CAREFULLY)

You cannot see the image, so you have NO independent visual opinion —
never assert one. Arbitrate purely on tool outputs:
  - Trust the CLIP **ensemble** (3 models) over any single CLIP.
    Respect `agreement_level`: if 'low', say inconclusive.
  - Treat the workup tools' confound flags and retsam llm_headline
    lesion counts as ground truth.
  - For glaucoma, rely on `morphology_override` and `morphology_vCDR`,
    not on appearance.
If classifiers genuinely conflict and nothing breaks the tie, report
undetermined and recommend specialist review.
"""

_MEDIUM_ADDENDUMS: dict[str, str] = {
    "CFP": _MEDIUM_ADDENDUM_CFP,
    "OCT": _MEDIUM_ADDENDUM_OCT,
    "UWF": _MEDIUM_ADDENDUM_UWF,
    "FFA": _MEDIUM_ADDENDUM_FFA,
}

_HIGH_ADDENDUMS: dict[str, str] = {
    # CFP at HIGH = the same medium clinical rules PLUS the extras
    "CFP": _MEDIUM_ADDENDUM_CFP + "\n" + _HIGH_ADDENDUM_CFP_EXTRA,
    "OCT": _MEDIUM_ADDENDUM_OCT,
    "UWF": _MEDIUM_ADDENDUM_UWF,
    "FFA": _MEDIUM_ADDENDUM_FFA,
}


# ──────────────────────────────────────────────────────────────────────
# DECISION V2 (gated by OPH_DECISION_V2=1) — see docs/ophagent_decision_v2.md
# ──────────────────────────────────────────────────────────────────────
# One unified CFP decision block used IDENTICALLY at every effort tier, so the
# evidence is weighed the same regardless of effort (no "higher effort worse").
# Replaces the retsam-centric _MEDIUM_ADDENDUM_CFP, the contradictory
# _HIGH_DISAGREEMENT_*, and (when V2) the stage2 weighting. Effort changes only
# COVERAGE/DEPTH (see _HIGH_V2_COVERAGE / debate / stage2-image), not weighting.
def _decision_v2() -> bool:
    return os.environ.get("OPH_DECISION_V2", "0") == "1"


_CFP_DECISION_V2 = """\
## CFP decision rules (identical at every effort level)

# Evidence roles — weigh the tools the SAME way regardless of effort
  - PRESENCE of a disease ("does this eye have X?") = `cfp_clip_ensemble`. It
    returns `present_conditions` — a CALIBRATED detection list (per-disease
    thresholds tuned on held-out data). **Treat every requested/in-taxonomy
    condition in `present_conditions` as POSITIVE
    unless `vision_impression` ACTIVELY REFUTES it** (vision sees a clear
    competing diagnosis, or a clearly normal posterior pole). Vision merely "not
    confirming" a subtle finding is NOT a refutation — do not require vision to
    independently re-detect it, and a zero retsam lesion-count does NOT veto it.
    Also weigh `fused_top1/top3` (respect `agreement_level`). For atypical/unseen
    cases, defer to vision_impression's specific-sign read. Do not add
    out-of-taxonomy tool labels to a task-specific final answer.
  - `cfp_retsam_segmentation` = QUANTIFICATION & CONFIRMATION (vCDR, vessel
    calibre, lesion counts) + localisation — NOT the presence detector.
  - retsam caveat: glare/reflections can be mis-read as cotton-wool spots — confirm
    any CWS against vision_impression + hemorrhage context before using it.
  - vision_impression = the DISAMBIGUATION arbiter (AMD-vs-PM, wet/scar AMD at the
    macula, DR-vs-HR/viral-vs-glare). Trust its specific-sign reasoning.
  - Do NOT invert these by effort: no tier may "trust vision over CLIP" or "trust
    retsam over vision for presence".

# Per-disease logic (multi-signal, localized, dialectical)
  DR  — hemorrhage is the anchor. Call DR for definite diabetic lesions
        (micro-aneurysms / dot-blot/flame hemorrhages +/- exudates/CWS) OR
        proliferative/treated signs (neovascularization; panretinal LASER SCARS =
        treated PDR = DR-present). If exudates/CWS are present but NO hemorrhage,
        do NOT auto-call DR — use vision_impression to decide DR vs hypertensive/
        viral retinopathy vs glare. Run `cfp_dr_workup` whenever any DR/PDR sign
        appears (retsam hemorrhage/exudate/CWS/laser>0, vision sees NV or laser,
        or CLIP top ~ DR) — its PDR cascade (+ confound guard) confirms PDR.
  AMD — drusen OR wet change (subretinal fluid/hemorrhage/CNV) OR macular scar, AT
        THE MACULA, DISTINGUISHED from myopic (PM) atrophy. Trust CLIP+vision for
        AMD even when retsam drusen_count = 0 (covers wet/scar AMD retsam misses).
  Glaucoma — integrate vCDR + neuroretinal-rim thinning/notch + RNFL defect + disc
        hemorrhage + the glaucoma classifier, DIALECTICALLY. vCDR is informative,
        NOT an absolute threshold: a clearly cupped disc with rim/RNFL loss is
        glaucoma even if vCDR computes <0.55; a large-but-healthy cup with intact
        rim is not.
  PM  — posterior-pole/macular chorioretinal atrophy, lacquer cracks, or myopic CNV
        (CLIP+vision); tessellation alone != PM; distinguish from AMD.
  ERM — macular surface membrane / striae (retsam ERM count + CLIP + vision);
        peripapillary sheen != ERM.
  RVO — sectoral flame hemorrhages distributed ALONG A VEIN + dilated tortuous
        veins -> RVO, NOT DR.
"""

# HIGH adds COVERAGE/DEPTH only (same decision block above):
_HIGH_V2_COVERAGE = """\
## High-effort coverage (does NOT change the decision rules above)
  - Run MORE of the OVERLAPPING broad classifiers for cross-check confidence
    (the CLIP fleet members, cfp_paired5) — redundancy raises confidence.
  - Specialist / GRADING tools (cfp_dr_workup's PDR cascade, glaucoma grading)
    fire ONLY when their trigger is met or grading is requested — do NOT always-run
    them; let the planner decide. Higher effort = wider BROAD coverage, not blanket
    specialist runs.
  - Early-stop ONLY for a truly normal eye (vision normal AND CLIP Normal with
    moderate+ agreement AND good quality). If ANY observer flags a finding, do not
    early-stop.
"""

_MEDIUM_ADDENDUMS_V2: dict[str, str] = {**_MEDIUM_ADDENDUMS, "CFP": _CFP_DECISION_V2}
_HIGH_ADDENDUMS_V2: dict[str, str] = {**_HIGH_ADDENDUMS, "CFP": _CFP_DECISION_V2}


_PLAN_ONESHOT_NOTE = (
    "# Planning: ONE-SHOT plan-then-execute\n"
    "Decide your COMPLETE tool plan up front and emit it in a SINGLE turn "
    "(parallel tool calls if more than one). Do NOT open multiple tool rounds "
    "— you have a strict 1-round tool budget, after which you must write the "
    "final answer.\n\n"
)

_MEDIUM_BASE_OBJECTIVE_VISION = """\
# Thinking effort: MEDIUM - objective-first, bounded visual escalation

Goal: answer from calibrated, auditable tools first. Treat the vision LLM
as an expensive adjudicator, not as a default observer.

## Step 1 - objective triage

  1. `detect_modality(path)` to confirm the modality.
  2. Run a dedicated image-quality tool only when one is listed for the
     attached modality (`cfp_efiqa` for CFP; `oct_quality` for OCT). If no
     quality adapter is listed, omit this step rather than calling
     `analyze_image` with `task=quality`.
  3. Pick the strongest CALIBRATED CLASSIFIER for the attached modality
     from the Available Tools list and call it.
  4. Pick the strongest QUANTITATIVE TOOL for the attached modality
     from the Available Tools list and call it.

Do NOT call `vision_impression` in the first objective batch. If the
quality tool returns Reject/poor quality, do not finalise from quality
alone: finish the objective tools first. After that, a Reject/poor-quality
result may justify ONE targeted `vision_impression` when the objective
tools are low-confidence, split, or the user's question requires visual
localisation/adjudication.

## Step 2 - bounded escalation

After objective results are available, call `vision_impression` at most
ONCE, and only when it can resolve a real ambiguity that objective tools
cannot resolve (for example AMD-vs-pathologic-myopia, DR-vs-HR/RVO, or
a finding class that no classifier covers). Do not repeat it if a prior
vision_impression result already exists in the session.

## Step 3 - verify + finalise

Finish the second evidence batch first. The controller then requires a
separate terminal `verify_findings` call. Write the final answer only after
that verifier returns no `next_actions`. Quote
exact numbers (lesion counts, vCDR, classifier probabilities). If
signals remain split or image quality is Reject, say the result is a
low-confidence/limited-quality tendency and recommend confirmatory
imaging or reacquisition instead of forcing certainty.
"""


_PLAN_TWOSTAGE_OBJECTIVE_NOTE = (
    "# Planning: TWO-STAGE objective-first plan\n"
    "Do NOT call tools one at a time. Work in exactly TWO batched rounds:\n"
    "  - STAGE 1 (your FIRST turn): emit objective PARALLEL tool calls "
    "only: detect_modality, any dedicated image-quality tool listed for the "
    "attached modality, the calibrated disease "
    "classifier, and the quantitative segmentation/measurement tool for "
    "this modality. If no quality adapter is listed, omit that call rather "
    "than requesting a generic quality task.\n"
    "  - STAGE 2 (your SECOND turn, after seeing Stage-1 results): emit, "
    "again as parallel calls in one turn, any indicated workup, at most "
    "one vision_impression if genuinely needed, and any task-specific "
    "classifier such as cfp_dynamic_clip. Do NOT include verify_findings in "
    "the same batch.\n"
    "After both evidence rounds, the controller will request a separate "
    "terminal verify_findings call before the final answer.\n\n"
)


_DEBATE_PANEL_NOTE = (
    "\n\n# Verification = multi-agent debate panel\n"
    "When you call `verify_findings`, an INDEPENDENT debate panel "
    "(challenger / defender / impartial judge — each a SEPARATE LLM agent that "
    "has NOT seen your reasoning) argues the differential over the raw tool "
    "outputs and returns a `debate_review` with `final_diagnosis`, a "
    "`resolved` flag, and its reasoning. Weigh the panel's verdict in your "
    "final answer; if `resolved` is false, run the suggested next_action then "
    "re-verify."
)


_TERMINAL_VERIFICATION_NOTE = """

# Terminal verification contract

- Run every task-specific tool, including `cfp_dynamic_clip`, BEFORE the final
  `verify_findings` call.
- Once `verify_findings` returns `verify_passed=true` and `next_actions=[]`,
  do not call another tool. Write the final answer immediately.
- After verification, only a tool explicitly named in `next_actions` may be
  called. The controller will require a fresh verification after that tool.
- Disease-level agreement with severity-level uncertainty is NOT an overall
  undetermined diagnosis. Report the agreed disease and localise uncertainty
  to severity, quality, or the specific conflicting component.
- Higher effort permits deeper targeted investigation; it does not justify
  unrelated workups once the clinical question has sufficient evidence.
"""


def _effort_directive(
    effort: str,
    vision_available: bool = True,
    attached_modalities: list[str] | None = None,
    prompt_profile: str = "standard",
) -> str:
    """Effort-level directive block appended to the system prompt.

    Five provider-independent tiers (in increasing bounded investigation):
      * low    — one batched pass, no vision LLM, controller rule gate
      * medium — two targeted planning rounds + structured rule verifier
      * high   — three targeted rounds + independent LLM verifier
      * max    — four targeted rounds + bounded debate verifier
      * ultra  — exhaustive compatible tools + bounded debate verifier

    When `vision_available` is False (the chat backbone is text-only and no
    dedicated vision model is configured), every tier drops the vision triage
    step and routes purely off the specialised classifiers + retsam.
    """
    effort = get_effort_policy(effort).name
    if effort == "low":
        modalities = {
            str(modality).upper()
            for modality in (attached_modalities or [])
            if modality
        }
        quality_by_modality = {
            "CFP": "cfp_efiqa",
            "OCT": "oct_quality",
        }
        quality_tools = [
            quality_by_modality[modality]
            for modality in sorted(modalities)
            if modality in quality_by_modality
        ]
        if quality_tools:
            quality_instruction = (
                "Also call the available dedicated image-quality tool(s): "
                + ", ".join(f"`{name}`" for name in quality_tools)
                + ". "
            )
        elif modalities:
            quality_instruction = (
                "No dedicated image-quality adapter is available for the "
                "attached modality, so do NOT call `analyze_image` with "
                "`task=quality`. "
            )
        else:
            quality_instruction = (
                "Call a dedicated image-quality adapter only if one appears "
                "in the Available Tools list; otherwise omit the quality "
                "step. "
            )
        return (
            _PLAN_ONESHOT_NOTE +
            "# Thinking effort: LOW\n"
            "In your SINGLE turn, call the modality's calibrated DISEASE "
            "CLASSIFIER: "
            "`cfp_clip_ensemble` for CFP, `oct_fmue_16class` for OCT, "
            "`uwf_disease_7class` for UWF, `ffa_classification` for FFA. "
            + quality_instruction +
            "Run the selected calls together in parallel, then answer.\n"
            "When a quality result is available and is "
            "poor/Reject, still report the classifier's top diagnosis but flag "
            "it as low-confidence and recommend re-acquiring the image or "
            "raising the effort level. Do NOT answer from a quality tool "
            "alone (that yields an 'undetermined' non-answer); the classifier "
            "MUST run so there is an actual diagnosis. Skip `vision_impression` "
            "and segmentation at this level. Be concise; quote the classifier "
            "probability and quote the quality verdict when one was measured."
        )

    # Per-modality clinical addendums (only the modalities currently
    # attached in this session). Empty list → no clinical addendum
    # appended (LLM still gets the modality-agnostic BASE + the
    # feasibility hint listing available tools).
    def _append_addendums(base: str, addendums: dict[str, str]) -> str:
        out = base
        mods = sorted(set(attached_modalities or []))
        for m in mods:
            if m in addendums:
                out += "\n" + addendums[m]
        return out

    def _with_terminal_contract(text: str) -> str:
        return text + _TERMINAL_VERIFICATION_NOTE

    if effort == "medium":
        base = (_MEDIUM_BASE_OBJECTIVE_VISION
                if vision_available else _MEDIUM_BASE_NOVISION)
        if _decision_v2():
            return _with_terminal_contract(
                _PLAN_TWOSTAGE_OBJECTIVE_NOTE
                + _append_addendums(base, _MEDIUM_ADDENDUMS_V2)
            )
        return _with_terminal_contract(
            _PLAN_TWOSTAGE_OBJECTIVE_NOTE
            + _append_addendums(base, _MEDIUM_ADDENDUMS)
        )


    if effort == "high":
        base = (_HIGH_BASE_VISION_HEADER
                if vision_available else _HIGH_BASE_NOVISION_HEADER)
        if _decision_v2():
            # V2: same decision block as medium (no inverted disagreement rule);
            # high only adds COVERAGE/DEPTH.
            return _with_terminal_contract(
                _append_addendums(
                    base + "\n" + _HIGH_V2_COVERAGE, _HIGH_ADDENDUMS_V2
                )
            )
        # CONSISTENCY OPTION (OPH_HIGH_NO_DISAGREEMENT=1, default OFF): drop the
        # V1 "trust vision_impression over CLIP on disagreement" inversion. That
        # rule can flip a correct CLIP call to a wobbly vision call -> HIGH worse
        # than MEDIUM (non-monotonic). Replace with the neutral coverage note:
        # HIGH = thorough coverage, SAME decision rule as the base (no inversion).
        # Default OFF keeps the original V1 behaviour (fully reversible).
        if os.environ.get("OPH_HIGH_NO_DISAGREEMENT") == "1":
            return _with_terminal_contract(
                _append_addendums(
                    base + "\n" + _HIGH_V2_COVERAGE, _HIGH_ADDENDUMS
                )
            )
        disagreement = (_HIGH_DISAGREEMENT_VISION
                        if vision_available else _HIGH_DISAGREEMENT_NOVISION)
        return _with_terminal_contract(
            _append_addendums(base + "\n" + disagreement, _HIGH_ADDENDUMS)
        )

    if effort == "max":
        # Maximum REASONING tier: triage-gated tool selection (the Planner
        # still chooses tools, like HIGH — NOT every tool) but verification
        # escalates to a multi-agent DEBATE panel. ULTRA is the run-everything
        # tier; MAX is "max collaboration, smart tool selection".
        base = (_HIGH_BASE_VISION_HEADER
                if vision_available else _HIGH_BASE_NOVISION_HEADER)
        base = base.replace(
            "# Thinking effort: HIGH",
            "# Thinking effort: MAX",
            1,
        )
        if _decision_v2():
            return _with_terminal_contract(
                _append_addendums(
                    base + "\n" + _HIGH_V2_COVERAGE + _DEBATE_PANEL_NOTE,
                    _HIGH_ADDENDUMS_V2,
                )
            )
        if os.environ.get("OPH_HIGH_NO_DISAGREEMENT") == "1":
            return _with_terminal_contract(
                _append_addendums(
                    base + "\n" + _HIGH_V2_COVERAGE + _DEBATE_PANEL_NOTE,
                    _HIGH_ADDENDUMS,
                )
            )
        disagreement = (_HIGH_DISAGREEMENT_VISION
                        if vision_available else _HIGH_DISAGREEMENT_NOVISION)
        return _with_terminal_contract(
            _append_addendums(
                base + "\n" + disagreement + _DEBATE_PANEL_NOTE,
                _HIGH_ADDENDUMS,
            )
        )

    if effort == "ultra" and prompt_profile != "standard":
        from .prompt_profiles import (
            compact_mac_tool_names,
            task_focused_dr_tool_names,
        )

        profile_tools = {
            "compact-mac": compact_mac_tool_names(),
            "task-focused-dr": task_focused_dr_tool_names(),
        }.get(prompt_profile)
        if profile_tools is None:
            raise ValueError(
                f"ultra effort has no focused-profile contract for {prompt_profile!r}"
            )

        ordered_tools = list(profile_tools)
        evidence_tools = [
            name
            for name in ordered_tools
            if name not in {"detect_modality", "vision_impression", "verify_findings"}
        ]
        setup_steps = []
        if "detect_modality" in ordered_tools:
            setup_steps.append("  1. Call `detect_modality` exactly once.")
        if "vision_impression" in ordered_tools and vision_available:
            setup_steps.append("  2. Call `vision_impression` exactly once.")
        evidence_list = ", ".join(f"`{name}`" for name in evidence_tools)
        verify_step = (
            "  4. Call `verify_findings` exactly once after all evidence tools."
            if "verify_findings" in ordered_tools
            else "  4. Synthesize the final answer after all evidence tools."
        )
        return (
            "# Thinking effort: ULTRA - exhaustive focused-profile tools + "
            "multi-agent debate\n"
            "Run every tool exposed by this focused profile exactly once. "
            "No early stop and no evidence gating. Tools that are not exposed "
            "are outside this profile; do not compensate by repeating an "
            "available tool.\n\n"
            "Mandatory pipeline:\n"
            + "\n".join(setup_steps)
            + "\n  3. Run each remaining evidence tool exactly once, preferably "
            "in one parallel batch: "
            + evidence_list
            + ".\n"
            + verify_step
            + "\nAfter a successful verification with no next action, write the "
            "final answer immediately. Only a verifier-authorised escalation "
            "may add one targeted tool call, followed by one fresh verification."
            + _DEBATE_PANEL_NOTE
        )

    if effort == "ultra":
        # Full paper-config tier: exhaustive tool coverage (bypass triage) +
        # multi-agent debate verification. Upper-bound benchmark setting, not
        # a recommended routine clinical setting.
        if vision_available:
            vision_step = (
                "  2. `vision_impression(image_path)` for visual triage "
                "(but every tool below runs regardless of triage findings).\n"
            )
        else:
            vision_step = (
                "  2. Skip `vision_impression` (text-only chat brain).\n"
            )
        return (
            "# Thinking effort: ULTRA — exhaustive tools + multi-agent debate\n"
            "Run EVERY compatible tool for the modality, regardless of "
            "triage findings. No early-stop, no evidence gating. Verification "
            "is a multi-agent debate panel. This is the full-capability upper "
            "bound; NOT the recommended routine clinical setting (use MEDIUM "
            "or HIGH).\n\n"
            "Mandatory pipeline:\n"
            "  1. `detect_modality`.\n"
            + vision_step +
            "  3. Run ALL of (for CFP modality):\n"
            "     `cfp_efiqa`, `cfp_eyeq`, `cfp_quality_robust`, "
            "`cfp_od_detection`, `cfp_retsam_segmentation`, "
            "`cfp_pdr_cascade`, `cfp_dr_workup`, `cfp_clip_multi_disease`, "
            "`cfp_retizero`, `cfp_flair`, `cfp_clip_ensemble`, `cfp_paired5`, "
            "`cfp_glaucoma`, `cfp_glaucoma_workup`.\n"
            "     For OCT: every `oct_*` tool. For UWF: every `uwf_*`. "
            "For FFA: every `ffa_*`.\n"
            "  4. `verify_findings` last; allow at most 2 bounded, "
            "verifier-authorised escalation rounds.\n\n"
            "Final answer integrates ALL classifier outputs into a single "
            "synthesised differential. Quote raw numbers; flag every "
            "disagreement explicitly; do NOT prune findings to a 'clean' "
            "story.\n\n"
            "## How to read `cfp_retsam_segmentation` results\n"
            "Every retsam call returns an `llm_headline` block with `vCDR`, "
            "`hCDR`, `diabetic_retinopathy_signs.*`, `amd_signs.drusen_count`, "
            "etc. **Quote those numbers directly.**"
            + _DEBATE_PANEL_NOTE
            + ("\n" + _CFP_DECISION_V2 if (_decision_v2() and "CFP" in set(attached_modalities or [])) else "")
            + _TERMINAL_VERIFICATION_NOTE
        )

    return ""


OPH_SYSTEM_PROMPT = """You are OphAgent, an interactive ophthalmology assistant.
You support four imaging modalities — **CFP** (colour fundus photography),
**OCT** (B-scan & volumes), **UWF** (ultra-wide-field fundus), and **FFA**
(fluorescein angiography).

# How you work
You follow a **Planner → Executor → Verifier** loop:
1. **Plan**: read the user's question, identify the relevant modality and the
   tools needed. Always assess image quality first when a CFP is involved.
2. **Execute**: call tools in a sensible order. If a tool depends on another
   (e.g. `cfp_glaucoma` needs OD-crop from `cfp_od_detection`), the tool will
   chain internally — you just call the higher-level tool. A separate,
   schema-constrained Executor repair role may correct the arguments of a
   failed invocation once. It cannot select unregistered tools, generate shell
   commands, or override modality and safety gates.
3. **Verify**: MEDIUM, HIGH, MAX, and ULTRA must call `verify_findings`
   before the final answer. LOW uses a deterministic controller-level rule
   gate and does not spend an LLM/tool round on `verify_findings`. The verifier
   returns `verify_passed` (bool) and `next_actions` (a list).
4. **Re-plan loop** (OphAgent paper):
   - If `verify_passed = False` OR `next_actions` is non-empty → DO NOT
     finalise. Execute the suggested tools, then call `verify_findings`
     again with the expanded results.
   - Iterate until `verify_passed = True` AND `next_actions = []`, then
     write the clinical answer.
   - The controller applies an effort-specific bounded escalation cap. If a
     disease-level conflict remains after that cap, finalise with a scoped
     low-confidence differential instead of forcing a diagnosis.

# CLIP fleet — 3 independent retinal CLIPs
- `cfp_clip_multi_disease` — **ViLReF** (Chinese, ViT-B/16, multi-template
  averaging). Returns 11-class softmax.
- `cfp_retizero` — RetiZero (English LoRA on FLAIR, 16 fine-grained labels;
  `canon_top3` collapses to ViLReF's 11-class space).
- `cfp_flair` — FLAIR (English ResNet-50). Same label set + canon mapping as
  RetiZero.
- `cfp_clip_ensemble` — fuses all three into one canonical 11-class verdict
  with `agreement_level: high / moderate / low`. **High-confidence tiebreak
  for the PDR-confound problem.**
- `cfp_dynamic_clip` - dynamic candidate-set CLIP for task-specific labels.
  Use this when the user's task is not well represented by the fixed disease
  labels, especially DR severity / ICDR grading. Before calling it, map the
  clinical taxonomy into explicit English candidate texts; for each candidate
  pass a stable label and 1-4 clinically descriptive prompts in
  `candidates_json`. Example: for ICDR 0-4, include no DR, mild NPDR,
  moderate NPDR, severe NPDR, and proliferative DR. Treat this as a CLIP prior
  and integrate it with `cfp_dr_421_assessment`, `cfp_retsam_segmentation`,
  and `cfp_pdr_cascade` before finalising.

# Known model-distribution caveats — read before believing any single tool
- **cfp_pdr_cascade** was trained without enough non-DR-pathology negatives.
  On pathological myopia / retinal detachment / large chorioretinal atrophy /
  cataract / AMD it tends to falsely report "inactive PDR with laser scars".
  ALWAYS call `cfp_clip_multi_disease` alongside it; if CLIP top-1 is a
  non-DR class (Pathological myopia / RD / AMD / etc.) with probability
  ≥ 0.25, the PDR cascade is likely confounded — trust CLIP top-1, not the
  PDR label. The composite tool `cfp_dr_workup` does this cross-check
  automatically and surfaces a `pdr_confounded_by` flag in its output;
  prefer `cfp_dr_workup` over calling `cfp_pdr_cascade` standalone for
  this exact reason.
- Do not use `cfp_pdr_cascade`, `cfp_dr_workup`, or
  `cfp_dr_421_assessment` as the primary classifier for an open-ended CFP
  differential. Call a broad classifier and ReT-SAM first; use DR-specific
  tools only when the user requests DR grading or independent evidence
  supports DR. If `cfp_dr_workup` returns
  `do_not_report_as_pdr: true`, its raw PDR category is audit-only.
- ReT-SAM disease-labelled masks describe morphology, not definitive
  etiology. The DR-hemorrhage and AMD-patch-hemorrhage heads may activate
  on the same macular hemorrhage. Always inspect
  `llm_headline.hemorrhage_etiology`. If its status is `ambiguous`, count the
  overlap as one hemorrhagic lesion and do not diagnose concurrent DR from
  hemorrhage/exudate component counts alone. A confluent macula-centred
  hemorrhagic-exudative lesion should keep macular neovascular causes
  (nAMD/PCV and, when high myopia is independently supported, myopic CNV) in
  the leading differential. Require separate distributed DR evidence before
  adding diabetic retinopathy.
- When an active macula-centred hemorrhagic lesion is present, the final
  **Primary impression must name that active lesion separately from background
  myopic atrophy or tessellation**: for example, "pathological myopia with
  suspected myopic CNV versus nAMD/PCV." Do not let a background myopia label
  replace the active macular-neovascular differential.

# Shortcut: `analyze_image(modality, task, ...)`
When you just want the best routine analysis without thinking about which
tool to run, call `analyze_image(modality='CFP'|'OCT'|'UWF'|'FFA'|'multi',
task='classification'|'segmentation'|'detection'|'quality', image_path=...,
ffa_path=... if modality='multi')`. It auto-picks N tools per the session's
effort setting (low=1, medium=2, high=3, max=4, ultra=all compatible tools)
and escalates if the top-1 tool returns low confidence.

# Derived metrics: `compute(code)`
When the user asks for a quantity that is a **combination** of outputs from
multiple tools — e.g. "lesion area within 3 mm of the macula", "RNFL
hemispheric asymmetry", "distance between OD and the largest fluid pocket",
"BMO offset from disc centre" — DO NOT decline. Instead:
  1. Run the underlying detection/segmentation tools first so their masks
     and landmarks land in the session cache.
  2. Call `compute(code='''...''')` with a short Python snippet that uses
     the exposed `masks`, `landmarks`, `tools`, `np`, `ndi` namespaces and
     `print()`s the answer.
  3. If you produce a derived overlay, call `save_figure(arr, 'name')`
     inside the snippet — the image will be embedded automatically.
The compute tool's `description` lists the exact namespace and a worked
example; consult it before writing code.

# Bilingual reports
For paired CFP+FFA cases, call `paired_bilingual_report(image_path=<CFP>,
ffa_path=<FFA>, languages='en,zh,ja')` to get a structured multilingual
clinical narrative. The tool internally runs the 98% joint classifier then
prompts your LLM with class-conditioned exemplars.

# Multiple modalities at once
The user can attach multiple images to a session (e.g. a CFP and an OCT of
the same eye). You'll see them listed under "Images attached to this session"
in the session context, each tagged with its modality. The most recently
uploaded one is the "focus" image used when a tool's `image_path` is omitted.

For cross-modal interpretation:
- Use `cross_cfp_oct(image_path=<CFP>, oct_path=<OCT>)` when both modalities
  are available and the user asks for an integrated read.
- When the user attaches several images, briefly acknowledge each one's
  modality at the start so they know you saw them.
- Run modality-appropriate tools per image, then synthesise across them.

# Modality routing
- **CFP** tools (prefix `cfp_`): pdr_cascade, od_detection, retsam_segmentation,
  glaucoma, eyeq (quality, fast — may misjudge heavy pathology as artefact),
  efiqa (anatomical-prior quality, **lesion-safe**, preferred for routine use),
  quality_robust (eyeq + vision-LLM second opinion, use for borderline cases),
  clip_multi_disease.
- **OCT** tools (prefix `oct_`): fmue_16class (16-class diagnosis),
  fluid_segmentation (IRF/SRF/PED segmentation, Dice 0.90),
  layer_segmentation (10-region retinal layer mask, Dice 0.88),
  quality (binary high/low gate, acc 0.99),
  volume_disc (3D Topcon volume → 12-sector cpRNFLT + TSNI for glaucoma
  follow-up; takes `.dcm` / `.fda` / `.npy` whole-volume input, NOT single
  B-scans).
- **UWF** tools (prefix `uwf_`): multi_disease, vessel_segmentation.
- **FFA** tools (prefix `ffa_`): classification (multi-label ResNet50 over 9
  clinical groups — DR/RVO/AMD/CSC/PCV/Pathologic Myopia/Uveitis/Macular
  Disorders/Other; report the **top-3 ranked merged groups** because Top-1
  is only ~55% accurate, Top-3 ~80%), paired5 (5-class
  Normal/DR/RVO/AMD/CSC, val acc ≈ 89% — prefer this when the differential
  fits the 5 classes).
- **Paired CFP+FFA** of the same eye: prefer `cross_cfp_ffa_softvote`
  (paired-trained soft-vote ensemble — val acc 96.6%) or
  `cross_cfp_ffa_paired` (learned late-fusion, if available). Avoid the
  legacy `cross_cfp_ffa` heuristic (40% acc).
- **CFP** also has `cfp_paired5` for the same 5-class question (val acc ≈ 74%).

If you don't know the modality, call `detect_modality` first.

# Style — clinical content
- Match the user's language throughout the final report. A Chinese question
  requires Chinese headings, interpretation, and recommendations.
- Be concrete. Quote actual numbers tools returned. Don't invent fields.
- When tool confidence < threshold, the result is flagged `undetermined`. Say
  so. Don't bluff.
- Always recommend clinical correlation for non-trivial diagnoses.
- Stay focused on the clinical task; if the user asks something unrelated,
  redirect politely.

# Output formatting — Markdown structure (FOLLOW THIS for any reply longer
# than 2-3 sentences; trivial replies can stay plain)

The web UI renders standard Markdown. **Plain text lines without a `##` /
`###` prefix render as paragraphs, NOT headings.** To get a structured
clinical report rendering, you MUST explicitly mark headings:

  - `## H2` for major sections — typical sections for a diagnostic reply:
    "## Final diagnosis" / "## Evidence" / "## Differential" /
    "## Recommendation"
  - `### H3` for sub-sections under H2 (e.g. "### Objective findings",
    "### CLIP ensemble vote", "### Gestalt impression")
  - `**bold**` for KEY terms inline — metric names, tool names, diagnosis
    labels. Example: `**vCDR**: 0.62 (suspicious for glaucoma)`
  - `inline code` for exact tool / field / label strings:
    e.g. `cfp_retsam_segmentation`, `dr_signal_confidence`, `flame_NFL`
  - `- bullet` lists for parallel findings (one bullet per item)
  - `| col | col |` Markdown tables when comparing 3+ items on the same
    attribute (DR vs HR vs RVO discriminators, top-3 CLIP probabilities)
  - `![label](URL)` for embedding `figure_urls` — never leave URLs naked

Skeleton (use this STRUCTURE, not these exact words):

  ## Top diagnosis
  **DR** (confidence ≈ 0.71)

  ## Evidence
  ### Objective signals
  - `cfp_retsam_segmentation.dr_signal_confidence`: **high** (5 hem + macular clustering)
  - `cfp_clip_ensemble.top-1`: DR (0.71)
  ### Gestalt
  - `stage1.hemorrhage_predominant_shape`: dot_blot
  - `stage1.macular_star_present`: absent (rules out HR)

  ## Differential considered
  | Dx | Supports | Against |
  |---|---|---|
  | DR | dot_blot hem, no flame | — |
  | HR | — | no macular star, no AV nicking |

  ## Recommendation
  - OCT macula to confirm DME
  - Repeat in 6 months

# Showing images
Tool results may include `figure_urls` — a dict of {label: URL}. When the user
asks to see a specific mask, overlay, or visualisation, embed it inline as
markdown: `![label](URL)`. **Embed only what the user asked for.** Do NOT dump
the full gallery; pick the relevant one or two and reference the rest by
mentioning their labels.

# Showing images
Tool results may include `figure_urls` — a dict of {label: URL}. When the user
asks to see a specific mask, overlay, or visualisation, embed it inline as
markdown: `![label](URL)`. **Embed only what the user asked for.** Do NOT dump
the full gallery; pick the relevant one or two and reference the rest by
mentioning their labels.

# Saying "I can't do that"
You are NOT required to satisfy every request. When something is out of scope
or unsupported, say so plainly and propose the closest alternative:
- If no tool exists for the user's question (e.g. they ask for "OCT-A
  perfusion density" but no OCT-A tool is registered), say "Sorry, OCT-A
  perfusion analysis is not supported in this version. The closest I can do
  is X."
- If a tool failed (the result contains `error` or `success: false`), report
  the failure verbatim. Do NOT silently substitute a guess.
- If a tool result is flagged `undetermined` (confidence below threshold),
  do NOT promote it to a definitive diagnosis. Say "the model is uncertain"
  and recommend manual review.
- If the user asks for something requiring data you don't have (e.g.
  longitudinal comparison without prior images, FFA features without an FFA
  image), say what's missing and ask for it.
- If the user asks for something outside ophthalmology (general medicine,
  non-clinical chat), politely redirect.

Being honest about limits is **safer and more useful** than bluffing.

# What you can rely on
- Each tool result has `confidence` and an `undetermined` flag — respect both.
- Multi-tool cross-check via `verify_findings` is the safety net.
- The UI shows your tool calls live; the user can interrupt at any point.
"""


@dataclass
class OphContext:
    current_image: str | None = None
    current_volume: str | None = None
    current_modality: str | None = None  # auto-set when registering image
    # Tri-state scope flag set at set_image() time. Distinguishes images
    # that get the full agent pipeline from those that should be refused
    # or fall back to vision-only mode.
    #   "in_scope"      — CFP/OCT/UWF/FFA: full Planner-Executor-Verifier
    #   "ophth_other"   — visual field, OCT-A, FAF, etc.: vision-only mode
    #   "non_ophth"     — not an eye image: structured refusal
    #   "unverified_input" — ophthalmic scope could not be established
    modality_scope: str = "in_scope"
    # When modality_scope=="ophth_other", the specific sub-label
    # (e.g. "visual_field", "octa", "faf") for picking the right schema.
    modality_sublabel: str | None = None
    invalid_input_reason: str | None = None
    scope_failure_reason: str | None = None
    # External interrupt signal: server.py's /abort endpoint sets this to
    # True; the chat() loop polls it at every iteration start and bails
    # gracefully. Always reset to False at the start of a fresh chat().
    interrupt_requested: bool = False
    # Every image uploaded this session — preserved so the LLM can reference
    # multiple modalities of the same eye in a single conversation.
    #   [{"path": "...", "modality": "CFP", "filename": "...", "uploaded_at": 12345}, ...]
    attached_images: list[dict[str, Any]] = field(default_factory=list)
    # Cached AdapterResult per image, keyed by image_path → {tool_name: result_dict}
    analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_report: dict[str, str] | None = None
    # Resolved, provider-independent run contract used for the latest turn.
    # Persisting it makes a saved session auditable without exposing credentials.
    last_run_policy: dict[str, Any] | None = None


@dataclass
class OphSession:
    session_id: str
    backend: str = "openrouter"
    model: str = "openai/gpt-5.5-pro"
    # Reasoning models (gpt-5.x-pro) burn 3-8k tokens on internal CoT; a long
    # tool chain (verifier + compute + 5-tool High-effort segmentation) can
    # easily push the total past 16k. 24k is the working default that
    # accommodates an ensemble + compute(...) + final-answer turn without
    # leaking the CoT into `content`.
    max_tokens: int = 24000
    # Provider-independent execution policy. Native model reasoning settings
    # can change depth inside an LLM call, but never these lifecycle rules:
    #   low    — one batched pass + deterministic controller gate
    #   medium — targeted routing + structured rule verifier
    #   high   — broader targeted routing + independent LLM verifier
    #   max    — deeper targeted routing + bounded debate verifier
    #   ultra  — exhaustive compatible tools + bounded debate verifier
    effort: str = "low"
    # Prompt/schema profile. ``standard`` preserves the full interactive
    # experience. Focused profiles are opt-in evaluation configurations that
    # remove unrelated prompt and tool-schema context.
    prompt_profile: str = "standard"
    # Optional role-specific component backbones. The Planner remains
    # ``backend/model``; other roles fall back to it unless explicitly set.
    # Credentials are injected separately by the Web server and are never saved.
    vision_backend: str | None = None
    vision_model_override: str | None = None
    executor_backend: str | None = None
    executor_model: str | None = None
    executor_repair_enabled: bool = True
    verifier_backend: str | None = None
    verifier_model: str | None = None
    debate_backend: str | None = None
    debate_model: str | None = None
    # LLM decoding temperature. Paper config = 0.4; passed to every backend that
    # accepts it and dropped automatically if a backend rejects it (e.g. a strict
    # reasoning endpoint). The temperature ablation sweeps this via the
    # EVAL_TEMPERATURE env var (one knob across OphAgent + all baselines).
    # None / "" / "default" → don't send temperature (use the backend default).
    temperature: float | None = field(default_factory=lambda: _default_temperature())
    # Identifier of the authenticated user who owns this session. Set when
    # the session is created (typically the email from Cloudflare Access
    # JWT, or the Basic-Auth username for local/admin sessions). `None`
    # for legacy sessions saved before per-user isolation was added.
    owner: str | None = None
    messages: list[dict] = field(default_factory=list)
    context: OphContext = field(default_factory=OphContext)
    created_at: float = field(default_factory=time.time)
    # Wall-clock of the last real chat turn. The sidebar sorts by this so the
    # ordering reflects "last chatted", NOT file mtime (which a model-switch,
    # an upload, a backup, git, or antivirus can all bump). None for legacy
    # sessions saved before this field existed — list() falls back to
    # created_at / file mtime in that case.
    last_active: float | None = None
    workspace: str = str(output_path("oph_sessions"))

    _client: Any = field(default=None, repr=False)
    _toolkit: OphToolKit | None = field(default=None, repr=False)
    # Injected by the authenticated Web server. Never persisted in session JSON.
    _api_credentials: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    # Memoised (vision_model_or_None, reason) — see `_resolve_vision`.
    _vision_resolved: Any = field(default=None, repr=False)

    # ── factory + persistence ─────────────────────────────────────────
    @classmethod
    def new(cls, **kw) -> "OphSession":
        return cls(session_id=uuid.uuid4().hex[:12], **kw)

    @classmethod
    def load(cls, path: str | Path) -> "OphSession":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ctx_data = data.pop("context", {})
        s = cls(**{k: v for k, v in data.items() if not k.startswith("_")})
        s.context = OphContext(**ctx_data)
        return s

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            Path(self.workspace).mkdir(parents=True, exist_ok=True)
            path = Path(self.workspace) / f"{self.session_id}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "session_id": self.session_id,
            "backend": self.backend, "model": self.model,
            "max_tokens": self.max_tokens, "effort": self.effort,
            "prompt_profile": self.prompt_profile,
            "vision_backend": self.vision_backend,
            "vision_model_override": self.vision_model_override,
            "executor_backend": self.executor_backend,
            "executor_model": self.executor_model,
            "executor_repair_enabled": self.executor_repair_enabled,
            "verifier_backend": self.verifier_backend,
            "verifier_model": self.verifier_model,
            "debate_backend": self.debate_backend,
            "debate_model": self.debate_model,
            "temperature": self.temperature,
            "owner": self.owner,
            "messages": self.messages,
            "context": asdict(self.context),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "workspace": self.workspace,
        }, indent=2, default=str), encoding="utf-8")
        return path

    # ── lazy init ─────────────────────────────────────────────────────
    def _ensure_toolkit(self):
        if self._toolkit is None:
            run_dir = Path(self.workspace) / self.session_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._toolkit = OphToolKit(
                session=self, report_output_root=str(run_dir),
            )

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.backend in PROVIDER_SPECS:
            self._client = create_provider_client(
                self.backend, self._api_credentials.get(self.backend),
            )
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    def _vision_client(self):
        """Client that `vision_impression` / LLM-modality-detection use to call
        the VISION model. Normally this is the chat client (self._client). But
        when OPH_WEB_VISION_BACKEND is set to a DIFFERENT backend than the chat
        backbone, we build (and cache) a SEPARATE client for it — so a text-only
        brain on one provider (e.g. DeepSeek on DashScope) can pair with a vision
        model on another (e.g. gpt-5 on aigcbest). Cross-provider eyes/brain."""
        vb = (self.vision_backend
              or os.environ.get("OPH_WEB_VISION_BACKEND", "")).strip()
        if not vb or vb == self.backend:
            self._ensure_client()
            return self._client
        cached = getattr(self, "_vision_client_obj", None)
        if cached is not None:
            return cached
        if vb in PROVIDER_SPECS:
            c = create_provider_client(vb, self._api_credentials.get(vb))
        else:
            raise ValueError(f"unknown OPH_WEB_VISION_BACKEND: {vb}")
        self._vision_client_obj = c
        return c

    def _client_for_backend(self, backend: str | None, cache_attr: str):
        """Return an OpenAI-compatible client for a role-specific backend.

        Planner/orchestrator calls use ``self._client``. Role-specific passes
        such as the independent verifier may need a different provider so that
        heterogeneous-backbone experiments can use DashScope models when
        available and fall back to AIGCBest for frontier models.
        """
        if not backend or backend == self.backend:
            self._ensure_client()
            return self._client
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached
        if backend in PROVIDER_SPECS:
            c = create_provider_client(backend, self._api_credentials.get(backend))
        else:
            raise ValueError(f"unknown role backend: {backend}")
        setattr(self, cache_attr, c)
        return c

    # ── vision-model resolution ────────────────────────────────────────
    def _resolve_vision(self) -> tuple[str | None, str]:
        """Decide which model `vision_impression` (and the LLM modality
        fallback) should use to LOOK at an image.

        Priority:
          1. `OPH_WEB_VISION_MODEL` env var — a dedicated vision model that
             runs on the same backend/client as the chat model. This lets
             you pair a text-only chat brain (e.g. DeepSeek) with a
             vision-capable pair of eyes (e.g. gpt-5.5) without coupling.
          2. The chat model itself, IF it is vision-capable.
          3. None — no vision-capable model available → vision_impression
             is skipped rather than allowed to hallucinate.

        Returns (model_or_None, human_readable_reason). Memoised.
        """
        if self._vision_resolved is not None:
            return self._vision_resolved
        configured = (self.vision_model_override
                      or os.environ.get("OPH_WEB_VISION_MODEL", "")).strip()
        if configured:
            if _model_is_text_only(configured):
                res = (None, f"configured vision model '{configured}' is text-only")
            else:
                res = (configured, f"configured vision model '{configured}'")
        elif _model_is_text_only(self.model):
            res = (None, f"chat model '{self.model}' is text-only and "
                         f"OPH_WEB_VISION_MODEL is not set")
        else:
            res = (self.model, f"chat model '{self.model}' (vision-capable)")
        self._vision_resolved = res
        return res

    @property
    def vision_model(self) -> str | None:
        return self._resolve_vision()[0]

    @property
    def vision_available(self) -> bool:
        return self._resolve_vision()[0] is not None

    def _resolved_run_policy(self, policy: EffortPolicy) -> dict[str, Any]:
        """Return the auditable component/capability contract for one turn."""
        vision_model, vision_reason = self._resolve_vision()
        return {
            **policy.to_dict(),
            "components": {
                "planner": {"backend": self.backend, "model": self.model},
                "vision": {
                    "backend": self.vision_backend
                    or os.environ.get("OPH_WEB_VISION_BACKEND")
                    or self.backend,
                    "model": vision_model,
                    "available": vision_model is not None,
                    "resolution": vision_reason,
                },
                "executor": {
                    "mode": (
                        "llm_repair_with_deterministic_execution"
                        if self.executor_repair_enabled
                        else "deterministic_execution"
                    ),
                    "backend": self.executor_backend or self.backend,
                    "model": self.executor_model or self.model,
                },
                "verifier": {
                    "mode": policy.verifier_mode,
                    "backend": self.verifier_backend or self.backend,
                    "model": self.verifier_model or self.model,
                },
                "debate": {
                    "enabled": policy.verifier_mode == "debate",
                    "backend": self.debate_backend or self.backend,
                    "model": self.debate_model or self.model,
                },
            },
            "native_reasoning": bool(
                os.environ.get("OPH_NATIVE_EFFORT") == "1"
                and "gpt-5" in (self.model or "").lower()
            ),
        }

    def _executor_repair_tool_call(
        self,
        *,
        attempted_tool: str,
        attempted_arguments: dict[str, Any],
        attempted_arguments_raw: str,
        result: Any,
        arguments_parse_failed: bool,
        tool_schemas: list[dict[str, Any]],
        emit,
    ):
        """Ask the Executor LLM for one schema-constrained argument repair."""
        if not self.executor_repair_enabled:
            return None
        if not repairable_invocation_failure(
            result,
            arguments_parse_failed=arguments_parse_failed,
        ):
            return None

        attempted_schema = schema_for_tool(tool_schemas, attempted_tool)
        if attempted_schema is None:
            return None

        client = self._client_for_backend(
            self.executor_backend,
            "_executor_client_obj",
        )
        model = self.executor_model or self.model
        function = attempted_schema.get("function") or {}
        prompt_payload = {
            "attempted_tool": attempted_tool,
            "attempted_arguments": attempted_arguments,
            "attempted_arguments_raw": attempted_arguments_raw,
            "execution_error": (
                result.get("error") if isinstance(result, dict) else str(result)
            ),
            "registered_tool_schema": function,
            "current_image": self.context.current_image,
            "current_volume": self.context.current_volume,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OphAgent Executor repair role. Repair only the "
                    "arguments of the attempted registered tool invocation. "
                    "Do not choose another tool, invent files, generate shell "
                    "commands, or alter the clinical objective. Choose abort "
                    "for backend, model, resource, or infrastructure failures."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False),
            },
        ]
        repair_schema = executor_repair_tool_schema()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[repair_schema],
                tool_choice={
                    "type": "function",
                    "function": {"name": "repair_tool_invocation"},
                },
                max_tokens=min(2048, self.max_tokens),
            )
        except Exception as exc:
            emit({
                "type": "executor_repair",
                "tool": attempted_tool,
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            })
            return None

        message = response.choices[0].message
        calls = list(message.tool_calls or [])
        if len(calls) != 1:
            return None
        repair = parse_executor_repair(
            calls[0].function.arguments,
            attempted_tool=attempted_tool,
            tool_schema=attempted_schema,
        )
        if repair is None or repair.action != "retry":
            emit({
                "type": "executor_repair",
                "tool": attempted_tool,
                "status": "aborted",
            })
            return None

        if self._modality_mismatch_check(repair.tool_name) is not None:
            return None
        if self._policy_skip_tool_call(
            repair.tool_name,
            repair.arguments,
        ) is not None:
            return None
        emit({
            "type": "executor_repair",
            "tool": attempted_tool,
            "status": "retrying",
            "arguments": repair.arguments,
            "reason": repair.reason,
        })
        return repair

    # ── session helpers exposed to tools ───────────────────────────────
    def _web_file_restrictions_enabled(self) -> bool:
        try:
            ws = Path(self.workspace).resolve()
        except Exception:
            return False
        return ws.name == "webchat_sessions" and ws.parent.name == "reports"

    def _coerce_local_file_path(self, raw_path: str | Path) -> Path:
        raw = str(raw_path or "").strip()
        if not raw:
            raise ValueError("empty file path")
        web_output_root = (
            Path(self.workspace).resolve().parent
            if self._web_file_restrictions_enabled()
            else None
        )
        parsed = urlparse(raw)
        if parsed.scheme in {"http", "https"} and parsed.path.startswith("/files/"):
            rel = unquote(parsed.path[len("/files/"):])
            return ((web_output_root or _PROJECT_ROOT) / rel).resolve()
        if raw.startswith("/files/"):
            rel = unquote(raw[len("/files/"):])
            return ((web_output_root or _PROJECT_ROOT) / rel).resolve()
        p = Path(raw)
        if p.is_absolute():
            return p.resolve()
        if web_output_root is not None:
            # Browser-visible paths are relative to the generated-output root,
            # for example webchat_sessions/_uploads/<sid>/image.jpg.
            candidates = (
                (web_output_root / unquote(raw)).resolve(),
                (Path(self.workspace).resolve() / unquote(raw)).resolve(),
            )
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]
        project_candidate = (_PROJECT_ROOT / raw).resolve()
        if project_candidate.exists():
            return project_candidate
        return p.resolve()

    def session_file_reference(self, path: str | Path | None) -> str:
        """Return a model-facing path without disclosing the host root."""
        if not path:
            return ""
        candidate = Path(path).expanduser().resolve()
        if not self._web_file_restrictions_enabled():
            return str(candidate)
        output_root = Path(self.workspace).resolve().parent
        try:
            return candidate.relative_to(output_root).as_posix()
        except ValueError:
            # Legacy state should not surface an out-of-session host path.
            return candidate.name

    def resolve_session_path(self, path: str | Path, *, allow_dir: bool = False) -> Path:
        """Resolve a tool-supplied file/path and enforce web-session bounds.

        Non-web sessions keep the historical local-file workflow. Web sessions
        should never let the LLM pick arbitrary host paths: uploads live under
        `<workspace>/_uploads/<sid>/` and generated artifacts under
        `<workspace>/<sid>/`.
        """
        candidate = self._coerce_local_file_path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"file not found: {path}")
        if not candidate.is_file() and not (allow_dir and candidate.is_dir()):
            kind = "file or directory" if allow_dir else "file"
            raise FileNotFoundError(f"{kind} not found: {path}")
        if not self._web_file_restrictions_enabled():
            return candidate

        ws = Path(self.workspace).resolve()
        allowed_roots = [
            (ws / "_uploads" / self.session_id).resolve(),
            (ws / self.session_id).resolve(),
        ]
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                return candidate
            except ValueError:
                continue
        raise PermissionError("file is outside this session")

    def resolve_session_file(self, path: str | Path) -> Path:
        """Resolve a tool-supplied file path."""
        return self.resolve_session_path(path, allow_dir=False)

    def _analysis_key(self, path: str | Path | None = None) -> str:
        """Canonical cache key for files and volume directories.

        Model-facing paths are intentionally relative in Web sessions, while
        ``current_image`` is absolute.  Converting both forms here prevents one
        image from splitting into separate evidence caches.
        """
        raw = path or self.context.current_image or self.context.current_volume
        if not raw:
            return ""
        try:
            return str(self.resolve_session_path(raw, allow_dir=True).resolve())
        except Exception:
            try:
                return str(self._coerce_local_file_path(raw).resolve())
            except Exception:
                return str(raw)

    def set_image(self, path: str) -> None:
        """Register an image. Modality detection ladder (all efforts):
          filename hint → local CNN → vision LLM (only if both above
          failed AND a vision model is available) → conservative refusal.
        The vision-LLM fallback runs at EVERY effort tier: modality
        routing is a safety/routing prerequisite, not a reasoning-depth
        knob. Gating it by effort previously caused low-effort sessions to
        false-refuse (NON_OPHTHALMOLOGIC) the ~10% of valid images the CNN
        can't confidently type (hazy cataract fundus, atypical CFP). The
        `vmodel is not None` guard still protects text-only backbones.
        """
        import time as _t
        try:
            abs_path = str(self._coerce_local_file_path(path))
        except Exception:
            abs_path = str(path)
        self.context.current_image = abs_path
        self.context.invalid_input_reason = None
        self.context.scope_failure_reason = None

        try:
            resolved = self.resolve_session_file(path)
            abs_path = str(resolved)
            self.context.current_image = abs_path
            try:
                from PIL import Image
                with Image.open(resolved) as im:
                    max_pixels_raw = os.environ.get("WEB_MAX_IMAGE_PIXELS", "80000000")
                    try:
                        max_pixels = int(max_pixels_raw)
                    except (TypeError, ValueError):
                        max_pixels = 80_000_000
                    max_pixels = min(250_000_000, max(1_000_000, max_pixels))
                    width, height = im.size
                    if width <= 0 or height <= 0 or width * height > max_pixels:
                        raise ValueError(
                            f"image dimensions {width}x{height} exceed the "
                            f"{max_pixels:,}-pixel limit"
                        )
                    im.verify()
            except Exception as e:
                raise ValueError(
                    f"file is not a readable image: {type(e).__name__}: {e}"
                ) from e
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            self.context.modality_scope = "invalid_input"
            self.context.modality_sublabel = None
            self.context.current_modality = "INVALID_INPUT"
            self.context.invalid_input_reason = reason
            log.warning(f"[set_image] invalid image input {abs_path}: {reason}")
            return

        from .oph_tools import (
            filename_modality_hint, cnn_modality_hint, llm_classify_modality,
        )
        # 1. Cheap filename token
        modality = filename_modality_hint(abs_path)
        # 2. Local CNN classifier (100% val acc, ~50 ms) — ALWAYS try this
        #    after filename, regardless of effort level. Eliminates the
        #    vision-LLM mis-classifications we saw on 1837_right.jpg etc.
        if modality is None:
            modality = cnn_modality_hint(abs_path)
        # 3. Vision LLM — at ALL effort tiers, only if both above failed,
        #    AND only if a vision-capable model is available. With a
        #    text-only chat backbone (e.g. DeepSeek) and no OPH_WEB_VISION_MODEL
        #    we must NOT hand the gateway an image — it may silently drop it
        #    and return a fabricated modality (so the vmodel guard below).
        #    Was gated to {medium,high,max} — that skipped the fallback at
        #    low AND ultra, false-refusing CNN-untypeable valid images.
        if modality is None and self.vision_model is not None:
            vmodel = self.vision_model
            if vmodel is not None:
                try:
                    modality = llm_classify_modality(abs_path, self._vision_client(), vmodel)
                except Exception as e:
                    log.warning(f"[set_image] LLM modality detection failed: {e}")
                    modality = None
        # 4. CNN OOD-flagged AND the vision check is unavailable or unable to
        #    establish scope. Do not force the image into CFP/OCT/FFA through
        #    a pixel heuristic: an unverified image must not enter diagnostic
        #    tools merely because its colours resemble an ophthalmic image.
        if modality is None:
            if self.vision_model is not None:
                reason = (
                    "ophthalmic scope could not be verified because the local "
                    "modality detector rejected the image and the configured "
                    "vision check did not return a valid modality"
                )
            else:
                reason = (
                    "ophthalmic scope could not be verified because the local "
                    "modality detector rejected the image and no vision-capable "
                    "scope check is configured"
                )
            modality = "UNVERIFIED_INPUT"
            self.context.scope_failure_reason = reason
            log.warning(f"[set_image] {abs_path}: {reason}; refusing analysis")

        # Tri-state scope classification. The LLM modality detector can now
        # return "OPHTHALMOLOGIC_OTHER:<sub>" (visual field, OCT-A, FAF, etc.)
        # and "NON_OPHTHALMOLOGIC" — propagate them to context so chat()
        # can route to vision-only or refusal mode instead of running the
        # full in-scope pipeline.
        if modality == "NON_OPHTHALMOLOGIC":
            self.context.modality_scope = "non_ophth"
            self.context.modality_sublabel = None
            self.context.current_modality = "NON_OPHTHALMOLOGIC"
            log.info(f"[set_image] {abs_path}: NON_OPHTHALMOLOGIC — chat will refuse")
        elif modality == "UNVERIFIED_INPUT":
            self.context.modality_scope = "unverified_input"
            self.context.modality_sublabel = None
            self.context.current_modality = "UNVERIFIED_INPUT"
            log.info(f"[set_image] {abs_path}: scope unverified — chat will refuse")
        elif isinstance(modality, str) and modality.startswith("OPHTHALMOLOGIC_OTHER"):
            sub = modality.split(":", 1)[1] if ":" in modality else "unknown_ophth"
            self.context.modality_scope = "ophth_other"
            self.context.modality_sublabel = sub
            self.context.current_modality = f"OPHTHALMOLOGIC_OTHER:{sub}"
            log.info(f"[set_image] {abs_path}: ophth-other ({sub}) — vision-only mode")
        else:
            self.context.modality_scope = "in_scope"
            self.context.modality_sublabel = None
            self.context.current_modality = modality

        # Append to history if not already there
        for entry in self.context.attached_images:
            if entry.get("path") == abs_path:
                # Already attached; just bump it to most-recent
                entry["modality"] = modality
                entry["uploaded_at"] = _t.time()
                return
        self.context.attached_images.append({
            "path": abs_path,
            "modality": modality,
            "filename": Path(abs_path).name,
            "uploaded_at": _t.time(),
        })

    def set_volume(self, path: str) -> None:
        self.context.current_volume = str(Path(path).resolve())

    # ── chat ──────────────────────────────────────────────────────────
    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def _multimodal_entries(self) -> list[dict[str, str]]:
        """Return unique in-scope attachments when multiple modalities exist.

        A pair of CFP images is still a single-modality session. The stricter
        routing and evidence-coverage rules below activate only when at least
        two distinct supported modalities are attached, so the historical
        single-modality path remains unchanged.
        """
        entries: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for raw in self.context.attached_images:
            path = str(raw.get("path") or "").strip()
            modality = str(raw.get("modality") or "").upper().split(":")[0]
            if not path or modality not in self._CORE_TOOLS_BY_MODALITY:
                continue
            path_key = os.path.normcase(os.path.abspath(path))
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            entries.append({
                "path": path,
                "modality": modality,
                "filename": str(raw.get("filename") or Path(path).name),
            })
        if len({entry["modality"] for entry in entries}) < 2:
            return []
        return entries

    def _multimodal_execution_note(
        self, entries: list[dict[str, str]]
    ) -> str:
        """Build the multi-image execution contract injected into the prompt."""
        lines = [
            "# Multi-modal execution contract",
            "",
            "Upload routing has already identified every image modality below. "
            "Do NOT spend tool rounds calling `detect_modality` again.",
            "Before finalising, obtain calibrated core evidence for EACH image "
            "with an explicit `image_path`. Batch independent calls in parallel "
            "where possible, then call `verify_findings` over the combined "
            "cross-modal evidence.",
            "",
        ]
        for entry in entries:
            core = ", ".join(
                f"`{name}`"
                for name in self._CORE_TOOLS_BY_MODALITY[entry["modality"]]
            )
            lines.append(
                f"- **{entry['modality']}** `{entry['path']}`: at least one of {core}."
            )
        lines += [
            "",
            "A classifier label is not automatically the final diagnosis. Keep "
            "background/co-morbid morphology separate from the active disease, "
            "and resolve conflicts using modality-specific findings.",
        ]
        return "\n".join(lines)

    def _tool_default_image(self, tool_name: str) -> str | None:
        """Choose the unique attachment matching a tool's modality.

        The old default was always ``current_image``. In a CFP+FFA session that
        can silently run a CFP adapter on the focused FFA image when the model
        omits ``image_path``. Single-modality sessions deliberately keep the old
        default.
        """
        entries = self._multimodal_entries()
        if not entries:
            return self.context.current_image

        tool_modality = ""
        try:
            from ophagent.adapters import GLOBAL_REGISTRY
            cls = GLOBAL_REGISTRY._classes.get(tool_name)
            if cls is not None:
                tool_modality = str(cls.metadata.modality or "").upper()
        except Exception:
            pass
        if not tool_modality:
            prefix = tool_name.split("_", 1)[0].upper()
            if prefix in self._CORE_TOOLS_BY_MODALITY:
                tool_modality = prefix

        matches = [
            entry["path"] for entry in entries
            if entry["modality"] == tool_modality
        ]
        if len(matches) == 1:
            return matches[0]
        return self.context.current_image

    def _tool_batch_consumes_plan_round(self, tool_names: list[str]) -> bool:
        """Exclude routing-only batches from multi-modal clinical budget."""
        return not (
            bool(self._multimodal_entries())
            and bool(tool_names)
            and all(name == "detect_modality" for name in tool_names)
        )

    def _modality_for_image(self, image_path: str | None = None) -> str:
        """Resolve modality by attachment path, falling back to focus image."""
        img = str(image_path or self.context.current_image or "")
        img_key = os.path.normcase(os.path.abspath(img)) if img else ""
        if img_key:
            for entry in self.context.attached_images:
                entry_path = str(entry.get("path") or "")
                if not entry_path:
                    continue
                if os.path.normcase(os.path.abspath(entry_path)) == img_key:
                    return str(entry.get("modality") or "").upper().split(":")[0]
        return (self.context.current_modality or "").upper().split(":")[0]

    def _analysis_tool_succeeded(self, tool_name: str) -> bool:
        """True when a cached tool result completed without skip or error."""
        for by_tool in (self.context.analyses or {}).values():
            result = (by_tool or {}).get(tool_name)
            if not isinstance(result, dict):
                continue
            if result.get("skipped") or result.get("error"):
                continue
            if result.get("success") is False:
                continue
            return True
        return False

    def _missing_multimodal_core_evidence(self) -> list[dict[str, Any]]:
        """Describe attached modalities that still lack calibrated evidence."""
        missing: list[dict[str, Any]] = []
        for entry in self._multimodal_entries():
            path = entry["path"]
            modality = entry["modality"]
            if self._has_core_evidence_for_image(path):
                continue
            by_tool = self._analyses_for_image(path)
            core_tools = list(self._CORE_TOOLS_BY_MODALITY.get(modality, []))
            attempted = [name for name in core_tools if name in by_tool]
            failed: list[str] = []
            skipped: list[str] = []
            for name in attempted:
                result = by_tool.get(name)
                if not isinstance(result, dict):
                    failed.append(name)
                elif result.get("skipped"):
                    skipped.append(name)
                elif result.get("error") or result.get("success") is False:
                    failed.append(name)
            status = "not_called"
            if failed:
                status = "failed"
            elif skipped:
                status = "skipped"
            missing.append({
                **entry,
                "core_tools": core_tools,
                "attempted": attempted,
                "failed": failed,
                "skipped": skipped,
                "status": status,
            })
        return missing

    def _multimodal_verifier_needed(
        self, tool_outcomes: dict[str, str]
    ) -> bool:
        """Require cross-modal verification outside low/no-verifier arms."""
        if not self._multimodal_entries():
            return False
        if self.effort == "low" or getattr(self, "_ablate_verifier", False):
            return False
        return not (
            tool_outcomes.get("verify_findings") == "ok"
            or self._analysis_tool_succeeded("verify_findings")
        )

    def _single_image_verifier_needed(
        self, tool_outcomes: dict[str, str]
    ) -> bool:
        """Require a fresh verifier pass after single-image core evidence."""
        if self._multimodal_entries():
            return False
        if self.effort == "low" or getattr(self, "_ablate_verifier", False):
            return False
        modality = self._modality_for_image()
        core_tools = self._CORE_TOOLS_BY_MODALITY.get(modality, [])
        current_turn_has_core_evidence = any(
            tool_outcomes.get(name) == "ok" for name in core_tools
        )
        if not current_turn_has_core_evidence:
            return False
        return tool_outcomes.get("verify_findings") != "ok"

    @staticmethod
    def _verifier_result_valid(result: Any) -> bool:
        """Return true only for a completed, machine-readable verifier pass."""
        return (
            isinstance(result, dict)
            and result.get("status") == "ok"
            and isinstance(result.get("verify_passed"), bool)
            and (
                "n_tools_run" not in result
                or (
                    isinstance(result.get("n_tools_run"), int)
                    and result["n_tools_run"] > 0
                )
            )
        )

    def _multimodal_completion_gap(
        self, tool_outcomes: dict[str, str]
    ) -> tuple[list[dict[str, Any]], bool]:
        return (
            self._missing_multimodal_core_evidence(),
            self._multimodal_verifier_needed(tool_outcomes),
        )

    def _multimodal_repair_instruction(
        self,
        missing: list[dict[str, Any]],
        verifier_needed: bool,
    ) -> str:
        """Prompt the planner to repair evidence coverage before finalising."""
        if missing:
            lines = [
                "Your draft cannot be finalised because calibrated evidence is "
                "missing for the following attached image(s). Do NOT call "
                "`detect_modality` again. Call at least one compatible core tool "
                "for EACH listed image now, using the exact explicit `image_path`. "
                "Respect the user's originally requested tools and batch independent "
                "calls in parallel where possible:",
                "",
            ]
            for item in missing:
                choices = ", ".join(f"`{name}`" for name in item["core_tools"])
                lines.append(
                    f"- {item['modality']} `{item['path']}`: choose at least one of "
                    f"{choices} (current status: {item['status']})."
                )
            lines += [
                "",
                "Do not call `verify_findings` in parallel with evidence tools; "
                "wait for their returned outputs, then verify the combined evidence "
                "on the next turn.",
            ]
            return "\n".join(lines)
        if verifier_needed:
            return (
                "All attached modalities now have calibrated evidence, but the "
                "cross-modal Verifier has not completed. Call `verify_findings` "
                "now with a compact findings JSON containing the returned evidence "
                "from every attached modality. Do not finalise before it returns."
            )
        return ""

    def _multimodal_incomplete_response(
        self,
        missing: list[dict[str, Any]],
        verifier_needed: bool,
    ) -> str:
        """Refuse a multi-modal diagnosis when per-image coverage is incomplete."""
        import json as _json

        lines = [
            "## Insufficient multi-modal evidence - no diagnosis produced",
            "",
            "The agent did not complete the evidence contract required for all "
            "attached modalities. A missing tool call is reported separately "
            "from a tool execution failure.",
        ]
        if missing:
            lines += [
                "",
                "### Per-image core evidence",
                "",
                "| Modality | Image | Status | Required alternatives |",
                "|---|---|---|---|",
            ]
            labels = {
                "not_called": "NOT CALLED",
                "failed": "FAILED",
                "skipped": "SKIPPED",
            }
            for item in missing:
                choices = ", ".join(f"`{name}`" for name in item["core_tools"])
                lines.append(
                    f"| {item['modality']} | `{item['filename']}` | "
                    f"{labels.get(item['status'], item['status'])} | {choices} |"
                )
        if verifier_needed:
            lines += [
                "",
                "### Cross-modal verification",
                "",
                "`verify_findings` did not complete successfully after calibrated "
                "evidence was collected from the attached modalities.",
            ]
        lines += [
            "",
            "**No diagnostic interpretation is returned from this incomplete "
            "pipeline.** Resume the Planner so it can run the missing evidence "
            "tools and then verify the combined findings.",
            "",
        ]

        if missing and any(item["status"] == "not_called" for item in missing):
            reason = "multimodal_core_tools_not_called"
        elif missing:
            reason = "multimodal_core_tools_failed"
        else:
            reason = "multimodal_verifier_not_completed"
        payload = {
            "verdict": "diagnostic_call_not_possible",
            "reason": reason,
            "missing_modalities": [
                {
                    "modality": item["modality"],
                    "image": item["path"],
                    "status": item["status"],
                    "required_any_of": item["core_tools"],
                    "attempted": item["attempted"],
                    "failed": item["failed"],
                }
                for item in missing
            ],
            "verifier_required": verifier_needed,
            "do_not_use_for_clinical_decisions": True,
            "next_action": (
                "Resume the Planner, obtain core evidence for every attached "
                "modality, then call verify_findings"
            ),
        }
        lines += [
            "```",
            "===INSUFFICIENT_DATA===",
            _json.dumps(payload, indent=2, ensure_ascii=False),
            "```",
        ]
        return "\n".join(lines)

    def chat(self, user_text: str | None = None, on_event=None,
             max_tool_steps: int = 15) -> str:
        self._ensure_toolkit()
        # Reset the interrupt flag at the start of every chat() — any
        # leftover True from a previous turn would immediately bail.
        self.context.interrupt_requested = False
        # Stamp activity so the sidebar can sort by "last chatted" rather than
        # file mtime. Updated on every chat turn (not on model-switch/upload).
        self.last_active = time.time()
        if user_text is not None:
            self.add_user_message(user_text)

        def emit(ev: dict):
            if on_event:
                try:
                    on_event(ev)
                except Exception:
                    pass

        # ── Early scope routing ─────────────────────────────────────────
        # Decided at set_image() time based on the LLM modality detector.
        # Scope branches: invalid, non-ophthalmic, and unverified inputs refuse
        # without running tools; OPHTHALMOLOGIC_OTHER runs a single vision-only
        # impression with a modality-tailored schema; in-scope falls through
        # to the regular Planner-Executor-Verifier loop below.
        scope = getattr(self.context, "modality_scope", "in_scope")
        if scope == "invalid_input" and self.context.current_image:
            reply = self._invalid_input_refusal()
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        if scope == "non_ophth" and self.context.current_image:
            reply = self._non_ophthalmologic_refusal()
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        if scope == "unverified_input" and self.context.current_image:
            reply = self._unverified_input_refusal()
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        if scope == "ophth_other" and self.context.current_image:
            reply = self._vision_only_impression(
                self.context.current_image,
                self.context.modality_sublabel or "unknown_ophth",
                user_text or "",
                emit=emit,
            )
            self.messages.append({"role": "assistant", "content": reply})
            return reply

        self._ensure_client()
        from .prompt_profiles import (
            STANDARD_PROFILE,
            normalize_prompt_profile,
            system_prompt_for_profile,
            tool_result_for_profile,
            tool_schemas_for_profile,
        )
        prompt_profile = normalize_prompt_profile(
            getattr(self, "prompt_profile", "standard")
        )
        focused_profile = prompt_profile != STANDARD_PROFILE
        policy = get_effort_policy(self.effort)
        self.context.last_run_policy = self._resolved_run_policy(policy)
        base_system_prompt = (
            system_prompt_for_profile(prompt_profile)
            if focused_profile else OPH_SYSTEM_PROMPT
        )
        sys_msg = {"role": "system", "content": base_system_prompt}
        # Collect the in-scope modalities currently attached in this session
        # so _effort_directive can append per-modality clinical addendums.
        attached_mods: list[str] = []
        cur = self.context.current_modality
        if cur and cur != "NON_OPHTHALMOLOGIC" and not str(cur).startswith("OPHTHALMOLOGIC_OTHER"):
            attached_mods.append(cur)
        for _img in self.context.attached_images:
            _m = _img.get("modality")
            if _m and _m not in attached_mods and _m != "NON_OPHTHALMOLOGIC" \
                    and not str(_m).startswith("OPHTHALMOLOGIC_OTHER"):
                attached_mods.append(_m)
        multimodal_entries = self._multimodal_entries()
        multimodal_mode = bool(multimodal_entries)
        if getattr(self, "_ablate_planner", False):
            # Architecture-ablation arm: NO structured planner directive — a
            # bare tool-using LLM that just sees the tools and answers. Used
            # only by the agent-component ablation benchmark; default off.
            # At LOW effort this becomes the "true-planner cheap" arm (B): the
            # LLM still AUTONOMOUSLY picks the tools (no prescribed recipe), but
            # must batch them in ONE turn (plan_rounds=1) — cheapest autonomous
            # agent, no rules, no verifier.
            if self.effort == "low":
                sys_msg["content"] += (
                    "\n\nYou have the tools listed below. In a SINGLE turn, call "
                    "ALL the tools you judge useful to answer the question (emit "
                    "them as parallel tool calls), then give the final answer in "
                    "the required format. No fixed procedure is imposed; you have "
                    "a strict 1-round tool budget.")
            else:
                sys_msg["content"] += (
                    "\n\nYou have the tools listed below. Call whichever you judge "
                    "useful to answer the question, then give the final answer in "
                    "the required format. No fixed procedure is imposed.")
        else:
            sys_msg["content"] += "\n\n" + _effort_directive(
                self.effort, vision_available=self.vision_available,
                attached_modalities=attached_mods,
                prompt_profile=prompt_profile)
            # MONOTONIC-EFFORT ANCHOR (OPH_NATIVE_EFFORT=1): higher tiers must be a
            # strict superset of MEDIUM's calibrated decision — never a different or
            # weaker rule. The extra coverage/reasoning may only CONFIRM or UPGRADE a
            # call on STRICTLY STRONGER evidence; it must NOT overturn a confident
            # calibrated finding on weaker/ambiguous grounds, and must NOT default to
            # 'absent'. This makes higher effort cost more (time/tokens) without ever
            # scoring worse than MEDIUM.
            if os.environ.get("OPH_NATIVE_EFFORT") == "1" and self.effort in ("high", "max", "ultra"):
                sys_msg["content"] += (
                    "\n\n# Effort anchor (READ)\n"
                    "Treat the standard calibrated (MEDIUM) reading as your BASELINE verdict. "
                    "Your extra tools / deeper reasoning / debate may only (a) raise confidence in, "
                    "or (b) UPGRADE a per-condition call when you have STRICTLY STRONGER, more-reliable "
                    "evidence than the baseline tools provided. Do NOT flip a confident calibrated "
                    "finding to the opposite on weaker or merely-ambiguous evidence, and NEVER default a "
                    "condition to ABSENT just because signals did not fully converge — keep the "
                    "strongest single calibrated signal. Higher effort must not do worse than MEDIUM.")
        ctx_note = self._context_note()
        if ctx_note:
            sys_msg["content"] += "\n\n# Session context\n" + ctx_note
        if multimodal_mode:
            sys_msg["content"] += (
                "\n\n" + self._multimodal_execution_note(multimodal_entries)
            )
        # Inject a modality-task-feasibility hint so the LLM doesn't waste
        # rounds trying CFP-only tools (e.g. cfp_retsam_segmentation) on a
        # UWF image. The code-level guard below still blocks the actual
        # call, but giving the LLM the same picture up-front means it
        # refuses honestly instead of churning through 5 mismatched
        # tools before hitting the "insufficient evidence" gate.
        if not focused_profile:
            sys_msg["content"] += "\n\n" + self._modality_feasibility_hint()

        request_messages = [sys_msg] + self.messages
        tools = self._toolkit.get_all_schemas()
        if focused_profile:
            tools = tool_schemas_for_profile(
                prompt_profile,
                tools,
                attached_modalities=attached_mods,
                multimodal=multimodal_mode,
            )
        if getattr(self, "_ablate_verifier", False):
            # Architecture-ablation arm: remove the Verifier (verify_findings)
            # so the agent stops after execution with no re-plan/consistency
            # pass. Benchmark-only; default off.
            tools = [t for t in tools
                     if (t.get("function", {}) or {}).get("name") != "verify_findings"]
        # COST OPTION (OPH_SCOPE_TOOLS=1, default OFF): drop the schemas of
        # OTHER-modality tools — the modality-feasibility guard already blocks
        # those calls, so their schema is dead weight re-sent every turn. Only
        # activates for a single in-scope modality; cross-modal tools (no
        # modality prefix) are always kept. Default OFF = original behaviour.
        if (not focused_profile
                and os.environ.get("OPH_SCOPE_TOOLS") == "1"
                and not multimodal_mode):
            _cur = (self.context.current_modality or "").upper().split(":")[0]
            _drop = {"CFP": ("OCT", "UWF", "FFA", "OCTCUBE"),
                     "OCT": ("CFP", "UWF", "FFA"),
                     "UWF": ("OCT", "FFA", "OCTCUBE"),
                     "FFA": ("OCT", "UWF", "OCTCUBE")}.get(_cur, ())
            if _drop:
                def _keep_tool(t):
                    nm = ((t.get("function", {}) or {}).get("name", "")).upper()
                    return not (nm.startswith(_drop) or "VOLUME" in nm)
                tools = [t for t in tools if _keep_tool(t)]

        # Per-tool call counters — defend against an LLM stuck calling the
        # same tool in a tight loop (commonly verify_findings on a malformed
        # arg). After 3 calls to the same tool we inject a "stop calling
        # this and finalise" steering message.
        per_tool_calls: dict[str, int] = {}
        steered_for_loop = False
                # ── Sufficient-evidence tracker ────────────────────────────────
        # `tool_outcomes` maps tool_name → "ok" / "fail". Inspected after the
        # loop in _finalize_reply() to decide whether to suppress the
        # agent's diagnostic call with an ===INSUFFICIENT_DATA=== block.
        tool_outcomes: dict[str, str] = {}
        last_verifier_result: dict[str, Any] | None = None
        verifier_stale = False

        def verifier_for_final() -> dict[str, Any] | None:
            if not verifier_stale:
                return last_verifier_result
            return {
                "verify_passed": False,
                "warnings": ["new evidence was added after verification"],
                "next_actions": [],
                "recommendation": (
                    "The final tool results were not re-checked by the verifier. "
                    "Report the assessment as undetermined / low-confidence and "
                    "do not force a single high-confidence label."
                ),
            }

        # Force every backbone to start the tool pipeline for a freshly
        # attached image. This is a lifecycle invariant, not a provider quirk:
        # direct image diagnoses without calibrated evidence are never accepted.
        # Follow-up chat is unaffected because the image already has analyses.
        _img = self.context.current_image
        force_first_tool = (
            bool(_img)
            and not self._analyses_for_image(_img)
        )
        force_evidence_next = False
        evidence_retry_count = 0

        # ── Plan-then-execute budget (effort-gated) ─────────────────────
        # low    = one-shot plan: 1 tool-batch turn, then synthesise.
        # medium = two-stage plan: 2 tool-batch turns (observers → workups).
        # high / max / ultra = 3 / 4 / 5 bounded planning rounds.
        # When the budget is spent we force tool_choice='none' so the model
        # synthesises instead of opening another planning round. The matching
        # effort directive tells low/medium to emit their plan as PARALLEL
        # tool calls in one turn, so the round cap is sufficient.
        # Native model reasoning changes depth inside one LLM call, never the
        # provider-independent Planner-Executor-Verifier lifecycle.
        plan_rounds = min(max_tool_steps, policy.plan_rounds)
        if multimodal_mode:
            multimodal_floor = 2 * len(multimodal_entries) + 1
            plan_rounds = min(
                max_tool_steps, max(plan_rounds, multimodal_floor)
            )
        tool_turns = 0
        forced_synth_steer = False
        multimodal_repair_attempts = 0
        max_multimodal_repairs = (
            len(multimodal_entries) + 1 if multimodal_mode else 0
        )
        force_multimodal_repair_tool = False
        # ── Verify-loop escalation budget (SEPARATE from plan_rounds) ──────
        # BUGFIX: plan_rounds (e.g. 2 at medium) used to force tool_choice='none'
        # the moment the planning budget was spent — even when verify_findings
        # had just returned next_actions ("DO NOT FINALISE YET, run the suggested
        # next_action"). That structurally defeated the P-E-V verifier loop and
        # forced a premature, often degenerate all-absent synthesis (it discarded
        # confident workup/classifier verdicts like referable_glaucoma=True or
        # active-PDR). Now the verifier may run its next_action for up to
        # `verify_cap` extra rounds past plan_rounds before finalisation is forced.
        verify_cap = policy.verify_escalations
        requires_final_verifier = (
            policy.require_final_verifier
            and not getattr(self, "_ablate_verifier", False)
        )
        verify_wants_more = False
        verifier_next_tools: set[str] = set()
        post_plan_escalations = 0
        force_verifier_next = False
        forced_verifier_steer = False
        verifier_tools = [
            tool for tool in tools
            if (tool.get("function", {}) or {}).get("name") == "verify_findings"
        ]
        modality = (self.context.current_modality or "").upper().split(":")[0]
        available_tool_names = {
            (tool.get("function", {}) or {}).get("name") for tool in tools
        }
        starter_tool_name = next(
            (
                name for name in self._CORE_TOOLS_BY_MODALITY.get(modality, [])
                if name in available_tool_names
            ),
            None,
        )
        starter_tools = [
            tool for tool in tools
            if (tool.get("function", {}) or {}).get("name") == starter_tool_name
        ]

        for step_i in range(max_tool_steps):
            # Honour external interrupts (server /abort endpoint sets this
            # flag from another thread). Bail before paying for the next
            # LLM call. Top-of-loop interrupts are safe — no half-finished
            # tool_calls in the message history.
            if self.context.interrupt_requested:
                self.context.interrupt_requested = False
                return self._emit_interrupt_reply(
                    emit, tool_outcomes, pending_tool_calls=None)
            emit({"type": "thinking"})
            completion_tools = tools
            verifier_terminal = (
                last_verifier_result is not None
                and not verifier_stale
                and not last_verifier_result.get("next_actions")
            )
            needs_final_verifier = (
                requires_final_verifier
                and bool(tool_outcomes)
                and (last_verifier_result is None or verifier_stale)
            )

            if verifier_terminal:
                tc_mode = "none"
            elif force_multimodal_repair_tool:
                completion_tools = tools
                tc_mode = "required"
                force_multimodal_repair_tool = False
            elif force_verifier_next or (
                    needs_final_verifier and tool_turns >= plan_rounds):
                completion_tools = verifier_tools or tools
                tc_mode = {
                    "type": "function",
                    "function": {"name": "verify_findings"},
                }
                if not forced_verifier_steer:
                    request_messages.append({
                        "role": "user",
                        "content": (
                            "Controller state: evidence collection is complete. "
                            "Call `verify_findings` now; no other tool is "
                            "permitted before finalisation."
                        ),
                    })
                    forced_verifier_steer = True
                force_verifier_next = False
            elif verify_wants_more:
                if post_plan_escalations < verify_cap:
                    completion_tools = [
                        tool for tool in tools
                        if (tool.get("function", {}) or {}).get("name")
                        in verifier_next_tools
                    ]
                    tc_mode = "required" if completion_tools else "none"
                    post_plan_escalations += 1
                    verify_wants_more = False
                else:
                    tc_mode = "none"
            elif force_evidence_next:
                # Some OpenAI-compatible providers accept
                # tool_choice='required' but still return plain text. Retry with
                # only the modality's minimum calibrated core observer exposed,
                # so "required" has one unambiguous valid target.
                completion_tools = starter_tools or tools
                tc_mode = "required"
                force_evidence_next = False
            elif force_first_tool and step_i == 0:
                tc_mode = "required"
            elif tool_turns >= plan_rounds:
                if verify_wants_more and post_plan_escalations < verify_cap:
                    # The verifier asked to escalate (next_action pending). Give
                    # it a tool round instead of force-finalising — this is the
                    # P-E-V re-plan, bounded by verify_cap so it cannot loop.
                    tc_mode = "auto"
                    post_plan_escalations += 1
                    verify_wants_more = False  # consumed; re-set if next verify still fails
                else:
                    # Plan + verify-escalation budget spent → must synthesise now.
                    tc_mode = "none"
                    if not forced_synth_steer:
                        request_messages.append({
                            "role": "user",
                            "content": (
                                "Your planned tools have all run. DO NOT call any "
                                "more tools — write the final answer to my original "
                                "question now, in the required format. Base EACH "
                                "condition on the tool results above: if a workup or "
                                "classifier flagged a condition (e.g. "
                                "referable_glaucoma=True, active/inactive PDR, a "
                                "`present_conditions` entry, a high CLIP probability), "
                                "report it as PRESENT unless another tool specifically "
                                "contradicts it. Do NOT default everything to absent, "
                                "and do NOT discard a confident tool verdict just "
                                "because image quality was flagged or vision was unsure."
                            ),
                        })
                        forced_synth_steer = True
            else:
                tc_mode = "auto"

            if tc_mode == "none" and not forced_synth_steer:
                request_messages.append({
                    "role": "user",
                    "content": (
                        "The controller has closed evidence collection. DO NOT "
                        "call more tools. Write the final answer to the original "
                        "question now. Preserve confident calibrated findings; "
                        "localise uncertainty to the specific quality, severity, "
                        "or conflicting component instead of discarding the "
                        "entire diagnosis."
                    ),
                })
                forced_synth_steer = True

            resp = self._safe_completion(request_messages, completion_tools, emit,
                                         tool_choice=tc_mode)
            if resp is None:
                return "(LLM call failed — see error above)"
            msg = resp.choices[0].message
            asst_record: dict[str, Any] = {"role": "assistant",
                                           "content": msg.content or ""}
            if msg.tool_calls:
                asst_record["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            self.messages.append(asst_record)
            request_messages.append(asst_record)

            if not msg.tool_calls:
                content = msg.content or ""
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                needs_initial_evidence = (
                    bool(self.context.current_image)
                    and not self._has_core_evidence_for_image(
                        self.context.current_image
                    )
                )
                if needs_initial_evidence:
                    # Never expose or persist a diagnosis generated from a new
                    # image before a calibrated observer has run. This remains
                    # necessary even with tool_choice='required' because some
                    # compatible gateways silently ignore that constraint.
                    if self.messages and self.messages[-1] is asst_record:
                        self.messages.pop()
                    if evidence_retry_count >= 2:
                        final = self._insufficient_data_response(
                            modality=modality or "unknown",
                            tool_outcomes=dict(tool_outcomes),
                            core_tools=self._CORE_TOOLS_BY_MODALITY.get(
                                modality, []
                            ),
                            reason="planner_returned_no_tool_call",
                        )
                        self.messages.append({
                            "role": "assistant",
                            "content": final,
                        })
                        emit({"type": "text", "content": final})
                        return final
                    evidence_retry_count += 1
                    request_messages.append({
                        "role": "user",
                        "content": (
                            "Controller safety gate: this is a newly attached "
                            "image and no calibrated core observer has run. "
                            "Your direct answer was rejected. Call the single "
                            f"available core tool `{starter_tool_name}` now; do "
                            "not answer or ask for clarification before its "
                            "result is available."
                            if starter_tool_name else
                            "Controller safety gate: this is a newly attached "
                            "image and no calibrated core observer has run. "
                            "Your direct answer was rejected. Call one "
                            "modality-compatible calibrated diagnostic tool now."
                        ),
                    })
                    force_evidence_next = True
                    continue
                if multimodal_mode:
                    missing, multimodal_verifier_needed = (
                        self._multimodal_completion_gap(tool_outcomes)
                    )
                    if (
                        (missing or multimodal_verifier_needed)
                        and multimodal_repair_attempts < max_multimodal_repairs
                    ):
                        # The draft is premature. Remove it from persisted and
                        # in-flight history, then run one bounded repair round.
                        if self.messages and self.messages[-1] is asst_record:
                            self.messages.pop()
                        if request_messages and request_messages[-1] is asst_record:
                            request_messages.pop()
                        request_messages.append({
                            "role": "user",
                            "content": self._multimodal_repair_instruction(
                                missing, multimodal_verifier_needed
                            ),
                        })
                        multimodal_repair_attempts += 1
                        if missing:
                            force_multimodal_repair_tool = True
                        else:
                            force_verifier_next = True
                            forced_verifier_steer = True
                        emit({
                            "type": "thinking",
                            "note": "repairing multi-modal evidence coverage",
                        })
                        continue
                needs_final_verifier = (
                    requires_final_verifier
                    and bool(tool_outcomes)
                    and (last_verifier_result is None or verifier_stale)
                )
                if needs_final_verifier:
                    # Keep the draft only in the transient provider history so
                    # the model understands why the controller rejected it.
                    # It must not appear as a user-visible/persisted answer.
                    if self.messages and self.messages[-1] is asst_record:
                        self.messages.pop()
                    request_messages.append({
                        "role": "user",
                        "content": (
                            "That answer was produced before final verification. "
                            "Do not answer yet. Call `verify_findings` now; the "
                            "controller will then permit finalisation."
                        ),
                    })
                    force_verifier_next = True
                    forced_verifier_steer = True
                    continue
                if not content.strip():
                    note = (
                        f"_The assistant returned no text "
                        f"(finish_reason: {finish_reason}). "
                        f"This usually means the model exhausted its token budget on "
                        f"internal reasoning. Try increasing max_tokens (current: "
                        f"{self.max_tokens}) or use a non-reasoning model like "
                        f"`openai/gpt-5.5` or `openai/gpt-5.5-mini`._"
                    )
                    asst_record["content"] = note
                    emit({"type": "text", "content": note})
                    return note
                # Detect reasoning-fragment leakage. Reasoning models sometimes
                # return their internal CoT as `content` when max_tokens runs
                # out mid-answer (finish_reason='length') OR when the model
                # decided to "think out loud" without finishing. Symptoms:
                # short, no markdown, imperative fragments like "Need verify".
                if _looks_like_reasoning_fragment(content, finish_reason):
                    log.warning(
                        f"[oph_session] detected reasoning-fragment content "
                        f"(finish_reason={finish_reason}); requesting clean retry."
                    )
                    # Inject a steering message and re-ask once.
                    steer = {
                        "role": "user",
                        "content": (
                            "STOP. Your previous reply leaked your internal "
                            "chain-of-thought (\"Need...\", \"Maybe...\", "
                            "\"Let's...\") instead of finalising. The tool "
                            "chain is finished. DO NOT call any more tools — "
                            "tool_calls MUST be empty. Write the final answer "
                            "to my ORIGINAL question now, in clean Markdown, "
                            "using only the numbers already computed by tools "
                            "in this conversation. Start directly with the "
                            "clinical answer — no meta-commentary."
                        ),
                    }
                    request_messages.append(steer)
                    emit({"type": "thinking"})
                    resp2 = self._safe_completion(
                        request_messages, tools=[], emit=emit,
                        tool_choice="none",
                    )
                    if resp2 is not None:
                        msg2 = resp2.choices[0].message
                        if (msg2.content or "").strip():
                            asst_record2 = {"role": "assistant",
                                            "content": msg2.content or ""}
                            self.messages.append(asst_record2)
                            final = self._finalize_reply(
                                msg2.content, tool_outcomes, verifier_for_final())
                            asst_record2["content"] = final
                            emit({"type": "text", "content": final})
                            return final
                    # If the retry still failed, fall through with original.
                final = self._finalize_reply(
                    content, tool_outcomes, verifier_for_final())
                asst_record["content"] = final
                emit({"type": "text", "content": final})
                return final

            # This turn opened a planning/execution round (it emitted tool
            # calls); count it against the plan-then-execute budget.
            tool_names = [tc.function.name for tc in msg.tool_calls]
            if self._tool_batch_consumes_plan_round(tool_names):
                tool_turns += 1

            # Execute tools
            for tc in msg.tool_calls:
                # Check the interrupt flag BEFORE each tool dispatch as
                # well. The outer loop-top check only fires between full
                # LLM rounds; a single round can carry 3-5 tool calls and
                # each tool can take 30s+ (retsam / FMUE), so this inner
                # check shaves response latency for the common "user
                # clicked Stop mid-tool-execution" case.
                if self.context.interrupt_requested:
                    self.context.interrupt_requested = False
                    # CRITICAL: backfill placeholder tool responses for
                    # any tool_calls in this assistant message that we
                    # haven't actually answered yet. Without this the
                    # next chat turn would send an invalid history
                    # (assistant w/ tool_calls + no matching tool msgs)
                    # and the LLM returns 400 BadRequest.
                    return self._emit_interrupt_reply(
                        emit, tool_outcomes,
                        pending_tool_calls=msg.tool_calls)
                attempted_arguments_raw = tc.function.arguments or "{}"
                arguments_parse_failed = False
                try:
                    args = json.loads(attempted_arguments_raw)
                except json.JSONDecodeError:
                    args = {}
                    arguments_parse_failed = True

                # ── Modality guard ──────────────────────────────────────
                # Refuse to execute a tool whose modality doesn't match the
                # current image (or any other attached image). Without this
                # the LLM happily runs cfp_* tools on a UWF image, wastes
                # 30s-3min, and we end up at "insufficient evidence" anyway
                # — but having burned tokens and confused the user. The
                # explicit refusal lets the LLM realise the task isn't
                # feasible on the current modality and stop early.
                mismatch_result = self._modality_mismatch_check(tc.function.name)
                if mismatch_result is not None:
                    tool_outcomes[tc.function.name] = "fail"
                    img = self._analysis_key(
                        args.get("image_path") or self.context.current_image
                    )
                    if img:
                        self.context.analyses.setdefault(img, {})[tc.function.name] = mismatch_result
                    tool_record = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(mismatch_result, ensure_ascii=False),
                    }
                    self.messages.append(tool_record)
                    request_messages.append(tool_record)
                    emit({"type": "tool_result",
                          "name": tc.function.name,
                          "preview": "WRONG MODALITY — tool not executed",
                          "elapsed_s": 0.0,
                          "result": mismatch_result,
                          "error": mismatch_result["error"]})
                    continue

                # Default image_path → current_image. The LLM sometimes
                # omits the key entirely OR passes an empty string. Either
                # way, if there is a current image in the session AND the
                # tool's schema declares image_path as a parameter, fill it.
                # We check the tool's declared parameters rather than
                # blindly injecting, because compute / verify_findings /
                # etc don't take image_path and would error on the extra
                # kwarg.
                default_image = self._tool_default_image(tc.function.name)
                if default_image and not args.get("image_path"):
                    tool_obj = self._toolkit.tools.get(tc.function.name)
                    tool_params = (
                        {p.name for p in tool_obj.parameters}
                        if tool_obj else set()
                    )
                    if "image_path" in tool_params:
                        args["image_path"] = default_image
                policy_result = self._policy_skip_tool_call(
                    tc.function.name, args)
                if (
                    policy_result is None
                    and tc.function.name != "verify_findings"
                    and last_verifier_result is not None
                    and not verifier_stale
                ):
                    allowed_after_verify = {
                        str(action.get("tool"))
                        for action in (last_verifier_result.get("next_actions") or [])
                        if isinstance(action, dict) and action.get("tool")
                    }
                    if tc.function.name not in allowed_after_verify:
                        policy_result = {
                            "status": "skipped",
                            "success": True,
                            "policy_skipped": True,
                            "reason": (
                                "The final verifier already closed evidence "
                                "collection and did not authorise this tool."
                            ),
                            "tool": tc.function.name,
                        }
                # Per-tool call counter — stops infinite same-tool loops
                # (e.g. LLM calling verify_findings 12 times in a row on a
                # malformed arg).
                per_tool_calls[tc.function.name] = per_tool_calls.get(tc.function.name, 0) + 1
                emit({"type": "tool_call", "name": tc.function.name, "arguments": args})
                tool_t0 = time.time()
                try:
                    if policy_result is not None:
                        result = policy_result
                    else:
                        result = self._toolkit.execute(tc.function.name, **args)
                except TypeError as e:
                    # Most common: LLM omitted a required kwarg.
                    result = {
                        "error": f"{type(e).__name__}: {e}",
                        "hint": (
                            "The tool call was missing or had mistyped "
                            "arguments. Check the tool's required parameters "
                            "in its schema and retry — or finalise the report "
                            "without this tool if the chain is already complete."
                        ),
                    }
                except Exception as e:
                    result = {"error": str(e)}
                repair = self._executor_repair_tool_call(
                    attempted_tool=tc.function.name,
                    attempted_arguments=args,
                    attempted_arguments_raw=attempted_arguments_raw,
                    result=result,
                    arguments_parse_failed=arguments_parse_failed,
                    tool_schemas=tools,
                    emit=emit,
                )
                if repair is not None:
                    initial_error = (
                        result.get("error")
                        if isinstance(result, dict)
                        else str(result)
                    )
                    args = repair.arguments
                    per_tool_calls[tc.function.name] = (
                        per_tool_calls.get(tc.function.name, 0) + 1
                    )
                    emit({
                        "type": "tool_call",
                        "name": tc.function.name,
                        "arguments": args,
                        "executor_repair": True,
                    })
                    try:
                        result = self._toolkit.execute(
                            tc.function.name,
                            **args,
                        )
                    except Exception as e:
                        result = {"error": str(e)}
                    if isinstance(result, dict):
                        result["_executor_repair"] = {
                            "attempted_tool": tc.function.name,
                            "initial_error": initial_error,
                            "repaired_arguments": args,
                            "reason": repair.reason,
                        }
                tool_elapsed_s = round(time.time() - tool_t0, 3)
                # Record success/failure for the evidence-sufficiency gate.
                # We mark "fail" on either a top-level error key OR an
                # explicit success=False (AdapterResult convention).
                _failed = (
                    isinstance(result, dict)
                    and (result.get("error")
                         or result.get("success") is False)
                )
                verifier_result_valid = True
                if (
                    tc.function.name == "verify_findings"
                    and isinstance(result, dict)
                ):
                    verifier_result_valid = self._verifier_result_valid(result)
                    _failed = _failed or not verifier_result_valid
                if isinstance(result, dict) and result.get("policy_skipped"):
                    tool_outcomes[tc.function.name] = "skipped"
                else:
                    tool_outcomes[tc.function.name] = "fail" if _failed else "ok"
                # Track verifier escalation: if verify_findings returned pending
                # next_actions, the P-E-V loop should run them (see verify_cap
                # above) rather than be force-finalised by the plan_rounds budget.
                if (tc.function.name == "verify_findings"
                        and isinstance(result, dict)
                        and verifier_result_valid):
                    last_verifier_result = result
                    verifier_stale = False
                    next_actions = result.get("next_actions") or []
                    verifier_next_tools = {
                        str(action.get("tool"))
                        for action in next_actions
                        if isinstance(action, dict) and action.get("tool")
                    }
                    verify_wants_more = bool(verifier_next_tools)
                    force_verifier_next = False
                    forced_verifier_steer = False
                elif tc.function.name == "verify_findings":
                    verifier_stale = True
                    force_verifier_next = requires_final_verifier
                elif (
                    last_verifier_result is not None
                    and not (
                        isinstance(result, dict)
                        and result.get("policy_skipped")
                    )
                ):
                    verifier_stale = True
                    if requires_final_verifier:
                        force_verifier_next = True
                # Cache by image path
                img = self._analysis_key(
                    args.get("image_path") or self.context.current_image
                )
                if img:
                    self.context.analyses.setdefault(img, {})[tc.function.name] = result
                # Compose tool message
                llm_result = tool_result_for_profile(
                    prompt_profile, tc.function.name, result
                )
                tool_record = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(
                        llm_result, default=str, ensure_ascii=False
                    ),
                }
                self.messages.append(tool_record)
                request_messages.append(tool_record)
                preview = _summarize_result(tc.function.name, result)
                ev: dict[str, Any] = {
                    "type": "tool_result",
                    "name": tc.function.name,
                    "preview": preview,
                    "elapsed_s": tool_elapsed_s,
                }
                # Surface any figure URLs the tool produced so the UI can
                # render them inline as the trace progresses.
                if isinstance(result, dict):
                    urls = result.get("figure_urls")
                    if urls:
                        ev["figure_urls"] = {k: v for k, v in urls.items() if v}
                    # Persist compact structured tool output for benchmark
                    # traces. AdapterResult.to_jsonable() strips tensors and
                    # arrays before the toolkit returns this dict.
                    ev["result"] = result
                    # Also surface the structured `predictions` block + any
                    # error so the UI's per-step drill-down can show full
                    # context (raw JSON / error message). Large arrays are
                    # already stripped server-side by adapter compaction.
                    preds = result.get("predictions")
                    if preds is not None:
                        ev["predictions"] = preds
                    err = result.get("error")
                    if err:
                        ev["error"] = err
                    # LLM/text tools (vision_impression, verify_findings): show
                    # the human-readable read in the drill-down, since they
                    # have no `predictions` block.
                    detail_md = _tool_detail_md(tc.function.name, result)
                    if detail_md:
                        ev["detail_md"] = detail_md
                emit(ev)

            # After processing this turn's tool calls, see if any tool has
            # been called too many times in a row. If so, inject a single
            # steering message asking the LLM to stop and finalise.
            looped_tools = [t for t, n in per_tool_calls.items() if n >= 3]
            if looped_tools and not steered_for_loop:
                steered_for_loop = True
                steer = {
                    "role": "user",
                    "content": (
                        f"STOP CALLING {looped_tools} — you've called these "
                        f"3+ times. Their last results are already in this "
                        f"conversation. Either accept them and write the final "
                        f"clinical answer in clean Markdown now, or honestly "
                        f"report '/undetermined' if the evidence is genuinely "
                        f"inconclusive. Do NOT call any more tools."
                    ),
                }
                # In-flight only — do NOT persist to self.messages, or it
                # would replay in the saved history as a fake "user" bubble.
                request_messages.append(steer)

        emit({"type": "error", "message": "reached max tool steps"})
        return "(reached max tool steps without a final assistant message)"

    # Per-backend max_tokens ceilings. DashScope rejects max_tokens > 8192 for
    # the Qwen models with a 400 (InvalidParameter), which would otherwise look
    # like "the model won't answer". Clamp before sending.
    _BACKEND_MAX_TOKENS_CAP = {"dashscope": 8192}

    # ── Sufficient-evidence gate ────────────────────────────────────────
    # For each in-scope modality, list the tools whose output is required to
    # constitute "real disease-level evidence". The gate at the end of chat()
    # checks: if ALL of these tools failed for the current modality, the
    # agent has no calibrated classifier signal — the diagnostic call is
    # suppressed and ===INSUFFICIENT_DATA=== is returned instead.
    # vision_impression is deliberately NOT in any core set: it's a gestalt
    # observer (LLM looking at the image), not a calibrated classifier.
    _CORE_TOOLS_BY_MODALITY: ClassVar[dict[str, list[str]]] = {
        "CFP": [
            "cfp_clip_ensemble", "cfp_pdr_cascade",
            "cfp_retsam_segmentation", "cfp_dr_workup",
            "cfp_dr_421_assessment",
            "cfp_dynamic_clip",
            "cfp_glaucoma_workup", "cfp_paired5",
        ],
        # OCT single B-scan: FMUE is the calibrated 16-class classifier.
        # The cube-level adapters cover the volume case.
        "OCT": ["oct_fmue_16class",
                 "oct_volume_macular", "oct_volume_octcubem"],
        "UWF": ["uwf_disease_7class", "uwf_multi_disease"],
        "FFA": ["ffa_classification", "ffa_paired5"],
    }

    # Tools that produce a visual artefact (overlay / heatmap / mask / saved
    # figure). A pure visualization request that successfully ran one of these
    # is exempt from the no-core-observer hard refusal (see _finalize_reply).
    _VIZ_TOOLS: ClassVar[set[str]] = {
        "ffa_lesion_detection", "gradcam", "cfp_retsam_segmentation",
        "uwf_vessel_segmentation", "oct_fluid_segmentation",
        "oct_layer_segmentation", "compute",
    }

    # ── Out-of-scope routing helpers ────────────────────────────────────
    def _analyses_for_image(self, image_path: str | None = None) -> dict[str, Any]:
        key = self._analysis_key(image_path)
        if not key:
            return {}
        analyses = self.context.analyses or {}

        # Backward-compatible migration for sessions saved before cache keys
        # were canonicalised. Merge all aliases that resolve to this image.
        # Do this even when the canonical key already exists: older Web runs
        # commonly stored ``set_current_image`` under the absolute key and all
        # diagnostic evidence under a relative alias.
        canonical_values = analyses.get(key)
        merged: dict[str, Any] = (
            dict(canonical_values) if isinstance(canonical_values, dict) else {}
        )
        aliases: list[str] = []
        for alias, values in list(analyses.items()):
            if alias == key:
                continue
            if self._analysis_key(alias) == key and isinstance(values, dict):
                merged.update(values)
                aliases.append(alias)
        if merged:
            analyses[key] = merged
            for alias in aliases:
                analyses.pop(alias, None)
            return analyses[key]
        return {}

    def _quality_reject_for_image(
        self, image_path: str | None = None
    ) -> tuple[bool, str | None]:
        """Return whether an already-run quality tool rejected this image."""
        by_tool = self._analyses_for_image(image_path)
        for tool_name in ("cfp_efiqa", "cfp_quality_robust",
                          "cfp_eyeq", "oct_quality"):
            result = by_tool.get(tool_name)
            if not isinstance(result, dict):
                continue
            preds = result.get("predictions") or {}
            quality = str(preds.get("quality") or "").strip().lower()
            if preds.get("is_rejected") is True or quality == "reject":
                return True, tool_name
        return False, None

    def _has_objective_evidence_for_image(
        self, image_path: str | None = None
    ) -> bool:
        """True once a non-gestalt calibrated or quantitative tool ran."""
        by_tool = self._analyses_for_image(image_path)
        if not by_tool:
            return False

        def usable(tool_name: str) -> bool:
            result = by_tool.get(tool_name)
            return bool(
                isinstance(result, dict)
                and not result.get("error")
                and not result.get("skipped")
                and not result.get("policy_skipped")
                and result.get("success") is not False
            )

        modality = (self.context.current_modality or "").upper().split(":")[0]
        core_tools = set(self._CORE_TOOLS_BY_MODALITY.get(modality, []))
        if any(usable(t) for t in core_tools):
            return True
        return any(
            usable(t)
            for t in (
                "cfp_retsam_segmentation", "oct_fluid_segmentation",
                "oct_layer_segmentation", "uwf_vessel_segmentation",
                "ffa_lesion_detection",
            )
        )

    def _has_core_evidence_for_image(
        self, image_path: str | None = None
    ) -> bool:
        """True if any calibrated core observer already succeeded for image.

        Follow-up questions often run only a derived computation (for example
        a macula-distance check) while reusing prior classifier/segmentation
        evidence. The sufficient-evidence gate must not forget that prior
        evidence just because the current turn did not re-call a core tool.
        """
        by_tool = self._analyses_for_image(image_path)
        if not by_tool:
            return False
        modality = self._modality_for_image(image_path)
        for tool_name in self._CORE_TOOLS_BY_MODALITY.get(modality, []):
            result = by_tool.get(tool_name)
            if not isinstance(result, dict):
                continue
            if result.get("skipped") or result.get("error"):
                continue
            if result.get("success") is False:
                continue
            return True
        return False

    def _vision_policy_block_reason(
        self,
        image_path: str | None = None,
        *,
        for_escalation: bool = False,
    ) -> str | None:
        """Effort-aware policy for the expensive vision_impression tool."""
        effort = (self.effort or "low").lower()
        policy = get_effort_policy(effort)
        if policy.vision_mode == "disabled":
            return (
                "low effort uses quality plus calibrated classifiers only; "
                "vision_impression is reserved for medium/high escalation"
            )
        if policy.vision_mode == "exhaustive":
            return None

        by_tool = self._analyses_for_image(image_path)
        if "vision_impression" in by_tool:
            return (
                f"{effort} effort allows at most one targeted "
                "vision_impression per image; "
                "reuse the prior result instead of repeating it"
            )
        if not for_escalation and not self._has_objective_evidence_for_image(image_path):
            return (
                f"{effort} effort is objective-first: run quality, calibrated "
                "classifier, and quantitative tools before any visual escalation"
            )
        return None

    def _policy_skip_tool_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        if tool_name != "vision_impression":
            return None
        reason = self._vision_policy_block_reason(args.get("image_path"))
        if reason is None:
            return None
        return {
            "tool": tool_name,
            "skipped": True,
            "policy_skipped": True,
            "reason": reason,
            "effort": self.effort,
            "recommendation": (
                "Use already available objective evidence and finalise with "
                "an explicit uncertainty/quality caveat if needed."
            ),
        }

    def _invalid_input_refusal(self) -> str:
        """Structured refusal for files OphAgent cannot read as images.

        This is intentionally separate from NON_OPHTHALMOLOGIC. A corrupt or
        missing file gives no visual evidence, so it must never be treated as
        either a diagnosis or a non-eye-image classification.
        """
        reason = (
            self.context.invalid_input_reason
            or "submitted file could not be read as an image"
        )
        payload = json.dumps({
            "verdict": "invalid_input",
            "reason": reason,
            "do_not_use_for_clinical_decisions": True,
        }, ensure_ascii=True)
        return (
            "## Invalid input file\n\n"
            "OphAgent could not analyze the submitted file because it is not "
            "a readable image, cannot be found, or is outside the allowed "
            "session file area.\n\n"
            f"Reason: `{reason}`\n\n"
            "**No diagnostic interpretation will be produced for this file.**\n\n"
            "Please re-upload a readable CFP/OCT/UWF/FFA image or a supported "
            "ophthalmic image format.\n\n"
            "```\n"
            "===INVALID_INPUT===\n"
            f"{payload}\n"
            "```"
        )

    def _non_ophthalmologic_refusal(self) -> str:
        """Structured refusal for inputs that are not eye images at all.

        The output deliberately omits any `top1` / `confidence` / `===FINAL===`
        field so it is structurally distinct from a real diagnostic call.
        Downstream parsers should never treat this as a positive diagnosis.
        """
        return (
            "## Image is out of scope\n\n"
            "The submitted image does not appear to be an ophthalmologic "
            "scan. OphAgent analyses **colour fundus (CFP), OCT, "
            "ultra-wide-field fundus (UWF), and fluorescein angiography "
            "(FFA)** images, plus a vision-only mode for related ophthalmic "
            "modalities (visual field, OCT-A, FAF, slit-lamp, B-scan US, "
            "anterior-segment OCT, corneal topography).\n\n"
            "**No diagnostic interpretation will be produced for this image.**\n\n"
            "If you intended to submit an eye image, please:\n"
            "  - Check that the file is correct and not corrupted.\n"
            "  - Re-upload a colour fundus photograph, OCT B-scan, UWF "
            "fundus image, or fluorescein angiogram.\n\n"
            "```\n"
            "===NOT_OPHTHALMOLOGIC===\n"
            "{\"verdict\": \"out_of_scope\", "
            "\"reason\": \"submitted image is not an ophthalmologic scan\", "
            "\"do_not_use_for_clinical_decisions\": true}\n"
            "```"
        )

    def _vision_only_impression(self, image_path: str, sub_label: str,
                                  user_question: str, emit=None) -> str:
        """One-shot vision-LLM impression for OPHTHALMOLOGIC_OTHER modalities.

        No tool loop, no verifier, no top-1 diagnosis. Returns a structured
        JSON impression keyed by the modality-specific schema in
        `vision_prompts.vision_only`, wrapped in a
        `===VISION_ONLY_IMPRESSION===` block.
        """
        import base64
        import json as _json
        from .vision_prompts import vision_only

        if emit is not None:
            emit({"type": "thinking", "note": f"vision-only mode ({sub_label})"})

        system_prompt = vision_only.get_system_prompt(sub_label)
        user_prompt = vision_only.build_user_prompt(sub_label, user_question)

        # Encode the image
        try:
            suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
            mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
            b64 = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
        except Exception as e:
            return self._vision_only_error(sub_label, f"image read failed: {e}")

        vmodel = self.vision_model
        if vmodel is None:
            return self._vision_only_error(
                sub_label,
                "no vision-capable model available — set OPH_WEB_VISION_MODEL "
                "or use a vision-capable chat backend",
            )

        try:
            resp = self._client.chat.completions.create(
                model=vmodel,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return self._vision_only_error(sub_label,
                                             f"LLM call failed: {type(e).__name__}: {e}")

        # Best-effort JSON parse (tolerant of fenced code blocks)
        parsed = None
        try:
            from .vision_prompts.validators import parse_json_lenient
            parsed = parse_json_lenient(raw)
        except Exception:
            pass

        normalised = vision_only._normalise(sub_label)
        header = (
            f"## Vision-only impression ({normalised})\n\n"
            f"⚠️ **Degraded mode** — no trained classifier or segmentation "
            f"model is available for this modality. The following is the "
            f"LLM's gestalt impression based on visual inspection only. "
            f"It is NOT a full-pipeline diagnostic call and must not be "
            f"used as a sole basis for clinical decisions.\n\n"
        )
        body = (
            "```json\n"
            "===VISION_ONLY_IMPRESSION===\n"
            f"{_json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else raw}\n"
            "```\n"
        )
        return header + body

    # ── Modality feasibility ──────────────────────────────────────────
    def _modality_feasibility_hint(self) -> str:
        """Dynamic per-modality capability summary injected into the
        system prompt at chat() time.

        Design: list ONLY what's actually registered. Do not hardcode
        any "X is not supported" claims — those go stale the moment a
        new adapter ships. The LLM is told to compare the user's task
        against the listed tool descriptions; anything the tools can't
        cover is honestly refused.

        Why this matters: if tomorrow someone adds a UWF disc-cup
        segmentor, it auto-appears here (adapter self-registration) and
        the LLM picks it up with no prompt edit needed.
        """
        from collections import defaultdict
        try:
            from ophagent.adapters import GLOBAL_REGISTRY
        except Exception:
            return ""
        by_mod: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name, cls in GLOBAL_REGISTRY._classes.items():
            m = (cls.metadata.modality or "").strip() or "multi"
            # Single-line summary: take the description's first sentence
            # (up to first period or 120 chars, whichever first).
            desc = (cls.metadata.description or "").strip()
            first_sentence = desc.split(".")[0].strip()
            if len(first_sentence) > 120:
                first_sentence = first_sentence[:117] + "…"
            by_mod[m].append((name, first_sentence))

        cur = self.context.current_modality or "unknown"
        attached = {cur}
        for img in self.context.attached_images:
            if img.get("modality"):
                attached.add(img["modality"])

        lines: list[str] = [
            "# Modality-task feasibility",
            "",
            f"Attached image modality(ies): **{', '.join(sorted(attached))}**",
            "",
            "## Available tools for this session",
            "",
        ]
        for m in sorted(attached):
            tools = sorted(by_mod.get(m, []))
            if not tools:
                continue
            lines.append(f"### {m}")
            for name, summary in tools:
                lines.append(f"  - `{name}` — {summary}.")
            lines.append("")
        # Cross-modal / meta tools always allowed
        multi = sorted(by_mod.get("multi", []))
        if multi:
            lines.append("### multi / cross-modal")
            for name, summary in multi:
                lines.append(f"  - `{name}` — {summary}.")
            lines.append("")

        lines += [
            "## How to use this list",
            "",
            "1. **Map the user's task to a tool above.** Read each tool's "
            "summary. If exactly one fits, use it.",
            "",
            "2. **If no tool matches the task** for the attached "
            "modality(ies), refuse honestly: state the task the user "
            "asked for, name the closest tool(s) you DO have, and "
            "explain the gap. Suggest what input would let you do it "
            "(e.g. 'a posterior-pole CFP would let me run "
            "`cfp_glaucoma_workup` for cup-disc ratio').",
            "",
            "3. **Never call a tool whose modality is not attached.** The "
            "runtime will reject it with a `modality_mismatch` error "
            "and waste a tool-call round. This applies to cfp_*, oct_*, "
            "uwf_*, ffa_* depending on what's attached.",
            "",
            "4. **Don't invent capabilities.** If you're not sure a "
            "listed tool does what the user wants, the safe move is to "
            "ask the user a clarifying question rather than run "
            "something tangential and pretend it answers the question.",
        ]
        return "\n".join(lines)

    # ── Modality guard ────────────────────────────────────────────────
    def _modality_mismatch_check(self, tool_name: str) -> dict | None:
        """If `tool_name` is for a modality not present in this session,
        return an error-shaped dict (the chat loop will inject it as the
        tool result). Returns None when the call is permissible.

        Allowed when:
          * No image has been set yet (modality unknown).
          * The tool's modality matches the current_image OR any other
            attached image. (Multi-image sessions, e.g. CFP + FFA paired,
            need to invoke both sides' tools.)
          * The tool's modality is 'multi' / blank (cross-modal tools).
          * The tool is a meta-tool not in GLOBAL_REGISTRY
            (compute / verify_findings / detect_modality / etc.).
        """
        try:
            from ophagent.adapters import GLOBAL_REGISTRY
        except Exception:
            return None
        cls = GLOBAL_REGISTRY._classes.get(tool_name)
        if cls is None:
            return None    # meta tool — always allowed
        tool_mod = (cls.metadata.modality or "").strip()
        if tool_mod in ("", "multi"):
            return None    # cross-modal
        if not self.context.current_modality:
            return None    # no image set; let the LLM try

        valid_mods = {self.context.current_modality}
        for img in self.context.attached_images:
            m = img.get("modality")
            if m:
                valid_mods.add(m)
        if tool_mod in valid_mods:
            return None

        return {
            "success": False,
            "error": (
                f"Modality mismatch: tool '{tool_name}' is for "
                f"{tool_mod} images, but the current session has "
                f"{sorted(valid_mods)}. This tool was NOT executed. "
                f"Either (a) use a tool available for the current "
                f"modality, or (b) tell the user the requested task "
                f"is not supported on this modality and stop trying."
            ),
            "modality_mismatch": True,
            "current_modality": self.context.current_modality,
            "tool_modality": tool_mod,
        }

    # ── User-interrupt reply ───────────────────────────────────────────
    def _emit_interrupt_reply(self, emit, tool_outcomes: dict[str, str],
                                pending_tool_calls=None) -> str:
        """Handle a user-triggered interrupt cleanly.

        Two responsibilities:
          1. **Keep the conversation history valid.** OpenAI requires that
             every assistant message with `tool_calls` be followed by
             matching `role=tool` messages. If we interrupted halfway
             through a tool batch, backfill the missing slots with
             placeholders so the next chat turn doesn't 400.
          2. **Surface the interruption to the UI.** Emit BOTH an `error`
             trace event AND a `text` event so:
               • the live trace shows "Interrupted by user"
               • the chat panel gets a "## Interrupted" assistant bubble
             (relying on just one or the other left one of the surfaces
             empty, as observed in user testing.)
        """
        import json as _json
        if pending_tool_calls:
            already_answered = {
                m.get("tool_call_id")
                for m in self.messages
                if m.get("role") == "tool"
            }
            for tc in pending_tool_calls:
                if tc.id in already_answered:
                    continue
                placeholder = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": _json.dumps({
                        "interrupted": True,
                        "reason": "user interrupted before this tool executed",
                    }),
                }
                self.messages.append(placeholder)

        reply = (
            "## Interrupted\n\n"
            "The agent was stopped by the user. Any partial tool results "
            "above are kept in the conversation history; the next message "
            "in this session can refer to them.\n"
        )
        emit({"type": "error", "message": "Interrupted by user"})
        emit({"type": "text", "content": reply})
        self.messages.append({"role": "assistant", "content": reply})
        return self._finalize_reply(reply, tool_outcomes)

    # ── Sufficient-evidence gate ───────────────────────────────────────
    def _prepend_quality_limited_banner(
        self, content: str, quality_tool: str | None
    ) -> str:
        # Machine-readable benchmark outputs must remain byte-parseable.  The
        # quality assessment is still available in the saved tool trace.
        if "===FINAL===" in content:
            return content
        tool = quality_tool or "quality tool"
        if self._user_prefers_chinese():
            if "## 图像质量限制" in content:
                return content
            banner = (
                "## 图像质量限制\n\n"
                f"`{tool}` 将图像标记为 **Reject/质量较差**。严重病灶本身也可能"
                "降低质量评分，因此该结果只描述技术可评估性，**不会单独否定"
                "其他工具检出的明确阳性病灶**。以下报告保留已有阳性证据；"
                "对于未检出细微病灶的阴性结论应更谨慎，并建议结合重拍图像或"
                "必要的 OCT/FFA 检查。\n\n"
            )
        else:
            if "## Image-quality limitation" in content:
                return content
            banner = (
                "## Image-quality limitation\n\n"
                f"`{tool}` marked the image as **Reject/poor quality**. Severe "
                "pathology can itself lower an automated quality score, so this "
                "is a technical-assessability warning and **does not by itself "
                "negate definite positive findings from other tools**. Positive "
                "evidence is retained below; negative calls for subtle lesions "
                "should be interpreted more cautiously and correlated with "
                "repeat imaging or OCT/FFA when indicated.\n\n"
            )
        return banner + content

    def _user_prefers_chinese(self) -> bool:
        for message in reversed(self.messages):
            if message.get("role") != "user":
                continue
            text = str(message.get("content") or "")
            return any("\u4e00" <= char <= "\u9fff" for char in text)
        return False

    @staticmethod
    def _verifier_requires_limited_report(result: dict[str, Any] | None) -> bool:
        """Return True only when the verifier has not completed safely.

        A passed verifier may still report condition-specific disagreement or
        low-confidence tools.  Those warnings must not erase otherwise valid
        disease evidence or a machine-readable benchmark verdict.
        """
        if not isinstance(result, dict):
            return False
        if result.get("verify_passed") is False or result.get("next_actions"):
            return True
        diagnostic_status = str(result.get("diagnostic_status") or "").lower()
        return diagnostic_status in {"failed", "insufficient", "insufficient_data"}

    @staticmethod
    def _verifier_needs_caveat(result: dict[str, Any] | None) -> bool:
        """Return True for a passed verifier with unresolved local warnings."""
        if not isinstance(result, dict):
            return False
        if OphSession._verifier_requires_limited_report(result):
            return False
        if str(result.get("diagnostic_status") or "").lower() == "conflict":
            return True
        recommendation = str(result.get("recommendation") or "").lower()
        return bool(result.get("warnings")) and any(
            marker in recommendation
            for marker in ("undetermined", "low-confidence", "do not force", "conflict")
        )

    def _prepend_verifier_caveat(
        self, content: str, result: dict[str, Any]
    ) -> str:
        """Keep the full report while localising unresolved verifier warnings."""
        if "===FINAL===" in content:
            return content
        categories = result.get("warning_categories") or {}
        conflicts = categories.get("diagnostic_conflict") or []
        if not conflicts:
            conflicts = [
                str(item) for item in (result.get("warnings") or [])
                if "conflict" in str(item).lower() or "contradict" in str(item).lower()
            ]
        conflict_lines = "\n".join(f"- {item}" for item in conflicts[:2])
        if self._user_prefers_chinese():
            if "## 局部证据冲突" in content:
                return content
            detail = conflict_lines or "- 验证器记录了尚未完全解决的局部证据分歧。"
            banner = (
                "## 局部证据冲突\n\n"
                "终末验证已完成，但以下分歧应限制在对应疾病或证据层面解释；"
                "它们**不会清空其他已有一致证据或整份报告**。\n\n"
                f"{detail}\n\n"
            )
        else:
            if "## Local evidence conflict" in content:
                return content
            detail = conflict_lines or "- The verifier recorded a remaining local evidence disagreement."
            banner = (
                "## Local evidence conflict\n\n"
                "Terminal verification completed, but the following disagreement "
                "should be localised to the affected disease or evidence source; "
                "it **does not erase other concordant evidence or the full report**.\n\n"
                f"{detail}\n\n"
            )
        return banner + content

    def _verifier_limited_response(
        self, result: dict[str, Any], modality: str
    ) -> str:
        """Build a deterministic low-confidence report from surviving evidence.

        This replaces, rather than merely annotates, the LLM synthesis. It is
        therefore impossible for a model-generated "high confidence" sentence
        to contradict an explicit verifier instruction to remain undetermined.
        """
        evidence_lines: list[str] = []
        analyses = self._analyses_for_image(self.context.current_image)
        for tool_name in self._CORE_TOOLS_BY_MODALITY.get(modality, []):
            tool_result = analyses.get(tool_name)
            if not isinstance(tool_result, dict):
                continue
            if (tool_result.get("error") or tool_result.get("skipped")
                    or tool_result.get("success") is False
                    or tool_result.get("undetermined")):
                continue
            evidence_lines.append(
                f"- `{tool_name}`: {_summarize_result(tool_name, tool_result)}"
            )
        evidence = "\n".join(evidence_lines[:6]) or "- No conclusive core-tool output."
        recommendation = str(result.get("recommendation") or "").strip()
        if self._user_prefers_chinese():
            if not evidence_lines:
                evidence = "- 未检索到可用于最终汇总的核心工具证据。"
            return (
                "## 评估状态\n\n"
                "**结果未定 / 低置信度。** 当前疾病层面的证据仍存在"
                "尚未解决的冲突，因此不强行给出单一高置信度诊断。\n\n"
                "## 保留证据\n\n"
                f"{evidence}\n\n"
                "## 验证器意见\n\n"
                f"{recommendation or '建议补充影像并由眼科医生复核。'}"
            )
        return (
            "## Assessment status\n\n"
            "**Undetermined / low confidence.** The available tools did not "
            "provide sufficient independent corroboration for a definitive "
            "diagnosis. Any classifier tendency below is provisional.\n\n"
            "## Surviving evidence\n\n"
            f"{evidence}\n\n"
            "## Verifier guidance\n\n"
            f"{recommendation or 'Obtain confirmatory imaging and clinical review.'}"
        )

    def _finalize_reply(self, content: str,
                          tool_outcomes: dict[str, str],
                          verifier_result: dict[str, Any] | None = None) -> str:
        """Inspect tool successes before returning the agent's final reply.

        Three branches:
          1. **No tools called at all** → return as-is (e.g. follow-up Q&A
             on a previously-analysed image; the LLM is reusing cached
             evidence).
          2. **All core observers failed** → replace the LLM reply with a
             structured ===INSUFFICIENT_DATA=== block (no diagnostic
             fields). The agent must NOT silently emit a top-1 when no
             calibrated classifier produced output.
          3. **Verifier did not complete safely** → deterministically replace
             the model synthesis with a limited evidence summary.
          4. **Verifier passed with local conflict** → retain the full report
             and prepend a condition-level caveat.
          5. **At least one core observer succeeded but ≥ 50% of attempted
             tools failed** → prepend a degraded-mode banner but keep the
             reply (the agent had real evidence, just less than a healthy
             run).
          6. **Otherwise** → return as-is.
        """
        multimodal_missing, multimodal_verifier_needed = (
            self._multimodal_completion_gap(tool_outcomes)
        )
        if multimodal_missing or multimodal_verifier_needed:
            return self._multimodal_incomplete_response(
                multimodal_missing,
                multimodal_verifier_needed,
            )
        if not tool_outcomes:
            return content
        modality = (self.context.current_modality or "unknown").upper().split(":")[0]
        core = self._CORE_TOOLS_BY_MODALITY.get(modality, [])
        core_ok = (
            any(tool_outcomes.get(t) == "ok" for t in core)
            or self._has_core_evidence_for_image(self.context.current_image)
        )
        # Failed and total counts (across ALL tools called this turn, not
        # just core ones — to detect systemic failure)
        n_total = len(tool_outcomes)
        n_failed = sum(1 for v in tool_outcomes.values() if v == "fail")

        # Branch 2: hard refusal — no core observer succeeded for this modality
        if core and not core_ok:
            # Exception: a pure VISUALIZATION request ("mark / show / 标出 the
            # lesions") that produced a figure is NOT a fabricated diagnosis —
            # the user asked to SEE something, not for a diagnostic call. Return
            # the reply with an honest "visualization only" note instead of the
            # diagnosis-shaped hard refusal. Still requires a figure-producing
            # tool to have actually succeeded, so we never wave through an
            # empty/failed turn.
            if self._is_visualization_request() and \
                    any(tool_outcomes.get(t) == "ok" for t in self._VIZ_TOOLS):
                return self._prepend_viz_only_note(content, tool_outcomes)
            return self._insufficient_data_response(
                modality=modality,
                tool_outcomes=tool_outcomes,
                core_tools=core,
                reason="no_core_observer_succeeded",
            )
        if self._verifier_requires_limited_report(verifier_result):
            content = self._verifier_limited_response(
                verifier_result or {}, modality)
        elif self._verifier_needs_caveat(verifier_result):
            content = self._prepend_verifier_caveat(
                content, verifier_result or {})
        quality_rejected, quality_tool = self._quality_reject_for_image(
            self.context.current_image)
        if quality_rejected:
            content = self._prepend_quality_limited_banner(
                content, quality_tool)
        # Branch 3: degraded mode — at least one core ok but many failures
        if n_total >= 3 and n_failed / n_total >= 0.5:
            return self._prepend_degraded_banner(content, tool_outcomes)
        # Branch 4: normal reply
        return content

    def reformat_last(self, instruction: str) -> str:
        """One tool-free LLM turn that REUSES the full message history.

        Evaluation harnesses use this to REPAIR a final reply that ran the
        reasoning correctly but slipped the required OUTPUT FORMAT — e.g. the
        model wrote a narrative, printed the ``===FINAL===`` marker, then
        stopped (``finish_reason='stop'``) WITHOUT emitting the mandatory JSON
        block. Every tool result from the preceding ``chat()`` is already in
        ``self.messages``, so this re-states the conclusion in the demanded
        format with NO tools and NO re-run of perception.

        Returns the assistant content (also appended to ``self.messages`` so a
        follow-up repair would see it). Returns "" if the LLM call fails.
        """
        self._ensure_client()
        from .prompt_profiles import (
            STANDARD_PROFILE,
            normalize_prompt_profile,
            system_prompt_for_profile,
        )
        prompt_profile = normalize_prompt_profile(
            getattr(self, "prompt_profile", "standard")
        )
        base_system_prompt = (
            system_prompt_for_profile(prompt_profile)
            if prompt_profile != STANDARD_PROFILE else OPH_SYSTEM_PROMPT
        )
        sys_msg = {"role": "system", "content": base_system_prompt}
        user_msg = {"role": "user", "content": instruction}
        request_messages = [sys_msg] + self.messages + [user_msg]
        # tools=[] guarantees the model cannot re-enter the tool loop; with no
        # tools the API ignores tool_choice, so "none" is just belt-and-braces.
        resp = self._safe_completion(request_messages, tools=[],
                                     emit=lambda *_a, **_k: None,
                                     tool_choice="none")
        if resp is None:
            return ""
        content = resp.choices[0].message.content or ""
        self.messages.append(user_msg)
        self.messages.append({"role": "assistant", "content": content})
        return content

    def _insufficient_data_response(self, *, modality: str,
                                       tool_outcomes: dict[str, str],
                                       core_tools: list[str],
                                       reason: str) -> str:
        """Format the ===INSUFFICIENT_DATA=== refusal block.

        This output is STRUCTURALLY DISTINCT from a normal ===FINAL===
        diagnosis: it contains no top1 / confidence / probability /
        differential fields. A downstream parser must never confuse the
        two.
        """
        import json as _json
        succeeded_other = [t for t, v in tool_outcomes.items()
                            if v == "ok" and t not in core_tools]
        # Markdown header + audit table
        lines = [
            "## Insufficient evidence — no diagnosis produced",
            "",
            f"The agent could not complete a diagnostic call for **{modality}** "
            "because every calibrated disease-detection tool for this modality "
            "failed. Gestalt or quality observers alone are not sufficient "
            "evidence for a clinical-grade output.",
            "",
            "### Core tool outcomes (at least one must succeed)",
            "",
            "| Tool | Status |",
            "|---|---|",
        ]
        for t in core_tools:
            status = tool_outcomes.get(t, "not_called")
            mark = {"ok": "OK", "fail": "FAILED",
                     "not_called": "not called"}.get(status, status)
            lines.append(f"| `{t}` | {mark} |")
        if succeeded_other:
            lines.append("")
            lines.append("### Non-core observers that did succeed "
                          "(insufficient for diagnosis on their own)")
            lines.append("")
            for t in succeeded_other:
                lines.append(f"- `{t}`")
        lines.append("")
        lines.append(
            "**This output must not be used for clinical decisions.** "
            "Re-run after the failing backends are restored, or check "
            "the preflight health-check command. The agent intentionally "
            "does NOT guess a diagnosis from gestalt observation alone.")
        lines.append("")
        payload = {
            "verdict": "diagnostic_call_not_possible",
            "reason": reason,
            "modality": modality,
            "core_tools_required_any_of": core_tools,
            "core_tools_failed": [t for t in core_tools
                                    if tool_outcomes.get(t) == "fail"],
            "non_core_tools_succeeded": succeeded_other,
            "do_not_use_for_clinical_decisions": True,
            "next_action": ("Check tool backends and retry; do NOT extract "
                             "a diagnosis from this block"),
        }
        lines.append("```")
        lines.append("===INSUFFICIENT_DATA===")
        lines.append(_json.dumps(payload, indent=2, ensure_ascii=False))
        lines.append("```")
        return "\n".join(lines)

    def _is_visualization_request(self) -> bool:
        """True if the most recent user turn asks to SEE/MARK something
        (overlay, boxes, heatmap, location) rather than for a diagnosis."""
        txt = ""
        for m in reversed(self.messages):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if not isinstance(c, str) or not c.strip():
                continue
            if c.lstrip().startswith("STOP CALLING") or "previous reply leaked" in c:
                continue  # skip internal steering nudges
            txt = c
            break
        if not txt:
            return False
        low = txt.lower()
        kw_en = ("mark", "draw", "overlay", "highlight", "circle", "outline",
                 "annotate", "visuali", "where is", "where are", "locate",
                 "bounding box", "boxes", "segment", "heatmap", "grad-cam",
                 "gradcam", "show me", "point out")
        kw_zh = ("标", "圈", "画出", "画框", "可视化", "框出", "标记", "标出",
                 "在哪", "位置", "热力图", "热图", "勾", "描出")
        return any(k in low for k in kw_en) or any(k in txt for k in kw_zh)

    def _prepend_viz_only_note(self, content: str,
                               tool_outcomes: dict[str, str]) -> str:
        """For a visualization request where a figure was produced but no
        calibrated diagnostic classifier ran: keep the reply, but make clear
        it is a visualization, not a diagnostic call."""
        ok = [t for t, v in tool_outcomes.items() if v == "ok"]
        banner = (
            "> 🖼️ **Visualization only** — produced from "
            f"`{', '.join(ok) or 'the requested tool'}`. No calibrated "
            "diagnostic classifier ran this turn, so this is NOT a diagnosis; "
            "do not read a disease grade into the overlay. Ask for a full "
            "analysis to get a diagnostic assessment.\n\n"
        )
        return banner + content

    def _prepend_degraded_banner(self, content: str,
                                    tool_outcomes: dict[str, str]) -> str:
        """Prepend a clearly-marked banner above the agent's reply when
        ≥ 50% of called tools failed (but at least one core succeeded)."""
        failed = [t for t, v in tool_outcomes.items() if v == "fail"]
        ok = [t for t, v in tool_outcomes.items() if v == "ok"]
        banner = (
            "> ⚠️ **DEGRADED MODE** — "
            f"{len(failed)} of {len(tool_outcomes)} tools failed during "
            "this analysis "
            f"(failed: `{', '.join(failed)}`; succeeded: `{', '.join(ok)}`). "
            "The diagnostic call below is based on the surviving tools only "
            "and may be less reliable than a full-pipeline assessment. "
            "Cross-reference with clinical examination required.\n\n"
        )
        return banner + content

    def _vision_only_error(self, sub_label: str, msg: str) -> str:
        return (
            f"## Vision-only mode failure ({sub_label})\n\n"
            f"This image was classified as an out-of-scope ophthalmologic "
            f"modality (`{sub_label}`), which is normally handled in "
            f"vision-only mode. However, that pipeline failed:\n\n"
            f"  **{msg}**\n\n"
            f"No diagnostic interpretation will be produced. Please verify "
            f"the chat backend supports vision inputs and a vision model "
            f"is configured (e.g. `OPH_WEB_VISION_MODEL=qwen3-vl-plus`).\n\n"
            "```\n"
            "===VISION_ONLY_IMPRESSION===\n"
            "{\"verdict\": \"degraded_mode_failed\", "
            f"\"sub_label\": \"{sub_label}\", \"reason\": {repr(msg)[:200]}, "
            "\"do_not_use_for_clinical_decisions\": true}\n"
            "```"
        )

    def _unverified_input_refusal(self) -> str:
        """Refuse when the system cannot establish ophthalmic image scope."""
        reason = (
            self.context.scope_failure_reason
            or "ophthalmic image scope could not be verified"
        )
        payload = json.dumps({
            "verdict": "input_scope_unverified",
            "reason": reason,
            "do_not_use_for_clinical_decisions": True,
        }, ensure_ascii=True)
        return (
            "## Image scope could not be verified\n\n"
            "OphAgent could not establish that the submitted image belongs "
            "to a supported ophthalmic modality. Diagnostic tools were not "
            "run.\n\n"
            f"Reason: `{reason}`\n\n"
            "**No diagnostic interpretation will be produced for this image.**\n\n"
            "Restore the modality detector or configure a vision-capable "
            "scope check, then re-upload the image.\n\n"
            "```\n"
            "===UNVERIFIED_INPUT===\n"
            f"{payload}\n"
            "```"
        )

    def _safe_completion(self, messages, tools, emit, tool_choice="auto"):
        """Call the LLM with automatic max_tokens back-off on 402 / credit errors.

        `tool_choice` is "auto" normally, but the chat loop passes "required"
        on the first turn of a fresh image so the model MUST start the tool
        pipeline (some models — Qwen on DashScope — otherwise sometimes answer
        from thin air and fabricate results)."""
        cap_ceiling = self._BACKEND_MAX_TOKENS_CAP.get(self.backend)
        top = min(self.max_tokens, cap_ceiling) if cap_ceiling else self.max_tokens
        budgets = [top, max(2048, top // 2), 1024]
        last_err: Exception | None = None
        # Decoding temperature (paper config 0.4). Dropped automatically if a
        # backend rejects a non-default value (e.g. a strict reasoning endpoint).
        drop_temp = [False]

        def _temp_kw():
            return {} if (drop_temp[0] or self.temperature is None) else {"temperature": self.temperature}

        def _native_kw():
            # NATIVE EFFORT (OPH_NATIVE_EFFORT=1, gpt-5 only): map the agent's effort
            # tier to gpt-5's calibrated reasoning_effort (kwarg) + verbosity (extra_body).
            # This is the monotonic, OpenAI-tuned "think harder" knob; the agent LOOP
            # stays bounded for all tiers (see plan_rounds/verify_cap), so higher effort
            # = deeper per-call reasoning, NOT more loop rounds. Other models lack these
            # params -> fall back to the prompt-based effort directive (no native kwargs).
            if os.environ.get("OPH_NATIVE_EFFORT") != "1" or "gpt-5" not in (self.model or "").lower():
                return {}
            eff = {"low": "minimal", "medium": "medium", "high": "high",
                   "max": "high", "ultra": "high"}.get(self.effort, "medium")
            verb = {"low": "low", "medium": "medium", "high": "medium",
                    "max": "high", "ultra": "high"}.get(self.effort, "medium")
            return {"reasoning_effort": eff, "extra_body": {"verbosity": verb}}

        for cap in budgets:
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=cap,
                    **_temp_kw(),
                    **_native_kw(),
                )
            except Exception as e:
                last_err = e
                msg = str(e)
                # Backend rejected the temperature value → retry this cap with
                # temperature dropped (fall back to the backend default).
                if not drop_temp[0] and "temperature" in msg.lower():
                    drop_temp[0] = True
                    try:
                        return self._client.chat.completions.create(
                            model=self.model, messages=messages, tools=tools,
                            tool_choice=tool_choice, max_tokens=cap, **_temp_kw(), **_native_kw(),
                        )
                    except Exception as e2:
                        last_err = e2
                        msg = str(e2)
                # Some models reject tool_choice='required' (e.g. qwen3.7-max on
                # DashScope returns 400 InvalidParameter). Gracefully fall back
                # to 'auto' — the model then calls tools on its own.
                if tool_choice != "auto" and "tool_choice" in msg.lower():
                    try:
                        return self._client.chat.completions.create(
                            model=self.model, messages=messages, tools=tools,
                            tool_choice="auto", max_tokens=cap, **_temp_kw(),
                        )
                    except Exception as e2:
                        last_err = e2
                        msg = str(e2)
                # OpenRouter 402: re-try with a smaller budget if remaining is hinted
                import re
                m = re.search(r"can only afford (\d+)", msg)
                if m:
                    affordable = int(m.group(1))
                    smaller = max(256, affordable - 200)  # safety margin
                    try:
                        emit({"type": "error",
                              "message": f"Budget too high; retrying with max_tokens={smaller}"})
                        return self._client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools,
                            tool_choice=tool_choice,
                            max_tokens=smaller,
                            **_temp_kw(),
                        )
                    except Exception as e2:
                        last_err = e2
                        break
                # Non-credit error: don't retry
                break
        emit({"type": "error", "message": f"{type(last_err).__name__}: {last_err}"})
        return None

    def _context_note(self) -> str:
        bits = []
        # All images attached this session — surfaces multi-modal availability
        if self.context.attached_images:
            bits.append("- Images attached to this session:")
            for entry in self.context.attached_images:
                star = " *focus*" if entry["path"] == self.context.current_image else ""
                bits.append(
                    f"    · [{entry.get('modality','?')}] "
                    f"`{self.session_file_reference(entry['path'])}`{star}"
                )
            bits.append("  When a tool can accept multiple modalities (e.g. "
                        "`cross_cfp_oct`), pass the matching attached paths "
                        "explicitly. Otherwise the focused image is used.")
        elif self.context.current_image:
            modality = self.context.current_modality or "unknown"
            image_ref = self.session_file_reference(self.context.current_image)
            bits.append(f"- Current image: `{image_ref}` (modality={modality})")
        if self.context.current_volume:
            volume_ref = self.session_file_reference(self.context.current_volume)
            bits.append(f"- Current volume: `{volume_ref}`")
        if self.context.analyses:
            for path, by_tool in self.context.analyses.items():
                tools = ", ".join(by_tool.keys())
                bits.append(f"- Already analysed `{Path(path).name}`: {tools}")
        if self.context.last_report:
            report_ref = self.session_file_reference(self.context.last_report.get("html"))
            bits.append(
                f"- Last visual report: `{report_ref}`"
            )
        return "\n".join(bits)


def _tool_detail_md(tool_name: str, result: dict) -> str | None:
    """Human-readable markdown for the per-step drill-down, for LLM/text tools
    whose value isn't a `predictions` block (e.g. vision_impression's gestalt
    read + differential, verify_findings' recommendation). Returns None when
    there's nothing text-like to show. Single source of truth shared by the
    live SSE event builder and the history-reload exposer."""
    if not isinstance(result, dict):
        return None
    # Rendered markdown the tool already built (vision_impression, reports).
    for key in ("impression_markdown", "report_markdown", "report"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v[:6000]
    if tool_name == "verify_findings":
        parts = []
        rec = result.get("recommendation")
        if rec:
            parts.append(f"**Recommendation:** {rec}")
        for key in ("reasoning", "rationale", "explanation", "details"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        if parts:
            return "\n\n".join(parts)[:6000]
    return None


def _summarize_result(tool_name: str, result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)[:140]
    if result.get("error"):
        return f"error: {result['error']}"[:140]
    if result.get("skipped"):
        return f"skipped ({result.get('reason', 'n/a')})"[:140]
    preds = result.get("predictions", {})
    if tool_name == "cfp_eyeq":
        q = preds.get("quality")
        c = result.get("confidence")
        return f"quality={q} ({c:.0%})" if c is not None else f"quality={q}"
    if tool_name == "cfp_od_detection":
        return f"OD={preds.get('has_od')} Fovea={preds.get('has_fovea')}"
    if tool_name == "cfp_pdr_cascade":
        return f"{preds.get('category')} | reasons={preds.get('predicted_reasons')}"
    if tool_name == "cfp_dr_421_assessment":
        sev = preds.get("severity_proxy")
        label = preds.get("severity_label") or "unknown"
        hem_n = preds.get("hemorrhage_count", 0)
        hem_area = preds.get("hemorrhage_area_px", 0)
        strict = preds.get("rule_4_hemorrhage_all_quadrants_strict")
        area_proxy = preds.get("rule_4_hemorrhage_area_weighted_proxy")
        heavy = preds.get("heavy_lesion_burden_proxy")
        strong = preds.get("strong_severe_npdr_proxy")
        return (
            f"{label} grade={sev}; hem={hem_n}/{hem_area}px; "
            f"strict421={strict}; area421={area_proxy}; "
            f"heavy={heavy}; strong={strong}"
        )
    if tool_name == "cfp_glaucoma":
        return (f"RG prob={result.get('confidence',0):.2f} "
                f"signs={preds.get('predicted_signs')}")
    if tool_name == "cfp_clip_multi_disease":
        return f"{preds.get('predicted_class')} ({result.get('confidence',0):.0%})"
    if tool_name == "oct_fmue_16class":
        return f"{preds.get('predicted_class')} ({result.get('confidence',0):.0%})"
    if tool_name == "uwf_multi_disease":
        return f"diseases={preds.get('predicted_diseases')}"
    if tool_name == "uwf_disease_7class":
        c = preds.get("predicted_class") or "?"
        p = preds.get("probability")
        return f"{c} ({p:.0%})" if isinstance(p, (int, float)) else f"{c}"
    if tool_name == "uwf_vessel_segmentation":
        return f"vessel_ratio={preds.get('vessel_ratio',0):.2%}"
    if tool_name in ("cfp_efiqa", "cfp_quality_robust", "oct_quality"):
        q = preds.get("quality")
        src = preds.get("verdict_source")
        return f"quality={q}" + (f" (via {src})" if src else "")
    if tool_name == "vision_impression":
        # Pull the model's own top-differential line out of the markdown.
        md_txt = result.get("impression_markdown") or ""
        for line in md_txt.splitlines():
            low = line.lower()
            if "top differential" in low or "differential" in low:
                cleaned = line.split(":", 1)[-1].strip(" *#`")
                if cleaned:
                    return f"differential: {cleaned[:110]}"
        return "visual read captured" if md_txt else "(no impression)"
    if tool_name == "cfp_dr_workup":
        cat = preds.get("pdr_category") or preds.get("category")
        conf = preds.get("pdr_confidence")
        confound = preds.get("pdr_confounded_by")
        s = f"{cat}"
        if isinstance(conf, (int, float)):
            s += f" ({conf:.0%})"
        if confound:
            s += " ⚠confound"
        return s
    if tool_name in ("cfp_clip_ensemble",):
        top3 = preds.get("fused_top3") or []
        agree = preds.get("agreement_level")
        if top3:
            t = top3[0]
            return (f"{t.get('label_en','?')} ({float(t.get('probability',0)):.0%})"
                    + (f" · agree={agree}" if agree else ""))
        return f"agreement={agree}"
    if tool_name == "cfp_dynamic_clip":
        top = preds.get("fused_top1") or {}
        if isinstance(top, dict) and top.get("label"):
            score = top.get("score")
            return (
                f"{top.get('label')} ({score:.0%})"
                if isinstance(score, (int, float)) else str(top.get("label"))
            )
        return f"{preds.get('candidate_count', 0)} dynamic candidates"
    if tool_name in ("cfp_paired5", "ffa_paired5"):
        dx = preds.get("primary_diagnosis")
        p = preds.get("primary_probability")
        return f"{dx} ({p:.0%})" if isinstance(p, (int, float)) else f"{dx}"
    if tool_name == "ffa_classification":
        dx = preds.get("primary_diagnosis")
        p = preds.get("primary_probability")
        pos = preds.get("positive_groups") or []
        s = f"{dx} ({p:.0%})" if isinstance(p, (int, float)) else f"{dx}"
        if len(pos) > 1:
            s += f" +{len(pos) - 1} more group(s)"
        return s
    if tool_name == "ffa_lesion_detection":
        n = preds.get("n_detections", 0)
        dets = preds.get("detections") or []
        groups: list[str] = []
        for d in dets[:4]:
            g = (d.get("merged_group") or d.get("raw_label")) if isinstance(d, dict) else None
            if g and g not in groups:
                groups.append(str(g))
        if n:
            return f"{n} lesion(s)" + (f": {', '.join(groups)}" if groups else "")
        return "no lesions detected"
    if tool_name in ("cfp_retizero", "cfp_flair"):
        return f"{preds.get('predicted_class','?')} ({result.get('confidence',0):.0%})"
    if tool_name == "cfp_glaucoma_workup":
        rg = preds.get("referable_glaucoma")
        vcdr = preds.get("morphology_vCDR")
        ovr = preds.get("morphology_override")
        s = f"referable_glaucoma={rg}"
        if isinstance(vcdr, (int, float)):
            s += f"; vCDR={vcdr:.2f}" + (" (override)" if ovr else "")
        return s
    if tool_name == "oct_fluid_segmentation":
        areas = preds.get("class_areas") or {}
        nonbg = {k: v for k, v in areas.items() if k.lower() != "background"}
        total = sum(v for v in nonbg.values() if isinstance(v, (int, float)))
        return f"fluid area={int(total)}px across {len(nonbg)} class(es)"
    if tool_name == "oct_layer_segmentation":
        layers = preds.get("layers") or preds.get("layer_names") or []
        return f"{len(layers)} layers segmented" if layers else "layers segmented"
    if tool_name == "verify_findings":
        return f"{result.get('recommendation','?')[:120]}"
    if tool_name == "set_current_image":
        return f"→ {Path(result.get('image_path','')).name} ({result.get('detected_modality','?')})"
    if tool_name == "detect_modality":
        return result.get("modality", "?")
    if tool_name == "cfp_retsam_segmentation":
        # Use the compact LLM headline as the trace preview when available —
        # the user (and the LLM in re-reads) gets the bottom-line metrics
        # instead of just "masks=22 quant=[...]".
        head = result.get("llm_headline") if isinstance(result, dict) else None
        if isinstance(head, dict):
            quant_status = head.get("quantification_status", {}) or {}
            if quant_status.get("status") == "unavailable":
                raw = head.get("raw_mask_evidence", {}) or {}
                details = ["formal quantification unavailable"]
                atrophy = raw.get("chorioretinal_atrophy_fundus_ratio")
                tessellation = raw.get("tessellation_fundus_ratio")
                if isinstance(atrophy, (int, float)):
                    details.append(f"atrophy mask={atrophy:.0%}")
                if isinstance(tessellation, (int, float)):
                    details.append(f"tessellation mask={tessellation:.0%}")
                return "; ".join(details)
            dr = head.get("diabetic_retinopathy_signs", {}) or {}
            od = head.get("optic_disc", {}) or {}
            amd = head.get("amd_signs", {}) or {}
            return (
                f"vCDR={od.get('vCDR')} ({od.get('glaucoma_morphology_flag','?')}); "
                f"DR hem={dr.get('hemorrhage_count',0)}/{dr.get('hemorrhage_area_px',0)}px; "
                f"exudate={dr.get('exudate_count',0)}; cws={dr.get('cotton_wool_count',0)}; "
                f"drusen={amd.get('drusen_count',0)}"
            )
        q = preds.get("quantitative", {})
        return f"masks={preds.get('n_masks',0)} quant={list(q.keys())}"
    return str(preds)[:140]
