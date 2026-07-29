"""Task protocols for reproducible OphAgent evaluations.

The protocol design keeps tool scheduling agent-native: the prompt does not
prescribe an ordered tool pipeline. Instead, each task declares an evidence
contract and a strict output schema. The planner can choose tools autonomously,
but the final answer must be auditable against the declared evidence fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent


EFFORT_ORDER = ("low", "medium", "high", "max", "ultra")


@dataclass(frozen=True)
class EffortPolicy:
    """Stable semantics for effort tiers in protocol-first experiments."""

    name: str
    label: str
    intent: str
    planning_budget: str
    verification_budget: str
    suitable_for_primary_eval: bool
    cautions: tuple[str, ...] = ()

    def render(self) -> str:
        cautions = "\n".join(f"- {c}" for c in self.cautions) or "- none"
        return dedent(
            f"""
            Effort policy: {self.name} ({self.label})
            Intent: {self.intent}
            Planning budget: {self.planning_budget}
            Verification budget: {self.verification_budget}
            Cautions:
            {cautions}
            """
        ).strip()


EFFORT_POLICIES: dict[str, EffortPolicy] = {
    "low": EffortPolicy(
        name="low",
        label="quick estimate",
        intent="Minimise cost. Produce a coarse answer when a fast estimate is acceptable.",
        planning_budget="One batched tool round.",
        verification_budget="No verifier escalation expected.",
        suitable_for_primary_eval=False,
        cautions=(
            "Not a primary setting for fine-grained grading.",
            "If evidence is thin, mark evidence_level as limited rather than overclaiming.",
        ),
    ),
    "medium": EffortPolicy(
        name="medium",
        label="standard",
        intent="Default clinical workflow: gather core independent evidence, then targeted workup.",
        planning_budget="Two batched tool rounds: initial evidence, then indicated workup.",
        verification_budget="Verifier may request bounded follow-up.",
        suitable_for_primary_eval=True,
    ),
    "high": EffortPolicy(
        name="high",
        label="thorough",
        intent="Broader cross-checking for hard or high-stakes cases.",
        planning_budget="Three bounded evidence rounds in evaluation harnesses.",
        verification_budget="Stronger verifier follow-up than medium.",
        suitable_for_primary_eval=True,
        cautions=("Watch for over-reading weak or conflicting secondary tools.",),
    ),
    "max": EffortPolicy(
        name="max",
        label="debate review",
        intent="Use a debate-style verifier for contested cases.",
        planning_budget="Three bounded evidence rounds in evaluation harnesses.",
        verification_budget="Debate review plus bounded verifier follow-up.",
        suitable_for_primary_eval=False,
        cautions=("Use as sensitivity analysis; do not treat as an exhaustive tool run.",),
    ),
    "ultra": EffortPolicy(
        name="ultra",
        label="exhaustive",
        intent="Upper-bound run that should gather every relevant compatible signal.",
        planning_budget="Exhaustive compatible-tool coverage.",
        verification_budget="Debate review plus maximum bounded follow-up.",
        suitable_for_primary_eval=False,
        cautions=("High cost and possible conflict amplification; best for ablation.",),
    ),
}


@dataclass(frozen=True)
class EvidenceRequirement:
    """One auditable evidence field required by a task protocol."""

    key: str
    description: str
    minimum_effort: str = "medium"
    low_effort_note: str = ""

    def render(self, effort: str) -> str:
        if EFFORT_ORDER.index(effort) < EFFORT_ORDER.index(self.minimum_effort):
            note = self.low_effort_note or "may be unavailable at this effort"
            return f"- {self.key}: {self.description} ({note})."
        return f"- {self.key}: {self.description}."


@dataclass(frozen=True)
class TaskProtocol:
    """A task-level contract that renders a user prompt."""

    task_id: str
    title: str
    modality: str
    minimum_effort: str
    grading_scale: str
    rubric: tuple[str, ...]
    evidence: tuple[EvidenceRequirement, ...]
    output_schema: str
    autonomy_clause: str = (
        "Use your available tools autonomously. Do not follow a fixed tool order "
        "unless the evidence itself makes that order necessary."
    )
    no_cot_clause: str = (
        "Do not reveal internal chain-of-thought. Provide concise evidence and "
        "a brief rationale only."
    )
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def build_user_prompt(
        self,
        effort: str = "medium",
        user_question: str | None = None,
    ) -> str:
        policy = get_effort_policy(effort)
        rubric = "\n".join(f"- {line}" for line in self.rubric)
        evidence = "\n".join(req.render(effort) for req in self.evidence)
        caveats = "\n".join(f"- {c}" for c in self.caveats) or "- none"
        question = user_question.strip() if user_question else (
            f"Perform {self.title} for the attached {self.modality} image."
        )

        return dedent(
            f"""
            {question}

            Task protocol: {self.task_id}
            Modality scope: {self.modality}
            Grading scale:
            {self.grading_scale}

            Agent autonomy:
            {self.autonomy_clause}
            The task is not to prove you called a particular tool; the task is to
            produce a grade whose evidence fields are supportable from the tool
            results and the visible image.

            Effort semantics:
            {policy.render()}

            Rubric:
            {rubric}

            Evidence contract:
            Before finalising, fill every evidence field below. If the current
            effort/tool evidence cannot support a field, say so in that field and
            lower confidence rather than inventing evidence.
            {evidence}

            Caveats:
            {caveats}

            Output rules:
            {self.no_cot_clause}
            End with the marker and exactly one JSON object matching this schema:

            ===FINAL===
            {self.output_schema}
            """
        ).strip()


DR_SEVERITY_ICDR_SINGLE_IMAGE = TaskProtocol(
    task_id="dr_severity_icdr_single_image_v2",
    title="single-image diabetic retinopathy severity grading",
    modality="CFP",
    minimum_effort="medium",
    grading_scale=(
        "ICDR 5-point scale: 0 no apparent DR; 1 mild NPDR; "
        "2 moderate NPDR; 3 severe NPDR; 4 proliferative DR."
    ),
    rubric=(
        "Grade 0: no apparent DR in the visible field.",
        "Grade 1: microaneurysm-like red dots only; no hemorrhage burden, exudates, CWS, IRMA, or NV.",
        "Grade 2: more than microaneurysms, such as dot/blot hemorrhage, hard exudate, or cotton-wool spot, but no severe-NPDR or PDR signs.",
        "Grade 3: severe NPDR by visible or tool-supported 4-2-1-type evidence: heavy four-quadrant hemorrhage, area-weighted/confluent hemorrhage burden, venous beading in at least two quadrants, or prominent IRMA, without neovascularisation.",
        "Grade 4: PDR or treated PDR evidence: NVD/NVE, fibrovascular proliferation, pre-retinal/vitreous hemorrhage, tractional RD, or convincing panretinal laser scars indicating treated PDR.",
    ),
    evidence=(
        EvidenceRequirement(
            key="image_quality",
            description="Whether the image is gradable and what limits the field or quality.",
            minimum_effort="low",
        ),
        EvidenceRequirement(
            key="lesion_burden",
            description="Visible DR lesion burden: red lesions/hemorrhage, exudate, cotton-wool spot, and distribution if available.",
            minimum_effort="medium",
            low_effort_note="low effort may only provide a classifier-driven estimate",
        ),
        EvidenceRequirement(
            key="severe_npdr_signs",
            description="Evidence for or against severe-NPDR 4-2-1 signs: strict four-quadrant hemorrhage, area-weighted/confluent hemorrhage proxy, heavy lesion burden, venous beading, IRMA.",
            minimum_effort="medium",
        ),
        EvidenceRequirement(
            key="pdr_signs",
            description="Evidence for or against PDR or treated PDR: NV, pre-retinal/vitreous hemorrhage, fibrovascular tissue, PRP/laser scars.",
            minimum_effort="medium",
        ),
        EvidenceRequirement(
            key="confounders",
            description="Non-DR findings or artifacts that could mimic DR lesions, such as hypertensive retinopathy, RVO, glare, drusen, tessellation, or dust.",
            minimum_effort="medium",
        ),
    ),
    caveats=(
        "This is a single-image grade. If the dataset label is eye-level or patient-level, note that unseen views may contain worse disease.",
        "Do not upgrade from grade 1 to grade 2 solely from isolated tiny artifacts.",
        "Do not call grade 3 from isolated CWS or tiny artifacts alone. However, do not downgrade to grade 2 solely because strict component-count 4-2-1 is false: confluent hemorrhages can be counted as few components. If `cfp_dr_421_assessment` reports `strong_severe_npdr_proxy`, `rule_4_hemorrhage_area_weighted_proxy`, or `heavy_lesion_burden_proxy`, treat this as tool-supported grade-3 evidence unless PDR evidence upgrades to grade 4.",
    ),
    output_schema=dedent(
        """
        {
          "task_id": "dr_severity_icdr_single_image_v2",
          "gradable": true,
          "grade": 0,
          "confidence": 0.0,
          "evidence_level": "sufficient",
          "evidence": {
            "image_quality": "",
            "lesion_burden": "",
            "severe_npdr_signs": "",
            "pdr_signs": "",
            "confounders": ""
          },
          "evidence_gaps": [],
          "rationale": ""
        }
        """
    ).strip(),
)


PROTOCOLS: dict[str, TaskProtocol] = {
    DR_SEVERITY_ICDR_SINGLE_IMAGE.task_id: DR_SEVERITY_ICDR_SINGLE_IMAGE,
    "dr_severity": DR_SEVERITY_ICDR_SINGLE_IMAGE,
}


def get_effort_policy(effort: str) -> EffortPolicy:
    try:
        return EFFORT_POLICIES[effort]
    except KeyError as exc:
        raise ValueError(
            f"unknown effort {effort!r}; expected one of {sorted(EFFORT_POLICIES)}"
        ) from exc


def get_protocol(task_id: str) -> TaskProtocol:
    try:
        return PROTOCOLS[task_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown protocol {task_id!r}; expected one of {sorted(PROTOCOLS)}"
        ) from exc
