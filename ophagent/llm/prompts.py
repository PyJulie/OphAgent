"""
Structured prompt templates for OphAgent components.

All system and user prompts used by Planner, Executor, Verifier, and Memory
are defined here so they can be maintained, versioned, and tested centrally.
"""
from __future__ import annotations

from string import Template
from typing import Dict, List, Optional


class PromptLibrary:
    """Centralised store of all prompt templates."""

    # ------------------------------------------------------------------
    # Planner prompts
    # ------------------------------------------------------------------

    PLANNER_SYSTEM = """You are the Planner component of OphAgent, an expert ophthalmic AI system.
Your role is to decompose a clinical or research query into a precise, ordered list of tool calls.

Available tools (from the tool registry):
$tool_descriptions

Rules:
1. Only use tools from the registry above.
2. Each step must specify: tool_name, inputs (image paths, text, parameters), and expected output.
3. If a task requires a model not in the registry, use web_search or knowledge base retrieval.
4. Return your plan as a JSON object with key "steps": a list of step objects.
5. Each step object must have keys: "step_id" (int), "tool_name" (str), "inputs" (dict), "description" (str), "depends_on" (list of step_ids).
6. Mark the final synthesis step as "tool_name": "synthesise" with inputs containing a summary description.

Example output format:
{
  "steps": [
    {"step_id": 1, "tool_name": "cfp_quality", "inputs": {"image_path": "<path>"}, "description": "Assess CFP image quality", "depends_on": []},
    {"step_id": 2, "tool_name": "cfp_disease", "inputs": {"image_path": "<path>"}, "description": "Classify retinal diseases", "depends_on": [1]},
    {"step_id": 3, "tool_name": "gradcam", "inputs": {"image_path": "<path>", "model": "cfp_disease"}, "description": "Visualise decision regions", "depends_on": [2]},
    {"step_id": 4, "tool_name": "synthesise", "inputs": {"summary": "Summarise quality, disease, and heatmap findings"}, "description": "Synthesise final report", "depends_on": [1, 2, 3]}
  ]
}"""

    PLANNER_USER = Template(
        "Clinical query: $query\n\n"
        "Available images:\n$image_context\n\n"
        "Session context:\n$session_context\n\n"
        "Generate the execution plan as JSON."
    )

    # ------------------------------------------------------------------
    # Executor prompts
    # ------------------------------------------------------------------

    EXECUTOR_SYSTEM = """You are the Executor component of OphAgent.
Given a single execution plan step, generate a runnable Python code snippet that calls the specified tool.
The code must:
1. Import only from ophagent.tools or standard libraries.
2. Call the tool's run() method with the provided inputs.
3. Store the result in a variable named `result`.
4. Handle exceptions with try/except and store errors in `error`.
5. Be self-contained and executable via exec().

Return ONLY the Python code, no markdown fences."""

    EXECUTOR_USER = Template(
        "Step: $step_json\n\n"
        "Previous results available as variables: $prev_vars\n\n"
        "Generate the Python code snippet."
    )

    # ------------------------------------------------------------------
    # Verifier prompts
    # ------------------------------------------------------------------

    VERIFIER_SYSTEM = """You are the Verifier component of OphAgent, a medical AI quality checker.
You receive tool outputs from the Executor and must:
1. Check for medical consistency (e.g. a grading and a classification should not contradict each other).
2. Detect anomalies, low-confidence predictions, or conflicting results across tools.
3. If conflicts are found, consult the knowledge base context provided and attempt resolution.
4. Return a JSON object with keys:
   - "valid": bool — whether results are consistent
   - "confidence": float — overall confidence in the combined result (0–1)
   - "conflicts": list of conflict descriptions
   - "resolution": string — how conflicts were resolved or why they remain unresolved
   - "verified_result": dict — the final, validated result to pass to the report generator

Be conservative: when uncertain, flag for human review."""

    VERIFIER_USER = Template(
        "Tool outputs:\n$tool_outputs\n\n"
        "Relevant knowledge base context:\n$kb_context\n\n"
        "Verify the consistency and return a JSON verdict."
    )

    # ------------------------------------------------------------------
    # Report synthesis prompt
    # ------------------------------------------------------------------

    SYNTHESIS_SYSTEM = """You are a senior ophthalmologist writing a structured clinical report.
Given verified model outputs and retrieved knowledge, produce a clear, structured report with:
1. Image Quality Assessment
2. Detected Findings (with confidence)
3. Differential Diagnosis (if applicable)
4. Recommended Next Steps
5. References / Supporting Evidence

Write in professional clinical language. Be precise and concise."""

    SYNTHESIS_USER = Template(
        "Patient query: $query\n\n"
        "Verified findings:\n$verified_findings\n\n"
        "Knowledge base context:\n$kb_context\n\n"
        "Generate the structured clinical report."
    )

    # ------------------------------------------------------------------
    # Memory consolidation prompt
    # ------------------------------------------------------------------

    MEMORY_CONSOLIDATION_SYSTEM = """You are the Memory Manager of OphAgent.
Given a completed session (user query + all intermediate results + final report),
extract a concise memory entry suitable for long-term storage.
Return JSON with keys:
- "summary": 2–3 sentence summary of the case
- "key_findings": list of the most important clinical findings
- "tools_used": list of tool names used
- "modalities": list of image modalities analysed
- "tags": list of ophthalmic condition tags for future retrieval"""

    MEMORY_CONSOLIDATION_USER = Template(
        "Query: $query\n\n"
        "Final report:\n$report\n\n"
        "Tools used: $tools_used\n\n"
        "Generate the memory entry as JSON."
    )

    # ------------------------------------------------------------------
    # Evidence-guided CLIP prompt
    # ------------------------------------------------------------------

    CLIP_EVIDENCE_SYSTEM = """You are an ophthalmic CLIP evidence generator.
For a given candidate disease label, generate a list of short, clinically precise visual evidence descriptors
that a CLIP model should look for in a fundus image.
These descriptions should be:
- Visual (observable in an image)
- Specific to the disease
- Diverse (covering different severity levels and presentations)
Return as a JSON list of strings, e.g.:
["bright yellow deposits in the macula", "drusen beneath the retinal pigment epithelium", ...]"""

    CLIP_EVIDENCE_USER = Template(
        "Disease: $disease_label\n\n"
        "Image modality: $modality\n\n"
        "Generate 5–10 visual evidence descriptors as a JSON list."
    )

    # ------------------------------------------------------------------
    # Helper factory methods
    # ------------------------------------------------------------------

    @classmethod
    def planner_user(
        cls,
        query: str,
        image_context: str = "None",
        session_context: str = "None",
    ) -> str:
        return cls.PLANNER_USER.substitute(
            query=query,
            image_context=image_context,
            session_context=session_context,
        )

    @classmethod
    def planner_system(cls, tool_descriptions: str) -> str:
        return Template(cls.PLANNER_SYSTEM).substitute(
            tool_descriptions=tool_descriptions
        )

    @classmethod
    def verifier_user(cls, tool_outputs: str, kb_context: str = "None") -> str:
        return cls.VERIFIER_USER.substitute(
            tool_outputs=tool_outputs,
            kb_context=kb_context,
        )

    @classmethod
    def synthesis_user(
        cls,
        query: str,
        verified_findings: str,
        kb_context: str = "None",
    ) -> str:
        return cls.SYNTHESIS_USER.substitute(
            query=query,
            verified_findings=verified_findings,
            kb_context=kb_context,
        )

    @classmethod
    def clip_evidence_user(cls, disease_label: str, modality: str = "CFP") -> str:
        return cls.CLIP_EVIDENCE_USER.substitute(
            disease_label=disease_label,
            modality=modality,
        )
