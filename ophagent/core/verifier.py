"""
OphAgent Verifier.

Validates Executor outputs for medical consistency, detects conflicts between
tools, and attempts RAG-based resolution using the Knowledge Base.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ophagent.core.executor import Executor, StepResult
from ophagent.llm.backbone import LLMBackbone
from ophagent.llm.prompts import PromptLibrary
from ophagent.utils.logger import get_logger

logger = get_logger("core.verifier")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    valid: bool
    confidence: float
    conflicts: List[str] = field(default_factory=list)
    resolution: str = ""
    verified_result: Dict[str, Any] = field(default_factory=dict)
    needs_human_review: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(
            valid=bool(d.get("valid", False)),
            confidence=float(d.get("confidence", 0.0)),
            conflicts=list(d.get("conflicts", [])),
            resolution=str(d.get("resolution", "")),
            verified_result=dict(d.get("verified_result", {})),
            needs_human_review=bool(d.get("needs_human_review", False)),
        )

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "conflicts": self.conflicts,
            "resolution": self.resolution,
            "verified_result": self.verified_result,
            "needs_human_review": self.needs_human_review,
        }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """
    Verifies the medical consistency of Executor results.

    Workflow:
      1. Collect all successful step results.
      2. Call the LLM with the Verifier system prompt + RAG context.
      3. Parse the JSON verdict.
      4. If conflicts found, attempt knowledge-base-guided resolution.
      5. If confidence is below threshold, flag for human review.
    """

    CONFIDENCE_THRESHOLD = 0.6
    MAX_RESOLUTION_ATTEMPTS = 2

    def __init__(
        self,
        llm: Optional[LLMBackbone] = None,
        knowledge_base=None,
    ):
        self.llm = llm or LLMBackbone()
        self._kb = knowledge_base   # injected lazily

    @property
    def kb(self):
        if self._kb is None:
            from ophagent.knowledge.knowledge_base import KnowledgeBase
            self._kb = KnowledgeBase()
        return self._kb

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def verify(
        self,
        results: Dict[int, StepResult],
        query: str,
        attempt: int = 0,
    ) -> Verdict:
        """
        Verify a set of step results and return a Verdict.

        Args:
            results: Dict of step_id -> StepResult from Executor.
            query: Original user query (used for KB retrieval).
            attempt: Current resolution attempt count (max = MAX_RESOLUTION_ATTEMPTS).
        """
        tool_outputs_str = Executor.format_results(results)

        # Retrieve relevant knowledge
        kb_context = self.kb.retrieve(query, top_k=5)

        system = PromptLibrary.VERIFIER_SYSTEM
        user_msg = PromptLibrary.verifier_user(
            tool_outputs=tool_outputs_str,
            kb_context=kb_context,
        )

        logger.info(f"Verifying {len(results)} step results (attempt {attempt + 1}).")
        raw = self.llm.chat_json(
            [{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.0,
        )

        verdict = Verdict.from_dict(raw)

        # Flag low-confidence cases
        if verdict.confidence < self.CONFIDENCE_THRESHOLD:
            verdict.needs_human_review = True
            logger.warning(
                f"Low confidence ({verdict.confidence:.2f}); flagging for human review."
            )

        # Attempt resolution if conflicts remain and attempts left
        if verdict.conflicts and attempt < self.MAX_RESOLUTION_ATTEMPTS:
            logger.info(
                f"Conflicts detected: {verdict.conflicts}. "
                f"Attempting resolution ({attempt + 1}/{self.MAX_RESOLUTION_ATTEMPTS})."
            )
            verdict = self._attempt_resolution(verdict, results, query, attempt)

        return verdict

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _attempt_resolution(
        self,
        verdict: Verdict,
        results: Dict[int, StepResult],
        query: str,
        attempt: int,
    ) -> Verdict:
        """
        Use a deeper KB search focused on the conflict topics to resolve
        discrepancies.
        """
        conflict_query = " ".join(verdict.conflicts) + " " + query
        expanded_context = self.kb.retrieve(conflict_query, top_k=10)

        # Re-run verification with richer context
        enriched_user = PromptLibrary.verifier_user(
            tool_outputs=Executor.format_results(results),
            kb_context=f"Expanded context for conflict resolution:\n{expanded_context}",
        )
        raw = self.llm.chat_json(
            [{"role": "user", "content": enriched_user}],
            system=PromptLibrary.VERIFIER_SYSTEM,
            temperature=0.0,
        )
        new_verdict = Verdict.from_dict(raw)
        if new_verdict.confidence < self.CONFIDENCE_THRESHOLD:
            new_verdict.needs_human_review = True
        return new_verdict

    # ------------------------------------------------------------------
    # Checks for specific medical inconsistencies
    # ------------------------------------------------------------------

    @staticmethod
    def check_quality_gate(quality_result: dict) -> bool:
        """
        Return True if image quality is acceptable for downstream analysis.
        Used by the Planner to gate further tool calls.
        """
        if quality_result is None:
            return True   # no quality check done, proceed
        score = quality_result.get("quality_score", 1.0)
        label = quality_result.get("quality_label", "acceptable")
        if isinstance(score, (int, float)) and score < 0.4:
            logger.warning(f"Image quality too low ({score:.2f}); downstream results unreliable.")
            return False
        if isinstance(label, str) and "ungradable" in label.lower():
            return False
        return True
