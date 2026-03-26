"""
OphAgent – Main Orchestrator.

Ties together Planner → Executor → Verifier → Memory into a complete
clinical AI agent loop.  This is the primary entry point for end users.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ophagent.core.executor import Executor
from ophagent.core.memory import MemoryManager
from ophagent.core.planner import ExecutionPlan, Planner
from ophagent.core.verifier import Verdict, Verifier
from ophagent.llm.backbone import LLMBackbone
from ophagent.llm.prompts import PromptLibrary
from ophagent.utils.logger import get_logger

logger = get_logger("core.agent")


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------

class AgentResponse:
    """Structured response returned by OphAgent.run()."""

    def __init__(
        self,
        query: str,
        report: str,
        verdict: Optional[Verdict] = None,
        plan: Optional[ExecutionPlan] = None,
        raw_results: Optional[Dict] = None,
        duration_s: float = 0.0,
        needs_human_review: bool = False,
    ):
        self.query = query
        self.report = report
        self.verdict = verdict
        self.plan = plan
        self.raw_results = raw_results or {}
        self.duration_s = duration_s
        self.needs_human_review = needs_human_review

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "report": self.report,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "needs_human_review": self.needs_human_review,
            "duration_s": self.duration_s,
        }

    def __str__(self) -> str:
        return self.report


# ---------------------------------------------------------------------------
# OphAgent
# ---------------------------------------------------------------------------

class OphAgent:
    """
    The main OphAgent orchestrator.

    Usage::

        agent = OphAgent()
        response = agent.run(
            query="Analyse this fundus image for diabetic retinopathy.",
            image_paths=["patient_cfp.jpg"],
        )
        print(response.report)
    """

    def __init__(
        self,
        llm: Optional[LLMBackbone] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        verifier: Optional[Verifier] = None,
        memory: Optional[MemoryManager] = None,
    ):
        self.llm = llm or LLMBackbone()
        self.planner = planner or Planner(llm=self.llm)
        self.executor = executor or Executor()
        self.verifier = verifier or Verifier(llm=self.llm)
        self.memory = memory or MemoryManager()
        logger.info("OphAgent initialised.")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        image_paths: Optional[List[Union[str, Path]]] = None,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process a clinical query through the full Planner → Executor →
        Verifier → Report pipeline.

        Args:
            query:       Natural-language clinical or research question.
            image_paths: Paths to uploaded images (any supported modality).
            session_id:  Optional session identifier for memory scoping.

        Returns:
            AgentResponse with the final clinical report and metadata.
        """
        t_start = time.monotonic()
        image_paths_str = [str(p) for p in (image_paths or [])]

        logger.info(f"New query: {query[:100]}...")

        # ---- Store user turn ----
        self.memory.add_turn("user", query)

        # ---- Retrieve long-term context ----
        past_context = self.memory.retrieve(query)

        # ---- Build session context string ----
        session_ctx = self._build_session_context(past_context)

        # ---- Plan ----
        plan = self.planner.plan(
            query=query,
            image_paths=image_paths_str,
            session_context=session_ctx,
        )

        # ---- Execute ----
        results = self.executor.execute(plan)

        # ---- Verify ----
        verdict = self.verifier.verify(results, query=query)

        # ---- Handle re-planning if needed ----
        if not verdict.valid and verdict.conflicts:
            logger.info("Verifier found conflicts; initiating re-plan.")
            revised_plan = self.planner.replan(
                plan, conflict_description=", ".join(verdict.conflicts)
            )
            revised_results = self.executor.execute(revised_plan)
            verdict = self.verifier.verify(revised_results, query=query, attempt=1)
            results = revised_results

        # ---- Image quality gate ----
        quality_result = self._extract_quality_result(results)
        if not Verifier.check_quality_gate(quality_result):
            report = (
                "Image quality assessment indicates the provided image(s) are "
                "ungradable. Please provide higher-quality images for reliable analysis."
            )
            response = AgentResponse(
                query=query,
                report=report,
                verdict=verdict,
                plan=plan,
                raw_results={k: v.to_dict() for k, v in results.items()},
                duration_s=time.monotonic() - t_start,
                needs_human_review=True,
            )
            self.memory.add_turn("assistant", report)
            return response

        degraded_execution = self._has_degraded_results(results)
        if degraded_execution:
            logger.warning("One or more tool steps ran in fallback/degraded mode.")
            verdict.needs_human_review = True

        # ---- Synthesise report ----
        report = self._synthesise_report(query, verdict, past_context)
        if degraded_execution:
            report = (
                "Note: one or more tool calls used degraded or fallback execution. "
                "Human review is recommended.\n\n"
                + report
            )

        # ---- Store assistant turn ----
        self.memory.add_turn("assistant", report)

        # ---- Consolidate session to long-term memory (async-like) ----
        tools_used = [s.tool_name for s in plan.steps]
        self.memory.consolidate_session(
            query=query,
            report=report,
            tools_used=tools_used,
            llm=self.llm,
        )

        duration = time.monotonic() - t_start
        logger.info(f"Query completed in {duration:.2f}s.")

        return AgentResponse(
            query=query,
            report=report,
            verdict=verdict,
            plan=plan,
            raw_results={k: v.to_dict() for k, v in results.items()},
            duration_s=duration,
            needs_human_review=verdict.needs_human_review,
        )

    # ------------------------------------------------------------------
    # Report synthesis
    # ------------------------------------------------------------------

    def _synthesise_report(
        self,
        query: str,
        verdict: Verdict,
        past_context: str,
    ) -> str:
        """Call the LLM to produce the final structured clinical report."""
        verified_str = json.dumps(verdict.verified_result, indent=2, default=str)
        user_msg = PromptLibrary.synthesis_user(
            query=query,
            verified_findings=verified_str,
            kb_context=past_context,
        )
        report = self.llm.chat(
            [{"role": "user", "content": user_msg}],
            system=PromptLibrary.SYNTHESIS_SYSTEM,
            temperature=0.2,
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_session_context(self, past_context: str) -> str:
        history = self.memory.get_history(last_n=6)
        hist_str = "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in history
        ) or "No prior turns."
        return f"Session history:\n{hist_str}\n\nRelevant past cases:\n{past_context}"

    @staticmethod
    def _extract_quality_result(results) -> Optional[dict]:
        """Find the quality assessment step result if it exists."""
        for step_id, res in results.items():
            if res.tool_name in ("cfp_quality", "uwf_quality_disease") and res.success:
                return res.output
        return None

    @staticmethod
    def _has_degraded_results(results) -> bool:
        for res in results.values():
            if not res.success:
                continue
            output = res.output
            if isinstance(output, dict) and (
                output.get("used_fallback")
                or output.get("degraded")
                or output.get("backend") == "heuristic-fallback"
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Interactive / streaming interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str, image_paths: Optional[List[str]] = None) -> str:
        """
        Simplified interactive interface.  Returns the report string directly.
        """
        response = self.run(query=user_message, image_paths=image_paths)
        return response.report

    def reset_session(self) -> None:
        """Clear short-term memory for a new patient session."""
        self.memory.clear_session()
        logger.info("Session reset.")
