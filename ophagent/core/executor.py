"""
OphAgent Executor.

Translates a validated ExecutionPlan into actual tool calls and collects results.
Two execution modes:
  - Direct: calls the tool's Python API directly (inline/fastapi tools)
  - Subprocess: spawns a Conda subprocess (on-demand tools)

The Executor maintains a result store keyed by step_id so downstream steps
can reference the outputs of completed steps.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ophagent.core.planner import ExecutionPlan, PlanStep
from ophagent.utils.logger import get_logger

logger = get_logger("core.executor")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class StepResult:
    def __init__(
        self,
        step_id: int,
        tool_name: str,
        output: Any = None,
        error: Optional[str] = None,
        duration_s: float = 0.0,
    ):
        self.step_id = step_id
        self.tool_name = tool_name
        self.output = output
        self.error = error
        self.duration_s = duration_s
        self.success = error is None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "error": self.error,
            "duration_s": self.duration_s,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Executes an ExecutionPlan step by step in topological order.

    For each step:
      1. Resolves input references (e.g. "${step_1.output.mask}") from previous results.
      2. Routes the call through the ToolScheduler.
      3. Stores the result.
      4. On error, marks step as failed but continues (unless a dependency failed).
    """

    def __init__(self, scheduler=None):
        if scheduler is None:
            from ophagent.tools.scheduler import ToolScheduler
            scheduler = ToolScheduler()
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, plan: ExecutionPlan) -> Dict[int, StepResult]:
        """
        Execute all steps in the plan and return a dict of step_id -> StepResult.
        """
        results: Dict[int, StepResult] = {}
        ordered_steps = plan.topological_order()

        for step in ordered_steps:
            if step.tool_name == "synthesise":
                # Synthesis is handled by OphAgent, not the executor
                continue

            # Check if dependencies succeeded
            failed_deps = [
                dep for dep in step.depends_on
                if dep in results and not results[dep].success
            ]
            if failed_deps:
                err = f"Dependency steps {failed_deps} failed; skipping step {step.step_id}."
                logger.warning(err)
                results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    error=err,
                )
                continue

            # Resolve dynamic input references
            resolved_inputs = self._resolve_inputs(step.inputs, results)

            # Execute
            logger.info(f"Executing step {step.step_id}: {step.tool_name}")
            t0 = time.monotonic()
            try:
                output = self.scheduler.run(step.tool_name, resolved_inputs)
                result = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    output=output,
                    duration_s=time.monotonic() - t0,
                )
                logger.info(
                    f"Step {step.step_id} ({step.tool_name}) completed "
                    f"in {result.duration_s:.2f}s"
                )
            except Exception as e:
                result = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    error=str(e),
                    duration_s=time.monotonic() - t0,
                )
                logger.error(f"Step {step.step_id} ({step.tool_name}) failed: {e}")

            results[step.step_id] = result

        return results

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    def _resolve_inputs(
        self,
        inputs: Dict[str, Any],
        results: Dict[int, StepResult],
    ) -> Dict[str, Any]:
        """
        Replace template references like "${step_1.output.mask}" with actual values.
        """
        resolved = {}
        for key, value in inputs.items():
            resolved[key] = self._resolve_value(value, results)
        return resolved

    def _resolve_value(self, value: Any, results: Dict[int, StepResult]) -> Any:
        if not isinstance(value, str) or "${" not in value:
            return value

        import re
        def replacer(m):
            ref = m.group(1)   # e.g. "step_1.output.mask"
            parts = ref.split(".")
            try:
                step_id = int(parts[0].replace("step_", ""))
                obj = results[step_id].output
                for attr in parts[2:]:   # skip "output"
                    if isinstance(obj, dict):
                        obj = obj[attr]
                    else:
                        if attr.startswith("_"):
                            raise AttributeError(
                                f"Access to private attribute '{attr}' is not permitted."
                            )
                        obj = getattr(obj, attr)
                return str(obj)
            except Exception as e:
                logger.warning(f"Could not resolve reference ${{{ref}}}: {e}")
                return m.group(0)

        return re.sub(r"\$\{([^}]+)\}", replacer, value)

    # ------------------------------------------------------------------
    # Convenience: format results for Verifier
    # ------------------------------------------------------------------

    @staticmethod
    def format_results(results: Dict[int, StepResult]) -> str:
        lines = []
        for step_id, res in sorted(results.items()):
            lines.append(
                f"Step {step_id} ({res.tool_name}): "
                + (json.dumps(res.output, default=str) if res.success else f"ERROR: {res.error}")
            )
        return "\n".join(lines)
