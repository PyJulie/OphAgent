"""
Composable Tool-Based VQA (Section 2.4.2).

Innovation: When no dedicated segmentation model exists for a particular
structure needed for VQA, we compose existing tools in a multi-step pipeline:
  1. Localise the structure of interest (disc/fovea detector or bounding box).
  2. Crop the ROI.
  3. Pass the cropped region to a VQA model with a targeted question.

This allows the agent to answer VQA questions about specific retinal regions
even without a purpose-built segmentation model for that region.

Example flow for "What is the cup-to-disc ratio?":
  -> disc_fovea (localise disc) -> roi_cropping (crop disc) -> fundus_expert VQA
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ast
import operator as _op

from ophagent.utils.logger import get_logger

logger = get_logger("strategies.vqa_composable")


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (replaces eval() for numeric template resolution)
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add:  _op.add,
    ast.Sub:  _op.sub,
    ast.Mult: _op.mul,
    ast.Div:  _op.truediv,
    ast.USub: _op.neg,
}


def _safe_eval_arithmetic(expr: str):
    """Evaluate a purely arithmetic expression without using eval().

    Accepts integer/float literals and the operators +, -, *, /.
    Returns the numeric result, or the original string if parsing fails.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return expr

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    try:
        return _eval(tree.body)
    except Exception:
        return expr


@dataclass
class VQACompositionStep:
    tool_id: str
    inputs: Dict[str, Any]
    description: str


class ComposableVQA:
    """
    Composes multiple tools to answer VQA questions about specific regions.

    The composition rules are defined per question-type.  New compositions
    can be registered at runtime via `register_composition`.

    Usage::

        vqa = ComposableVQA(scheduler=scheduler)
        answer = vqa.answer(
            image_path="fundus.jpg",
            question="Describe the optic disc appearance.",
        )
    """

    def __init__(self, scheduler=None):
        if scheduler is None:
            from ophagent.tools.scheduler import ToolScheduler
            scheduler = ToolScheduler()
        self.scheduler = scheduler
        self._compositions: Dict[str, List[Dict[str, Any]]] = {}
        self._register_default_compositions()

    # ------------------------------------------------------------------
    # Default composition rules
    # ------------------------------------------------------------------

    def _register_default_compositions(self) -> None:
        """Register built-in composition pipelines."""

        # --- Optic disc VQA ---
        self.register_composition(
            keywords=["optic disc", "cup disc", "cup-to-disc", "rim", "RNFL"],
            steps=[
                {
                    "tool_id": "disc_fovea",
                    "input_template": {"image_path": "{image_path}"},
                    "output_ref": "disc_coords",
                    "description": "Localise optic disc",
                },
                {
                    "tool_id": "roi_cropping",
                    "input_template": {
                        "image_path": "{image_path}",
                        "x": "{disc_coords.disc_center[0] - disc_coords.disc_radius * 1.5}",
                        "y": "{disc_coords.disc_center[1] - disc_coords.disc_radius * 1.5}",
                        "width": "{disc_coords.disc_radius * 3}",
                        "height": "{disc_coords.disc_radius * 3}",
                    },
                    "output_ref": "disc_crop",
                    "description": "Crop disc region",
                },
                {
                    "tool_id": "fundus_expert",
                    "input_template": {
                        "image_path": "{disc_crop.cropped_b64}",
                        "question": "{question}",
                    },
                    "output_ref": "vqa_answer",
                    "description": "Answer VQA on disc region",
                },
            ],
        )

        # --- Macula / fovea VQA ---
        self.register_composition(
            keywords=["macula", "fovea", "macular", "central vision", "drusen"],
            steps=[
                {
                    "tool_id": "disc_fovea",
                    "input_template": {"image_path": "{image_path}"},
                    "output_ref": "fovea_coords",
                    "description": "Localise fovea",
                },
                {
                    "tool_id": "roi_cropping",
                    "input_template": {
                        "image_path": "{image_path}",
                        "x": "{fovea_coords.fovea_center[0] - 150}",
                        "y": "{fovea_coords.fovea_center[1] - 150}",
                        "width": "300",
                        "height": "300",
                    },
                    "output_ref": "macula_crop",
                    "description": "Crop macular region",
                },
                {
                    "tool_id": "vision_unite",
                    "input_template": {
                        "image_path": "{macula_crop.cropped_b64}",
                        "question": "{question}",
                    },
                    "output_ref": "vqa_answer",
                    "description": "Answer VQA on macular region",
                },
            ],
        )

        # --- General/full-image VQA (fallback) ---
        self.register_composition(
            keywords=["__default__"],
            steps=[
                {
                    "tool_id": "fundus_expert",
                    "input_template": {
                        "image_path": "{image_path}",
                        "question": "{question}",
                    },
                    "output_ref": "vqa_answer",
                    "description": "Full-image VQA",
                }
            ],
        )

    def register_composition(self, keywords: List[str], steps: List[Dict[str, Any]]) -> None:
        for kw in keywords:
            self._compositions[kw.lower()] = steps

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _select_composition(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        q_lower = question.lower()
        for keyword, steps in self._compositions.items():
            if keyword == "__default__":
                continue
            if keyword in q_lower:
                logger.debug(f"Matched composition keyword: '{keyword}'")
                return keyword, steps
        return "__default__", self._compositions["__default__"]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def answer(
        self,
        image_path: str,
        question: str,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the appropriate composition pipeline and return the VQA answer.

        Returns:
            {"answer": str, "composition": str, "intermediate": dict}
        """
        keyword, steps = self._select_composition(question)
        logger.info(f"ComposableVQA: using composition '{keyword}' for question: {question[:60]}")

        context: Dict[str, Any] = {
            "image_path": image_path,
            "question": question,
            **(extra_inputs or {}),
        }
        intermediate: Dict[str, Any] = {}

        for step_def in steps:
            tool_id = step_def["tool_id"]
            output_ref = step_def["output_ref"]
            input_template = step_def["input_template"]

            # Resolve template variables from context
            resolved_inputs = self._resolve_template(input_template, context)

            try:
                result = self.scheduler.run(tool_id, resolved_inputs)
                context[output_ref] = result
                intermediate[output_ref] = result
                logger.debug(f"Step {tool_id} -> {output_ref}: OK")
            except Exception as e:
                logger.error(f"ComposableVQA step '{tool_id}' failed: {e}")
                return {
                    "answer": f"Unable to complete analysis at step '{tool_id}': {e}",
                    "composition": keyword,
                    "intermediate": intermediate,
                    "error": str(e),
                }

        # Extract final answer from last step's output
        final_output = context.get("vqa_answer", {})
        answer = (
            final_output.get("answer", str(final_output))
            if isinstance(final_output, dict)
            else str(final_output)
        )

        return {
            "answer": answer,
            "composition": keyword,
            "intermediate": intermediate,
        }

    # ------------------------------------------------------------------
    # Template resolver
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_template(template: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Replace {variable} tokens in template values using context dict."""
        import re

        def resolve_value(val: str) -> Any:
            if not isinstance(val, str) or "{" not in val:
                return val
            # Simple expression evaluation for arithmetic in templates
            try:
                # Replace {var} with context values
                def replacer(m):
                    expr = m.group(1)
                    parts = expr.split(".")
                    obj = context
                    for part in parts:
                        # Handle array indexing: key[n]
                        if "[" in part:
                            k, idx = part.rstrip("]").split("[")
                            obj = obj[k][int(idx)]
                        elif isinstance(obj, dict):
                            obj = obj[part]
                        else:
                            obj = getattr(obj, part)
                    return str(obj)

                resolved = re.sub(r"\{([^}]+)\}", replacer, val)
                # Try numeric evaluation if it looks like arithmetic
                try:
                    return _safe_eval_arithmetic(resolved)
                except Exception:
                    return resolved
            except Exception as e:
                logger.warning(f"Template resolve failed for '{val}': {e}")
                return val

        return {k: resolve_value(v) for k, v in template.items()}
