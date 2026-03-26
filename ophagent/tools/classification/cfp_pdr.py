"""PDR Activity Grading (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPPDRTool(FastAPIToolMixin, BaseTool):
    """
    Grades proliferative diabetic retinopathy (PDR) activity level.
    Input:  {"image_path": str}
    Output: {"grade": str, "grade_index": int, "probability": float,
             "findings": list[str]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(inputs["image_path"]),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import pdr_prediction
        return pdr_prediction(inputs["image_path"], error=error)
