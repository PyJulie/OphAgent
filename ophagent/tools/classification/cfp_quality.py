"""CFP Image Quality Reasoning Model (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPQualityTool(FastAPIToolMixin, BaseTool):
    """
    Assesses CFP image quality and provides a reasoning explanation.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str}
    Output: {"quality_score": float, "quality_label": str, "reasoning": str}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(inputs["image_path"]),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import quality_assessment
        return quality_assessment(inputs["image_path"], error=error)
