"""Multiple Retinal Disease Classification from CFP (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPDiseaseTool(FastAPIToolMixin, BaseTool):
    """
    Multi-label retinal disease classification from colour fundus photographs.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str}
    Output: {"labels": list[str], "probabilities": dict[str, float]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(inputs["image_path"]),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import cfp_disease_prediction
        return cfp_disease_prediction(inputs["image_path"], error=error)
