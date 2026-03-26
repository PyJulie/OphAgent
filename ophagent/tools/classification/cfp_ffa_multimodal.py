"""Multimodal CFP + FFA Classification (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPFFATool(FastAPIToolMixin, BaseTool):
    """
    Joint classification using both CFP and FFA images.
    Input:  {"cfp_path": str, "ffa_path": str}
    Output: {"labels": list[str], "probabilities": dict[str, float]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._dual_image_payload(inputs["cfp_path"], inputs["ffa_path"]),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import cfp_ffa_multimodal_prediction
        return cfp_ffa_multimodal_prediction(
            inputs["cfp_path"],
            inputs["ffa_path"],
            error=error,
        )
