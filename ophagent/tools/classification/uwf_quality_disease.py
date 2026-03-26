"""UWF Quality + Disease Classification (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class UWFQualityDiseaseTool(FastAPIToolMixin, BaseTool):
    """
    Ultra-widefield image quality assessment and disease classification.
    Input:  {"image_path": str}
    Output: {"quality_score": float, "quality_label": str, "disease_labels": list, "probabilities": dict}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(inputs["image_path"]),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import uwf_quality_disease_prediction
        return uwf_quality_disease_prediction(inputs["image_path"], error=error)
