"""FFA Multi-Lesion Detection (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class FFALesionTool(FastAPIToolMixin, BaseTool):
    """
    Detects multiple lesion types in fluorescein angiography (FFA) images.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str, "confidence_threshold": float (default 0.5)}
    Output: {"detections": list[{"label": str, "bbox": list, "confidence": float}],
             "lesion_types": list[str]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(
                inputs["image_path"],
                params={"confidence_threshold": inputs.get("confidence_threshold", 0.5)},
            ),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import ffa_lesion_detection
        return ffa_lesion_detection(
            inputs["image_path"],
            confidence_threshold=inputs.get("confidence_threshold", 0.5),
            error=error,
        )
