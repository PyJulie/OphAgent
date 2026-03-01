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
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={
                "image_b64": image_to_base64(inputs["image_path"]),
                "confidence_threshold": inputs.get("confidence_threshold", 0.5),
            },
        )
