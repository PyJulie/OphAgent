"""VisionUnite VQA Tool (reused model)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class VisionUniteTool(FastAPIToolMixin, BaseTool):
    """
    Unified ophthalmic VQA using VisionUnite.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str, "question": str}
    Output: {"answer": str}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(
                inputs["image_path"],
                params={"question": inputs.get("question", "Describe the findings in this image.")},
            ),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import vqa_response
        return vqa_response(
            inputs["image_path"],
            inputs.get("question", "Describe the findings in this image."),
            error=error,
        )
