"""CFP Image Quality Reasoning Model (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
import torch
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPQualityTool(FastAPIToolMixin, BaseTool):
    """
    Assesses CFP image quality and provides a reasoning explanation.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str}
    Output: {"quality_score": float, "quality_label": str, "reasoning": str}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        payload = {
            "image_b64": image_to_base64(inputs["image_path"]),
        }
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=payload,
        )
