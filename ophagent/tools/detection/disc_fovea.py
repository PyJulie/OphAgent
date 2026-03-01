"""Optic Disc & Fovea Localisation (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class DiscFoveaTool(FastAPIToolMixin, BaseTool):
    """
    Localises the optic disc and fovea in CFP images.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str}
    Output: {"disc_center": [x, y], "disc_radius": float,
             "fovea_center": [x, y]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={"image_b64": image_to_base64(inputs["image_path"])},
        )
