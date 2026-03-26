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
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload=self._single_image_payload(inputs["image_path"]),
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import disc_fovea_localisation
        return disc_fovea_localisation(inputs["image_path"], error=error)
