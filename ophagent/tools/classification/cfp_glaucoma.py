"""Referable Glaucoma + Structural Features Detection (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPGlaucomaTool(FastAPIToolMixin, BaseTool):
    """
    Detects referable glaucoma and extracts structural optic disc features.
    Input:  {"image_path": str}
    Output: {"referable_glaucoma": bool, "probability": float,
             "cup_disc_ratio": float, "structural_features": dict}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={"image_b64": image_to_base64(inputs["image_path"])},
        )
