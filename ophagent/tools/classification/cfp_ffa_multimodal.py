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
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={
                "cfp_b64": image_to_base64(inputs["cfp_path"]),
                "ffa_b64": image_to_base64(inputs["ffa_path"]),
            },
        )
