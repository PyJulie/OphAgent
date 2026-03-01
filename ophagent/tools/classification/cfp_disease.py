"""Multiple Retinal Disease Classification from CFP (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPDiseaseTool(FastAPIToolMixin, BaseTool):
    """
    Multi-label retinal disease classification from colour fundus photographs.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str}
    Output: {"labels": list[str], "probabilities": dict[str, float]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={"image_b64": image_to_base64(inputs["image_path"])},
        )
