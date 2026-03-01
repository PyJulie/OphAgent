"""FundusExpert VQA Tool (reused model)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class FundusExpertTool(FastAPIToolMixin, BaseTool):
    """
    Visual question answering on fundus images using the FundusExpert model.
    Runs as a FastAPI microservice.
    Input:  {"image_path": str, "question": str}
    Output: {"answer": str, "confidence": float}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={
                "image_b64": image_to_base64(inputs["image_path"]),
                "question": inputs.get("question", "What abnormalities are visible?"),
            },
        )
