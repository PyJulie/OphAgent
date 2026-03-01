"""PDR Activity Grading (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class CFPPDRTool(FastAPIToolMixin, BaseTool):
    """
    Grades proliferative diabetic retinopathy (PDR) activity level.
    Input:  {"image_path": str}
    Output: {"grade": str, "grade_index": int, "probability": float,
             "findings": list[str]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={"image_b64": image_to_base64(inputs["image_path"])},
        )
