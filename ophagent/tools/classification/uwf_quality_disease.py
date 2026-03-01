"""UWF Quality + Disease Classification (newly developed)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, FastAPIToolMixin, ToolMetadata


class UWFQualityDiseaseTool(FastAPIToolMixin, BaseTool):
    """
    Ultra-widefield image quality assessment and disease classification.
    Input:  {"image_path": str}
    Output: {"quality_score": float, "quality_label": str, "disease_labels": list, "probabilities": dict}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import image_to_base64
        return self._post(
            port=self.metadata.fastapi_port,
            endpoint="/run",
            payload={"image_b64": image_to_base64(inputs["image_path"])},
        )
