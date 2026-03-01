"""ROI Cropping auxiliary tool."""
from __future__ import annotations
import base64
import io
from typing import Any, Dict
from PIL import Image
from ophagent.tools.base import BaseTool, ToolMetadata


class ROICroppingTool(BaseTool):
    """
    Crops a region of interest from an image.
    Input:  {"image_path": str,
             "x": int, "y": int, "width": int, "height": int,
             "output_path": str (optional)}
    Output: {"cropped_b64": str, "output_path": str}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from ophagent.utils.image_utils import crop_region, load_image_pil
        img = load_image_pil(inputs["image_path"])
        try:
            x = int(inputs.get("x", 0))
            y = int(inputs.get("y", 0))
            w = int(inputs.get("width", img.width))
            h = int(inputs.get("height", img.height))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid crop parameters: {exc}") from exc
        # Clamp coordinates to valid image bounds to prevent out-of-range crops
        x = max(0, min(x, img.width - 1))
        y = max(0, min(y, img.height - 1))
        w = max(1, min(w, img.width - x))
        h = max(1, min(h, img.height - y))
        cropped = crop_region(img, x, y, w, h)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        output_path = inputs.get("output_path", "")
        if output_path:
            cropped.save(output_path)
        return {"cropped_b64": b64, "output_path": output_path}
