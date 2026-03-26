"""RetSAM - CFP retinal segmentation via SAM adaptation (reused model)."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch
from ophagent.tools.base import BaseTool, ToolMetadata


class RetSAMTool(BaseTool):
    """
    Segments retinal structures in CFP images using a SAM-adapted model.
    Runs inline (model loaded in main process).
    Input:  {"image_path": str, "prompt_type": str ("auto"|"point"|"box"),
             "prompt_coords": list (optional)}
    Output: {"mask_path": str, "mask_array_b64": str, "area_fraction": float}
    """
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        self._model = None
        self._predictor = None

    def load_model(self) -> None:
        from segment_anything import sam_model_registry, SamPredictor
        import os
        weight = str(self.metadata.model_weight)
        if not os.path.exists(weight):
            raise FileNotFoundError(f"RetSAM weights not found: {weight}")
        sam = sam_model_registry["vit_h"](checkpoint=weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device)
        self._predictor = SamPredictor(sam)
        self._model_loaded = True

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import io
        import numpy as np
        from PIL import Image
        from ophagent.utils.image_utils import load_image_pil
        from config.settings import get_settings

        img = load_image_pil(inputs["image_path"])
        img_np = np.array(img)
        self._predictor.set_image(img_np)

        prompt_type = inputs.get("prompt_type", "auto")
        if prompt_type == "box":
            box = np.array(inputs.get("prompt_coords", [0, 0, img.width, img.height]))
            masks, _, _ = self._predictor.predict(box=box, multimask_output=False)
        else:
            # Auto: use image centre as point prompt
            cx, cy = img.width // 2, img.height // 2
            masks, _, _ = self._predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )

        mask = masks[0].astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "mask_array_b64": mask_b64,
            "area_fraction": float(mask.sum()) / (mask.size * 255),
        }

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import segmentation_prediction
        return segmentation_prediction(inputs["image_path"], error=error)
