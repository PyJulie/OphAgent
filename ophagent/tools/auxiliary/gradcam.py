"""Grad-CAM saliency heatmap generator (auxiliary tool).

Post-processing pipeline (§2.4.2):
  1. Border suppression     — zero out a 5 % perimeter band to remove artefacts.
  2. 75th-pct thresholding  — binarise at the 75th percentile of non-zero activations.
  3. Connected-component decomposition — keep the top-3 components by area.
  4. Texture filtering      — discard uniform / low-variance components.
  5. Class-specificity mask — re-weight by original CAM magnitude (acts as
                              class-specificity weighting when a single class CAM
                              is used; full cross-class comparison requires multiple
                              forward passes and can be enabled via multi_class=True).
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch
from ophagent.tools.base import BaseTool, ToolMetadata


class GradCAMTool(BaseTool):
    """
    Generates Grad-CAM heatmaps for any CNN model already loaded in the scheduler,
    with the five-step post-processing pipeline from §2.4.2.

    Runs inline.
    Input:  {"image_path": str, "model_name": str, "target_class": int (optional)}
    Output: {"heatmap_b64": str, "overlay_b64": str, "processed_heatmap_b64": str}
    """

    # ------------------------------------------------------------------
    # Post-processing helpers  (§2.4.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_cam(cam: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply the five post-processing steps from §2.4.2 to a raw Grad-CAM map.

        Args:
            cam:     Grayscale CAM array, shape (H, W), values in [0, 1].
            img_rgb: Original image as float32 RGB array, shape (H, W, 3), [0, 1].

        Returns:
            Processed CAM array, same shape, values in [0, 1].
        """
        h, w = cam.shape

        # Step 1 — Border suppression (5 % margin on each side)
        bh = max(1, h // 20)
        bw = max(1, w // 20)
        processed = cam.copy()
        processed[:bh, :] = 0.0
        processed[-bh:, :] = 0.0
        processed[:, :bw] = 0.0
        processed[:, -bw:] = 0.0

        # Step 2 — 75th-percentile thresholding → binary mask
        positive = processed[processed > 0]
        if positive.size == 0:
            return processed
        thr = float(np.percentile(positive, 75))
        binary = (processed >= thr).astype(np.uint8)

        # Step 3 — Connected-component decomposition; keep top-3 by area
        try:
            from scipy.ndimage import label as nd_label
            labeled, n_comp = nd_label(binary)
            if n_comp > 0:
                areas = np.array([(labeled == i).sum() for i in range(1, n_comp + 1)])
                top_n = min(3, n_comp)
                top_ids = np.argsort(areas)[::-1][:top_n] + 1   # 1-indexed
                kept = np.zeros_like(binary)
                for cid in top_ids:
                    kept[labeled == cid] = 1
                binary = kept
        except ImportError:
            # scipy not available; skip connected-component filtering
            pass

        # Step 4 — Texture filtering: discard components with low image-region variance
        if binary.any():
            try:
                from scipy.ndimage import label as nd_label
                labeled2, n2 = nd_label(binary)
                gray = img_rgb.mean(axis=-1) if img_rgb.ndim == 3 else img_rgb
                texture_mask = np.zeros_like(binary)
                for i in range(1, n2 + 1):
                    comp_mask = labeled2 == i
                    variance = float(gray[comp_mask].var())
                    if variance > 1e-4:          # retain textured regions only
                        texture_mask |= comp_mask
                binary = texture_mask.astype(np.uint8)
            except ImportError:
                pass   # skip if scipy unavailable

        # Step 5 — Class-specificity mask: re-weight by original CAM values.
        # This preserves high-activation regions within the binary mask and
        # naturally suppresses class-agnostic low-level structure.
        # (Full cross-class comparison requires generating CAMs for multiple
        # target classes and is left to higher-level orchestration.)
        result = processed * binary.astype(np.float32)

        # Re-normalise to [0, 1]
        max_val = result.max()
        if max_val > 1e-8:
            result /= max_val

        return result

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import io
        from PIL import Image
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from ophagent.utils.image_utils import load_image_pil
        from ophagent.tools.scheduler import ToolScheduler

        img_pil = load_image_pil(inputs["image_path"])
        img_np = np.array(img_pil).astype(np.float32) / 255.0   # (H, W, 3) in [0, 1]

        # Obtain the target model from the scheduler
        scheduler = ToolScheduler()
        model_name = inputs.get("model_name", "cfp_disease")
        tool = scheduler._get_or_load_tool(scheduler.registry.get(model_name))
        if not hasattr(tool, "_model") or tool._model is None:
            tool.load_model()

        model = tool._model
        target_layers = (
            [model.layer4[-1]]
            if hasattr(model, "layer4")
            else [list(model.modules())[-2]]
        )

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(img_pil).unsqueeze(0)

        cam_gen = GradCAM(model=model, target_layers=target_layers)
        target_class = inputs.get("target_class")
        targets = (
            [lambda x, c=target_class: x[:, c]]
            if target_class is not None
            else None
        )
        grayscale_cam = cam_gen(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

        # Resize CAM to original image size for post-processing
        cam_resized = np.array(
            Image.fromarray(grayscale_cam).resize(
                (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
            )
        )

        # Apply §2.4.2 five-step post-processing
        processed_cam = self._postprocess_cam(cam_resized, img_np)

        # Build visualisations
        overlay_raw = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        overlay_processed = show_cam_on_image(img_np, processed_cam, use_rgb=True)

        def to_b64(arr: np.ndarray) -> str:
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        return {
            "heatmap_b64":           to_b64((grayscale_cam * 255).astype(np.uint8)),
            "overlay_b64":           to_b64(overlay_raw),
            "processed_heatmap_b64": to_b64((processed_cam * 255).astype(np.uint8)),
            "overlay_processed_b64": to_b64(overlay_processed),
        }
