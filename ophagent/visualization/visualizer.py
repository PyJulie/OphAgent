"""
Visualization utilities for OCT analysis:
  - Grad-CAM heatmaps for the classifier
  - Segmentation overlays with class colors
  - Bounding boxes extracted from heatmaps / segmentation masks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ── Color palettes ──────────────────────────────────────────────────────────

FLUID_COLORS = {
    0: (0, 0, 0),         # background  - transparent
    1: (66, 135, 245),    # IRF - blue
    2: (245, 134, 66),    # SRF - orange
    3: (245, 66, 156),    # PED - pink
}

LAYER_COLORS = {
    0: (0, 0, 0),         # background-above
    1: (255, 99, 99),     # ILM-NFL
    2: (255, 158, 99),    # NFL-IPL
    3: (255, 217, 99),    # IPL-INL
    4: (164, 255, 99),    # INL-OPL
    5: (99, 255, 145),    # OPL-ONL
    6: (99, 222, 255),    # ONL-ISM
    7: (99, 145, 255),    # ISM-ISOS
    8: (158, 99, 255),    # ISOS-RPE
    9: (40, 40, 40),      # below-RPE
}


# ── Grad-CAM ────────────────────────────────────────────────────────────────

@dataclass
class GradCAMResult:
    heatmap: np.ndarray          # (H, W) float in [0, 1]
    overlay: np.ndarray          # (H, W, 3) uint8 BGR
    boxes: list[tuple[int, int, int, int]]  # list of (x, y, w, h)
    predicted_class: str
    confidence: float


class GradCAM:
    """Compute Grad-CAM heatmap for a classifier given an input image.

    Works with any model exposing a backbone feature map; for OCTClassifier,
    we hook the timm backbone's last conv stage.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = self._auto_pick_target_layer(model)
        if target_layer is None:
            raise ValueError("No suitable Conv2d layer found for Grad-CAM target")

        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    @staticmethod
    def _auto_pick_target_layer(model: torch.nn.Module) -> torch.nn.Module | None:
        """Pick the deepest Conv2d (most semantic features)."""
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv

    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.enable_grad()
    def compute(
        self,
        image_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute Grad-CAM heatmap.

        Args:
            image_tensor: shape (1, C, H, W), already preprocessed
            class_idx: target class for CAM; if None, use predicted class

        Returns:
            (heatmap normalized to [0,1] shape (H_input, W_input), predicted_class_idx, confidence)
        """
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device).requires_grad_(True)

        self.model.zero_grad()
        out = self.model(image_tensor)
        logits = out["logits"] if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=-1)

        if class_idx is None:
            class_idx = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, class_idx].item())

        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # weight each activation channel by mean gradient
        grads = self._gradients[0]      # (C, h, w)
        acts = self._activations[0]      # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (h, w)
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        # resize to input size
        h, w = image_tensor.shape[-2:]
        cam = cv2.resize(cam, (w, h))
        return cam, class_idx, confidence


# ── Overlays ────────────────────────────────────────────────────────────────

def to_3ch_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    return image


def heatmap_overlay(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Overlay a grayscale heatmap on the image using JET colormap."""
    img = to_3ch_bgr(image)
    h, w = img.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))
    cmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, cmap, alpha, 0)
    return overlay


def segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    palette: dict[int, tuple[int, int, int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a class-index mask on the image using the given palette."""
    img = to_3ch_bgr(image)
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(img)
    for cls_idx, color in palette.items():
        color_mask[mask == cls_idx] = color

    nonbg = (mask > 0)[..., None]
    out = img.copy()
    out = np.where(nonbg, cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0), img)
    return out.astype(np.uint8)


# ── Box extraction ──────────────────────────────────────────────────────────

def boxes_from_heatmap(
    heatmap: np.ndarray, threshold: float = 0.5, min_area: int = 200
) -> list[tuple[int, int, int, int]]:
    """Threshold heatmap and return bounding boxes of connected hotspots.

    Returns list of (x, y, w, h) in pixel coords matching heatmap shape.
    """
    binary = (heatmap >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def boxes_from_mask(
    mask: np.ndarray, class_indices: list[int], min_area: int = 50
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Bounding boxes per class from a segmentation mask."""
    result: dict[int, list[tuple[int, int, int, int]]] = {}
    for cls in class_indices:
        binary = (mask == cls).astype(np.uint8) * 255
        if binary.sum() == 0:
            continue
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((int(x), int(y), int(w), int(h)))
        if boxes:
            result[cls] = boxes
    return result


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    out = to_3ch_bgr(image).copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if label:
            txt = f"{label}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y), color, -1)
            cv2.putText(out, txt, (x + 3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ── Image I/O helpers ───────────────────────────────────────────────────────

def save_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def load_image_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
