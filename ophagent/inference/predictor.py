"""
Standardized inference interface for all OCT models.

Provides a uniform API: load image → preprocess → predict → postprocess → return result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..data.preprocessing.transforms import (
    get_classification_transforms,
    get_segmentation_transforms,
    get_generation_transforms,
)
from .model_registry import ModelRegistry


class OphPredictor:
    """Unified prediction interface for all OCT models."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def predict(
        self,
        model_name: str,
        image: np.ndarray | str | Path,
    ) -> dict[str, Any]:
        card = self.registry.get_card(model_name)
        model = self.registry.load_model(model_name)
        device = next(model.parameters()).device

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot load image: {image}")

        original_shape = image.shape[:2]

        if card.task == "classification":
            return self._predict_classification(model, card, image, device)
        elif card.task == "quality":
            return self._predict_quality(model, card, image, device)
        elif card.task == "segmentation":
            return self._predict_segmentation(model, card, image, device, original_shape)
        elif card.task == "denoising":
            return self._predict_denoising(model, card, image, device)
        elif card.task == "super_resolution":
            return self._predict_super_resolution(model, card, image, device)
        else:
            raise ValueError(f"Unknown task: {card.task}")

    def _preprocess(
        self, image: np.ndarray, size: int, task: str
    ) -> torch.Tensor:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if task in ("classification", "quality"):
            transform = get_classification_transforms(size, is_training=False)
        elif task == "segmentation":
            transform = get_segmentation_transforms((size, size), is_training=False)
        else:
            transform = get_generation_transforms(size, is_training=False)

        result = transform(image=image)
        return result["image"].unsqueeze(0)

    def _predict_classification(
        self, model, card, image, device
    ) -> dict[str, Any]:
        tensor = self._preprocess(image, card.input_size, "classification").to(device)

        with torch.no_grad():
            output = model(tensor)

        probs = output["probabilities"][0].cpu().numpy()
        pred_idx = int(probs.argmax())
        class_name = card.class_names[pred_idx] if card.class_names else str(pred_idx)

        return {
            "task": "classification",
            "model": card.name,
            "predicted_class": class_name,
            "predicted_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(p)
                for name, p in zip(card.class_names or range(len(probs)), probs)
            },
        }

    def _predict_quality(
        self, model, card, image, device
    ) -> dict[str, Any]:
        tensor = self._preprocess(image, card.input_size, "quality").to(device)

        with torch.no_grad():
            output = model(tensor)

        # support both OCTQualityAssessor (quality_score) and OCTClassifier (probabilities)
        probs_t = output.get("quality_score") or output.get("probabilities")
        probs = probs_t[0].cpu().numpy()
        quality_idx = int(probs.argmax())
        quality_label = card.class_names[quality_idx]

        return {
            "task": "quality_assessment",
            "model": card.name,
            "quality": quality_label,
            "quality_index": quality_idx,
            "confidence": float(probs[quality_idx]),
            "scores": {
                name: float(p)
                for name, p in zip(card.class_names, probs)
            },
        }

    def _predict_segmentation(
        self, model, card, image, device, original_shape
    ) -> dict[str, Any]:
        tensor = self._preprocess(image, card.input_size, "segmentation").to(device)

        with torch.no_grad():
            output = model(tensor)

        logits = output["logits"]
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        pred_resized = cv2.resize(
            pred_mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        class_areas = {}
        for idx, name in enumerate(card.class_names or range(pred_mask.max() + 1)):
            area = int((pred_resized == idx).sum())
            if area > 0:
                class_areas[str(name)] = area

        return {
            "task": "segmentation",
            "model": card.name,
            "mask": pred_resized,
            "class_areas": class_areas,
            "num_classes_detected": len([a for a in class_areas.values() if a > 0]),
        }

    def _predict_denoising(
        self, model, card, image, device
    ) -> dict[str, Any]:
        if image.ndim == 2:
            image = image[..., np.newaxis]

        h, w = image.shape[:2]
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = F.interpolate(tensor, size=(card.input_size, card.input_size), mode="bilinear")
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)

        denoised = output["denoised"][0].cpu()
        denoised = F.interpolate(denoised.unsqueeze(0), size=(h, w), mode="bilinear")[0]
        denoised = ((denoised * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        return {
            "task": "denoising",
            "model": card.name,
            "denoised_image": denoised.squeeze(),
        }

    def _predict_super_resolution(
        self, model, card, image, device
    ) -> dict[str, Any]:
        if image.ndim == 2:
            image = image[..., np.newaxis]

        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)

        sr = output["super_resolved"][0].cpu()
        sr = ((sr * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        return {
            "task": "super_resolution",
            "model": card.name,
            "super_resolved_image": sr.squeeze(),
            "output_size": sr.shape[:2],
        }
