"""Inference-only implementation of the OphAgent PDR cascade."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CATEGORY_NAMES = [
    "\u65e0PDR",
    "\u975e\u6d3b\u52a8\u6027PDR",
    "\u6d3b\u52a8\u6027PDR",
    "\u65e0\u6cd5\u5224\u65ad",
]
CATEGORY_EN = ["no_PDR", "inactive_PDR", "active_PDR", "ungradable"]
ACTIVE_REASON_LABELS = [
    "\u89c6\u76d8\u65b0\u751f\u8840\u7ba1",
    "\u89c6\u76d8\u4ee5\u5916\u65b0\u751f\u8840\u7ba1",
    "\u73bb\u7483\u4f53\u6216\u89c6\u7f51\u819c\u524d\u51fa\u8840",
    "\u7ea4\u7ef4\u8840\u7ba1\u589e\u6b96\u819c",
]
INACTIVE_REASON_LABELS = [
    "\u64ad\u6563\u6027\u6fc0\u5149\u6591",
    "\u7622\u75d5\u5316\u589e\u6b96\u819c",
]
MAIN_INPUT_SIZE = 384
REASONS_INPUT_SIZE = 224


def create_main_model(num_classes: int = 4) -> nn.Module:
    import timm

    return timm.create_model(
        "convnext_base.fb_in22k_ft_in1k_384",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )


def create_reasons_model(num_labels: int, dropout: float = 0.0) -> nn.Module:
    import timm

    model = timm.create_model(
        "efficientnet_b0.ra_in1k",
        pretrained=False,
        num_classes=0,
    )
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.num_features, num_labels),
    )
    return model


def get_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


def _state_dict(path: str | Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(str(path), map_location=device, weights_only=True)
    if isinstance(state, dict):
        for key in ("model", "state_dict"):
            value = state.get(key)
            if isinstance(value, dict):
                state = value
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _thresholds(path: str | Path) -> Any:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "thresholds" in payload:
        return payload["thresholds"]
    return payload


class PDRCascadePipeline:
    """Four-class category head plus active and inactive sign heads."""

    def __init__(
        self,
        main_model_path: str | Path,
        active_model_path: str | Path,
        inactive_model_path: str | Path,
        active_thresholds_path: str | Path,
        inactive_thresholds_path: str | Path,
        device: str = "cuda",
    ):
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)

        self.main_model = create_main_model(len(CATEGORY_NAMES))
        self.main_model.load_state_dict(
            _state_dict(main_model_path, self.device),
            strict=True,
        )
        self.main_model.to(self.device).eval()

        self.active_model = create_reasons_model(len(ACTIVE_REASON_LABELS))
        self.active_model.load_state_dict(
            _state_dict(active_model_path, self.device),
            strict=True,
        )
        self.active_model.to(self.device).eval()

        self.inactive_model = create_reasons_model(len(INACTIVE_REASON_LABELS))
        self.inactive_model.load_state_dict(
            _state_dict(inactive_model_path, self.device),
            strict=True,
        )
        self.inactive_model.to(self.device).eval()

        self.active_thresholds = _thresholds(active_thresholds_path)
        self.inactive_thresholds = _thresholds(inactive_thresholds_path)
        self.main_transform = get_eval_transform(MAIN_INPUT_SIZE)
        self.reasons_transform = get_eval_transform(REASONS_INPUT_SIZE)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda")
        return nullcontext()

    @torch.no_grad()
    def predict(self, image_path: str) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        main_input = self.main_transform(image).unsqueeze(0).to(self.device)
        with self._autocast():
            category_logits = self.main_model(main_input)
        category_probs = F.softmax(category_logits.float(), dim=1)[0].cpu()
        category_id = int(category_probs.argmax().item())

        result: dict[str, Any] = {
            "image": Path(image_path).name,
            "category": CATEGORY_NAMES[category_id],
            "category_en": CATEGORY_EN[category_id],
            "category_id": category_id,
            "category_probs": {
                CATEGORY_NAMES[index]: round(float(category_probs[index]), 4)
                for index in range(len(CATEGORY_NAMES))
            },
            "category_probs_en": {
                CATEGORY_EN[index]: round(float(category_probs[index]), 4)
                for index in range(len(CATEGORY_EN))
            },
            "reasons": {},
        }

        reason_input = self.reasons_transform(image).unsqueeze(0).to(self.device)
        with self._autocast():
            active_logits = self.active_model(reason_input)
            inactive_logits = self.inactive_model(reason_input)
        active_scores = torch.sigmoid(active_logits.float())[0].cpu().numpy()
        inactive_scores = torch.sigmoid(inactive_logits.float())[0].cpu().numpy()

        for name, score, threshold in zip(
            ACTIVE_REASON_LABELS,
            active_scores,
            self.active_thresholds,
        ):
            result["reasons"][name] = {
                "score": round(float(score), 4),
                "predicted": bool(score > threshold),
                "threshold": round(float(threshold), 4),
                "head": "active",
            }
        for name, score, threshold in zip(
            INACTIVE_REASON_LABELS,
            inactive_scores,
            self.inactive_thresholds,
        ):
            result["reasons"][name] = {
                "score": round(float(score), 4),
                "predicted": bool(score > threshold),
                "threshold": round(float(threshold), 4),
                "head": "inactive",
            }

        has_active = any(
            result["reasons"][name]["predicted"] for name in ACTIVE_REASON_LABELS
        )
        has_inactive = any(
            result["reasons"][name]["predicted"] for name in INACTIVE_REASON_LABELS
        )
        result["has_active_signs"] = has_active
        result["has_inactive_signs"] = has_inactive
        result["mixed_pattern"] = bool(has_active and has_inactive)
        return result
