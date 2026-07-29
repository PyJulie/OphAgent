"""
Adapter: CFP image quality assessment (ResNet18, 3-class EyeQ).

Wraps an external EyeQ-style image-quality checkpoint. Configure the project
with ``OPHAGENT_EYEQ_SRC`` or the checkpoint with ``OPHAGENT_EYEQ_WEIGHTS``.
Outputs: quality category (Good / Usable / Reject) + per-class probabilities.

NOTE: Known to occasionally misjudge images with very large central lesions
as "Reject" (the lesions look like artefacts to the model). The verifier
should weight EyeQ output against the disease-classifier's confidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_file, external_dir


EYEQ_SRC = external_dir("OPHAGENT_EYEQ_SRC", "eyeq")
CKPT_PATH = external_file(
    "OPHAGENT_EYEQ_WEIGHTS", "external", "eyeq", "outputs_eyeq_resnet18", "best_val.pth"
)
if not CKPT_PATH.exists() and not os.environ.get("OPHAGENT_EYEQ_WEIGHTS"):
    CKPT_PATH = checkpoint_file("OPHAGENT_EYEQ_WEIGHTS", "cfp", "eyeq.pth")

LABELS = ["Good", "Usable", "Reject"]


@register
class EyeQAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_eyeq",
        modality="CFP",
        task="quality",
        description=(
            "Colour-fundus image quality assessment — outputs one of three "
            "categories: Good (clean, ready for diagnosis), Usable (acceptable "
            "but some artefact), Reject (severe blur / shadow / over-exposure). "
            "Always call this first; if quality is Reject, downstream models "
            "should be interpreted with caution or the image re-acquired."
        ),
        input_size=(224, 224),
        labels=LABELS,
        confidence_threshold=0.55,
        limitations=[
            "Sometimes flags images with very large lesions as low-quality — "
            "cross-check against disease classifier output",
            "Trained on EyePACS / Kaggle DR images — UWF or non-standard FOVs "
            "may bias toward 'Reject'",
        ],
        cost_class="fast",
        source_dir=str(EYEQ_SRC),
    )

    def _load_impl(self) -> None:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                "EyeQ checkpoint not found. Set OPHAGENT_EYEQ_WEIGHTS or "
                f"OPHAGENT_EYEQ_SRC. Expected: {CKPT_PATH}"
            )
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        ckpt = torch.load(str(CKPT_PATH), map_location=self.device, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self._impl = model
        self._transform = transforms.Compose([
            transforms.Resize(int(224 * 1.15)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        img = Image.open(image_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(self.device)
        logits = self._impl(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="quality",
            predictions={
                "quality": LABELS[pred_idx],
                "quality_id": pred_idx,
                "probabilities": {
                    LABELS[i]: float(probs[i]) for i in range(len(LABELS))
                },
                "is_usable": pred_idx < 2,  # Good or Usable
                "is_rejected": pred_idx == 2,
            },
            confidence=float(probs[pred_idx]),
        )
