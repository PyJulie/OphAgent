"""
Adapter: UWF (ultra-wide-field) multi-label disease classifier.

Wraps an external UWF multi-label classifier. Configure the project with
``OPHAGENT_UWF_SRC`` or the checkpoint with ``OPHAGENT_UWF_WEIGHTS``.
Multi-label sigmoid over 8 conditions: MH, RP, AMD, RVO, RD, Glaucoma, DR,
any retinal disease (umbrella).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


UWF_SRC = external_dir("OPHAGENT_UWF_SRC", "uwf")
CKPT_PATH = checkpoint_file("OPHAGENT_UWF_WEIGHTS", "uwf", "multi_disease.pt")

LABELS = ["MH", "RP", "AMD", "RVO", "RD", "Glaucoma", "DR", "any_retina_disease"]
LABEL_DESCRIPTIONS = {
    "MH": "Macular hole",
    "RP": "Retinitis pigmentosa",
    "AMD": "Age-related macular degeneration",
    "RVO": "Retinal vein occlusion",
    "RD": "Retinal detachment",
    "Glaucoma": "Glaucomatous optic neuropathy",
    "DR": "Diabetic retinopathy",
    "any_retina_disease": "Any retinal abnormality (umbrella label)",
}


@register
class UWFClassifierAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="uwf_multi_disease",
        modality="UWF",
        task="classification",
        description=(
            "Multi-label disease classifier on an Ultra-Wide-Field (UWF) "
            "fundus image. Independently scores 8 conditions: macular hole, "
            "retinitis pigmentosa, AMD, RVO, retinal detachment, glaucoma, "
            "diabetic retinopathy, and an umbrella 'any retinal disease' "
            "label. Multi-label — multiple findings can coexist."
        ),
        input_size=(384, 512),
        labels=LABELS,
        confidence_threshold=0.50,   # sigmoid threshold for binary flag
        limitations=[
            "Uses only Red + Green channels (Blue is masked) — Optos device specific",
            "May under-detect peripheral lesions outside standard 200° FOV",
        ],
        cost_class="medium",
        source_dir=str(UWF_SRC),
    )

    def _load_impl(self) -> None:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                "UWF checkpoint not found. Set OPHAGENT_UWF_WEIGHTS or "
                f"OPHAGENT_UWF_SRC. Expected: {CKPT_PATH}"
            )
        # The .pt was saved with an older timm where adaptive_avgmax_pool
        # lived under timm.models.layers. Stub the old path -> new path.
        import sys
        try:
            import timm.layers.adaptive_avgmax_pool as _new
            sys.modules.setdefault("timm.models.layers.adaptive_avgmax_pool", _new)
        except ImportError:
            pass
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        from timm.models.resnet import BasicBlock, ResNet

        safe_model_classes = [
            torch.nn.Sequential,
            torch.nn.Dropout,
            torch.nn.BatchNorm2d,
            torch.nn.ReLU,
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.PReLU,
            torch.nn.MaxPool2d,
            torch.nn.Linear,
            torch.nn.Conv2d,
            BasicBlock,
            ResNet,
            (
                SelectAdaptivePool2d,
                "timm.models.layers.adaptive_avgmax_pool.SelectAdaptivePool2d",
            ),
        ]
        with torch.serialization.safe_globals(safe_model_classes):
            model = torch.load(
                str(CKPT_PATH), map_location=self.device, weights_only=True
            )
        if hasattr(model, "global_pool"):
            try:
                model.global_pool.flatten = torch.nn.Flatten(1)
            except Exception:
                pass
        # Patch attribute / behavior incompatibilities between the old timm
        # the .pt was pickled with and the current timm:
        for sub in model.modules():
            if not hasattr(sub, "grad_checkpointing"):
                try:
                    sub.grad_checkpointing = False
                except Exception:
                    pass
            # Old timm ResNet had self.drop_block = None when unused;
            # newer code calls it unconditionally → wrap in Identity.
            if hasattr(sub, "drop_block") and sub.drop_block is None:
                sub.drop_block = torch.nn.Identity()
            # Same for aa (anti-alias pool)
            if hasattr(sub, "aa") and sub.aa is None:
                sub.aa = torch.nn.Identity()
        model.to(self.device).eval()
        self._impl = model
        self._transform = T.Compose([
            T.Resize((384, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.22578795, 0.23797078, 1.0],
                        std=[0.14651306, 0.11282759, 1.0]),
            T.Lambda(lambda x: x[:2, ...]),   # drop blue channel
        ])

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        img = Image.open(image_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(self.device)
        logits = self._impl(x).flatten()
        probs = torch.sigmoid(logits).cpu().float().numpy()

        per_label = {
            LABELS[i]: {
                "probability": float(probs[i]),
                "predicted": bool(probs[i] > 0.5),
                "description": LABEL_DESCRIPTIONS[LABELS[i]],
            }
            for i in range(len(LABELS))
        }
        positive = [n for n, v in per_label.items() if v["predicted"]]
        # confidence = max prob (among disease classes, excluding umbrella)
        disease_probs = [probs[i] for i, n in enumerate(LABELS) if n != "any_retina_disease"]
        max_conf = float(max(disease_probs)) if disease_probs else 0.0

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="UWF",
            task="classification",
            predictions={
                "per_label": per_label,
                "predicted_diseases": positive,
                "any_disease_probability": float(probs[LABELS.index("any_retina_disease")]),
            },
            confidence=max_conf,
        )
