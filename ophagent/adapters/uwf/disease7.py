"""
Adapter: UWF 7-class single-label disease classifier (ConvNeXt-Tiny).

Complementary to `uwf_multi_disease` (8-label multi-label, ResNet, R+G only):
  - single-label softmax over 7 classes — adds Healthy / PM / Uveitis the
    multi-label model lacks;
  - different architecture (ConvNeXt-Tiny) + full RGB @768px → ensemble
    diversity for cross-checking UWF reads.

Trained on an internal UWF development set (700 images, 100/class).
Val: acc 0.886, macro-F1 0.886 (vs ResNet50 reference 0.829).
Checkpoint: checkpoints/uwf_classifier7/best.pt
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T, models

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file


CKPT_PATH = checkpoint_file("OPHAGENT_UWF_DISEASE7_WEIGHTS", "uwf", "disease7", "best.pt")

# Folder labels → clean names + descriptions
_CLEAN = {
    "0_Healthy": ("Healthy", "No referable retinal disease"),
    "1_DR": ("DR", "Diabetic retinopathy"),
    "2_AMD": ("AMD", "Age-related macular degeneration"),
    "3_PM": ("PM", "Pathologic myopia"),
    "4_RVO": ("RVO", "Retinal vein occlusion"),
    "5_Uveitis": ("Uveitis", "Uveitis / intraocular inflammation"),
    "6_RD": ("RD", "Retinal detachment"),
}


@register
class UWFDisease7Adapter(AdapterBase):
    metadata = ToolMetadata(
        name="uwf_disease_7class",
        modality="UWF",
        task="classification",
        description=(
            "Single-label 7-class disease classifier for Ultra-Wide-Field "
            "fundus (ConvNeXt-Tiny, full RGB @768px). Classes: Healthy, DR, "
            "AMD, Pathologic Myopia, RVO, Uveitis, Retinal Detachment. "
            "COMPLEMENTARY to uwf_multi_disease (which is multi-label and "
            "covers MH/RP/Glaucoma but NOT Healthy/PM/Uveitis) — run both and "
            "cross-check. Returns a softmax probability per class + top-3."
        ),
        input_size=(768, 768),
        labels=[_CLEAN[k][0] for k in
                ["0_Healthy", "1_DR", "2_AMD", "3_PM", "4_RVO", "5_Uveitis", "6_RD"]],
        confidence_threshold=0.40,
        limitations=[
            "Single-label — assumes one dominant pathology per image; for "
            "co-existing diseases cross-check with uwf_multi_disease",
            "Trained on 700 images (100/class); Uveitis is the weakest class "
            "(val F1 0.73), sometimes confused with Healthy",
            "No glaucoma / macular-hole / RP head — use uwf_multi_disease for those",
        ],
        cost_class="medium",
        source_dir=str(CKPT_PATH.parent),
    )

    def _load_impl(self) -> None:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"UWF 7-class checkpoint not found: {CKPT_PATH}")
        ck = torch.load(str(CKPT_PATH), map_location=self.device, weights_only=True)
        self._classes = ck["classes"]                       # folder-prefixed
        n = len(self._classes)
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n)
        model.load_state_dict(ck["model_state"])
        model.to(self.device).eval()
        self._impl = model
        size = ck.get("input_size", 768)
        self._transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=ck.get("norm_mean", [0.485, 0.456, 0.406]),
                        std=ck.get("norm_std", [0.229, 0.224, 0.225])),
        ])

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        img = Image.open(image_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(self.device)
        logits = self._impl(x).flatten()
        probs = torch.softmax(logits, dim=0).cpu().float().numpy()

        ranked = sorted(
            ((self._classes[i], float(probs[i])) for i in range(len(self._classes))),
            key=lambda kv: -kv[1],
        )
        top_class_raw, top_p = ranked[0]
        clean, desc = _CLEAN.get(top_class_raw, (top_class_raw, top_class_raw))
        top3 = [
            {"label": _CLEAN.get(c, (c, c))[0],
             "description": _CLEAN.get(c, (c, c))[1],
             "probability": round(p, 4)}
            for c, p in ranked[:3]
        ]
        all_probs = {
            _CLEAN.get(self._classes[i], (self._classes[i],))[0]: round(float(probs[i]), 4)
            for i in range(len(self._classes))
        }
        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="UWF",
            task="classification",
            predictions={
                "predicted_class": clean,
                "predicted_description": desc,
                "probability": round(top_p, 4),
                "top_3": top3,
                "all_probabilities": all_probs,
            },
            confidence=float(top_p),
        )
