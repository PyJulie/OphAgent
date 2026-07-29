"""
Adapter: FFA multi-task multi-label classification.

Wraps an external FFA multi-task classifier. Configure the local project with
``OPHAGENT_FFA_PROJECT`` or the checkpoint with
``OPHAGENT_FFA_CLASSIFIER_WEIGHTS``.

Architecture (from `mtml/modeling.py`):
  - Backbone: timm ResNet-50, num_classes=0, global_pool='avg'   →  2048-dim
  - head_raw    : Linear(2048, 47)   — fine-grained FFAIR lesion codes
  - head_merged : Linear(2048,  9)   — clinical groups:
        AMD / CSC / DR / Macular Disorders / Other / PCV /
        Pathologic Myopia / RVO / Uveitis

Multi-label sigmoid outputs (an FFA image can have several findings at once).

Saved checkpoint (`outputs_refactored/checkpoints/best.pt`):
  {
    'model'            : OrderedDict[str, Tensor],
    'epoch'            : 9,
    'summary'          : {'raw_auc_macro': 0.906, 'merged_auc_macro': 0.884, ...},
    'raw_class_to_idx' : {raw_label_id_str: idx, ...},   # 47 entries
    'merged_to_idx'    : {merged_name: idx,    ...},     # 9 entries
  }

Preprocessing (matches training):
  Resize(448, 448) → ToTensor → ImageNet normalisation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


FFA_PROJECT = external_dir("OPHAGENT_FFA_PROJECT", "ffa")
FFA_CLS_WEIGHTS = checkpoint_file("OPHAGENT_FFA_CLASSIFIER_WEIGHTS", "ffa", "classification.pt")
BACKBONE_NAME = "resnet50"
IMG_SIZE = 448
DEFAULT_THRESHOLD = 0.5


class _MTMLModel(nn.Module):
    """Mirrors `mtml/modeling.py::MultiTaskModel` exactly."""

    def __init__(self, backbone_name: str, num_raw: int, num_merged: int):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = getattr(self.backbone, "num_features", 2048) or 2048
        self.head_raw = nn.Linear(feat_dim, num_raw)
        self.head_merged = nn.Linear(feat_dim, num_merged)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_raw(f), self.head_merged(f)


@register
class FFAClassificationAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="ffa_classification",
        modality="FFA",
        task="classification",
        description=(
            "Multi-task multi-label disease classification on an FFA "
            "(fluorescein angiography) image. Returns: (1) probabilities over 9 "
            "merged clinical groups (DR, RVO, AMD, CSC, PCV, Pathologic Myopia, "
            "Uveitis, Macular Disorders, Other) — the headline output; (2) "
            "fine-grained probabilities over 47 raw FFAIR lesion codes for "
            "power users. Use this when the user has an FFA image and asks "
            "'what disease is this?' or wants a diagnostic shortlist. Pair "
            "with `ffa_lesion_detection` for localisation."
        ),
        input_size=(IMG_SIZE, IMG_SIZE),
        labels=[],   # filled at load (from checkpoint metadata)
        confidence_threshold=0.5,
        limitations=[
            "Multi-label: probabilities are NOT mutually exclusive; >1 disease "
            "can co-occur on the same image.",
            "Trained on FFAIR / Tongren cohort — generalisation to other "
            "angiograph hardware or unusual acquisition phases may degrade.",
            "Validation performance: raw AUROC macro ≈ 0.91, merged AUROC "
            "macro ≈ 0.88 (epoch 9); merged F1 macro ≈ 0.52 at threshold 0.5. "
            "On a held-out spot-check (n=20 patients): Top-1 ≈ 55%, Top-3 ≈ 80% "
            "— report the **top-3** ranked list, not just argmax, for clinical use.",
            "The 'Other' merged group is heterogeneous — interpret with care.",
            "Threshold 0.5 is not calibrated per class; use the ranked merged "
            "probability list as the primary signal.",
        ],
        cost_class="fast",
        source_dir=str(FFA_PROJECT),
    )

    def _load_impl(self) -> None:
        if not FFA_CLS_WEIGHTS.exists():
            raise FileNotFoundError(
                "FFA classifier weights not found. Set "
                "OPHAGENT_FFA_CLASSIFIER_WEIGHTS or OPHAGENT_FFA_PROJECT. "
                f"Expected: {FFA_CLS_WEIGHTS}"
            )
        ckpt = torch.load(FFA_CLS_WEIGHTS, map_location="cpu", weights_only=True)
        self._raw_class_to_idx: dict[str, int] = ckpt["raw_class_to_idx"]
        self._merged_to_idx: dict[str, int]    = ckpt["merged_to_idx"]
        self._raw_idx_to_class: dict[int, str] = {v: k for k, v in self._raw_class_to_idx.items()}
        self._merged_idx_to_name: dict[int, str] = {v: k for k, v in self._merged_to_idx.items()}
        num_raw = len(self._raw_class_to_idx)        # 47
        num_merged = len(self._merged_to_idx)        # 9

        model = _MTMLModel(BACKBONE_NAME, num_raw=num_raw, num_merged=num_merged)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        # both should be empty if architecture matches
        self._impl = model.to(self.device).eval()

        # publish the merged-group labels to ToolMetadata
        self.metadata.labels = [self._merged_idx_to_name[i] for i in range(num_merged)]

        self._tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self._train_summary = ckpt.get("summary", {})

    @torch.no_grad()
    def _predict_impl(
        self,
        image_path: str,
        threshold: float = DEFAULT_THRESHOLD,
        topk_raw: int = 5,
        **_,
    ) -> AdapterResult:
        img = Image.open(image_path).convert("RGB")
        x = self._tf(img).unsqueeze(0).to(self.device)

        logits_raw, logits_merged = self._impl(x)
        prob_raw    = torch.sigmoid(logits_raw)[0].cpu().tolist()    # (47,)
        prob_merged = torch.sigmoid(logits_merged)[0].cpu().tolist() # (9,)

        # Merged: full distribution
        merged_probs = {
            self._merged_idx_to_name[i]: float(prob_merged[i])
            for i in range(len(prob_merged))
        }
        positives = sorted(
            [(n, p) for n, p in merged_probs.items() if p >= threshold],
            key=lambda kv: kv[1], reverse=True,
        )
        # Sort the full distribution to give the planner a ranked view
        merged_ranked = sorted(merged_probs.items(), key=lambda kv: kv[1], reverse=True)
        top_merged = merged_ranked[0]   # (name, prob)

        # Raw: top-K is enough for the LLM (47 numbers is noisy)
        raw_ranked = sorted(
            [(self._raw_idx_to_class[i], float(prob_raw[i])) for i in range(len(prob_raw))],
            key=lambda kv: kv[1], reverse=True,
        )
        top_raw = raw_ranked[:topk_raw]

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="FFA",
            task="classification",
            predictions={
                "primary_diagnosis": top_merged[0],
                "primary_probability": float(top_merged[1]),
                "positive_groups": [
                    {"label": n, "probability": float(p)} for n, p in positives
                ],
                "merged_probabilities": {
                    n: float(p) for n, p in merged_ranked
                },
                "topk_raw_lesion_codes": [
                    {"raw_id": rid, "probability": float(p)} for rid, p in top_raw
                ],
                "threshold": threshold,
            },
            confidence=float(top_merged[1]),
            raw_output={
                "merged_probs_ordered": merged_ranked,
                "raw_probs_ordered": raw_ranked,
            },
            metadata={
                "weights": str(FFA_CLS_WEIGHTS),
                "backbone": BACKBONE_NAME,
                "input_size": IMG_SIZE,
                "training_epoch": int(self._train_summary.get("epoch", -1)),
                "val_raw_auc_macro": self._train_summary.get("raw_auc_macro"),
                "val_merged_auc_macro": self._train_summary.get("merged_auc_macro"),
            },
        )
