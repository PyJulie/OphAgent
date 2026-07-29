"""
Adapters: Paired_600-trained 5-class classifiers.

These wrap the models trained by `scripts/train_paired_cfp_ffa.py`:
  - `paired_cfp_5class`  — CFP-only ResNet50, val acc ≈ 73.9%
  - `paired_ffa_5class`  — FFA-only ResNet50 (FFAIR-pretrained backbone),
                            val acc ≈ 89.1%
  - `paired_joint_5class` — late-feature fusion of the two backbones,
                            val acc TBD (training in progress)

Classes: Normal / DR / RVO / AMD / CSC
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file


CFP_CKPT = checkpoint_file(
    "OPHAGENT_PAIRED_CFP_WEIGHTS", "paired", "cfp", "best.pt"
)
FFA_CKPT = checkpoint_file(
    "OPHAGENT_PAIRED_FFA_WEIGHTS", "paired", "ffa", "best.pt"
)
JOINT_CKPT = checkpoint_file(
    "OPHAGENT_PAIRED_JOINT_WEIGHTS", "paired", "joint", "best.pt"
)
CLASSES = ["Normal", "DR", "RVO", "AMD", "CSC"]
IMG_SIZE = 448

_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def _build_resnet50_with_head(num_classes: int = 5) -> nn.Module:
    import timm
    bb = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="avg")
    head = nn.Linear(bb.num_features, num_classes)
    return nn.Sequential(bb, head)


def _preprocess(image_path: str, device: str, tta: bool = False) -> torch.Tensor:
    """Return a batch tensor on `device`. If `tta`, returns [original, hflip]
    stacked (B=2) for test-time augmentation."""
    img = Image.open(image_path).convert("RGB")
    t = _TF(img).unsqueeze(0)
    if tta:
        t = torch.cat([t, torch.flip(t, dims=(3,))], dim=0)   # add hflip
    return t.to(device)


def _softmax_logits(impl: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Softmax logits, averaged over batch (TTA)."""
    logits = impl(x)
    probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
    return probs


def _ranked_probs(probs: np.ndarray) -> list[dict]:
    idx = np.argsort(-probs)
    return [{"label": CLASSES[i], "probability": float(probs[i])} for i in idx]


def _make_predictions(probs: np.ndarray) -> dict:
    ranked = _ranked_probs(probs)
    return {
        "primary_diagnosis": ranked[0]["label"],
        "primary_probability": ranked[0]["probability"],
        "all_probabilities": {r["label"]: r["probability"] for r in ranked},
        "ranked": ranked,
        "top_3": ranked[:3],
    }


# ════════════════════════════════════════════════════════════════════════════
# 1. CFP-only 5-class
# ════════════════════════════════════════════════════════════════════════════
@register
class PairedCFP5ClassAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_paired5",
        modality="CFP",
        task="classification",
        description=(
            "5-class CFP disease classifier (Normal / DR / RVO / AMD / CSC) "
            "trained on the Paired_600_CFPFFA_MPOS dataset. ResNet-50 backbone, "
            "stratified 80/20 split, weighted sampling. Validation accuracy "
            "≈ 73.9%. Use when the user wants a direct 5-way verdict on a "
            "single CFP image; combine with `ffa_paired5` if an FFA is "
            "available for the same eye (use `cross_cfp_ffa_paired` for fused "
            "inference)."
        ),
        input_size=(IMG_SIZE, IMG_SIZE),
        labels=CLASSES,
        confidence_threshold=0.5,
        limitations=[
            "Only 5 classes — anything outside this set will be forced into "
            "the closest match.",
            "Per-class val accuracy: Normal 55%, DR 80%, RVO 85%, AMD 67%, "
            "CSC 68% (data-limited on Normal: only 11 val samples).",
        ],
        cost_class="fast",
        source_dir=str(CFP_CKPT.parent),
    )

    def _load_impl(self) -> None:
        ckpt_path = CFP_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Paired CFP weights missing: {ckpt_path}")
        m = _build_resnet50_with_head(5)
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model"], strict=True)
        self._impl = m.to(self.device).eval()
        self._val_acc = float(ck.get("val_acc", 0))

    @torch.no_grad()
    def _predict_impl(self, image_path: str, tta: bool = True, **_) -> AdapterResult:
        x = _preprocess(image_path, self.device, tta=tta)
        probs = _softmax_logits(self._impl, x)
        pred = _make_predictions(probs)
        return AdapterResult(
            success=True, tool=self.metadata.name, modality="CFP",
            task="classification",
            predictions=pred,
            confidence=pred["primary_probability"],
            metadata={"val_acc": self._val_acc, "input_size": IMG_SIZE, "tta": tta},
        )


# ════════════════════════════════════════════════════════════════════════════
# 2. FFA-only 5-class
# ════════════════════════════════════════════════════════════════════════════
@register
class PairedFFA5ClassAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="ffa_paired5",
        modality="FFA",
        task="classification",
        description=(
            "5-class FFA disease classifier (Normal / DR / RVO / AMD / CSC). "
            "ResNet-50 backbone *initialised from the FFAIR multi-label "
            "classifier* and fine-tuned on Paired_600. Validation accuracy "
            "≈ 89.1% — substantially better than the FFAIR-9-class projection "
            "baseline (41% on this benchmark). Prefer this over "
            "`ffa_classification` when the differential is among these 5 "
            "diseases."
        ),
        input_size=(IMG_SIZE, IMG_SIZE),
        labels=CLASSES,
        confidence_threshold=0.5,
        limitations=[
            "Only 5 classes; smaller differentials (e.g. PCV vs nAMD) collapse "
            "into AMD. Use `ffa_classification` for a broader 9-class view.",
            "Per-class val accuracy: Normal 91%, DR 97%, RVO 89%, AMD 81%, "
            "CSC 84%.",
        ],
        cost_class="fast",
        source_dir=str(FFA_CKPT.parent),
    )

    def _load_impl(self) -> None:
        ckpt_path = FFA_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Paired FFA weights missing: {ckpt_path}")
        m = _build_resnet50_with_head(5)
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model"], strict=True)
        self._impl = m.to(self.device).eval()
        self._val_acc = float(ck.get("val_acc", 0))

    @torch.no_grad()
    def _predict_impl(self, image_path: str, tta: bool = True, **_) -> AdapterResult:
        x = _preprocess(image_path, self.device, tta=tta)
        probs = _softmax_logits(self._impl, x)
        pred = _make_predictions(probs)
        return AdapterResult(
            success=True, tool=self.metadata.name, modality="FFA",
            task="classification",
            predictions=pred,
            confidence=pred["primary_probability"],
            metadata={"val_acc": self._val_acc, "input_size": IMG_SIZE, "tta": tta},
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. Joint CFP+FFA fusion (5-class)
# ════════════════════════════════════════════════════════════════════════════
class _JointFusion(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.cfp_bb = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="avg")
        self.ffa_bb = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.cfp_bb.num_features + self.ffa_bb.num_features
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 5),
        )

    def forward(self, x_cfp, x_ffa):
        f1 = self.cfp_bb(x_cfp)
        f2 = self.ffa_bb(x_ffa)
        return self.fuse(torch.cat([f1, f2], dim=1))


@register
class PairedSoftVoteAdapter(AdapterBase):
    """Convenience composite: soft-vote ensemble of `cfp_paired5` +
    `ffa_paired5`. No extra training required — averages the two posteriors.

    Empirically the single best paired-modality strategy: 96.6% val acc on
    Paired_600, vs 90.8% (FFA alone) and 85.7% (CFP alone). Use this whenever
    a paired CFP+FFA is available *unless* the learned joint fusion (which
    requires explicit training) has finished and is wired."""

    metadata = ToolMetadata(
        name="cross_cfp_ffa_softvote",
        modality="multi",
        task="classification",
        description=(
            "Soft-vote ensemble of `cfp_paired5` and `ffa_paired5` for paired "
            "CFP+FFA inputs. Averages the two posteriors. 96.6% val accuracy "
            "on Paired_600 — beats either single modality by ~7-10 pts and "
            "the GPT-5 baseline by ~42 pts. Use this for any paired-modality "
            "5-class question."
        ),
        input_size=(IMG_SIZE, IMG_SIZE),
        labels=CLASSES,
        confidence_threshold=0.5,
        limitations=[
            "Requires BOTH a CFP and an FFA path (kwargs: image_path, ffa_path).",
            "Restricted to 5 classes (Normal/DR/RVO/AMD/CSC).",
        ],
        cost_class="medium",
        source_dir="(composite)",
    )

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, ffa_path: str = "", **_) -> AdapterResult:
        from ..base import GLOBAL_REGISTRY
        if not ffa_path:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="classification",
                error="cross_cfp_ffa_softvote requires both image_path (CFP) and ffa_path",
            )
        rc = GLOBAL_REGISTRY.predict("cfp_paired5", image_path)
        rf = GLOBAL_REGISTRY.predict("ffa_paired5", ffa_path)
        if not (rc.success and rf.success):
            err = rc.error or rf.error
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="classification", error=err,
            )
        pc = rc.predictions["all_probabilities"]
        pf = rf.predictions["all_probabilities"]
        fused = {c: 0.5 * pc.get(c, 0) + 0.5 * pf.get(c, 0) for c in CLASSES}
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        primary, prob = ranked[0]
        return AdapterResult(
            success=True, tool=self.metadata.name, modality="multi",
            task="classification",
            predictions={
                "primary_diagnosis": primary,
                "primary_probability": float(prob),
                "all_probabilities": {c: float(p) for c, p in ranked},
                "ranked": [{"label": c, "probability": float(p)} for c, p in ranked],
                "cfp_top1": rc.predictions.get("primary_diagnosis"),
                "ffa_top1": rf.predictions.get("primary_diagnosis"),
                "conflict_flag": rc.predictions.get("primary_diagnosis") !=
                                 rf.predictions.get("primary_diagnosis"),
            },
            confidence=float(prob),
            raw_output={"cfp": rc.to_jsonable(), "ffa": rf.to_jsonable()},
        )


@register
class PairedJointAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cross_cfp_ffa_paired",
        modality="multi",
        task="classification",
        description=(
            "Joint CFP+FFA 5-class classifier (Normal / DR / RVO / AMD / CSC) "
            "trained on Paired_600. Late-feature fusion of two ResNet-50 "
            "backbones (one CFP, one FFA), 512-d MLP head. Takes both images "
            "in one pass — use when you have paired CFP+FFA of the same eye "
            "and want the best single-model verdict (replaces the heuristic "
            "weighted-fuse `cross_cfp_ffa`)."
        ),
        input_size=(IMG_SIZE, IMG_SIZE),
        labels=CLASSES,
        confidence_threshold=0.5,
        limitations=[
            "Requires BOTH a CFP and an FFA path (kwargs: image_path, ffa_path).",
            "Trained on Paired_600 (only 600 image pairs, 5 classes); "
            "out-of-distribution diseases will be misclassified.",
        ],
        cost_class="medium",
        source_dir=str(JOINT_CKPT.parent),
    )

    def _load_impl(self) -> None:
        ckpt_path = JOINT_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Joint fusion weights missing: {ckpt_path}. Run "
                "`python scripts/train_paired_cfp_ffa.py --mode joint` first."
            )
        m = _JointFusion()
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model"], strict=True)
        self._impl = m.to(self.device).eval()
        self._val_acc = float(ck.get("val_acc", 0))

    @torch.no_grad()
    def _predict_impl(self, image_path: str, ffa_path: str = "", **_) -> AdapterResult:
        if not ffa_path:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="classification",
                error="cross_cfp_ffa_paired requires both image_path (CFP) and ffa_path",
            )
        x_cfp = _preprocess(image_path, self.device)
        x_ffa = _preprocess(ffa_path, self.device)
        logits = self._impl(x_cfp, x_ffa)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        pred = _make_predictions(probs)
        return AdapterResult(
            success=True, tool=self.metadata.name, modality="multi",
            task="classification",
            predictions=pred,
            confidence=pred["primary_probability"],
            metadata={"val_acc": self._val_acc, "input_size": IMG_SIZE},
        )
