"""
Adapter: FMUE OCT 16-class multi-pathology classifier (ViT-L/16 + LoRA + evidential).

Wraps an external FMUE checkpoint. Configure the project with
``OPHAGENT_FMUE_SRC`` or the checkpoint with ``OPHAGENT_FMUE_WEIGHTS``.
16 classes: Normal, dAMD, nAMD, PCV, DME, DR without ME, iERM, iMH, MTM,
mCNV, RD, acute CSC, acute RAO, acute RVO, acute VKH, RP.

Uses evidential deep-learning output: softplus → α evidence, predictions
are E/S (Dirichlet mean), confidence = max probability.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


FMUE_SRC = external_dir("OPHAGENT_FMUE_SRC", "fmue")
CKPT_PATH = checkpoint_file("OPHAGENT_FMUE_WEIGHTS", "oct", "fmue.pth")

LABELS = [
    "Normal", "dAMD", "nAMD", "PCV", "DME", "DR_without_ME", "iERM", "iMH",
    "MTM", "mCNV", "RD", "acute_CSC", "acute_RAO", "acute_RVO", "acute_VKH", "RP",
]
LABEL_DESCRIPTIONS = {
    "Normal": "Normal retina",
    "dAMD": "Dry age-related macular degeneration",
    "nAMD": "Neovascular (wet) age-related macular degeneration",
    "PCV": "Polypoidal choroidal vasculopathy",
    "DME": "Diabetic macular edema",
    "DR_without_ME": "Diabetic retinopathy without macular edema",
    "iERM": "Idiopathic epiretinal membrane",
    "iMH": "Idiopathic macular hole",
    "MTM": "Myopic traction maculopathy",
    "mCNV": "Myopic choroidal neovascularisation",
    "RD": "Retinal detachment",
    "acute_CSC": "Acute central serous chorioretinopathy",
    "acute_RAO": "Acute retinal artery occlusion",
    "acute_RVO": "Acute retinal vein occlusion",
    "acute_VKH": "Acute Vogt-Koyanagi-Harada disease",
    "RP": "Retinitis pigmentosa",
}


@register
class FMUEAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="oct_fmue_16class",
        modality="OCT",
        task="classification",
        description=(
            "OCT B-scan 16-class disease classifier (FMUE: ViT-Large/16 with "
            "LoRA + evidential learning). Covers AMD subtypes (dAMD, nAMD, "
            "PCV), DME, DR, ERM, MH, myopic disease, retinal detachment, and "
            "acute conditions (CSC, RAO, RVO, VKH). The evidential head also "
            "gives a per-class uncertainty bound."
        ),
        input_size=(224, 224),
        labels=LABELS,
        confidence_threshold=0.50,
        limitations=[
            "Acute conditions and rare diseases (PCV, VKH) labelled from a "
            "single Asian cohort — out-of-distribution accuracy may vary",
            "Requires foveal-centred B-scans; peripheral scans bias toward Normal",
        ],
        cost_class="medium",
        source_dir=str(FMUE_SRC),
    )

    def _load_impl(self) -> None:
        # The OCTCubeM adapter (OCTCube/) also ships a top-level `util` package.
        # If it loaded first its `util.*` is cached in sys.modules and would
        # shadow FMUE's own `util.pos_embed`. Drop any stale `util` modules and
        # put FMUE's source at the front of sys.path so the import below
        # resolves against THIS tree, load-order-independent.
        import sys
        for _name in [m for m in list(sys.modules)
                      if m == "util" or m.startswith("util.")]:
            del sys.modules[_name]
        _src = str(FMUE_SRC)
        while _src in sys.path:
            sys.path.remove(_src)
        sys.path.insert(0, _src)
        import vit_model  # type: ignore  # from FMUE source
        from util.pos_embed import interpolate_pos_embed  # type: ignore

        model = vit_model.__dict__["vit_large_patch16"](
            img_size=224,
            num_classes=len(LABELS),
            drop_path_rate=0.1,
            global_pool=True,
        )
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                "FMUE checkpoint not found. Set OPHAGENT_FMUE_WEIGHTS or "
                f"OPHAGENT_FMUE_SRC. Expected: {CKPT_PATH}"
            )
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(
                str(CKPT_PATH), map_location="cpu", weights_only=True
            )
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # drop head if shape mismatch
        for k in ["head.weight", "head.bias"]:
            if k in state and state[k].shape != model.state_dict()[k].shape:
                del state[k]
        interpolate_pos_embed(model, state)
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        # New timm's forward_head double-pools when global_pool is a bool;
        # call forward_features + head ourselves to avoid the LayerNorm shape mismatch.
        orig_features = model.forward_features
        orig_head = model.head

        def _forward(x):
            feats = orig_features(x)   # custom forward_features already pooled
            return orig_head(feats)
        model.forward = _forward
        self._impl = model
        self._transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        img = Image.open(image_path).convert("RGB")
        x = self._transform(img).unsqueeze(0).to(self.device)
        out = self._impl(x)
        # Evidential learning: alpha = softplus(out) + 1; probs = (alpha-1)/S
        evidence = F.softplus(out)
        alpha = evidence + 1
        S = alpha.sum(dim=1, keepdim=True)
        probs = (alpha - 1) / S
        probs_np = probs[0].cpu().float().numpy()
        # Uncertainty: K / S (lower = more confident)
        uncertainty = float((LABELS.__len__() / S[0]).item())

        idxs = list(range(len(LABELS)))
        idxs.sort(key=lambda i: -probs_np[i])
        top_idx = idxs[0]

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="classification",
            predictions={
                "predicted_class": LABELS[top_idx],
                "description": LABEL_DESCRIPTIONS[LABELS[top_idx]],
                "probabilities": {
                    LABELS[i]: float(probs_np[i]) for i in idxs[:8]
                },
                "top_3": [
                    {"label": LABELS[i], "probability": float(probs_np[i]),
                     "description": LABEL_DESCRIPTIONS[LABELS[i]]}
                    for i in idxs[:3]
                ],
                "evidential_uncertainty": uncertainty,
            },
            confidence=float(probs_np[top_idx]),
        )

    # ── 2D-on-3D support ────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_arrays(self, images: list, batch_size: int = 16) -> list[dict]:
        """Classify a LIST of 2D B-scan arrays (numpy, any dtype/shape HxW or
        HxWx{1,3}) — the per-slice entry point for running this 2D model over a
        3D volume (see ophagent.agent.volume_processor.analyze_volume with
        classifier_adapter). Reuses the SAME loaded model, transform and
        evidential decode as predict(); just skips PIL file I/O. Returns one
        dict per image: {predicted_class, confidence, probabilities{16},
        evidential_uncertainty} — matching what analyze_volume expects.
        """
        import numpy as np
        if not self._loaded:
            self.load()
        rows: list[dict] = []
        buf: list = []

        def _flush(batch):
            if not batch:
                return
            x = torch.stack(batch).to(self.device)
            out = self._impl(x)
            evidence = F.softplus(out)
            alpha = evidence + 1
            S = alpha.sum(dim=1, keepdim=True)
            probs = ((alpha - 1) / S).cpu().float().numpy()
            unc = (len(LABELS) / S).squeeze(1).cpu().float().numpy()
            for i in range(probs.shape[0]):
                p = probs[i]
                top = int(np.argmax(p))
                rows.append({
                    "predicted_class": LABELS[top],
                    "confidence": float(p[top]),
                    "probabilities": {LABELS[j]: float(p[j]) for j in range(len(LABELS))},
                    "evidential_uncertainty": float(unc[i]),
                })

        for arr in images:
            a = np.asarray(arr)
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            elif a.ndim == 3 and a.shape[2] == 1:
                a = np.repeat(a, 3, axis=2)
            if a.dtype != np.uint8:
                lo, hi = float(a.min()), float(a.max())
                a = ((a.astype(np.float32) - lo) / (hi - lo) * 255.0) if hi > lo else a.astype(np.float32)
                a = np.clip(a, 0, 255).astype(np.uint8)
            buf.append(self._transform(Image.fromarray(a)))
            if len(buf) >= batch_size:
                _flush(buf); buf = []
        _flush(buf)
        return rows

    def predict_array(self, image) -> dict:
        """Single-slice convenience wrapper around predict_arrays."""
        return self.predict_arrays([image])[0]
