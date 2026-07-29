"""
Adapters: OCT fluid + retinal layer segmentation, and OCT image quality.

Wraps the three trained UNet/ResNet models in `checkpoints/`:
  - `fluid_segmentor/best.pt`   — UNet ResNet50, 4 classes  (BG/IRF/SRF/PED),
                                  val Dice ≈ 0.90 on RETOUCH multi-device data
  - `layer_segmentor/best.pt`   — UNet ResNet50, 10 classes (10 retinal regions),
                                  val Dice ≈ 0.88 on Duke DME (Chiu 2015)
  - `quality_assessor/best.pt`  — ResNet18 classifier, 2 classes (high/low),
                                  val acc ≈ 0.995 on synthetic-degraded OCT

Preprocessing follows the training pipeline exactly:
  Resize → Normalize(mean=0.5, std=0.5) → CHW float32 tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, output_path


FLUID_CKPT = checkpoint_file(
    "OPHAGENT_OCT_FLUID_WEIGHTS", "oct", "fluid_segmentor", "best.pt"
)
LAYER_CKPT = checkpoint_file(
    "OPHAGENT_OCT_LAYER_WEIGHTS", "oct", "layer_segmentor", "best.pt"
)
QUAL_CKPT = checkpoint_file(
    "OPHAGENT_OCT_QUALITY_WEIGHTS", "oct", "quality_assessor", "best.pt"
)

FLUID_CLASSES = ["Background", "IRF", "SRF", "PED"]
# Region names match `checkpoints/registry.json` — 10-region scheme from the
# Duke DME (Chiu 2015) layer-segmentation training.
LAYER_REGIONS = [
    "BG-above",  "ILM-NFL",  "NFL-IPL",  "IPL-INL",  "INL-OPL",
    "OPL-ONL",   "ONL-ISM",  "ISM-ISOS", "ISOS-RPE", "Below-RPE",
]
QUALITY_LABELS = ["high", "low"]


# ── shared utility: image → tensor (grayscale → 3ch, mean=0.5/std=0.5) ──────
def _preprocess(image_path: str, size: int, device: str) -> tuple[torch.Tensor, Image.Image]:
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t, img


@torch.no_grad()
def _predict_seg_tta(impl: torch.nn.Module, image_path: str,
                     primary_size: int, device: str,
                     tta_hflip: bool = True,
                     multi_scale: tuple[int, ...] = (384, 512)) -> tuple[np.ndarray, Image.Image]:
    """Run segmentation with horizontal-flip TTA + multi-scale averaging.
    Returns (mask at primary_size, original PIL image at primary_size)."""
    orig_pil = Image.open(image_path).convert("RGB").resize(
        (primary_size, primary_size), Image.BICUBIC)
    accum: torch.Tensor | None = None
    for s in multi_scale:
        x, _ = _preprocess(image_path, s, device)
        prob = impl(x)["probabilities"]                # (1, C, s, s)
        prob = torch.nn.functional.interpolate(
            prob, size=(primary_size, primary_size),
            mode="bilinear", align_corners=False,
        )
        accum = prob if accum is None else accum + prob
        if tta_hflip:
            x_flip = torch.flip(x, dims=(3,))
            prob_f = impl(x_flip)["probabilities"]
            prob_f = torch.flip(prob_f, dims=(3,))
            prob_f = torch.nn.functional.interpolate(
                prob_f, size=(primary_size, primary_size),
                mode="bilinear", align_corners=False,
            )
            accum = accum + prob_f
    mask = accum[0].argmax(dim=0).cpu().numpy().astype(np.int32)
    return mask, orig_pil


def _seg_overlay(orig: Image.Image, mask: np.ndarray, palette: list[tuple[int, int, int]],
                 alpha: float = 0.45) -> Image.Image:
    """Blend a class-index mask on top of the original image."""
    h, w = mask.shape
    base = np.asarray(orig.resize((w, h), Image.BICUBIC), dtype=np.float32)
    overlay = np.zeros_like(base)
    for cls_idx, color in enumerate(palette):
        if cls_idx == 0:
            continue   # background not coloured
        sel = (mask == cls_idx)
        if not sel.any():
            continue
        overlay[sel] = color
    blended = np.clip(base * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ════════════════════════════════════════════════════════════════════════════
# 1. OCT fluid segmentor
# ════════════════════════════════════════════════════════════════════════════
@register
class OCTFluidSegAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="oct_fluid_segmentation",
        modality="OCT",
        task="segmentation",
        description=(
            "Retinal fluid segmentation on an OCT B-scan. UNet (ResNet-50 + "
            "attention gates) trained on RETOUCH multi-device data; segments "
            "intra-retinal fluid (IRF), sub-retinal fluid (SRF), and "
            "pigment-epithelial detachment (PED). Returns per-class pixel area "
            "ratios, an overlay PNG, and a per-class bounding-box list. Use "
            "when the user has an OCT B-scan and asks about edema, fluid "
            "pockets, or wants to quantify exudation."
        ),
        input_size=(384, 384),
        labels=FLUID_CLASSES,
        confidence_threshold=0.0,         # area metrics, not a class probability
        limitations=[
            "Trained on RETOUCH (Spectralis/Cirrus/Topcon) — atypical device "
            "scans may need rescaling.",
            "Best results on fovea-centred B-scans; peripheral scans are noisier.",
            "Validation Dice ≈ 0.90 macro across IRF/SRF/PED.",
        ],
        cost_class="medium",
        source_dir=str(FLUID_CKPT.parent),
    )

    INPUT_SIZE = 384
    PALETTE = [(0, 0, 0), (231, 76, 60), (52, 152, 219), (241, 196, 15)]  # BG/IRF/SRF/PED

    def _load_impl(self) -> None:
        from ...models.segmentation.unet import OCTFluidSegmentor
        if not FLUID_CKPT.exists():
            raise FileNotFoundError(f"Fluid segmentor weights missing: {FLUID_CKPT}")
        m = OCTFluidSegmentor(backbone="resnet50", pretrained=False)
        ck = torch.load(str(FLUID_CKPT), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model_state_dict"], strict=True)
        self._impl = m.to(self.device).eval()
        self._best_dice = float(ck.get("best_metric", 0))

    @torch.no_grad()
    def _predict_impl(self, image_path: str, save_overlay: bool = True,
                      tta: bool = True, **_) -> AdapterResult:
        if tta:
            mask, orig = _predict_seg_tta(
                self._impl, image_path, self.INPUT_SIZE, self.device,
                tta_hflip=True, multi_scale=(384, 512),
            )
        else:
            x, orig = _preprocess(image_path, self.INPUT_SIZE, self.device)
            out = self._impl(x)
            mask = out["probabilities"][0].argmax(dim=0).cpu().numpy().astype(np.int32)

        n_pix = mask.size
        ratios = {
            cls: float((mask == i).sum()) / n_pix
            for i, cls in enumerate(FLUID_CLASSES)
        }
        total_fluid_ratio = sum(ratios[c] for c in ["IRF", "SRF", "PED"])

        # per-class bbox (simple: convex bounding rect of the class)
        bboxes = {}
        for i, cls in enumerate(FLUID_CLASSES):
            if i == 0 or ratios[cls] < 1e-4:
                continue
            ys, xs = np.where(mask == i)
            bboxes[cls] = [int(xs.min()), int(ys.min()),
                           int(xs.max()), int(ys.max())]

        figures: dict[str, str] = {}
        # Always save the raw class-index mask as .npy so downstream
        # `compute(code)` calls can reuse it.
        fig_dir = output_path("adapter_figures", self.metadata.name)
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        np.save(fig_dir / f"{stem}_mask.npy", mask.astype(np.int16))
        figures["mask_npy"] = str(fig_dir / f"{stem}_mask.npy")
        if save_overlay:
            overlay = _seg_overlay(orig, mask, self.PALETTE)
            p_overlay = fig_dir / f"{stem}_overlay.png"
            overlay.save(p_overlay)
            figures["overlay"] = str(p_overlay)

        # Primary verdict
        positives = [c for c in ["IRF", "SRF", "PED"] if ratios[c] > 5e-4]
        verdict = "fluid present" if positives else "no fluid detected"

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="segmentation",
            predictions={
                "verdict": verdict,
                "fluid_present": bool(positives),
                "positive_classes": positives,
                "area_ratios": ratios,
                "total_fluid_ratio": total_fluid_ratio,
                "bounding_boxes": bboxes,
                "class_labels": FLUID_CLASSES,
            },
            confidence=float(min(1.0, total_fluid_ratio * 20)) if positives else 1.0,
            figures=figures,
            metadata={"val_dice": self._best_dice, "input_size": self.INPUT_SIZE},
        )


# ════════════════════════════════════════════════════════════════════════════
# 2. OCT retinal layer segmentor
# ════════════════════════════════════════════════════════════════════════════
@register
class OCTLayerSegAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="oct_layer_segmentation",
        modality="OCT",
        task="segmentation",
        description=(
            "Retinal layer segmentation on an OCT B-scan. UNet (ResNet-50) "
            "trained on Duke DME (Chiu 2015) — outputs a 10-region mask: "
            "Above-ILM / NFL / GCL+IPL / INL / OPL / ONL+ELM / IS / OS / RPE / "
            "Below-RPE. Returns per-layer pixel area, average column thickness "
            "(microns-style A-line counts), and an overlay PNG. Use to inspect "
            "RNFL/GCC thinning, drusen elevation of RPE, or to anatomically "
            "locate other findings."
        ),
        input_size=(384, 384),
        labels=LAYER_REGIONS,
        confidence_threshold=0.0,
        limitations=[
            "Trained on Duke DME — heavily distorted retinas (large detachment, "
            "macular hole) can confuse the layer ordering.",
            "Validation Dice ≈ 0.88 macro across 10 regions.",
            "Layer 'thickness' here is in pixel rows (A-line counts); convert "
            "to microns using the device's axial resolution if available.",
        ],
        cost_class="medium",
        source_dir=str(LAYER_CKPT.parent),
    )

    INPUT_SIZE = 384
    # 10 colours for the 10 regions (Above-ILM stays black)
    PALETTE = [
        (0, 0, 0),
        (231,  76,  60),  # NFL
        (230, 126,  34),  # GCL+IPL
        (241, 196,  15),  # INL
        (39,  174,  96),  # OPL
        (26,  188, 156),  # ONL+ELM
        (52,  152, 219),  # IS
        (155,  89, 182),  # OS
        (192,  57,  43),  # RPE
        (127, 140, 141),  # Below-RPE
    ]

    def _load_impl(self) -> None:
        from ...models.segmentation.unet import OCTLayerSegmentor
        if not LAYER_CKPT.exists():
            raise FileNotFoundError(f"Layer segmentor weights missing: {LAYER_CKPT}")
        m = OCTLayerSegmentor(backbone="resnet50", num_layers=len(LAYER_REGIONS), pretrained=False)
        ck = torch.load(str(LAYER_CKPT), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model_state_dict"], strict=True)
        self._impl = m.to(self.device).eval()
        self._best_dice = float(ck.get("best_metric", 0))

    @torch.no_grad()
    def _predict_impl(self, image_path: str, save_overlay: bool = True,
                      tta: bool = True, **_) -> AdapterResult:
        if tta:
            mask, orig = _predict_seg_tta(
                self._impl, image_path, self.INPUT_SIZE, self.device,
                tta_hflip=True, multi_scale=(384, 512),
            )
        else:
            x, orig = _preprocess(image_path, self.INPUT_SIZE, self.device)
            out = self._impl(x)
            mask = out["probabilities"][0].argmax(dim=0).cpu().numpy().astype(np.int32)

        n_pix = mask.size
        ratios = {
            name: float((mask == i).sum()) / n_pix
            for i, name in enumerate(LAYER_REGIONS)
        }

        # Mean column-wise thickness (pixel rows) per layer.
        h, w = mask.shape
        thicknesses: dict[str, float] = {}
        for i, name in enumerate(LAYER_REGIONS):
            if i in (0, len(LAYER_REGIONS) - 1):
                continue            # ignore Above/Below
            col_counts = (mask == i).sum(axis=0)               # (W,) pixel-rows per column
            present_cols = col_counts > 0
            if present_cols.sum() > 0:
                thicknesses[name] = float(col_counts[present_cols].mean())
            else:
                thicknesses[name] = 0.0

        figures: dict[str, str] = {}
        fig_dir = output_path("adapter_figures", self.metadata.name)
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        np.save(fig_dir / f"{stem}_mask.npy", mask.astype(np.int16))
        figures["mask_npy"] = str(fig_dir / f"{stem}_mask.npy")
        if save_overlay:
            overlay = _seg_overlay(orig, mask, self.PALETTE, alpha=0.50)
            p_overlay = fig_dir / f"{stem}_overlay.png"
            overlay.save(p_overlay)
            figures["overlay"] = str(p_overlay)

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="segmentation",
            predictions={
                "region_area_ratios": ratios,
                "mean_column_thickness_px": thicknesses,
                "input_size_px": [self.INPUT_SIZE, self.INPUT_SIZE],
            },
            confidence=1.0,
            figures=figures,
            metadata={"val_dice": self._best_dice, "input_size": self.INPUT_SIZE},
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. OCT quality assessor
# ════════════════════════════════════════════════════════════════════════════
@register
class OCTQualityAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="oct_quality",
        modality="OCT",
        task="quality",
        description=(
            "OCT image quality classifier (binary: high / low). ResNet-18 "
            "trained with synthetic degradations (blur, noise, low-dynamic-range) "
            "on Kermany / OCTDL / OCT-C8. Gate before running downstream OCT "
            "analysis — when 'low', warn the user and rerun acquisition if "
            "possible. Validation accuracy ≈ 0.995."
        ),
        input_size=(224, 224),
        labels=QUALITY_LABELS,
        confidence_threshold=0.5,
        limitations=[
            "Trained on synthetic degradations — real-world artefact types not "
            "in the synthesis bank may fool the model.",
            "Binary only: doesn't tell you *why* an image is low quality.",
        ],
        cost_class="fast",
        source_dir=str(QUAL_CKPT.parent),
    )

    INPUT_SIZE = 224

    def _load_impl(self) -> None:
        from ...models.classification.classifier import OCTClassifier
        if not QUAL_CKPT.exists():
            raise FileNotFoundError(f"Quality weights missing: {QUAL_CKPT}")
        m = OCTClassifier(backbone="resnet18", num_classes=2, pretrained=False)
        ck = torch.load(str(QUAL_CKPT), map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model_state_dict"], strict=True)
        self._impl = m.to(self.device).eval()
        self._best_acc = float(ck.get("best_metric", 0))

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        x, _ = _preprocess(image_path, self.INPUT_SIZE, self.device)
        out = self._impl(x)
        probs = out["probabilities"][0].cpu().numpy()
        idx = int(probs.argmax())
        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="quality",
            predictions={
                "quality": QUALITY_LABELS[idx],
                "is_usable": QUALITY_LABELS[idx] == "high",
                "probabilities": {
                    lbl: float(probs[i]) for i, lbl in enumerate(QUALITY_LABELS)
                },
            },
            confidence=float(probs[idx]),
            metadata={"val_acc": self._best_acc, "input_size": self.INPUT_SIZE},
        )
