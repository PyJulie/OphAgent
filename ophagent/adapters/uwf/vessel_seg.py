"""
Adapter: UWF retinal vessel segmentation (PRIME-FP20, U-Net + ResNet34).

Wraps an external PRIME-FP20 vessel-segmentation checkpoint. Configure the
project with ``OPHAGENT_PRIME_FP20_SRC`` or the checkpoint with
``OPHAGENT_PRIME_FP20_WEIGHTS``.
Uses sliding-window inference (256×256 patches, stride 128) — handles
arbitrarily large UWF images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


PRIME_SRC = external_dir("OPHAGENT_PRIME_FP20_SRC", "prime_fp20")
CLI_DIR = PRIME_SRC / "release" / "vessel_seg_cli"
CKPT_PATH = checkpoint_file("OPHAGENT_PRIME_FP20_WEIGHTS", "uwf", "vessel_seg.pth")


@register
class UWFVesselSegAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="uwf_vessel_segmentation",
        modality="UWF",
        task="segmentation",
        description=(
            "Retinal vessel segmentation on an Ultra-Wide-Field fundus image "
            "(U-Net + ResNet34, sliding-window 256×256 patches). Returns a "
            "binary vessel mask, probability map, and a global vessel-area "
            "ratio (% of FOV covered by vessels)."
        ),
        input_size=(256, 256),
        labels=["vessel"],
        confidence_threshold=0.0,
        limitations=[
            "Sliding-window inference; ~10-30s on 4K UWF",
            "Threshold defaulted to 0.5 — fine vessels at the periphery may be missed",
        ],
        cost_class="slow",
        source_dir=str(PRIME_SRC),
    )

    def _load_impl(self) -> None:
        import segmentation_models_pytorch as smp
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                "PRIME-FP20 checkpoint not found. Set OPHAGENT_PRIME_FP20_WEIGHTS "
                f"or OPHAGENT_PRIME_FP20_SRC. Expected: {CKPT_PATH}"
            )
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        model.load_state_dict(
            torch.load(str(CKPT_PATH), map_location=self.device, weights_only=True)
        )
        model.to(self.device).eval()
        self._impl = model
        # patch params
        self._patch = 256
        self._stride = 128

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """R + G + CLAHE(G) — matches PRIME-FP20 training."""
        if img.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return np.stack([img, img, clahe.apply(img)], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        r, g = img[:, :, 0], img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return np.stack([r, g, clahe.apply(g)], axis=-1)

    def _fov_mask(self, image_3ch: np.ndarray) -> np.ndarray:
        g = image_3ch[:, :, 1]
        mask = (g > 10).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _render_overlay(original_rgb: np.ndarray, mask: np.ndarray,
                        out_dir: Path, stem: str) -> dict[str, str]:
        """Composite the vessel mask (bright green) over the original image and
        also save the raw binary mask. Returns {label: abs_path}."""
        out_dir.mkdir(parents=True, exist_ok=True)
        H, W = mask.shape[:2]
        base = original_rgb
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
        if base.shape[2] == 4:
            base = base[:, :, :3]
        if base.shape[:2] != (H, W):
            base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)
        base_bgr = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Overlay: dim the fundus, paint vessels bright green.
        overlay = (base_bgr.astype(np.float32) * 0.6).astype(np.uint8)
        vessel = mask.astype(bool)
        overlay[vessel] = (0, 255, 0)            # BGR green
        comp = cv2.addWeighted(base_bgr, 0.5, overlay, 0.5, 0)
        comp[vessel] = (0, 255, 0)

        # White-on-black binary mask, too.
        mask_img = (mask.astype(np.uint8) * 255)

        out: dict[str, str] = {}
        p_over = out_dir / f"{stem}_vessel_overlay.png"
        p_mask = out_dir / f"{stem}_vessel_mask.png"
        cv2.imwrite(str(p_over), comp)
        cv2.imwrite(str(p_mask), mask_img)
        out["vessel_overlay"] = str(p_over)
        out["vessel_mask"] = str(p_mask)
        return out

    @torch.no_grad()
    def _predict_impl(self, image_path: str, output_dir: str | None = None,
                      **_) -> AdapterResult:
        pil = Image.open(image_path).convert("RGB")
        arr = np.asarray(pil)
        img3 = self._preprocess(arr)
        fov = self._fov_mask(img3)

        h, w = img3.shape[:2]
        ps, st = self._patch, self._stride
        pred_sum = np.zeros((h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.float64)

        # iterate patches (no albumentations dep here — just float32/255)
        for y in range(0, max(1, h - ps + 1), st):
            for x in range(0, max(1, w - ps + 1), st):
                y0, x0 = min(y, h - ps), min(x, w - ps)
                patch = img3[y0:y0 + ps, x0:x0 + ps].astype(np.float32) / 255.0
                t = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                p = torch.sigmoid(self._impl(t)).squeeze().cpu().numpy()
                pred_sum[y0:y0 + ps, x0:x0 + ps] += p
                count[y0:y0 + ps, x0:x0 + ps] += 1

        count = np.maximum(count, 1)
        prob = (pred_sum / count) * fov
        mask = (prob >= 0.5).astype(np.uint8)
        fov_px = float(fov.sum())
        vessel_ratio = float(mask.sum() / fov_px) if fov_px > 0 else 0.0

        # Render a viewable overlay + binary mask so the UI can show the
        # segmentation (the user asks "分割血管给我看看"). Without a persistent
        # output_dir there is nowhere web-served to write, so we skip.
        figures: dict[str, str] = {}
        if output_dir:
            try:
                figures = self._render_overlay(
                    arr, mask, Path(output_dir) / f"vessel_{Path(image_path).stem}",
                    Path(image_path).stem,
                )
            except Exception as _e:
                figures = {"error": f"overlay rendering failed: {_e}"}

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="UWF",
            task="segmentation",
            predictions={
                "vessel_ratio": vessel_ratio,
                "vessel_pixel_count": int(mask.sum()),
                "fov_pixel_count": int(fov_px),
                "mask_shape": [h, w],
            },
            confidence=None,
            figures={k: v for k, v in figures.items() if k != "error"},
            raw_output={"prob": prob, "mask": mask, "fov": fov},
            metadata={"patch_size": ps, "stride": st},
        )
