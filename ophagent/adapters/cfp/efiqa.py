"""Explainable fundus-image quality assessment with anatomical priors.

The inference glue ships with OphAgent. The small EFIQA adapter parameters and
the gated DINOv3 backbone remain external model assets.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ...utils.paths import checkpoint_file
from ..base import AdapterBase, AdapterResult, ToolMetadata, register


ADAPTER_WEIGHTS_FILE = checkpoint_file(
    "OPHAGENT_EFIQA_WEIGHTS",
    "cfp",
    "efiqa_adapter.npz",
)
DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@register
class EFIQAAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_efiqa",
        modality="CFP",
        task="quality",
        description=(
            "Lesion-aware fundus image quality assessment using EFIQA. The "
            "anatomical-integrity quality map reports global degradation and "
            "the proportion of usable retinal area."
        ),
        input_size=(224, 224),
        labels=["Good", "Usable", "Reject"],
        confidence_threshold=0.0,
        limitations=[
            "Requires access to the gated DINOv3 model on Hugging Face",
            "The first uncached call downloads the DINOv3 backbone",
            "The 14 by 14 patch grid provides a coarse spatial quality map",
        ],
        cost_class="medium",
        source_dir="ophagent.adapters.cfp.efiqa",
    )

    def _load_impl(self) -> None:
        from transformers import AutoModel

        try:
            self._dino = AutoModel.from_pretrained(
                DINO_MODEL_ID,
                revision=DINO_REVISION,
            )
        except Exception as exc:
            raise RuntimeError(
                f"DINOv3 backbone load failed: {exc}. The model is gated; "
                "authenticate with Hugging Face before retrying."
            ) from exc
        self._dino.to(self.device).eval()
        self._num_reg = getattr(self._dino.config, "num_register_tokens", 0)

        if not ADAPTER_WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                "EFIQA adapter weights not found. Set OPHAGENT_EFIQA_WEIGHTS "
                f"or place the asset at: {ADAPTER_WEIGHTS_FILE}"
            )
        with np.load(ADAPTER_WEIGHTS_FILE, allow_pickle=False) as weights:
            self._W = np.asarray(weights["W"], dtype=np.float32)
            self._b = float(np.asarray(weights["b"]).item())
        if self._W.shape != (1024,):
            raise ValueError(
                "EFIQA adapter weight W must have shape (1024,), "
                f"found {self._W.shape}"
            )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        output = self._dino(self._preprocess(image_path))
        features = output.last_hidden_state[0, 1 + self._num_reg :, :].cpu().numpy()
        features = features.reshape(14, 14, -1).astype(np.float32)
        logits = features @ self._W + self._b
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        mean_degradation = float(probabilities.mean())
        max_degradation = float(probabilities.max())
        usable_ratio = float((probabilities < 0.5).mean())
        if mean_degradation < 0.20:
            quality = "Good"
        elif mean_degradation < 0.45:
            quality = "Usable"
        else:
            quality = "Reject"

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="quality",
            predictions={
                "quality": quality,
                "mean_degradation": mean_degradation,
                "max_degradation": max_degradation,
                "usable_area_ratio": usable_ratio,
                "is_usable": quality != "Reject",
                "is_rejected": quality == "Reject",
                "spatial_map_shape": list(probabilities.shape),
            },
            confidence=1.0 - mean_degradation,
            raw_output={"degradation_map": probabilities.tolist()},
            metadata={
                "backbone": DINO_MODEL_ID,
                "backbone_revision": DINO_REVISION,
                "adapter_params": int(self._W.size + 1),
            },
        )
