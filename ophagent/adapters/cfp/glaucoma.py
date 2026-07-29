"""Glaucoma classification on an optic-disc crop.

The inference architecture ships with OphAgent. Model weights remain external
and are configured with ``OPHAGENT_GLAUCOMA_WEIGHTS``.
"""

from __future__ import annotations

import torch

from ...components.glaucoma import GlaucomaModel, build_eval_transform
from ...utils.paths import checkpoint_file
from ..base import AdapterBase, AdapterResult, ToolMetadata, register


CKPT_PATH = checkpoint_file("OPHAGENT_GLAUCOMA_WEIGHTS", "cfp", "glaucoma.pth")
INPUT_SIZE = 392
AUX_LABELS = [
    "ANRS",
    "ANRI",
    "RNFLDS",
    "RNFLDI",
    "BCLVS",
    "BCLVI",
    "NVT",
    "DH",
    "LD",
    "LC",
]
AUX_LABEL_DESCRIPTIONS = {
    "ANRS": "Appearance of Nasal RNFL defect - Superior",
    "ANRI": "Appearance of Nasal RNFL defect - Inferior",
    "RNFLDS": "Retinal Nerve Fibre Layer Defect - Superior",
    "RNFLDI": "Retinal Nerve Fibre Layer Defect - Inferior",
    "BCLVS": "Bayonet Configuration of Lamina Cribrosa Vessels - Superior",
    "BCLVI": "Bayonet Configuration of Lamina Cribrosa Vessels - Inferior",
    "NVT": "Notch in the Vertical neuro-retinal-rim Thinning",
    "DH": "Disc Haemorrhage",
    "LD": "Laminar Dots",
    "LC": "Large Cup",
}


@register
class GlaucomaAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_glaucoma",
        modality="CFP",
        task="classification",
        description=(
            "Glaucoma referability and 10 morphological signs on the optic-disc "
            "crop of a colour fundus photograph. The ViT-L DINOv2 model returns "
            "a referable-glaucoma probability and sign-specific scores."
        ),
        input_size=(INPUT_SIZE, INPUT_SIZE),
        labels=["referable_glaucoma", *AUX_LABELS],
        confidence_threshold=0.45,
        limitations=[
            "Requires accurate optic-disc localisation through cfp_od_detection",
            "Trained on JustRAIGS cohorts; out-of-distribution images may underperform",
            "Auxiliary-sign labels are noisy; threshold each at 0.5 with caution",
        ],
        requires_tools=["cfp_od_detection"],
        cost_class="medium",
        source_dir="ophagent.components.glaucoma",
    )

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._od_adapter = None
        self._model = None
        self._transform = None

    def _load_impl(self) -> None:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                "Glaucoma checkpoint not found. Set OPHAGENT_GLAUCOMA_WEIGHTS "
                f"or place the checkpoint at: {CKPT_PATH}"
            )

        model = GlaucomaModel(num_aux=len(AUX_LABELS))
        state = torch.load(str(CKPT_PATH), map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        self._model = model
        self._transform = build_eval_transform(size=INPUT_SIZE)

        from . import od_detection as _od  # noqa: F401
        from ..base import GLOBAL_REGISTRY

        self._od_adapter = GLOBAL_REGISTRY.get("cfp_od_detection", device=self.device)

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        crop_info = self._od_adapter.crop_disc(image_path, padding_ratio=1.4)
        if crop_info is None:
            return AdapterResult(
                success=False,
                tool=self.metadata.name,
                modality="CFP",
                task="classification",
                error="OD detector failed to find an optic disc in this image",
            )
        crop, meta = crop_info

        image = self._transform(crop).unsqueeze(0).to(self.device)
        out_main, out_aux = self._model(image)
        rg_prob = float(torch.sigmoid(out_main.float()).item())
        aux_probs = torch.sigmoid(out_aux.squeeze().float()).cpu().numpy()
        aux_signs = {
            name: {
                "probability": float(aux_probs[index]),
                "predicted": bool(aux_probs[index] > 0.5),
                "description": AUX_LABEL_DESCRIPTIONS[name],
            }
            for index, name in enumerate(AUX_LABELS)
        }

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions={
                "referable_glaucoma_probability": rg_prob,
                "referable_glaucoma": bool(rg_prob > 0.5),
                "auxiliary_signs": aux_signs,
                "predicted_signs": [
                    name for name, value in aux_signs.items() if value["predicted"]
                ],
            },
            confidence=max(rg_prob, 1.0 - rg_prob),
            metadata={
                "od_crop_box": meta["crop_box"],
                "od_detection_confidence": meta["od_confidence"],
            },
        )
