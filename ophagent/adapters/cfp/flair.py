"""
Adapter: FLAIR — A Foundation Language Image model of the Retina.

ResNet-50 backbone + Bio_ClinicalBERT text encoder (Silva-Rodríguez et al.,
Medical Image Analysis 2024). Pretrained on aggregated public fundus
datasets with expert-knowledge text supervision.

Configure weights with ``OPHAGENT_FLAIR_WEIGHTS`` or place them under
``external/flair/flair/modeling/flair_resnet.pth``.

Uses the same `CLIPRModel` class as RetiZero (shared FLAIR architecture)
— only `vision_type` differs: 'resnet_v1' here, 'lora' for RetiZero.
"""

from __future__ import annotations

from pathlib import Path

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ._flair_arch import build_flair_resnet, classify
from ._clip_native_vocab import FLAIR_NATIVE, CANON_TO_NATIVE
from ...utils.paths import checkpoint_file, external_file, first_existing_path


_FLAIR_CKPT_CANDIDATES = [
    checkpoint_file("OPHAGENT_FLAIR_WEIGHTS", "cfp", "flair.pth"),
    external_file("OPHAGENT_FLAIR_WEIGHTS", "external", "flair", "flair", "modeling", "flair_resnet.pth"),
    external_file("OPHAGENT_FLAIR_WEIGHTS", "weights", "flair_resnet.pth"),
]


def _find_ckpt() -> Path:
    p = first_existing_path("OPHAGENT_FLAIR_WEIGHTS", _FLAIR_CKPT_CANDIDATES)
    if p.exists():
        return p
    raise FileNotFoundError(
        "FLAIR weights not found. Set OPHAGENT_FLAIR_WEIGHTS or place the file at:\n  - "
        + "\n  - ".join(str(p) for p in _FLAIR_CKPT_CANDIDATES)
    )


@register
class FLAIRAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_flair",
        modality="CFP",
        task="classification",
        description=(
            "FLAIR (Silva-Rodríguez et al. 2024 MedIA) — English zero-shot "
            "CFP CLIP. ResNet-50 vision + Bio_ClinicalBERT text. Trained on "
            "37 public fundus datasets with expert-knowledge text supervision. "
            "Returns softmax probabilities over FLAIR's native paper/repository "
            "label vocabulary and a canonical projection for direct comparison. "
            "Use as a THIRD independent vote (alongside ViLReF "
            "Chinese and RetiZero English-LoRA) — three-CLIP majority makes "
            "the differential robust against any single model's biases."
        ),
        input_size=(224, 224),
        labels=FLAIR_NATIVE,
        confidence_threshold=0.18,
        limitations=[
            "Older / smaller training corpus than RetiZero (RetiZero is a "
            "LoRA-finetune of this on more data).",
            "ResNet-50 vision encoder — pure CNN, no ViT.",
        ],
        cost_class="medium",
        source_dir=None,
    )

    def _load_impl(self) -> None:
        ckpt = _find_ckpt()
        self._impl = build_flair_resnet(ckpt, device=self.device)
        # populate source_dir post-load for transparency
        self.metadata.source_dir = str(ckpt.parent)

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        ranked = classify(
            self._impl, image_path, FLAIR_NATIVE,
            use_domain_knowledge=True, device=self.device,
        )
        top1_label, top1_prob = ranked[0]
        canon_ranked: dict[str, float] = {}
        flair_to_canon: dict[str, str] = {}
        for canon, by_model in CANON_TO_NATIVE.items():
            for lab in (by_model.get("flair") or []):
                flair_to_canon[lab] = canon
        for lab, prob in ranked:
            canon = flair_to_canon.get(lab, lab)
            canon_ranked[canon] = max(canon_ranked.get(canon, 0.0), prob)
        canon_sorted = sorted(canon_ranked.items(), key=lambda kv: -kv[1])
        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions={
                "predicted_class": top1_label,
                "predicted_class_canon": flair_to_canon.get(top1_label, top1_label),
                "probabilities": dict(ranked[:8]),
                "top_3": [{"label_en": lab, "probability": prob}
                          for lab, prob in ranked[:3]],
                "canon_top3": [{"label_en": c, "probability": p}
                               for c, p in canon_sorted[:3]],
            },
            confidence=float(top1_prob),
        )
