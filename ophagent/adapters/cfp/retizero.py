"""
Adapter: RetiZero (LoRA-finetuned FLAIR) — English zero-shot CFP CLIP.

Wraps an external RetiZero checkpoint. Configure it with
``OPHAGENT_RETIZERO_WEIGHTS`` or place it under
``external/retizero/CLIP/model.pth``.

This gives us a SECOND independent CLIP signal alongside ViLReF (Chinese),
trained on a much larger English fundus-text corpus. Disagreement between
the two is a strong signal for the verifier / planner.
"""

from __future__ import annotations

from pathlib import Path

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ._flair_arch import build_retizero, classify
from ._clip_native_vocab import RETIZERO_NATIVE, CANON_TO_NATIVE
from ...utils.paths import checkpoint_file


RETIZERO_CKPT = checkpoint_file("OPHAGENT_RETIZERO_WEIGHTS", "cfp", "retizero.pth")


# Categories — English; the architecture's built-in multi-template prompt
# bank (`dictionary.py::definitions`) covers most of these with 2-6 templates
# each, averaged automatically via `compute_text_embeddings(domain_knowledge=True)`.
RETIZERO_LABELS = [
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
    "age related macular degeneration",
    "pathologic myopia",
    "central serous retinopathy",
    "central retinal vein occlusion",
    "branch retinal vein occlusion",
    "epiretinal membrane",
    "macular hole",
    "retinal detachment",
    "glaucoma",
    "cataract",
    "normal retina",
]

# Map RetiZero verbose labels → the canonical 11-class space ViLReF uses.
# Used by the ensemble composite to vote across CLIPs.
RETIZERO_TO_CANON = {
    "no diabetic retinopathy": "Normal retina",
    "mild diabetic retinopathy": "Diabetic retinopathy",
    "moderate diabetic retinopathy": "Diabetic retinopathy",
    "severe diabetic retinopathy": "Diabetic retinopathy",
    "proliferative diabetic retinopathy": "Diabetic retinopathy",
    "age related macular degeneration": "Age-related macular degeneration",
    "pathologic myopia": "Pathological myopia",
    "central serous retinopathy": "Central serous chorioretinopathy",
    "central retinal vein occlusion": "Retinal vein occlusion",
    "branch retinal vein occlusion": "Retinal vein occlusion",
    "epiretinal membrane": "Epiretinal membrane",
    "macular hole": "Macular hole",
    "retinal detachment": "Retinal detachment",
    "glaucoma": "Glaucoma",
    "cataract": "Suspected cataract",
    "normal retina": "Normal retina",
}


@register
class RetiZeroAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_retizero",
        modality="CFP",
        task="classification",
        description=(
            "RetiZero — English zero-shot CFP CLIP (FLAIR architecture, "
            "LoRA-finetuned on a larger fundus-text corpus). Returns "
            "softmax probabilities over 16 fine-grained labels (5 DR severities, "
            "AMD, pathologic myopia, CSC, CRVO/BRVO, ERM, MH, RD, glaucoma, "
            "cataract, normal). Use as an INDEPENDENT vote alongside "
            "`cfp_clip_multi_disease` (ViLReF, Chinese) — disagreement between "
            "the two is a strong signal that the PDR cascade's standalone "
            "prediction needs scrutiny. The output's `canon_top3` projects "
            "labels into ViLReF's 11-class space for direct comparison."
        ),
        input_size=(224, 224),
        labels=RETIZERO_NATIVE,
        confidence_threshold=0.20,   # 16-class softmax is naturally diffuse
        limitations=[
            "Bio_ClinicalBERT text encoder — English only.",
            "LoRA-finetuned on FLAIR backbone; size 1.65 GB.",
            "Templates use the FLAIR `definitions` dict for multi-template "
            "averaging — already activated.",
        ],
        cost_class="medium",
        source_dir=str(RETIZERO_CKPT.parent),
    )

    def _load_impl(self) -> None:
        if not RETIZERO_CKPT.exists():
            raise FileNotFoundError(f"RetiZero weights not found: {RETIZERO_CKPT}")
        # RetiZero used LoRA rank 8 in upstream Zeroshot.py.
        self._impl = build_retizero(RETIZERO_CKPT, R=8, device=self.device)

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        ranked = classify(
            self._impl, image_path, RETIZERO_NATIVE,
            use_domain_knowledge=True, device=self.device,
        )
        top1_label, top1_prob = ranked[0]
        canon_ranked: dict[str, float] = {}
        retizero_to_canon: dict[str, str] = {}
        for canon, by_model in CANON_TO_NATIVE.items():
            for native_label in (by_model.get("retizero") or []):
                retizero_to_canon[native_label] = canon
        for lab, prob in ranked:
            canon = retizero_to_canon.get(lab, lab)
            canon_ranked[canon] = max(canon_ranked.get(canon, 0.0), prob)
        canon_sorted = sorted(canon_ranked.items(), key=lambda kv: -kv[1])
        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions={
                "predicted_class": top1_label,
                "predicted_class_canon": retizero_to_canon.get(top1_label, top1_label),
                "probabilities": dict(ranked[:8]),
                "top_3": [{"label_en": lab, "probability": prob}
                          for lab, prob in ranked[:3]],
                "canon_top3": [{"label_en": c, "probability": p}
                               for c, p in canon_sorted[:3]],
            },
            confidence=float(top1_prob),
        )
