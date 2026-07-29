"""Open-vocabulary zero-shot CFP classifier (paper §S2.4.1, open-vocab path).

WHY THIS EXISTS
---------------
The paper describes CLIP as an OPEN-VOCABULARY / zero-shot engine able to
recognise rare conditions "without retraining" (RetiZero / ViLReF), and
§S2.4.1 specifies a scalable zero-shot pipeline that, when no lesion narrows
the candidate set, "degrades gracefully to unconstrained CLIP inference over
the full disease label space D". The shipped `cfp_clip_multi_disease` /
`cfp_retizero` adapters are instead PINNED to a fixed 11-class space, so a
disease outside that list (e.g. retinitis / inflammatory / infectious) can
never be named — it gets mapped to the nearest vascular class (the observed
retinitis→HR miss).

This adapter implements that open-vocab path WITHOUT touching the closed-set
tools: it REUSES the already-loaded CVL CLIP (`cfp_clip_multi_disease`) image
+ text encoders to score the image against a BROADER disease vocabulary that
adds inflammatory / infectious / rare entities on top of the closed 11. When
the top match is OUTSIDE the closed set, or confidence is low, it raises a
flag so the orchestrator can widen the differential and recommend FFA / OCT /
clinical correlation instead of confidently mislabelling.

SELF-CONTAINED & ROLLBACK-SAFE: this is a new, additive tool. To remove it,
delete this file and its import line in `cfp/__init__.py`. Nothing else is
modified; the closed-set classifiers are untouched.

NOTE (scope): this is the unconstrained full-D open-vocab path. The
lesion-guided candidate-set reduction (Dx via a lesion→disease table) from
§S2.4.1 is a precision refinement for IN-set diseases and is left as a
follow-up — it does not help surface out-of-set diseases like retinitis
(retsam has no inflammation signal), which is exactly what the full-D path
catches.
"""
from __future__ import annotations

import torch
from PIL import Image

from ..base import AdapterBase, ToolMetadata, AdapterResult, register, GLOBAL_REGISTRY
from .clip_disease import LABEL_ZH, LABEL_EN, TEMPLATES_ZH

# ── Extra (OUT-OF-CLOSED-SET) disease entries: zh, en, templates_zh ──────────
# Focus on inflammatory / infectious / rare entities the closed 11 cannot name.
_EXTRA = [
    ("视网膜炎", "Retinitis", [
        "视网膜炎", "感染性或炎症性视网膜炎症，视网膜白色坏死灶伴血管鞘",
        "视网膜可见渗出、出血及血管炎性改变，符合视网膜炎"]),
    ("巨细胞病毒性视网膜炎", "CMV retinitis", [
        "巨细胞病毒性视网膜炎", "沿血管分布的视网膜坏死伴出血（番茄酱样）"]),
    ("急性视网膜坏死", "Acute retinal necrosis", [
        "急性视网膜坏死", "周边视网膜融合性白色坏死伴闭塞性血管炎"]),
    ("葡萄膜炎", "Uveitis", [
        "葡萄膜炎", "后葡萄膜炎，玻璃体混浊、视网膜血管鞘、脉络膜病灶"]),
    ("视网膜血管炎", "Retinal vasculitis", [
        "视网膜血管炎", "视网膜血管壁白鞘、血管闭塞伴周边渗出"]),
    ("中心性浆液性脉络膜视网膜病变", "Central serous chorioretinopathy", [
        "中心性浆液性脉络膜视网膜病变", "黄斑区局限性浆液性视网膜神经上皮脱离"]),
    ("视网膜色素变性", "Retinitis pigmentosa", [
        "视网膜色素变性", "骨细胞样色素沉着、视网膜血管变细、蜡黄色视盘"]),
    ("视盘水肿", "Optic disc edema", [
        "视盘水肿", "视盘边界模糊、隆起、充血，符合视盘水肿或视神经炎"]),
    ("脉络膜新生血管", "Choroidal neovascularization", [
        "脉络膜新生血管", "黄斑下出血、灰绿色膜状渗出性病灶"]),
    ("眼底肿瘤", "Intraocular tumor", [
        "眼底肿瘤", "脉络膜黑色素瘤或视网膜母细胞瘤样隆起占位"]),
]

# Unified vocabulary D: (zh, en, in_closed_set, templates)
_VOCAB: list[tuple[str, str, bool, list[str]]] = [
    (zh, en, True, TEMPLATES_ZH.get(zh, [zh]))
    for zh, en in zip(LABEL_ZH, LABEL_EN)
] + [(zh, en, False, tmpls) for (zh, en, tmpls) in _EXTRA]

_CLIP_TOOL = "cfp_clip_multi_disease"   # the loaded CVL CLIP we reuse


@register
class CFPOpenVocabZeroShotAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_openvocab_zeroshot",
        modality="CFP",
        task="classification",
        description=(
            "OPEN-VOCABULARY zero-shot CFP classifier over a BROAD disease "
            "space that extends beyond the closed 11 classes to include "
            "inflammatory / infectious / rare entities (retinitis, CMV "
            "retinitis, acute retinal necrosis, uveitis, retinal vasculitis, "
            "CSC, retinitis pigmentosa, optic disc edema, choroidal "
            "neovascularisation, intraocular tumour). CALL THIS when the "
            "standard closed-set classifiers are low-confidence or mutually "
            "conflicting, or to check whether the case may be an OUT-OF-SCOPE "
            "disease the closed-set tools cannot name. If the top match is "
            "outside the closed set (`out_of_closed_set=true`) or confidence "
            "is low, treat it as a differential WIDENING signal — do NOT "
            "over-call; recommend FFA / OCT and clinical correlation."
        ),
        input_size=(224, 224),
        labels=[en for (_zh, en, _c, _t) in _VOCAB],
        confidence_threshold=0.15,   # softmax over ~21 prompts is diffuse
        limitations=[
            "Reuses the CVL (ViLReF) Chinese CLIP encoders; open-vocab scores "
            "on rare entities are uncalibrated and should WIDEN the "
            "differential, not confirm a diagnosis.",
            "CFP-only: inflammatory/infectious diseases usually need FFA + "
            "clinical/lab context to confirm; this only flags the possibility.",
            "No lesion-guided candidate reduction yet (full-D path only).",
        ],
        cost_class="medium",
        source_dir="(reuses cfp_clip_multi_disease)",
    )

    def _load_impl(self) -> None:
        # Reuse the already-loaded CVL CLIP — do NOT load a second model.
        clip = GLOBAL_REGISTRY.get(_CLIP_TOOL, device=self.device)
        clip.load()
        self._clip = clip
        model = clip._impl
        # Pre-encode the broad vocabulary's text features (multi-template
        # averaging, same convention as cfp_clip_multi_disease).
        with torch.no_grad():
            feats = []
            for (_zh, _en, _closed, tmpls) in _VOCAB:
                tokens = clip._tokenize(tmpls).to(self.device)
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb.mean(dim=0, keepdim=True))
            t = torch.cat(feats, dim=0)
            self._text_features = t / t.norm(dim=-1, keepdim=True)
        self._impl = model   # mark loaded

    @torch.no_grad()
    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        clip = self._clip
        img = Image.open(image_path).convert("RGB")
        x = clip._image_transform(img).unsqueeze(0).to(self.device)
        img_feat = self._impl.encode_image(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ self._text_features.T) * self._impl.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        order = sorted(range(len(_VOCAB)), key=lambda i: -probs[i])
        top = order[0]
        zh, en, in_closed, _ = _VOCAB[top]
        out_of_set = not in_closed
        conf = float(probs[top])
        low_conf = conf < self.metadata.confidence_threshold

        if out_of_set:
            flag = (f"Top open-vocab match '{en}' is OUTSIDE the closed 11-class "
                    f"set — a possible out-of-scope disease. Widen the "
                    f"differential and recommend FFA/OCT + clinical correlation; "
                    f"do not finalise on CFP alone.")
        elif low_conf:
            flag = ("All candidates are low-confidence — evidence is ambiguous; "
                    "recommend further imaging / clinical context.")
        else:
            flag = None

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="classification",
            predictions={
                "top1": en, "top1_zh": zh,
                "out_of_closed_set": out_of_set,
                "low_confidence": low_conf,
                "flag": flag,
                "top5": [
                    {"label_en": _VOCAB[i][1], "label_zh": _VOCAB[i][0],
                     "in_closed_set": _VOCAB[i][2],
                     "probability": round(float(probs[i]), 4)}
                    for i in order[:5]
                ],
            },
            confidence=conf,
        )
