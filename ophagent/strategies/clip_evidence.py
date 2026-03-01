"""
Evidence-Guided CLIP Zero-Shot Classification (Section 2.4.1).

Innovation: Instead of using raw disease label names as CLIP text prompts,
we use the LLM to generate clinically precise visual evidence descriptors
for each candidate label.  This dramatically improves zero-shot CLIP
performance on ophthalmic images.

Pipeline:
  1. For each candidate disease label, generate K evidence descriptors via LLM.
  2. Embed all descriptors with the CLIP text encoder.
  3. Embed the query image with the CLIP image encoder.
  4. Aggregate per-label similarity scores (mean or max).
  5. Apply confidence threshold θ_CLIP = 0.5 (§2.4.1).
  6. Optionally refine with evidence consistency scoring (§2.4.2):
       Score(d_k) = λ·Match+(d_k) − (1−λ)·Match−(d_k),  λ = 0.6
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ophagent.llm.backbone import LLMBackbone
from ophagent.llm.prompts import PromptLibrary
from ophagent.utils.logger import get_logger

logger = get_logger("strategies.clip_evidence")


# ---------------------------------------------------------------------------
# Disease ↔ finding consistency knowledge (§2.4.2)
# Keys: lowercase disease label
# Values: (positive_findings, negative_findings) — sets of lowercase finding tokens
# ---------------------------------------------------------------------------
_DISEASE_FINDING_MAP: Dict[str, Tuple[set, set]] = {
    "diabetic retinopathy": (
        {"microaneurysm", "hemorrhage", "hard exudate", "cotton wool spot", "neovascularisation"},
        {"drusen", "geographic atrophy"},
    ),
    "glaucoma": (
        {"cup disc enlargement", "disc pallor", "rnfl defect", "notching"},
        {"drusen", "microaneurysm", "hard exudate"},
    ),
    "age-related macular degeneration": (
        {"drusen", "geographic atrophy", "choroidal neovascularisation", "subretinal fluid"},
        {"microaneurysm", "cup disc enlargement"},
    ),
    "hypertensive retinopathy": (
        {"av nicking", "arteriolar narrowing", "flame hemorrhage", "cotton wool spot", "disc edema"},
        {"drusen", "microaneurysm"},
    ),
    "retinal vein occlusion": (
        {"flame hemorrhage", "disc edema", "venous dilation", "cotton wool spot"},
        {"drusen"},
    ),
    "myopia": (
        {"tessellation", "lacquer cracks", "myopic conus", "posterior staphyloma"},
        {"drusen", "microaneurysm"},
    ),
    "normal": (
        set(),
        {"hemorrhage", "drusen", "cup disc enlargement", "microaneurysm", "hard exudate"},
    ),
}


class EvidenceGuidedCLIP:
    """
    Evidence-guided zero-shot CLIP classifier for ophthalmic images.

    Usage::

        clf = EvidenceGuidedCLIP(clip_tool=retizero_tool, llm=llm)
        result = clf.classify(
            image_path="fundus.jpg",
            candidate_labels=["diabetic retinopathy", "glaucoma", "AMD", "normal"],
            modality="CFP",
            findings=["microaneurysm", "hard exudate"],   # optional: from seg tools
        )
    """

    # Number of evidence descriptors to generate per label (K in §2.4.1)
    N_EVIDENCE = 7
    # Score aggregation: "mean" | "max"
    AGGREGATION = "mean"
    # Confidence threshold below which the top prediction is marked as uncertain (§2.4.1)
    CONF_THRESHOLD: float = 0.5
    # Consistency score weight λ (§2.4.2)
    LAMBDA: float = 0.6

    def __init__(
        self,
        clip_tool=None,     # RetiZeroTool or ViLReFTool instance
        llm: Optional[LLMBackbone] = None,
        cache_evidence: bool = True,
    ):
        self.llm = llm or LLMBackbone()
        self._clip_tool = clip_tool
        self._evidence_cache: Dict[str, List[str]] = {}
        self._cache_enabled = cache_evidence

    # ------------------------------------------------------------------
    # Evidence generation
    # ------------------------------------------------------------------

    def generate_evidence(
        self,
        label: str,
        modality: str = "CFP",
        n: int = N_EVIDENCE,
    ) -> List[str]:
        """
        Generate visual evidence descriptors for *label* using the LLM.
        Results are cached to avoid redundant API calls.
        """
        cache_key = f"{label}|{modality}"
        if self._cache_enabled and cache_key in self._evidence_cache:
            return self._evidence_cache[cache_key]

        system = PromptLibrary.CLIP_EVIDENCE_SYSTEM
        user = PromptLibrary.clip_evidence_user(label, modality)
        raw = self.llm.chat(
            [{"role": "user", "content": user}],
            system=system,
            temperature=0.3,
        )

        # Parse JSON list
        from ophagent.utils.text_utils import extract_json_block
        try:
            descriptors = json.loads(extract_json_block(raw))
            if not isinstance(descriptors, list):
                raise ValueError
        except Exception:
            logger.warning(f"Failed to parse evidence for '{label}'; using label as-is.")
            descriptors = [label]

        descriptors = [d for d in descriptors if isinstance(d, str)][:n]

        if self._cache_enabled:
            self._evidence_cache[cache_key] = descriptors
        logger.debug(f"Evidence for '{label}': {descriptors}")
        return descriptors

    # ------------------------------------------------------------------
    # CLIP inference
    # ------------------------------------------------------------------

    def _get_clip_model_and_preprocess(self):
        """Obtain the CLIP model and preprocessor from the clip_tool."""
        if self._clip_tool is None:
            raise ValueError("No CLIP tool provided to EvidenceGuidedCLIP.")
        if not self._clip_tool._model_loaded:
            self._clip_tool.load_model()
        return (
            self._clip_tool._model,
            self._clip_tool._preprocess,
            self._clip_tool._tokenizer,
        )

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of text strings, returns (N, D) normalised tensor."""
        model, _, tokenizer = self._get_clip_model_and_preprocess()
        device = next(model.parameters()).device
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        return feats  # (N, D)

    def _embed_image(self, image_path: str) -> torch.Tensor:
        """Embed a single image, returns (1, D) normalised tensor."""
        from ophagent.utils.image_utils import load_image_pil
        model, preprocess, _ = self._get_clip_model_and_preprocess()
        device = next(model.parameters()).device
        img = preprocess(load_image_pil(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = F.normalize(feat, dim=-1)
        return feat  # (1, D)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        image_path: str,
        candidate_labels: List[str],
        modality: str = "CFP",
        aggregation: Optional[str] = None,
        findings: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Classify *image_path* against *candidate_labels* using evidence-guided CLIP.

        Args:
            image_path:       Path to the query image.
            candidate_labels: Pre-screened disease candidates (caller may reduce this
                              set via segmentation-based lesion-to-disease mapping,
                              i.e. D_x = ∪_{ℓ∈L_x} R(ℓ) from §2.4.1).
            modality:         Image modality string ("CFP", "OCT", …).
            aggregation:      "mean" or "max" similarity aggregation over K descriptors.
            findings:         Optional list of detected finding tokens (from segmentation /
                              detection tools) used to compute the evidence consistency
                              score (§2.4.2): Score(d_k) = λ·Match+(d_k) − (1−λ)·Match−(d_k).

        Returns:
            {
                "label": str | None,       # top predicted label; None if below θ_CLIP
                "confident": bool,         # True if top prob ≥ CONF_THRESHOLD (θ_CLIP = 0.5)
                "probabilities": {label: float, ...},
                "consistency_scores": {label: float, ...} | None,  # if findings given
                "evidence": {label: [str, ...], ...},              # LLM descriptors used
            }
        """
        agg = aggregation or self.AGGREGATION
        img_feat = self._embed_image(image_path)  # (1, D)

        label_scores: Dict[str, float] = {}
        label_evidence: Dict[str, List[str]] = {}

        for label in candidate_labels:
            evidence = self.generate_evidence(label, modality=modality)
            label_evidence[label] = evidence
            text_feats = self._embed_texts(evidence)  # (K, D)
            sims = (img_feat @ text_feats.T).squeeze(0)  # (K,)
            label_scores[label] = float(sims.max().item() if agg == "max" else sims.mean().item())

        # Softmax over raw CLIP scores
        scores_arr = torch.tensor(list(label_scores.values()))
        probs = F.softmax(scores_arr, dim=0).tolist()
        label_probs = dict(zip(label_scores.keys(), probs))

        top_label = max(label_probs, key=lambda k: label_probs[k])
        top_prob = label_probs[top_label]
        confident = top_prob >= self.CONF_THRESHOLD  # θ_CLIP = 0.5  (§2.4.1)

        logger.info(
            f"Evidence-CLIP: {top_label} (prob={top_prob:.3f}, "
            f"{'confident' if confident else 'uncertain — below θ_CLIP'})"
        )

        # Evidence consistency scoring (§2.4.2) when segmentation findings are provided
        consistency_scores: Optional[Dict[str, float]] = None
        if findings:
            consistency_scores = self._score_consistency(label_probs, findings)
            # Use consistency-refined ranking if findings are available
            top_label = max(consistency_scores, key=lambda k: consistency_scores[k])  # type: ignore[arg-type]
            logger.info(f"After consistency re-rank: {top_label}")

        return {
            "label": top_label if confident else None,
            "confident": confident,
            "probabilities": label_probs,
            "consistency_scores": consistency_scores,
            "evidence": label_evidence,
        }

    # ------------------------------------------------------------------
    # Evidence consistency scoring  (§2.4.2)
    # ------------------------------------------------------------------

    def _score_consistency(
        self,
        label_probs: Dict[str, float],
        findings: List[str],
    ) -> Dict[str, float]:
        """
        Refine CLIP probabilities using detected visual findings.

        Score(d_k) = λ·Match⁺(d_k) − (1−λ)·Match⁻(d_k)

        where Match⁺ counts findings consistent with d_k and Match⁻ counts
        findings that contradict d_k, according to _DISEASE_FINDING_MAP.
        The raw consistency score is added to the CLIP probability and the
        result is re-normalised so it remains a valid probability distribution.

        Args:
            label_probs: CLIP softmax probabilities per label.
            findings:    Detected finding tokens (lowercase) from segmentation tools.

        Returns:
            Combined and re-normalised probability dict.
        """
        findings_set = {f.lower() for f in findings}
        combined: Dict[str, float] = {}

        for label, prob in label_probs.items():
            pos_set, neg_set = _DISEASE_FINDING_MAP.get(label.lower(), (set(), set()))
            match_pos = len(findings_set & pos_set)
            match_neg = len(findings_set & neg_set)
            consistency = self.LAMBDA * match_pos - (1 - self.LAMBDA) * match_neg
            combined[label] = prob + consistency

        # Re-normalise (clamp negatives to 0 to keep probabilities valid)
        total = sum(max(0.0, v) for v in combined.values())
        if total > 0:
            return {k: max(0.0, v) / total for k, v in combined.items()}
        # If all scores are 0 or negative fall back to uniform
        n = len(label_probs)
        return {k: 1.0 / n for k in label_probs}
