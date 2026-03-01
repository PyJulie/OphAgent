"""ViLReF CLIP Tool - Chinese zero-shot CFP classification (reused)."""
from __future__ import annotations
from typing import Any, Dict, List
import torch
from ophagent.tools.clip_models.retizero import RetiZeroTool
from ophagent.tools.base import ToolMetadata


class ViLReFTool(RetiZeroTool):
    """
    Zero-shot classification of CFP images using the ViLReF CLIP model (Chinese).
    Inherits inference logic from RetiZeroTool; only the weights and tokeniser differ.
    Input:  {"image_path": str, "candidate_labels": list[str]}
    Output: {"label": str, "probabilities": dict[str, float]}
    """
    def load_model(self) -> None:
        import open_clip
        # ViLReF uses a Chinese-aware CLIP checkpoint
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(self.metadata.model_weight)
        )
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device).eval()
        self._model_loaded = True
