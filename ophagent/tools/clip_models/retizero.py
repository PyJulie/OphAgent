"""RetiZero CLIP Tool - English zero-shot CFP classification (reused)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from ophagent.tools.base import BaseTool, ToolMetadata


class RetiZeroTool(BaseTool):
    """
    Zero-shot classification of CFP images using the RetiZero CLIP model (English).
    Runs inline.
    Input:  {"image_path": str, "candidate_labels": list[str]}
    Output: {"label": str, "probabilities": dict[str, float]}
    """
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def load_model(self) -> None:
        import open_clip
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(self.metadata.model_weight)
        )
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device).eval()
        self._model_loaded = True

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch.nn.functional as F
        from ophagent.utils.image_utils import load_image_pil
        device = next(self._model.parameters()).device
        img = self._preprocess(load_image_pil(inputs["image_path"])).unsqueeze(0).to(device)
        labels: List[str] = inputs.get("candidate_labels", ["normal", "abnormal"])
        text = self._tokenizer(labels).to(device)
        with torch.no_grad():
            img_feat = self._model.encode_image(img)
            txt_feat = self._model.encode_text(text)
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            logits = (img_feat @ txt_feat.T).squeeze(0)
            probs = logits.softmax(dim=-1).cpu().tolist()
        top_idx = int(torch.tensor(probs).argmax())
        return {
            "label": labels[top_idx],
            "probabilities": {l: float(p) for l, p in zip(labels, probs)},
        }
