"""
Shared loader for the FLAIR / RetiZero family of retinal CLIPs.

The two ship as separate source trees with slightly different `VisionModel`
implementations:

  * **Upstream FLAIR**  — configure with ``OPHAGENT_FLAIR_ROOT``.
                          Supports `resnet_v1`, `resnet_v2`, `efficientnet`.
  * **RetiZero (LoRA)** — configure with ``OPHAGENT_RETIZERO_ROOT``.
                          Supports `lora` only.

Both share the same Bio_ClinicalBERT text encoder + an expert-knowledge
prompt dictionary (`dictionary.definitions`) which we leverage for
automatic multi-template averaging.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ...utils.paths import external_dir


FLAIR_ROOT = external_dir("OPHAGENT_FLAIR_ROOT", "flair")
RETIZERO_ROOT = external_dir("OPHAGENT_RETIZERO_ROOT", "retizero")


def _ensure_on_path(root: Path):
    p = str(root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _purge_modules(prefix: str):
    """Drop cached imports from sys.modules — the two CLIP source trees
    share top-level module names ('modeling', 'dictionary' etc.) that
    would collide if loaded in the same process."""
    for m in list(sys.modules):
        if m == prefix or m.startswith(prefix + "."):
            del sys.modules[m]


def _set_upstream_device(model, device: str) -> None:
    """Keep upstream module-level tensor placement aligned with the adapter."""
    module = sys.modules.get(type(model).__module__)
    if module is not None and hasattr(module, "device"):
        module.device = device


def build_flair_resnet(weights_path: str | Path, device: str = "cuda"):
    """Pure FLAIR ResNet-50. Uses the upstream package."""
    _ensure_on_path(FLAIR_ROOT)
    _purge_modules("flair")
    from flair import FLAIRModel  # type: ignore

    model = FLAIRModel(
        from_checkpoint=False,
        weights_path=str(weights_path),
        vision_type="resnet_v1",
        vision_pretrained=False,
    )
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # `strict=False` because newer transformers versions removed the
    # `position_ids` buffer from BertEmbeddings; old FLAIR/RetiZero
    # checkpoints still ship it. The actual learnable weights all match.
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        import logging
        logging.getLogger(__name__).warning(
            f"FLAIR/RetiZero load mismatches: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    model.to(device).eval()
    _set_upstream_device(model, device)
    return model


def build_retizero(weights_path: str | Path, R: int = 8, device: str = "cuda"):
    """RetiZero (LoRA-finetuned FLAIR). Uses the 多病种3 zeroshot package."""
    _ensure_on_path(RETIZERO_ROOT)
    _purge_modules("zeroshot")
    from zeroshot import CLIPRModel  # type: ignore

    model = CLIPRModel(
        vision_type="lora",
        from_checkpoint=False,
        weights_path=str(weights_path),
        R=R,
        caption="[CLS]",
    )
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # `strict=False` because newer transformers versions removed the
    # `position_ids` buffer from BertEmbeddings; old FLAIR/RetiZero
    # checkpoints still ship it. The actual learnable weights all match.
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        import logging
        logging.getLogger(__name__).warning(
            f"FLAIR/RetiZero load mismatches: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    model.to(device).eval()
    _set_upstream_device(model, device)
    return model


# ── inference helpers (both model classes share the same API surface) ─────
def _encode_image(model, image_path: str, device: str = "cuda"):
    """The two source trees disagree on `preprocess_image`'s expected type:
    upstream FLAIR wants a numpy ndarray, RetiZero wants a PIL.Image. Try
    one, fall back to the other."""
    import numpy as np
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")
    try:
        x = model.preprocess_image(np.asarray(pil))
    except (AttributeError, TypeError):
        x = model.preprocess_image(pil)
    x = x.to(device)
    with torch.no_grad():
        emb = model.vision_model(x)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


def _encode_text_multi_template(model, categories: list[str],
                                use_domain_knowledge: bool = True,
                                device: str = "cuda"):
    embeds_dict = model.compute_text_embeddings(
        categories, domain_knowledge=use_domain_knowledge,
    )
    # Some upstream versions return a tuple (dict, concat-tensor); take the dict.
    if isinstance(embeds_dict, tuple):
        embeds_dict = embeds_dict[0]
    embeds = torch.cat([embeds_dict[c] for c in categories], dim=0)
    embeds = torch.nn.functional.normalize(embeds, dim=-1).to(device)
    return embeds


@torch.no_grad()
def classify(model, image_path: str, categories: list[str],
             use_domain_knowledge: bool = True,
             device: str = "cuda") -> list[tuple[str, float]]:
    img_emb = _encode_image(model, image_path, device=device)
    txt_emb = _encode_text_multi_template(
        model, categories, use_domain_knowledge=use_domain_knowledge, device=device
    )
    logit_scale = model.logit_scale.exp()
    logits = (img_emb @ txt_emb.T) * logit_scale
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    pairs = list(zip(categories, probs))
    pairs.sort(key=lambda kv: -kv[1])
    return pairs
