"""Native-vocab CLIP ensemble scorer (V2). Scores a set of CANONICAL conditions
by mapping each to every model's NATIVE label(s) (CANON_TO_NATIVE), running each
open-vocab CLIP (FLAIR + RetiZero) over its native panel, aggregating per-canon
(max over native synonyms), and fusing with the per-disease calibrated weights +
thresholds learned on the MAC held-out split (_native_clip_calib.json).

This restores the models' real capability (the legacy ensemble truncated FLAIR's
~109 / RetiZero's vocab to a canon-11 with NO retinal artery occlusion). Gated by
OPH_DECISION_V2 at the ensemble adapter.
"""
from __future__ import annotations
import os, json
from functools import lru_cache

from ._flair_arch import classify
from ._clip_native_vocab import CANON_TO_NATIVE, TASK7_TO_CANON
from ...utils.paths import runtime_path

_CANON2TASK = {v: k for k, v in TASK7_TO_CANON.items()}
_DEFAULT_THR = 0.12
_DEFAULT_W = 0.5  # RetiZero weight when a disease wasn't calibrated


@lru_cache(maxsize=1)
def _calib() -> dict:
    p = os.environ.get("OPHAGENT_NATIVE_CLIP_CALIB", "").strip()
    if not p:
        p = str(runtime_path("config", "native_clip_calibration.json"))
    try:
        return json.load(open(os.path.abspath(p), encoding="utf-8"))
    except Exception:
        return {}


def _per_canon(impl, mkey, image_path, target_canons, dk=True):
    panel = []
    for canon in target_canons:
        panel += (CANON_TO_NATIVE.get(canon, {}) or {}).get(mkey) or []
    if len(panel) < 2:
        return {}
    ranked = dict(classify(impl, str(image_path), panel, use_domain_knowledge=dk))
    out = {}
    for canon in target_canons:
        nats = (CANON_TO_NATIVE.get(canon, {}) or {}).get(mkey) or []
        vals = [ranked.get(n, 0.0) for n in nats if n in ranked]
        out[canon] = max(vals) if vals else None
    return out


@lru_cache(maxsize=1)
def _vilref_adapter():
    """Load the CVL Chinese CLIP (ViLReF) adapter once and expose its encoders."""
    from ..base import GLOBAL_REGISTRY
    ad = GLOBAL_REGISTRY.get("cfp_clip_multi_disease"); ad.load()
    return ad


def _vilref_classify(image_path, zh_labels):
    """Open-vocab softmax over arbitrary Chinese labels via ViLReF's own
    image/text encoders (image_size=224, CVL tokenize). Native label store, so
    we score the bare clinical phrase — that IS ViLReF's training vocabulary."""
    import torch
    from PIL import Image
    ad = _vilref_adapter()
    impl, tok, tf, dev = ad._impl, ad._tokenize, ad._image_transform, ad.device
    with torch.no_grad():
        img = Image.open(str(image_path)).convert("RGB")
        x = tf(img).unsqueeze(0).to(dev)
        img_feat = impl.encode_image(x); img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        tokens = tok(list(zh_labels)).to(dev)
        txt = impl.encode_text(tokens); txt = txt / txt.norm(dim=-1, keepdim=True)
        logits = (img_feat @ txt.T) * impl.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    return dict(zip(zh_labels, probs))


def _per_canon_vilref(image_path, target_canons):
    """ViLReF Chinese open-vocab analogue of _per_canon. Builds a panel from each
    canon's ViLReF native synonym(s) (CANON_TO_NATIVE[...]['vilref']) plus a
    '正常眼底' anchor, softmaxes over the panel, returns per-canon = max over its
    synonyms. canon -> None when ViLReF has no native concept (RAO, ERM)."""
    panel = ["正常眼底"]
    for canon in target_canons:
        panel += (CANON_TO_NATIVE.get(canon, {}) or {}).get("vilref") or []
    panel = list(dict.fromkeys(panel))  # dedupe, preserve order
    if len(panel) < 2:
        return {}
    ranked = _vilref_classify(image_path, panel)
    out = {}
    for canon in target_canons:
        nats = (CANON_TO_NATIVE.get(canon, {}) or {}).get("vilref") or []
        vals = [ranked.get(n, 0.0) for n in nats if n in ranked]
        out[canon] = max(vals) if vals else None
    return out


def native_ensemble(image_path, target_canons, dk=True):
    """Returns (ranked[(canon,score)], present_list, fused_dict, per_model_top1)."""
    from ..base import GLOBAL_REGISTRY  # lazy to avoid import cycle
    impls = {}
    for tool, mkey in (("cfp_retizero", "retizero"), ("cfp_flair", "flair")):
        try:
            ad = GLOBAL_REGISTRY.get(tool); ad.load(); impls[mkey] = ad._impl
        except Exception:
            pass
    if not impls:
        return [], [], {}, {}
    per_model = {mk: _per_canon(im, mk, image_path, target_canons, dk) for mk, im in impls.items()}
    calib = _calib()
    fused = {}
    for canon in target_canons:
        tk = _CANON2TASK.get(canon)
        w = calib.get(tk, {}).get("w_retizero", _DEFAULT_W) if tk else _DEFAULT_W
        a = per_model.get("retizero", {}).get(canon)
        b = per_model.get("flair", {}).get(canon)
        # if one model has no native mapping, put full weight on the other
        if a is None and b is None:
            continue
        if a is None:
            fused[canon] = b
        elif b is None:
            fused[canon] = a
        else:
            fused[canon] = w * a + (1 - w) * b
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    present = []
    for canon, score in fused.items():
        if canon == "Normal":
            continue
        tk = _CANON2TASK.get(canon)
        thr = calib.get(tk, {}).get("threshold", _DEFAULT_THR) if tk else _DEFAULT_THR
        if score >= thr:
            present.append(canon)
    per_model_top1 = {mk: (max(d.items(), key=lambda kv: (kv[1] or 0))[0] if d else None)
                      for mk, d in per_model.items()}
    return ranked, present, fused, per_model_top1
