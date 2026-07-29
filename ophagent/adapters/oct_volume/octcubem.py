"""
Adapter: OCTCubeM 3D foundation model for macular OCT volume classification.

Wraps the OCTCube multitask classifier (Liu et al. 2024, arXiv:2408.11227,
BSD 2-Clause). Source repo: https://github.com/ZucksLiu/OCTCubeM

The model is a 3D spatio-temporal Vision Transformer (ViT-Large with
t_patch_size=3, num_frames=48, input 256x256) trained on 26,685 OCT volumes
(1.62M B-scans). It emits 16 logits arranged as 8 independent binary heads
(per-disease neg/pos), one head per disease in:

  [DME, AMD, POAG, EPM, DR, VD, RAO/RVO, RNV]

This means the output is **multi-label** (a single eye can have e.g. both DR
and DME fire simultaneously). The downstream agent should reason from the
raw 8-d probability vector rather than collapsing to a strict top-1.

## On the published threshold arrays

OCTCubeM ships two per-class threshold sets (notebook cell 15):
  octcube_threshold:    [0.583, 0.753, 0.567, 0.464, 0.586, 0.446, 0.126, 0.368]
  screening_threshold:  [0.847, 0.793, 0.821, 0.777, 0.829, 0.738, 0.715, 0.734]

We expose all three operating points (0.5 default + the two paper sets) but
flag the RAO/RVO threshold of 0.126 in the octcube set as suspect — it almost
certainly reflects class-imbalance in the authors' validation cohort and may
not transfer to other distributions. The agent should use raw probabilities
preferentially and treat the threshold-derived "fires" lists as advisory.

## Implementation notes

- Runs WITHOUT flash_attn (Windows compatibility). The model class supports
  use_flash_attn=False (regular separate-q/k/v attention).
- The published checkpoint stores attention as a single concatenated Wqkv
  weight (flash_attn convention). We translate to separate q/k/v Linears
  before load_state_dict (192 keys reshaped → 0 missing / 0 unexpected).
- Lazy-loads the registry + ckpt on first call; ~1.2 GB GPU after load.
- 0.4 s per volume inference at 48-frame / 256x256 resolution.
"""
from __future__ import annotations

import functools
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


log = logging.getLogger(__name__)


# External OCTCubeM source + checkpoint location.
# Configure with OPHAGENT_OCTCUBEM_ROOT / OPHAGENT_OCTCUBEM_WEIGHTS.
OCTCUBEM_ROOT = external_dir("OPHAGENT_OCTCUBEM_ROOT", "OCTCubeM")
CKPT_PATH = checkpoint_file("OPHAGENT_OCTCUBEM_WEIGHTS", "oct_volume", "octcube.pth")


def _ensure_home_env_for_octcubem() -> None:
    """OCTCubeM's dataset modules assume POSIX-style HOME exists.

    Windows shells often expose USERPROFILE/HOMEDRIVE/HOMEPATH but not HOME,
    especially after launching from GUI apps. The upstream code does
    os.getenv("HOME") + "/", so provide a stable fallback before importing it.
    """
    if os.environ.get("HOME"):
        return
    fallback = os.environ.get("USERPROFILE")
    if not fallback:
        drive = os.environ.get("HOMEDRIVE", "")
        path = os.environ.get("HOMEPATH", "")
        fallback = (drive + path) or str(Path.home())
    os.environ["HOME"] = fallback

CLASSES = ['DME', 'AMD', 'POAG', 'EPM', 'DR', 'VD', 'RAO/RVO', 'RNV']
CLASS_DESCRIPTIONS = {
    'DME':     "Diabetic macular edema",
    'AMD':     "Age-related macular degeneration",
    'POAG':    "Primary open-angle glaucoma",
    'EPM':     "Epiretinal membrane (same as ERM)",
    'DR':      "Diabetic retinopathy",
    'VD':      "Vitreous detachment / degeneration",
    'RAO/RVO': "Retinal artery or vein occlusion",
    'RNV':     "Retinal neovascularization (retinal-source, distinct from CNV)",
}

# Per-class thresholds published by the OCTCubeM authors in their notebook
# (cell 15). The RAO/RVO=0.126 value in octcube_threshold is suspect; we
# pass it through but the agent should weight it lightly.
OCTCUBE_THRESHOLD   = [0.583, 0.753, 0.567, 0.464, 0.586, 0.446, 0.126, 0.368]
SCREENING_THRESHOLD = [0.847, 0.793, 0.821, 0.777, 0.829, 0.738, 0.715, 0.734]


def _install_flash_attn_stub() -> None:
    """Stub `flash_attn.models.vit.create_block` so OCTCube's models_vit_st_*
    files can be imported without flash_attn installed. We only invoke the
    non-flash code path, so create_block should never actually be called.
    """
    if "flash_attn" in sys.modules:
        return
    stub  = types.ModuleType("flash_attn")
    sm    = types.ModuleType("flash_attn.models")
    sv    = types.ModuleType("flash_attn.models.vit")
    def _not_supported(*a, **kw):
        raise RuntimeError(
            "flash_attn.models.vit.create_block called — OCTCube model was "
            "built with use_flash_attn=True but flash_attn is not installed.")
    sv.create_block = _not_supported
    sm.vit = sv
    stub.models = sm
    sys.modules["flash_attn"] = stub
    sys.modules["flash_attn.models"] = sm
    sys.modules["flash_attn.models.vit"] = sv


def _translate_flash_attn_state_dict(state: dict) -> dict:
    """Convert OCTCubeM checkpoint (flash_attn convention) to regular q/k/v.

    flash_attn blocks store attention as `mixer.Wqkv` (concat [Q;K;V]) +
    `mixer.out_proj`. Our non-flash Block uses separate Linears named
    `attn.q`, `attn.k`, `attn.v`, `attn.proj`. Translate 192 keys per
    24-block model so load_state_dict has 0 missing / 0 unexpected.
    """
    new: dict = {}
    for k, v in state.items():
        if ".mixer.Wqkv.weight" in k:
            base = k.replace(".mixer.Wqkv.weight", ".attn")
            d = v.shape[0] // 3
            new[f"{base}.q.weight"] = v[0:d]
            new[f"{base}.k.weight"] = v[d:2 * d]
            new[f"{base}.v.weight"] = v[2 * d:3 * d]
        elif ".mixer.Wqkv.bias" in k:
            base = k.replace(".mixer.Wqkv.bias", ".attn")
            d = v.shape[0] // 3
            new[f"{base}.q.bias"] = v[0:d]
            new[f"{base}.k.bias"] = v[d:2 * d]
            new[f"{base}.v.bias"] = v[2 * d:3 * d]
        elif ".mixer.out_proj.weight" in k:
            new[k.replace(".mixer.out_proj.weight", ".attn.proj.weight")] = v
        elif ".mixer.out_proj.bias" in k:
            new[k.replace(".mixer.out_proj.bias", ".attn.proj.bias")] = v
        else:
            new[k] = v
    return new


@register
class OCTCubeMAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="oct_volume_octcubem",
        modality="OCT",
        task="classification",
        description=(
            "OCTCubeM 3D foundation model (Liu et al. 2024, arXiv:2408.11227, "
            "BSD-2). 3D ViT-Large trained on 26,685 macular OCT volumes — "
            "8-disease MULTI-LABEL classifier covering DME, AMD, POAG, EPM "
            "(=ERM), DR, VD, RAO/RVO, RNV. Strictly broader than our "
            "broad/FMUE class set: it has dedicated heads for VD and RNV, "
            "and a POAG head that's relevant when peripapillary RNFL is "
            "compromised. Returns RAW per-class probabilities + 3 candidate "
            "threshold operating points. Reason from the raw probabilities, "
            "not the strict top-1 — the model is multi-label by design. "
            "Accepts .dcm / .nii / .npy macular cubes; runs ~0.4 s per "
            "volume on a 24 GB GPU."
        ),
        input_size=None,
        labels=CLASSES,
        confidence_threshold=0.0,
        limitations=[
            "Multi-label by design — top-1 argmax is a poor summary; use raw "
            "per-class probabilities.",
            "The paper's octcube_threshold[6]=0.126 for RAO/RVO is very low "
            "(likely a class-imbalance calibration); treat as advisory.",
            "Trained primarily on Heidelberg Spectralis cubes; other vendors "
            "(Topcon, Zeiss) may need recalibration.",
            "POAG signal here reflects macular GCL/GCIPL changes — not a "
            "substitute for peripapillary RNFL/disc analysis.",
            "Requires the bundled OCTCubeM source tree and the 3D OCTCube "
            "checkpoint.",
        ],
        cost_class="medium",  # 0.4s GPU but 3.4 GB ckpt on first load
        source_dir=str(OCTCUBEM_ROOT),
    )

    def _load_impl(self) -> None:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(
                f"OCTCubeM checkpoint not found at {CKPT_PATH}. "
                "Download with: huggingface-cli download zucksliu/OCTCubeM "
                "OCTCube_multitask_cls.pth --local-dir "
                f"{CKPT_PATH.parent}")

        # Add OCTCubeM dirs to sys.path so `OCTCube.*` and the top-level
        # `util.*` (used by their video_vit) both resolve.
        #
        # NOTE: the FMUE adapter ALSO ships a top-level `util` package. If it
        # loaded first it cached its own `util.*` in sys.modules, which shadows
        # OCTCube's `util.*` and breaks OCTCube's internal
        # `from util.misc import master_print`. Drop any stale `util` modules
        # and force OCTCube's dirs to the front of sys.path so its imports
        # resolve against THIS tree, regardless of which adapter loaded first.
        for _name in [m for m in list(sys.modules)
                      if m == "util" or m.startswith("util.")]:
            del sys.modules[_name]
        for p in (str(OCTCUBEM_ROOT), str(OCTCUBEM_ROOT / "OCTCube")):
            while p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)

        _ensure_home_env_for_octcubem()
        _install_flash_attn_stub()

        import torch
        from OCTCube import models_vit_st_flash_attn
        from OCTCube.util.misc import (
            interpolate_pos_embed,
            interpolate_temporal_pos_embed,
        )
        from OCTCube.util.PatientDataset_inhouse import create_3d_transforms

        # Build model — DIRECT VisionTransformer instantiation so we can pass
        # use_flash_attn=False (the published factory hardcodes True).
        model = models_vit_st_flash_attn.VisionTransformer(
            num_frames=48, t_patch_size=3,
            img_size=256, patch_size=16, in_chans=1,
            num_classes=2 * 8,
            embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
            no_qkv_bias=False, drop_path_rate=0.2,
            norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6),
            global_pool=True, sep_pos_embed=True, cls_embed=True,
            use_flash_attn=False,
        )
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(
                str(CKPT_PATH), map_location="cpu", weights_only=True
            )
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        interpolate_pos_embed(model, state)
        interpolate_temporal_pos_embed(model, state)
        state = _translate_flash_attn_state_dict(state)
        miss, unex = model.load_state_dict(state, strict=False)
        if miss or unex:
            log.warning(
                f"OCTCubeM load: {len(miss)} missing + {len(unex)} unexpected "
                f"keys after flash-attn translation. First missing: {miss[:3]}")

        model.to(self.device).eval()

        # Build the val transform (same as the official notebook). It
        # resamples to 48 frames of 256x256.
        _, val_t = create_3d_transforms(
            input_size=256, num_frames=48, t_patch_size=3)

        self._impl = model
        self._val_transform = val_t

    def _predict_impl(self, image_path: str, **_) -> AdapterResult:
        import torch
        import pydicom as dcm

        src = Path(image_path).resolve()
        if not src.exists():
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="classification",
                error=f"input does not exist: {src}",
            )

        # Load volume — pydicom for .dcm, otherwise via our existing loader
        try:
            if src.suffix.lower() == ".dcm":
                data = dcm.dcmread(str(src), force=True)
                arr = data.pixel_array
                if arr.ndim == 2:
                    arr = arr[None, ...]
            else:
                # Re-use our existing volume loader for .nii/.npy/folder
                from ophagent.data.volume_loader import load_volume
                vol = load_volume(str(src))
                arr = vol.volume
        except Exception as e:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="classification",
                error=f"volume load failed: {type(e).__name__}: {e}",
            )

        # Preprocess using OCTCubeM's published transform
        try:
            from inference_utils import process_dicom_array
        except ImportError:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="classification",
                error="inference_utils not on sys.path (OCTCubeM root install)",
            )

        t0 = time.time()
        tensor, _ = process_dicom_array(arr, self._val_transform)
        tensor = (tensor / 255.0).unsqueeze(0).to(self.device)
        # AdapterBase stores `device` as a STRING ("cuda" / "cpu") rather
        # than a torch.device object — strip out the index portion if any.
        amp_dev = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.no_grad():
            with torch.amp.autocast(amp_dev):
                pred = self._impl(tensor)
                pred = pred.reshape(1, -1, 2)
            pred = torch.softmax(pred, dim=2)
        # .float() FIRST: CPU autocast emits bfloat16, which numpy cannot
        # convert ("unsupported ScalarType BFloat16"); GPU's float16 happened
        # to work, but float() makes both paths safe.
        probs = pred[0, :, 1].float().cpu().numpy().astype(np.float32)
        elapsed = time.time() - t0

        # Build the multi-perspective output for the agent
        raw_probs = {c: float(p) for c, p in zip(CLASSES, probs)}
        order = np.argsort(-probs)
        top_3 = [{"class": CLASSES[i],
                  "probability": float(probs[i]),
                  "description": CLASS_DESCRIPTIONS[CLASSES[i]]}
                 for i in order[:3]]

        fires_05 = [c for c, p in zip(CLASSES, probs) if p > 0.5]
        fires_octcube = [c for c, p, t in zip(CLASSES, probs, OCTCUBE_THRESHOLD)
                         if p > t]
        fires_screening = [c for c, p, t in zip(CLASSES, probs,
                                                 SCREENING_THRESHOLD) if p > t]

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="classification",
            predictions={
                "raw_probabilities": raw_probs,
                "top_3": top_3,
                "argmax_top1": CLASSES[order[0]],
                "fires_at_0.5_threshold": fires_05,
                "fires_at_octcube_threshold": fires_octcube,
                "fires_at_screening_threshold": fires_screening,
                "_thresholds": {
                    "octcube": dict(zip(CLASSES, OCTCUBE_THRESHOLD)),
                    "screening": dict(zip(CLASSES, SCREENING_THRESHOLD)),
                },
                "_note": (
                    "OCTCubeM is multi-label: each class has an independent "
                    "binary head. Treat raw_probabilities as the primary "
                    "signal. The 'octcube_threshold' fires_at list uses the "
                    "paper-published per-class thresholds — note the "
                    "suspiciously low 0.126 cutoff for RAO/RVO."
                ),
            },
            confidence=float(probs[order[0]]),
            metadata={
                "elapsed_s": round(elapsed, 2),
                "model": "OCTCube_multitask_cls (ViT-L 3D, 303M params)",
                "source": "Liu et al. 2024, arXiv:2408.11227",
                "license": "BSD 2-Clause",
            },
        )
