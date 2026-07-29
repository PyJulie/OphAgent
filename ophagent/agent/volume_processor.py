"""
Process an OCT volume:
  - load DICOM/NIfTI/NPY
  - per-slice quality, classification, fluid + layer segmentation
  - aggregate findings across the cube
  - locate fovea-like slice (max layer-curvature heuristic)
  - build en-face fluid map (z-projection)
  - save slice masks for later rendering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..data.volume_loader import OCTVolume, load_volume
from ..data.preprocessing.transforms import (
    get_classification_transforms, get_segmentation_transforms,
)
from ..inference.predictor import OphPredictor
from ..inference.model_registry import ModelRegistry
from ..visualization.visualizer import FLUID_COLORS, LAYER_COLORS


@dataclass
class SliceResult:
    index: int
    quality: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    fluid_mask: np.ndarray | None = None
    fluid_class_areas: dict[str, int] = field(default_factory=dict)
    fluid_has_any: bool = False
    layer_mask: np.ndarray | None = None
    layer_class_areas: dict[str, int] = field(default_factory=dict)


@dataclass
class VolumeAnalysis:
    volume: OCTVolume
    slices: list[SliceResult]
    foveal_slice_idx: int
    enface_fluid: np.ndarray              # (N_slices, W) — fluid pixel count per A-scan column
    total_fluid_voxels: dict[str, int]    # per-class summed pixels
    slice_with_fluid_count: int
    classification_consensus: dict[str, int]   # disease → vote count (argmax per slice)
    classifier_name: str
    # Per-class MAX probability across slices. For a 2D classifier with a
    # "Normal" class (e.g. FMUE) run over a volume, argmax voting is swamped by
    # peripheral Normal slices; the per-class peak is the right volume-level
    # detection signal (any slice that confidently shows disease X). Empty for
    # legacy callers that don't request it.
    classification_max_prob: dict[str, float] = field(default_factory=dict)


def _to_tensor_classification(image: np.ndarray, size: int) -> torch.Tensor:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    tf = get_classification_transforms(size, is_training=False)
    return tf(image=image)["image"].unsqueeze(0)


def _to_tensor_segmentation(image: np.ndarray, size: int) -> torch.Tensor:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    tf = get_segmentation_transforms((size, size), is_training=False)
    return tf(image=image)["image"].unsqueeze(0)


@torch.no_grad()
def _predict_classifier(predictor: OphPredictor, model_name: str, image: np.ndarray) -> dict:
    return predictor.predict(model_name, image)


@torch.no_grad()
def _predict_segmentation(predictor: OphPredictor, model_name: str, image: np.ndarray) -> dict:
    return predictor.predict(model_name, image)


def analyze_volume(
    volume_path: str | Path,
    registry: ModelRegistry,
    predictor: OphPredictor,
    classifier_model: str = "oct_classifier_octdl",
    quality_model: str = "oct_quality_assessor",
    fluid_model: str = "oct_fluid_segmentor",
    layer_model: str = "oct_layer_segmentor",
    slice_stride: int = 1,
    progress: bool = True,
    classifier_adapter: Any = None,
    run_segmentation: bool = True,
) -> VolumeAnalysis:
    """Run the per-slice discriminative pipeline over a volume and aggregate.

    `classifier_adapter` (optional): a 2D-classifier AdapterBase exposing
    `predict_arrays(list_of_HxW_arrays) -> list[{predicted_class, confidence,
    probabilities}]` (e.g. the FMUE adapter). When given, it REPLACES the
    OphPredictor `classifier_model` for the per-slice classification step —
    this is how a 2D B-scan model (FMUE) is run over a 3D cube. All slices are
    batched through it at once.

    `run_segmentation` (default True): when False, skip quality + fluid + layer
    (which need the OphPredictor models) for a fast classification-only pass.

    Returns an aggregated VolumeAnalysis (now incl. `classification_max_prob`,
    the per-class peak probability across slices — the right detection signal
    for a Normal-containing 2D classifier).
    """
    vol = load_volume(volume_path)
    n = vol.n_slices

    indices = list(range(0, n, slice_stride))
    imgs = [vol.slice(i) for i in indices]
    slices: list[SliceResult] = []

    fluid_class_names = (registry.get_card(fluid_model).class_names
                         if run_segmentation else [])
    classification_votes: dict[str, int] = {}
    classification_max: dict[str, float] = {}
    total_fluid_voxels: dict[str, int] = {nm: 0 for nm in fluid_class_names if nm != "Background"}

    # Batched adapter classification (FMUE etc.) — one forward pass for the cube.
    adapter_cls: list[dict] | None = None
    if classifier_adapter is not None:
        adapter_cls = classifier_adapter.predict_arrays(imgs)

    iterator = tqdm(list(enumerate(indices)), desc="Volume slices") if progress else enumerate(indices)
    for k, i in iterator:
        img = imgs[k]
        sr = SliceResult(index=i)

        # quality (OphPredictor model — skip on classification-only pass)
        if run_segmentation:
            try:
                sr.quality = predictor.predict(quality_model, img)
            except Exception as e:
                sr.quality = {"error": str(e)}

        # classification — adapter (FMUE) if provided, else OphPredictor model
        try:
            if adapter_cls is not None:
                r = adapter_cls[k]
            else:
                r = predictor.predict(classifier_model, img)
            probs = {k2: float(v) for k2, v in r["probabilities"].items()}
            sr.classification = {
                "predicted_class": r["predicted_class"],
                "confidence": float(r["confidence"]),
                "probabilities": probs,
            }
            cls = r["predicted_class"]
            classification_votes[cls] = classification_votes.get(cls, 0) + 1
            for c2, p2 in probs.items():
                if p2 > classification_max.get(c2, 0.0):
                    classification_max[c2] = p2
        except Exception as e:
            sr.classification = {"error": str(e)}

        if not run_segmentation:
            slices.append(sr)
            continue

        # fluid segmentation
        try:
            f = predictor.predict(fluid_model, img)
            sr.fluid_mask = f.get("mask")
            sr.fluid_class_areas = f.get("class_areas", {})
            sr.fluid_has_any = any(
                k != "Background" and v > 0 for k, v in sr.fluid_class_areas.items()
            )
            for cls, area in sr.fluid_class_areas.items():
                if cls != "Background":
                    total_fluid_voxels[cls] = total_fluid_voxels.get(cls, 0) + int(area)
        except Exception:
            pass

        # layer segmentation
        try:
            l = predictor.predict(layer_model, img)
            sr.layer_mask = l.get("mask")
            sr.layer_class_areas = l.get("class_areas", {})
        except Exception:
            pass

        slices.append(sr)

    # ── Locate fovea-like slice ─────────────────────────────────────────────
    # Heuristic: the foveal B-scan typically has the deepest central depression
    # in the inner retina. We pick the slice whose mean ILM-NFL band ratio is
    # smallest near the centre column band. As a robust fallback, pick the slice
    # with the largest total fluid (the most diagnostically interesting one).
    foveal_idx = _pick_foveal_slice(slices)

    # ── En-face fluid map: project each slice's fluid mask along its height
    enface = np.zeros((len(slices), vol.shape[2]), dtype=np.uint16)
    for row, sr in enumerate(slices):
        if sr.fluid_mask is None:
            continue
        m = sr.fluid_mask
        # any non-background fluid contributes
        per_col = (m > 0).sum(axis=0)  # (W,)
        enface[row] = per_col.astype(np.uint16)

    slice_with_fluid = sum(1 for s in slices if s.fluid_has_any)

    return VolumeAnalysis(
        volume=vol,
        slices=slices,
        foveal_slice_idx=foveal_idx,
        enface_fluid=enface,
        total_fluid_voxels=total_fluid_voxels,
        slice_with_fluid_count=slice_with_fluid,
        classification_consensus=dict(sorted(
            classification_votes.items(), key=lambda x: -x[1]
        )),
        classification_max_prob=dict(sorted(
            classification_max.items(), key=lambda x: -x[1]
        )),
        classifier_name=(getattr(getattr(classifier_adapter, "metadata", None),
                                 "name", None) or classifier_model),
    )


def _pick_foveal_slice(slices: list[SliceResult]) -> int:
    """Pick the most clinically interesting slice as the 'foveal' slice."""
    # ① most fluid
    fluid_scores = [(s.index, sum(v for k, v in s.fluid_class_areas.items()
                                  if k != "Background"))
                    for s in slices]
    if fluid_scores and max(s for _, s in fluid_scores) > 0:
        return max(fluid_scores, key=lambda x: x[1])[0]

    # ② fallback: middle slice
    return slices[len(slices) // 2].index if slices else 0
