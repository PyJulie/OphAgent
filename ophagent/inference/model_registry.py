"""
Model registry: manages loading, caching, and routing to trained OCT models.

Each model is registered with its task type, checkpoint path, and configuration.
The registry enables the agent to discover and load models dynamically.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.paths import CKPT_DIR

# Serialise checkpoint loading across threads (multi-worker evals) — concurrent
# torch.load + .to(cuda) of several models races and can OOM the GPU. See the
# matching note in ophagent/adapters/base.py.
_REGISTRY_LOAD_LOCK = threading.RLock()


@dataclass
class ModelCard:
    name: str
    task: str  # classification, segmentation, generation, quality, denoising, super_resolution
    description: str
    checkpoint_path: str
    model_class: str
    config: dict = field(default_factory=dict)
    diseases: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    input_size: int = 224
    device: str = "cuda"
    version: str = "1.0"


class ModelRegistry:
    """Central registry of all trained OCT models."""

    def __init__(self, registry_path: str | Path | None = None):
        self._cards: dict[str, ModelCard] = {}
        self._loaded_models: dict[str, nn.Module] = {}
        self.registry_path = Path(registry_path) if registry_path else None

        if self.registry_path and self.registry_path.exists():
            self._load_registry()

    def register(self, card: ModelCard) -> None:
        self._cards[card.name] = card
        if self.registry_path:
            self._save_registry()

    def get_card(self, name: str) -> ModelCard:
        if name not in self._cards:
            available = list(self._cards.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._cards[name]

    def list_models(self, task: str | None = None) -> list[ModelCard]:
        cards = list(self._cards.values())
        if task:
            cards = [c for c in cards if c.task == task]
        return cards

    def load_model(self, name: str) -> nn.Module:
        if name in self._loaded_models:
            return self._loaded_models[name]
        with _REGISTRY_LOAD_LOCK:
            if name in self._loaded_models:   # re-check inside lock
                return self._loaded_models[name]

            card = self.get_card(name)
            model = self._instantiate_model(card)

            checkpoint_path = Path(card.checkpoint_path)
            if checkpoint_path.exists():
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint for '{name}' from {checkpoint_path}")

            device = torch.device(card.device if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()
            self._loaded_models[name] = model
            return model

    def unload_model(self, name: str) -> None:
        if name in self._loaded_models:
            del self._loaded_models[name]
            torch.cuda.empty_cache()

    def _instantiate_model(self, card: ModelCard) -> nn.Module:
        from ..models.classification.classifier import (
            OCTClassifier, OCTQualityAssessor, MultiModalClassifier,
        )
        from ..models.segmentation.unet import (
            OCTUNet, OCTFluidSegmentor, OCTLayerSegmentor,
        )
        from ..models.generation.diffusion import (
            OCTDiffusionModel, OCTDenoiser, OCTSuperResolver,
        )

        model_map = {
            "OCTClassifier": OCTClassifier,
            "OCTQualityAssessor": OCTQualityAssessor,
            "MultiModalClassifier": MultiModalClassifier,
            "OCTUNet": OCTUNet,
            "OCTFluidSegmentor": OCTFluidSegmentor,
            "OCTLayerSegmentor": OCTLayerSegmentor,
            "OCTDiffusionModel": OCTDiffusionModel,
            "OCTDenoiser": OCTDenoiser,
            "OCTSuperResolver": OCTSuperResolver,
        }

        cls = model_map.get(card.model_class)
        if cls is None:
            raise ValueError(f"Unknown model class: {card.model_class}")
        return cls(**card.config)

    def _save_registry(self) -> None:
        # Thread-safe ATOMIC write. The old code did open(path,"w") (which
        # truncates the file to empty before writing) with NO lock, so under a
        # multi-worker eval one thread's register()->_save_registry() would
        # leave registry.json momentarily empty while ANOTHER thread's
        # _load_registry() did json.load() on it -> JSONDecodeError "Expecting
        # value: line 1 column 1". We now serialise on the registry lock and
        # write to a per-thread temp file + os.replace() (atomic rename), so a
        # concurrent reader NEVER sees a half-written/empty file.
        data = {name: asdict(card) for name, card in self._cards.items()}
        with _REGISTRY_LOAD_LOCK:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.registry_path.with_name(
                f"{self.registry_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            for attempt in range(8):
                try:
                    os.replace(tmp, self.registry_path)
                    break
                except PermissionError:
                    if attempt == 7:
                        # The registry is a rebuildable cache. Under very high
                        # thread counts on Windows another process can briefly
                        # hold registry.json; do not fail an image benchmark.
                        try:
                            tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return
                    time.sleep(0.05 * (attempt + 1))

    def _load_registry(self) -> None:
        with _REGISTRY_LOAD_LOCK:
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError, OSError):
                # Empty / half-written / missing — benign: create_default_registry()
                # re-register()s every card right after, so we self-heal.
                return
        for name, card_dict in data.items():
            self._cards[name] = ModelCard(**card_dict)


def create_default_registry(
    checkpoints_dir: str | Path | None = None,
) -> ModelRegistry:
    """Create a registry with default model configurations.

    Models are registered but checkpoints may not exist yet (need training).
    """
    checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else CKPT_DIR
    registry = ModelRegistry(registry_path=checkpoints_dir / "registry.json")

    registry.register(ModelCard(
        name="oct_classifier_kermany",
        task="classification",
        description="4-class OCT disease classifier trained on Kermany dataset "
        "(CNV, DME, Drusen, Normal)",
        checkpoint_path=f"{checkpoints_dir}/oct/classification/kermany/best.pt",
        model_class="OCTClassifier",
        config={"backbone": "resnet50", "num_classes": 4, "pretrained": False},
        diseases=["CNV", "DME", "Drusen", "Normal"],
        class_names=["CNV", "DME", "DRUSEN", "NORMAL"],
        input_size=224,
    ))

    registry.register(ModelCard(
        name="oct_classifier_octdl",
        task="classification",
        description="7-class OCT disease classifier trained on OCTDL dataset "
        "(AMD, DME, ERM, Normal, RAO, RVO, VID)",
        checkpoint_path=f"{checkpoints_dir}/oct/classification/octdl/best.pt",
        model_class="OCTClassifier",
        config={"backbone": "resnet50", "num_classes": 7, "pretrained": False},
        diseases=["AMD", "DME", "ERM", "Normal", "RAO", "RVO", "VID"],
        class_names=["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"],
        input_size=224,
    ))

    registry.register(ModelCard(
        name="oct_classifier_broad",
        task="classification",
        description="8-class OCT disease classifier trained on OCT-C8 "
        "(AMD, CNV, CSR, DME, DR, Drusen, MH, Normal)",
        checkpoint_path=f"{checkpoints_dir}/oct/classification/oct_c8/best.pt",
        model_class="OCTClassifier",
        config={"backbone": "swin_tiny", "num_classes": 8, "pretrained": False},
        diseases=["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "MH", "Normal"],
        class_names=["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"],
        input_size=224,
    ))

    registry.register(ModelCard(
        name="oct_quality_assessor",
        task="quality",
        description="OCT image quality assessment (high vs low) "
        "trained with synthetic degradations on Kermany/OCTDL/OCT-C8",
        checkpoint_path=f"{checkpoints_dir}/oct/quality_assessor/best.pt",
        model_class="OCTClassifier",
        config={"backbone": "resnet18", "num_classes": 2, "pretrained": False},
        class_names=["high", "low"],
        input_size=224,
    ))

    registry.register(ModelCard(
        name="oct_fluid_segmentor",
        task="segmentation",
        description="Retinal fluid segmentation (IRF, SRF, PED) "
        "trained on RETOUCH multi-device data",
        checkpoint_path=f"{checkpoints_dir}/oct/fluid_segmentor/best.pt",
        model_class="OCTUNet",
        config={"backbone": "resnet50", "num_classes": 4, "pretrained": False},
        diseases=["AMD", "DME", "RVO"],
        class_names=["Background", "IRF", "SRF", "PED"],
        input_size=384,
    ))

    registry.register(ModelCard(
        name="oct_layer_segmentor",
        task="segmentation",
        description="Retinal layer segmentation (8 boundaries → 10-region mask) "
        "trained on Duke DME (Chiu 2015)",
        checkpoint_path=f"{checkpoints_dir}/oct/layer_segmentor/best.pt",
        model_class="OCTUNet",
        config={"backbone": "resnet50", "num_classes": 10, "pretrained": False},
        class_names=["BG-above", "ILM-NFL", "NFL-IPL", "IPL-INL", "INL-OPL",
                     "OPL-ONL", "ONL-ISM", "ISM-ISOS", "ISOS-RPE", "Below-RPE"],
        input_size=384,
    ))

    registry.register(ModelCard(
        name="oct_denoiser",
        task="denoising",
        description="OCT speckle noise removal",
        checkpoint_path=f"{checkpoints_dir}/oct/denoiser/best.pt",
        model_class="OCTDenoiser",
        config={"in_channels": 1},
        input_size=256,
    ))

    registry.register(ModelCard(
        name="oct_super_resolver",
        task="super_resolution",
        description="2x super-resolution for low-quality OCT images",
        checkpoint_path=f"{checkpoints_dir}/oct/super_resolver/best.pt",
        model_class="OCTSuperResolver",
        config={"scale_factor": 2, "in_channels": 1},
        input_size=256,
    ))

    return registry
