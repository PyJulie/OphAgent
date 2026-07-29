"""
Adapter base classes.

Every external model (CFP / OCT / UWF / FFA) is wrapped behind a single
interface so the agent can call them uniformly. We never modify the user's
original code or weights — we add the model's source dir to `sys.path` and
import its inference module on demand.

ToolMetadata follows the OphAgent paper convention (modality + inputs/outputs
+ labels + confidence threshold τ + known limitations L).
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar


log = logging.getLogger(__name__)

# Global lock serialising model LOADING across threads. Multi-worker evals
# (ThreadPoolExecutor) otherwise hit two races on the first image: (1) several
# threads doing `from transformers import AutoModel` at once — transformers'
# lazy `_LazyModule` is not thread-safe and raises "cannot import name
# AutoModel"; (2) several multi-GB checkpoints (retsam 10.8 GB, the FLAIR +
# RetiZero CLIP fleet) allocating GPU memory simultaneously → CUDA OOM / hard
# process abort. Loading is a one-time cost per model, so serialising it is
# free after warm-up; INFERENCE is unaffected and stays fully concurrent.
# RLock (not Lock) so a `_load_impl` that itself triggers a nested adapter
# load on the same thread cannot self-deadlock.
_LOAD_LOCK = threading.RLock()


@dataclass
class ToolMetadata:
    """Per-tool descriptor used by the planner to pick + interpret tools."""
    name: str
    modality: str               # "CFP" | "OCT" | "UWF" | "FFA"
    task: str                   # "classification" | "segmentation" | "detection" | "quality" | "captioning"
    description: str
    input_size: tuple[int, ...] | None = None
    labels: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.5     # τ — below this we mark "undetermined"
    limitations: list[str] = field(default_factory=list)   # L — known failure modes
    requires_tools: list[str] = field(default_factory=list)  # dependencies (e.g. "od_detection")
    cost_class: str = "fast"    # "fast" | "medium" | "slow" — guides planner
    # extra:
    source_dir: str | None = None  # filesystem path to the user's project dir


@dataclass
class AdapterResult:
    """Uniform return type from any adapter."""
    success: bool
    tool: str
    modality: str
    task: str
    predictions: dict[str, Any] = field(default_factory=dict)  # standardized payload
    confidence: float | None = None        # primary scalar confidence
    undetermined: bool = False             # below threshold → uncertain
    raw_output: Any = None                 # full original output for debugging
    metadata: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)   # name → file path
    error: str | None = None

    def to_jsonable(self) -> dict:
        """Strip numpy / tensors before serialising for the LLM."""
        import numpy as np
        def _clean(v):
            if isinstance(v, (np.ndarray, )):
                return f"<ndarray shape={v.shape} dtype={v.dtype}>"
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                return f"<tensor shape={tuple(v.shape)} dtype={v.dtype}>"
            if isinstance(v, dict):
                return {k: _clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            return v
        return {
            "tool": self.tool,
            "modality": self.modality,
            "task": self.task,
            "success": self.success,
            "predictions": _clean(self.predictions),
            "confidence": self.confidence,
            "undetermined": self.undetermined,
            "figures": self.figures,
            "metadata": _clean(self.metadata),
            "error": self.error,
        }


class AdapterBase:
    """Adapter abstract base — subclasses MUST set `metadata` and implement
    `_load_impl()` + `_predict_impl()`."""

    metadata: ClassVar[ToolMetadata]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._loaded = False
        self._impl: Any = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._loaded:
            return
        # Double-checked locking: cheap path above for the already-loaded
        # common case; under the lock we re-check so only the FIRST thread
        # actually loads while the rest wait, then return immediately.
        with _LOAD_LOCK:
            if self._loaded:
                return
            log.info(f"[adapter:{self.metadata.name}] loading from {self.metadata.source_dir}")
            try:
                self._load_impl()
                self._loaded = True
            except Exception as e:
                log.exception(f"[adapter:{self.metadata.name}] load failed")
                raise RuntimeError(f"{self.metadata.name} failed to load: {e}") from e

    def unload(self) -> None:
        self._impl = None
        self._loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def is_loaded(self) -> bool:
        return self._loaded

    # ── inference ──────────────────────────────────────────────────────────
    def predict(self, image_path: str | Path, **kwargs) -> AdapterResult:
        try:
            if not self._loaded:
                self.load()
            result = self._predict_impl(str(image_path), **kwargs)
            # Apply confidence threshold gating
            if result.confidence is not None and result.confidence < self.metadata.confidence_threshold:
                result.undetermined = True
            return result
        except Exception as e:
            log.exception(f"[adapter:{self.metadata.name}] predict failed")
            return AdapterResult(
                success=False,
                tool=self.metadata.name,
                modality=self.metadata.modality,
                task=self.metadata.task,
                error=f"{type(e).__name__}: {e}",
            )

    # ── subclass hooks ─────────────────────────────────────────────────────
    def _load_impl(self) -> None:
        raise NotImplementedError

    def _predict_impl(self, image_path: str, **kwargs) -> AdapterResult:
        raise NotImplementedError

    # ── helper for subclasses ──────────────────────────────────────────────
    @staticmethod
    def _ensure_path(src_dir: str | Path) -> None:
        """Add the user's project dir to sys.path so its code is importable."""
        p = str(Path(src_dir).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    @staticmethod
    def _import_fresh(module_name: str, src_dir: str | Path):
        """Import (or re-import) a module from src_dir."""
        AdapterBase._ensure_path(src_dir)
        if module_name in sys.modules:
            return sys.modules[module_name]
        return importlib.import_module(module_name)


class AdapterRegistry:
    """Lightweight registry of available adapters keyed by tool name."""

    def __init__(self):
        self._classes: dict[str, type[AdapterBase]] = {}
        self._instances: dict[str, AdapterBase] = {}

    def register(self, cls: type[AdapterBase]) -> None:
        name = cls.metadata.name
        self._classes[name] = cls
        log.info(f"[registry] registered {name} ({cls.metadata.modality}/{cls.metadata.task})")

    def get(self, name: str, device: str = "cuda") -> AdapterBase:
        from ..checkpoint_config import tool_is_enabled

        if not tool_is_enabled(name):
            raise RuntimeError(f"Adapter '{name}' is disabled in checkpoint settings")
        if name not in self._instances:
            if name not in self._classes:
                raise KeyError(f"Unknown adapter '{name}'. Registered: {list(self._classes)}")
            self._instances[name] = self._classes[name](device=device)
        return self._instances[name]

    def list_tools(self, modality: str | None = None) -> list[ToolMetadata]:
        out = []
        for cls in self._classes.values():
            if modality is None or cls.metadata.modality == modality:
                out.append(cls.metadata)
        return out

    def tools_for(self, modality: str, task: str) -> list[ToolMetadata]:
        """Return single-modality tools matching (modality, task), sorted by
        preferred-use rank (composite/strongest first)."""
        # Curated ranking — higher rank = run first.
        priority = {
            # CFP quality
            "cfp_efiqa": 90, "cfp_quality_robust": 70, "cfp_eyeq": 50,
            # CFP classification — workups bundle PDR + CLIP with confound
            # cross-check, so they outrank the bare PDR cascade. The CLIP
            # ensemble (3 independent CLIPs) outranks any single CLIP.
            "cfp_paired5": 90, "cfp_dr_workup": 85, "cfp_dr_421_assessment": 83,
            "cfp_glaucoma_workup": 80,
            "cfp_clip_ensemble": 78,
            "cfp_clip_multi_disease": 70, "cfp_retizero": 68, "cfp_flair": 66,
            "cfp_pdr_cascade": 55, "cfp_glaucoma": 50,
            # OCT classification / segmentation
            "oct_fmue_16class": 90, "oct_fluid_segmentation": 70,
            "oct_layer_segmentation": 60, "oct_quality": 50,
            # FFA
            "ffa_paired5": 90, "ffa_classification": 60,
            # UWF — two complementary classifiers; at high effort both run and
            # cross-check (7-class single-label + 8-label multi-label).
            "uwf_disease_7class": 82, "uwf_multi_disease": 80,
            "uwf_vessel_segmentation": 50,
        }
        out = [c.metadata for c in self._classes.values()
               if c.metadata.modality == modality and c.metadata.task == task]
        out.sort(key=lambda m: -priority.get(m.name, 0))
        return out

    def cross_modal_tools_for(self, modalities: list[str]) -> list[ToolMetadata]:
        """Return multi-modality (composite) tools applicable when these
        modalities are co-present. Sorted by ranked preference."""
        modset = set(modalities)
        # Mapping from cross-tool name to the modality set it consumes.
        cross_needs = {
            "cross_cfp_ffa_paired":   {"CFP", "FFA"},
            "cross_cfp_ffa_softvote": {"CFP", "FFA"},
            "cross_cfp_ffa":          {"CFP", "FFA"},   # legacy 40% — last resort
            "cross_cfp_oct":          {"CFP", "OCT"},
        }
        priority = {
            "cross_cfp_ffa_paired": 100, "cross_cfp_ffa_softvote": 80,
            "cross_cfp_oct": 70, "cross_cfp_ffa": 10,
        }
        out = []
        for cls in self._classes.values():
            name = cls.metadata.name
            if name in cross_needs and cross_needs[name].issubset(modset):
                out.append(cls.metadata)
        out.sort(key=lambda m: -priority.get(m.name, 0))
        return out

    def predict(self, name: str, image_path: str | Path, **kwargs) -> AdapterResult:
        return self.get(name).predict(image_path, **kwargs)

    def unload_all(self) -> None:
        for inst in self._instances.values():
            inst.unload()
        self._instances.clear()


# Singleton — adapters self-register on import via the helper below.
GLOBAL_REGISTRY = AdapterRegistry()


def register(cls: type[AdapterBase]) -> type[AdapterBase]:
    """Class decorator: register the adapter into the global registry."""
    GLOBAL_REGISTRY.register(cls)
    return cls
