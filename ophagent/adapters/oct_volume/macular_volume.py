"""
Adapter: macular OCT volume → per-slice ensemble + en-face fluid map.

Wraps ``ophagent.agent.volume_processor.analyze_volume`` (which iterates a
volume's B-scans through 4 trained discriminative models — quality,
classification, fluid segmentation, layer segmentation) and packs the
aggregated result into the standard AdapterResult schema so the chat agent
can consume it like any other tool.

Inputs accepted:
  - ``.dcm`` Spectralis / 3rd-party DICOM cube
  - ``.nii`` / ``.nii.gz`` NIfTI
  - ``.npy`` / ``.npz`` pre-processed 3D array
  - a folder of greyscale image B-scans (sorted lexically as the slice order)

Returns:
  - ``classification_consensus``  dict<class, n_slices>  the per-slice vote
                                  histogram across the cube (descending)
  - ``n_slices_total`` / ``n_slices_analyzed`` / ``stride``
  - ``n_slices_with_fluid``       how many B-scans had any IRF/SRF/PED
  - ``total_fluid_voxels``        per-fluid-class pixel sum across the cube
  - ``foveal_slice_idx``          the "most clinically interesting" B-scan
                                  (heuristic: max fluid; mid-cube fallback)
  - ``manufacturer`` / ``model``  decoded from the DICOM header when present
  - ``spacing``                   (z, y, x) mm per voxel if header had it

Two PNGs are also returned for the agent to view:
  - ``foveal_slice``     the foveal B-scan as a JPEG (vision_impression
                         can be run on this directly afterward)
  - ``enface_fluid_map`` a (n_slices × W) heatmap of summed fluid pixels per
                         slice column (the rough en-face fluid distribution)

Runtime: ~70–90s per volume at stride=4 (12–15 slices analysed) on a single
24 GB GPU.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import RELEASE_ROOT, output_path


log = logging.getLogger(__name__)


OUT_BASE = output_path("adapter_figures", "oct_volume_macular")


@register
class OCTVolumeMacularAdapter(AdapterBase):
    """Per-slice ensemble + cube-level aggregation for a macular OCT volume."""

    metadata = ToolMetadata(
        name="oct_volume_macular",
        modality="OCT",
        task="aggregation",
        description=(
            "Macular OCT cube (3D volume) end-to-end analysis. Iterates every "
            "N-th B-scan (default N=4) through quality assessment, an 8-class "
            "disease classifier (AMD/CNV/CSR/DME/DR/DRUSEN/MH/NORMAL), fluid "
            "segmentation (IRF/SRF/PED), and layer segmentation. Returns the "
            "volume-level classification consensus (vote histogram across the "
            "cube), total fluid burden (pixel sum per fluid class), the "
            "heuristically-selected foveal B-scan, and an en-face fluid map. "
            "Accepts .dcm / .nii / .nii.gz / .npy / .npz / image folder. The "
            "agent should call this whenever a volume (rather than a single "
            "B-scan) is attached. After it returns, run vision_impression on "
            "the returned foveal_slice JPEG to add a vision-LLM gestalt."
        ),
        input_size=None,
        labels=[
            # Union of all classifier class sets (broad ∪ octdl ∪ kermany)
            "AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "ERM", "MH",
            "NO", "NORMAL", "RAO", "RVO", "VID",
        ],
        confidence_threshold=0.0,
        limitations=[
            "POAG / glaucoma cannot be diagnosed from a macular cube alone — "
            "use oct_volume_disc (peripapillary RNFL) or fundus glaucoma tools.",
            "Per-slice models are 2D; no true 3D spatial context is used.",
            "The foveal-slice heuristic picks the slice with the most fluid (or "
            "the middle slice as fallback); this is not the anatomic fovea — "
            "in normal cubes the picked slice IS the anatomic centre but in "
            "diseased cubes it is the most pathologic slice.",
            "Trained on publicly-available Kermany / OCTDL / RETOUCH / Duke-DME "
            "datasets — out-of-distribution acquisition (vendor-specific "
            "denoising, non-Heidelberg cubes, peripheral cubes) may degrade.",
        ],
        cost_class="slow",  # 70-90s wall at stride=4
        source_dir=str(RELEASE_ROOT / "ophagent" / "agent"),
    )

    def _load_impl(self) -> None:
        """Lazy-load: build the shared registry + predictor once per adapter."""
        from ophagent.inference.model_registry import create_default_registry
        from ophagent.inference.predictor import OphPredictor
        self._registry = create_default_registry()
        self._predictor = OphPredictor(self._registry)
        OUT_BASE.mkdir(parents=True, exist_ok=True)
        self._impl = "ready"

    def _predict_impl(
        self,
        image_path: str,
        classifier_model: str = "oct_classifier_broad",
        stride: int = 4,
        segment: bool = True,
        **_,
    ) -> AdapterResult:
        from ophagent.agent.volume_processor import analyze_volume

        src = Path(image_path).resolve()
        if not src.exists():
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="aggregation",
                error=f"input does not exist: {src}",
            )

        # Lightweight existence check on the classifier choice (so a bad arg
        # fails loudly rather than silently picking 'broad'). `oct_fmue_16class`
        # routes through the FMUE adapter as a 2D-on-3D volume classifier
        # (every slice classified + aggregated, not just the foveal slice).
        if classifier_model not in (
            "oct_classifier_broad", "oct_classifier_octdl",
            "oct_classifier_kermany", "oct_fmue_16class",
        ):
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="aggregation",
                error=(f"unknown classifier_model '{classifier_model}'. Pick one "
                       "of: oct_classifier_broad, oct_classifier_octdl, "
                       "oct_classifier_kermany, oct_fmue_16class"),
            )

        try:
            stride = max(1, int(stride))
        except Exception:
            stride = 4
        if isinstance(segment, str):
            segment_bool = segment.strip().lower() not in {
                "0", "false", "no", "off", "none", "",
            }
        else:
            segment_bool = bool(segment)

        # Resolve an adapter-based classifier (FMUE) — it replaces the
        # OphPredictor per-slice classifier inside analyze_volume.
        classifier_adapter = None
        if classifier_model == "oct_fmue_16class":
            from ophagent.adapters import GLOBAL_REGISTRY
            classifier_adapter = GLOBAL_REGISTRY.get("oct_fmue_16class")

        t0 = time.time()
        try:
            analysis = analyze_volume(
                volume_path=str(src),
                registry=self._registry,
                predictor=self._predictor,
                classifier_model=classifier_model,
                classifier_adapter=classifier_adapter,
                run_segmentation=segment_bool,
                slice_stride=stride,
                progress=False,
            )
        except Exception as e:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="OCT",
                task="aggregation",
                error=f"analyze_volume failed: {type(e).__name__}: {e}",
            )
        elapsed = time.time() - t0

        # ── Persist the foveal slice + en-face fluid map as figures ──
        run_token = f"{src.stem}_{int(time.time())}"
        out_dir = OUT_BASE / run_token
        out_dir.mkdir(parents=True, exist_ok=True)

        figures: dict[str, str] = {}

        # 1) foveal slice as JPEG
        try:
            fov_img = analysis.volume.slice(analysis.foveal_slice_idx)
            fov_path = out_dir / f"foveal_slice_{analysis.foveal_slice_idx}.jpg"
            ok, buf = cv2.imencode(".jpg", fov_img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                buf.tofile(str(fov_path))
                figures["foveal_slice"] = str(fov_path)
        except Exception as _e:
            figures["foveal_slice_error"] = f"{type(_e).__name__}: {_e}"

        # 2) en-face fluid map (slice × column → grey heatmap)
        try:
            enface = analysis.enface_fluid
            if enface is not None and enface.size > 0:
                if enface.max() > 0:
                    norm = (enface.astype(np.float32) / enface.max()
                            * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    norm = enface.astype(np.uint8)
                # Apply a colormap to make it readable as an "en-face fluid map"
                heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                enf_path = out_dir / "enface_fluid_map.png"
                ok, buf = cv2.imencode(".png", heat)
                if ok:
                    buf.tofile(str(enf_path))
                    figures["enface_fluid_map"] = str(enf_path)
        except Exception as _e:
            figures["enface_error"] = f"{type(_e).__name__}: {_e}"

        # ── Build the flat predictions dict for the agent ──
        meta = analysis.volume.metadata or {}
        predictions: dict[str, Any] = {
            "classification_consensus": dict(analysis.classification_consensus),
            "classification_max_prob": {k: round(float(v), 4) for k, v
                                        in analysis.classification_max_prob.items()},
            "classifier_used": analysis.classifier_name,
            "n_slices_total": int(analysis.volume.n_slices),
            "n_slices_analyzed": int(len(analysis.slices)),
            "stride": stride,
            "n_slices_with_fluid": int(analysis.slice_with_fluid_count),
            "total_fluid_voxels": {k: int(v)
                                   for k, v in analysis.total_fluid_voxels.items()},
            "foveal_slice_idx": int(analysis.foveal_slice_idx),
            "manufacturer": meta.get("Manufacturer", "unknown"),
            "model": meta.get("Model", "unknown"),
            "study_date": meta.get("StudyDate", ""),
            "spacing_mm_zyx": (list(analysis.volume.spacing)
                               if analysis.volume.spacing else None),
            "overlay_files": figures,
        }

        # ── FMUE 16-class evidential classifier on the foveal slice ──
        # FMUE has a wider class set than oct_classifier_broad (covers iERM,
        # acute_RAO, mCNV, PCV, MTM, RP, VKH that broad lacks) AND emits an
        # EDL vacuity score in [0,1] that flags out-of-distribution inputs
        # (POAG, VD-only cubes, etc.) where the classifier consensus alone
        # would mislead. We call FMUE ONLY on the foveal slice because FMUE
        # was trained on foveal-centred B-scans — running it on peripheral
        # slices would inflate uncertainty for the wrong reason.
        if "foveal_slice" in figures:
            try:
                from ophagent.adapters import GLOBAL_REGISTRY
                fmue = GLOBAL_REGISTRY.get("oct_fmue_16class")
                if fmue is not None:
                    fr = fmue.predict(figures["foveal_slice"])
                    if fr.success:
                        predictions["fmue_foveal"] = {
                            "predicted_class": fr.predictions["predicted_class"],
                            "description": fr.predictions.get("description"),
                            "top_3": fr.predictions["top_3"],
                            "evidential_uncertainty":
                                fr.predictions["evidential_uncertainty"],
                            "_note": (
                                "EDL vacuity score in [0,1]. "
                                "<0.3 = confident; 0.3-0.5 = moderate; "
                                ">0.5 = likely out-of-distribution"
                            ),
                        }
                    else:
                        predictions["fmue_foveal_error"] = (fr.error or "")[:300]
            except Exception as _e:
                predictions["fmue_foveal_error"] = f"{type(_e).__name__}: {_e}"

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="OCT",
            task="aggregation",
            predictions=predictions,
            figures={k: v for k, v in figures.items() if not k.endswith("_error")},
            metadata={
                "elapsed_s": round(elapsed, 1),
                "output_dir": str(out_dir),
            },
        )
