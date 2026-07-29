#!/usr/bin/env python3
"""
retsam-2.0 — mask-only inference with DataLoader + GPU batching.

Throughput features:
    * **GPU batching** (``--batch_size N``) — one model.forward over N images.
    * **CPU preprocessing in parallel** (``--num_workers W``) — image I/O,
      black-edge removal, and resize run in worker processes, overlapped
      with GPU compute.
    * **Mixed precision** (``--fp16``) — ``torch.autocast`` for ~1.5-2× GPU
      throughput at negligible segmentation-accuracy cost.
    * **torch.compile** (``--torch_compile``) — opt-in; compile overhead
      pays off over ~50+ images.
    * **Resume** (``--resume``) — skip images whose ``infer_summary.json``
      is newer than the source image.

Example (a few thousand images on a single GPU):
    python scripts/infer.py \\
        --input_dir ./images \\
        --output_dir ./out \\
        --model_path retsam.ckpt \\
        --output_channels "(2,3,2,4,6,11,6,2)" \\
        --has_coordinate_head --num_coordinates 2 \\
        --batch_size 8 --num_workers 4 --fp16 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.config import InferenceConfig
from inference.image_processor import ImageProcessor
from inference.model_loader import ModelLoader
from inference.postprocessing import NoiseFilter
from inference.utils import (
    Logger,
    convert_numpy_types,
    find_image_files,
    parse_output_channels,
)
from inference.visualizer import Visualizer


# --------------------------------------------------------------------------- Dataset

class _InferDataset(Dataset):
    """Apply ImageProcessor in DataLoader workers so preprocessing overlaps
    with the GPU forward pass. Each item yields:

        (tensor, image_data_without_tensor)

    where ``image_data_without_tensor`` is the dict produced by
    ``ImageProcessor.load_and_preprocess`` minus the raw tensor — we keep the
    ``original_shape``, ``crop_coords``, and raw image arrays so the visualizer
    can paste masks back onto the original canvas without a second disk read.
    """

    def __init__(self, paths: List[str], image_processor: ImageProcessor):
        self.paths = list(paths)
        self.ip = image_processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        path = self.paths[idx]
        data = self.ip.load_and_preprocess(path)
        tensor = data.pop("tensor").squeeze(0)  # (3, H, W)
        data["image_path"] = path
        return tensor, data


def _collate(items):
    """Stack tensors, keep per-image metadata as a Python list."""
    tensors = torch.stack([t for t, _ in items], dim=0)
    metas = [m for _, m in items]
    return tensors, metas


# --------------------------------------------------------------------------- core runner

class BatchedMaskInference:
    """Run mask-only inference with GPU batching + CPU preprocessing workers.

    The predictor path consciously avoids ``inference.predictor.Predictor``
    because that class's noise-filter assumes batch_size=1. We call the model
    directly on a batched tensor and fan out the per-image noise filtering
    afterwards.
    """

    def __init__(self,
                 model_path: str,
                 output_dir: str,
                 device: str = "cuda",
                 output_channels: Optional[Tuple[int, ...]] = None,
                 has_coordinate_head: Optional[bool] = None,
                 num_coordinates: Optional[int] = None,
                 enable_noise_filter: bool = True,
                 save_overlay: bool = False,
                 input_root: Optional[str] = None,
                 quiet: bool = True,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 use_fp16: bool = False,
                 torch_compile: bool = False,
                 resume: bool = False):
        self.config = InferenceConfig()
        self.device = device
        self.output_dir = output_dir
        self.input_root = Path(input_root).resolve() if input_root else None
        self.save_overlay = save_overlay
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(0, int(num_workers))
        self.use_fp16 = bool(use_fp16) and device.startswith("cuda")
        self.enable_noise_filter = bool(enable_noise_filter)
        self.resume = bool(resume)
        self.quiet = quiet
        Logger.set_quiet(quiet)

        self.model_loader = ModelLoader(device)
        self.image_processor = ImageProcessor()
        self.visualizer = Visualizer(self.config)
        self.noise_filter = NoiseFilter(self.config) if enable_noise_filter else None

        Logger.info(f"Loading model from {model_path}")
        self.model = self.model_loader.load(
            model_path, multitask=True,
            output_channels=output_channels,
            has_coordinate_head=has_coordinate_head,
            num_coordinates=num_coordinates,
        )
        self.model.eval()

        if torch_compile:
            Logger.info("Wrapping model with torch.compile (first forward will be slow)")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                Logger.warning(f"torch.compile failed: {e}. Falling back to eager.")

        self.output_channels = output_channels or self.config.default_output_channels
        os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------- paths

    def _output_subdir(self, image_path: str) -> str:
        p = Path(image_path).resolve()
        try:
            rel = (p.parent.relative_to(self.input_root)
                   if self.input_root else Path())
        except Exception:
            rel = Path()
        out = os.path.join(self.output_dir, str(rel), p.stem)
        return out

    def _should_skip(self, image_path: str) -> bool:
        if not self.resume:
            return False
        summary_path = os.path.join(self._output_subdir(image_path), "infer_summary.json")
        if not os.path.exists(summary_path):
            return False
        try:
            return os.path.getmtime(summary_path) >= os.path.getmtime(image_path)
        except OSError:
            return False

    # ----------------------------------------------------------------- predict

    def _split_batched_outputs(self, outputs, batch_size: int) -> List[List[torch.Tensor]]:
        """Turn model outputs into per-image lists of tensors with the batch
        dim stripped. Coordinate head (if present) follows the seg tensors.
        """
        if isinstance(outputs, (list, tuple)):
            if (len(outputs) == 2 and isinstance(outputs[0], (list, tuple))
                    and torch.is_tensor(outputs[1])):
                task_outputs = list(outputs[0]) + [outputs[1]]
            else:
                task_outputs = list(outputs)
        else:
            task_outputs = [outputs]
        return [[to[i] for to in task_outputs] for i in range(batch_size)]

    def _apply_noise_filter(self, predictions: List[torch.Tensor],
                            task_names: List[str]) -> List[torch.Tensor]:
        """Per-image noise filtering. ``NoiseFilter.filter_prediction`` expects
        a batched tensor (B=1) — we unsqueeze and squeeze around the call.
        """
        if self.noise_filter is None:
            return predictions

        # od_oc context for lesion filters
        lesion_context: Dict[str, torch.Tensor] = {}
        if "od_oc" in task_names:
            idx = task_names.index("od_oc")
            if idx < len(predictions):
                lesion_context["od_oc"] = predictions[idx].unsqueeze(0)

        num_tasks = len(task_names)
        filtered: List[torch.Tensor] = []
        for i, pred in enumerate(predictions):
            if i < num_tasks:
                task_name = task_names[i]
                ctx = lesion_context if self.config.is_lesion_task(task_name) else None
                fp = self.noise_filter.filter_prediction(
                    pred.unsqueeze(0), task_name, ctx
                ).squeeze(0)
                filtered.append(fp)
            else:
                filtered.append(pred)
        return filtered

    # ----------------------------------------------------------------- per-image save

    @staticmethod
    def _compute_fundus_mask(image_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
        b, g, r = cv2.split(image_bgr) if image_bgr.ndim == 3 else (image_bgr, image_bgr, image_bgr)
        mask = ((b > threshold) | (g > threshold) | (r > threshold)).astype(np.uint8)
        k_open = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            mask = (labels == largest).astype(np.uint8)
        k_close = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        return (mask * 255).astype(np.uint8)

    def _save_fundus_mask(self, original_image_bgr: np.ndarray,
                          image_output_dir: str) -> Optional[str]:
        mask = self._compute_fundus_mask(original_image_bgr)
        out_path = os.path.join(image_output_dir, "masks", "fundus_mask.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, mask)
        return out_path

    def _extract_macula(self, predictions: List[torch.Tensor],
                        original_shape: Tuple[int, int]) -> Optional[Dict[str, float]]:
        if len(predictions) <= len(self.config.task_names):
            return None
        coord_tensor = predictions[-1]
        if not torch.is_tensor(coord_tensor) or coord_tensor.numel() < 2:
            return None
        coord = coord_tensor.squeeze().detach().cpu().numpy()
        if len(coord) < 2:
            return None
        w, h = original_shape
        return {
            "y": float(coord[0]) * float(h),
            "x": float(coord[1]) * float(w),
            "unit": "pixels",
            "frame": "original_image",
        }

    def _save_masks(self, predictions, image_data, image_output_dir) -> Dict[str, Any]:
        mask_paths: Dict[str, Any] = {}
        lesion_task_masks: Dict[str, np.ndarray] = {}

        for task_idx, pred in enumerate(predictions):
            if task_idx >= len(self.config.task_names):
                break
            task_name = self.config.get_task_name(task_idx)

            mask_model_frame = self.visualizer._get_task_mask(
                pred, task_name,
                (self.config.image_size, self.config.image_size),
                task_idx=task_idx,
            )
            mask_arr = self.visualizer._place_on_original_canvas(
                mask_model_frame, image_data
            )

            mask_path = self.visualizer.save_task_masks(
                pred, task_name, image_output_dir, image_data, task_idx=task_idx
            )

            if task_name in ("lesion_s1", "lesion_s2", "lesion_s3", "possible_lesions"):
                lesion_task_masks[task_name] = mask_arr

            legacy_tasks = {"lesion_s1", "lesion_s2", "lesion_s3"}
            if (mask_path is not None
                    and not (isinstance(mask_path, str) and mask_path.startswith("placeholder_"))
                    and task_name not in legacy_tasks):
                mask_paths[task_name] = mask_path

        if lesion_task_masks:
            grouped_paths = self.visualizer.save_grouped_lesion_masks(
                lesion_task_masks, image_output_dir, image_data,
                save_visuals=self.save_overlay, save_masks=True,
            )
            mask_paths.update(grouped_paths)
        return mask_paths

    def _process_one(self, image_data: Dict[str, Any],
                     predictions: List[torch.Tensor]) -> None:
        image_path = image_data["image_path"]
        image_output_dir = self._output_subdir(image_path)
        os.makedirs(image_output_dir, exist_ok=True)

        mask_paths = self._save_masks(predictions, image_data, image_output_dir)
        # Fundus mask uses the already-loaded original image (no extra I/O).
        fundus_path = self._save_fundus_mask(image_data["original"], image_output_dir)
        if fundus_path:
            mask_paths["fundus"] = fundus_path

        macula = self._extract_macula(predictions, image_data["original_shape"])

        overlay_path = None
        if self.save_overlay:
            overlay_path = self.visualizer.create_combined_visualization(
                predictions, image_path, image_output_dir, image_data,
                save_visuals=True,
            )

        summary = {
            "image_name": os.path.basename(image_path),
            "image_path": str(Path(image_path).resolve()),
            "original_shape_hw": list(image_data["original_shape"]),
            "analysis_shape_hw": [self.config.image_size, self.config.image_size],
            "output_channels": list(self.output_channels),
            "mask_paths": {k: (v if isinstance(v, str) else None) for k, v in mask_paths.items()},
            "overlay_path": overlay_path,
            "macula_center": macula,
            "device": self.device,
        }
        with open(os.path.join(image_output_dir, "infer_summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(summary), f, ensure_ascii=False, indent=2)
        if macula is not None:
            with open(os.path.join(image_output_dir, "coords.json"),
                      "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types({"macula_center": macula}),
                          f, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------- run

    def run(self, image_paths: List[str]) -> Dict[str, int]:
        total = len(image_paths)
        if self.resume:
            image_paths = [p for p in image_paths if not self._should_skip(p)]
            skipped = total - len(image_paths)
            if skipped:
                print(f"Resumed: skipping {skipped}/{total} already-processed images",
                      file=sys.stderr)
        else:
            skipped = 0

        if not image_paths:
            return {"processed": 0, "failed": 0, "skipped": skipped, "total": total}

        dataset = _InferDataset(image_paths, self.image_processor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate,
            shuffle=False,
            pin_memory=self.device.startswith("cuda"),
            persistent_workers=(self.num_workers > 0),
        )

        processed, failed = 0, 0
        t0 = time.time()
        pbar = tqdm(total=len(image_paths), desc="Inference", unit="img")
        try:
            for batch_tensors, batch_metas in loader:
                try:
                    batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        if self.use_fp16:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = self.model(batch_tensors)
                        else:
                            outputs = self.model(batch_tensors)
                    per_image = self._split_batched_outputs(outputs, batch_tensors.shape[0])
                except Exception as e:
                    failed += len(batch_metas)
                    print(f"[fail] batch forward: {e}", file=sys.stderr)
                    pbar.update(len(batch_metas))
                    continue

                for preds, meta in zip(per_image, batch_metas):
                    try:
                        preds = self._apply_noise_filter(preds, self.config.task_names)
                        self._process_one(meta, preds)
                        processed += 1
                    except Exception as e:
                        failed += 1
                        print(f"[fail] {meta.get('image_path')}: {e}", file=sys.stderr)
                    pbar.update(1)
        finally:
            pbar.close()

        dt = time.time() - t0
        throughput = processed / dt if dt > 0 else 0.0
        print(
            f"Done. processed={processed} failed={failed} skipped={skipped} "
            f"total={total} wall_time={dt:.1f}s throughput={throughput:.2f} img/s",
            file=sys.stderr,
        )
        return {"processed": processed, "failed": failed,
                "skipped": skipped, "total": total}


# --------------------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="retsam-2.0 batched mask inference")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default=str(PROJECT_ROOT / "retsam.ckpt"))
    p.add_argument("--output_channels", default="(2,3,2,4,6,11,6,2)")
    p.add_argument("--has_coordinate_head", action="store_true", default=True)
    p.add_argument("--no_coordinate_head", dest="has_coordinate_head", action="store_false")
    p.add_argument("--num_coordinates", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--disable_noise_filter", action="store_true")
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--image_extensions", default="jpg,jpeg,png,bmp,tiff,tif")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories when collecting images. "
                        "Combined with --input_dir pointing at a dataset root, "
                        "outputs preserve the relative subfolder structure.")

    # Throughput knobs
    p.add_argument("--batch_size", type=int, default=1,
                   help="GPU batch size for the model forward pass.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers for CPU preprocessing.")
    p.add_argument("--fp16", action="store_true",
                   help="Run forward pass under torch.autocast(float16) on CUDA.")
    p.add_argument("--torch_compile", action="store_true",
                   help="Wrap the model with torch.compile (long warm-up, pays off >50 images).")
    p.add_argument("--resume", action="store_true",
                   help="Skip images whose infer_summary.json is newer than the source.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2
    extensions = [e.strip() for e in args.image_extensions.split(",") if e.strip()]
    image_paths = find_image_files(input_dir, extensions=extensions,
                                   recursive=args.recursive)
    if not image_paths:
        print(f"No images with extensions {extensions} found in {input_dir}", file=sys.stderr)
        return 2
    output_channels = parse_output_channels(args.output_channels)

    runner = BatchedMaskInference(
        model_path=args.model_path,
        output_dir=output_dir,
        device=args.device,
        output_channels=output_channels,
        has_coordinate_head=args.has_coordinate_head,
        num_coordinates=args.num_coordinates,
        enable_noise_filter=not args.disable_noise_filter,
        save_overlay=args.save_overlay,
        input_root=input_dir,
        quiet=not args.verbose,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fp16=args.fp16,
        torch_compile=args.torch_compile,
        resume=args.resume,
    )
    stats = runner.run(image_paths)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
