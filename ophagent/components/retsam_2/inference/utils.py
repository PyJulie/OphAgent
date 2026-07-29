"""
Utility functions and classes for the mask-generating inference pipeline.

Trimmed from the original `inference_modules/utils.py`:
- dropped `FileManager.save_json` (only used by the analyzer)
- dropped `ImageUtils` (only used for RGB overlay rendering)
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch


class Logger:
    """Unified logging utilities."""
    quiet_mode: bool = False

    @staticmethod
    def info(message: str):
        if Logger.quiet_mode:
            return
        print(f"ℹ️ {message}")

    @staticmethod
    def success(message: str):
        if Logger.quiet_mode:
            return
        print(f"✅ {message}")

    @staticmethod
    def warning(message: str):
        print(f"⚠️ {message}")

    @staticmethod
    def error(message: str):
        print(f"❌ {message}")

    @staticmethod
    def debug(message: str):
        if Logger.quiet_mode:
            return
        print(f"🐛 {message}")

    @staticmethod
    def progress(current: int, total: int, item_name: str = ""):
        if Logger.quiet_mode:
            return
        print(f"🔄 Processing [{current}/{total}]{': ' + item_name if item_name else ''}")

    @staticmethod
    def saved(description: str, filepath: str = None):
        if Logger.quiet_mode:
            return
        if filepath:
            print(f"💾 Saved {description}: {filepath}")
        else:
            print(f"💾 Saved {description}")

    @staticmethod
    def set_quiet(quiet: bool):
        Logger.quiet_mode = quiet


class FileManager:
    """File management utilities used by mask saving."""

    @staticmethod
    def create_output_paths(base_output_dir: str, image_name: str) -> Dict[str, str]:
        paths = {
            'base': base_output_dir,
            'masks': os.path.join(base_output_dir, 'masks'),
            'individual_lesions': os.path.join(base_output_dir, 'masks', 'individual_lesions'),
            'individual_channels': os.path.join(base_output_dir, 'individual_channels'),
            'visualizations': os.path.join(base_output_dir, 'visualizations'),
        }
        for key in ('base', 'masks', 'visualizations'):
            os.makedirs(paths[key], exist_ok=True)
        return paths

    @staticmethod
    def save_image_with_log(image: np.ndarray, filepath: str,
                            description: str = "", original_size: Tuple[int, int] = None):
        cv2.imwrite(filepath, image)
        if description and original_size:
            width, height = original_size
            Logger.saved(f"{description} at original size {width}x{height}", filepath)
        elif description:
            Logger.saved(description, filepath)


class MaskProcessor:
    """Mask processing utilities used by visualizer and postprocess callers."""

    @staticmethod
    def apply_threshold(prediction: Union[np.ndarray, torch.Tensor],
                        threshold: float = 0.5) -> np.ndarray:
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        return (prediction > threshold).astype(np.uint8)

    @staticmethod
    def apply_argmax(prediction: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        return np.argmax(prediction, axis=0).astype(np.uint8)

    @staticmethod
    def process_multilabel_prediction(prediction: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        pred_probs = torch.sigmoid(prediction).squeeze().cpu().numpy()
        pred_binary = (pred_probs > 0.5).astype(np.uint8)
        if pred_binary.ndim == 3:
            combined_mask = np.sum(pred_binary, axis=0)
        else:
            combined_mask = pred_binary
        combined_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
        return combined_mask, pred_binary

    @staticmethod
    def get_largest_component(mask: np.ndarray) -> np.ndarray:
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        if num_labels <= 1:
            return mask
        largest_size = 0
        largest_label = 0
        for label in range(1, num_labels):
            size = int(np.sum(labels == label))
            if size > largest_size:
                largest_size = size
                largest_label = label
        return (labels == largest_label).astype(np.uint8)

    @staticmethod
    def morphological_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return cleaned


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)) or \
       str(type(obj)) == "<class 'numpy.bool_'>" or \
       (hasattr(obj, 'dtype') and 'bool' in str(obj.dtype)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    if obj is None:
        return None
    return obj


def parse_output_channels(output_channels_str: str) -> Union[Tuple[int, ...], int, None]:
    """Parse `--output_channels` CLI arg."""
    if output_channels_str is None:
        return None
    try:
        import ast
        return ast.literal_eval(output_channels_str)
    except (ValueError, SyntaxError):
        try:
            return int(output_channels_str)
        except ValueError as exc:
            raise ValueError(f"Invalid output_channels format: {output_channels_str}") from exc


def find_image_files(directory: str, extensions: List[str] = None,
                     recursive: bool = False) -> List[str]:
    """Find image files in directory. Supports recursive descent when needed
    for datasets organised into split / class subfolders.
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    image_files: List[str] = []
    pattern_prefix = "**/" if recursive else ""
    for ext in extensions:
        image_files.extend(glob.glob(
            os.path.join(directory, f"{pattern_prefix}*.{ext.lower()}"),
            recursive=recursive))
        image_files.extend(glob.glob(
            os.path.join(directory, f"{pattern_prefix}*.{ext.upper()}"),
            recursive=recursive))
    return sorted(set(image_files))
