"""I/O utilities for image loading, saving, and format conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def load_oct_image(path: str | Path, grayscale: bool = True) -> np.ndarray:
    path = Path(path)
    if path.suffix in (".npy",):
        return np.load(path)
    if path.suffix in (".npz",):
        data = np.load(path)
        return data[list(data.keys())[0]]
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        img = np.array(Image.open(path).convert("L" if grayscale else "RGB"))
    return img


def load_oct_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data[list(data.keys())[0]]
    if path.suffix in (".nii", ".gz"):
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj)
    raise ValueError(f"Unsupported volume format: {path.suffix}")


def save_image(img: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, img)
    else:
        cv2.imwrite(str(path), img)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
