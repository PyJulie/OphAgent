"""
Unified dataset classes for OCT tasks.

Provides consistent interface regardless of source dataset,
with patient-level splits to prevent data leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..download.registry import TaskType


class OCTClassificationDataset(Dataset):
    """Generic classification dataset that reads from a directory structure:
    root/
      class_0/
        img1.png
        img2.png
      class_1/
        ...
    """

    def __init__(
        self,
        root: str | Path,
        transform: Any = None,
        class_names: list[str] | None = None,
        grayscale: bool = True,
    ):
        self.root = Path(root)
        self.transform = transform
        self.grayscale = grayscale

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = sorted([
                d.name for d in self.root.iterdir() if d.is_dir()
            ])

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            idx = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_path, label = self.samples[index]
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(img_path), flag)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        if self.grayscale and image.ndim == 2:
            image = np.expand_dims(image, -1)
            image = np.repeat(image, 3, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {"image": image, "label": label, "path": str(img_path)}


class OCTSegmentationDataset(Dataset):
    """Generic segmentation dataset.

    Expects paired images and masks:
      images_dir/  img001.png, img002.png, ...
      masks_dir/   img001.png, img002.png, ...  (pixel values = class indices)
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Any = None,
        num_classes: int = 2,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.num_classes = num_classes

        image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in image_suffixes
        ])

        self.mask_files = []
        for img_file in self.image_files:
            mask_candidates = [
                self.masks_dir / img_file.name,
                self.masks_dir / img_file.with_suffix(".png").name,
                self.masks_dir / img_file.with_suffix(".tif").name,
            ]
            mask_path = next((m for m in mask_candidates if m.exists()), None)
            if mask_path is None:
                raise FileNotFoundError(
                    f"No mask found for {img_file.name} in {self.masks_dir}"
                )
            self.mask_files.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = cv2.imread(str(self.image_files[index]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_files[index]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to load image: {self.image_files[index]}")
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {self.mask_files[index]}")

        image = np.stack([image] * 3, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        return {
            "image": image,
            "mask": mask,
            "path": str(self.image_files[index]),
        }


class OCTVolumeDataset(Dataset):
    """Dataset for 3D OCT volumes.

    Expects .npy or .nii.gz files containing (D, H, W) arrays.
    """

    def __init__(
        self,
        volume_dir: str | Path,
        labels: dict[str, int] | str | Path | None = None,
        slice_transform: Any = None,
        num_slices: int | None = None,
    ):
        self.volume_dir = Path(volume_dir)
        self.slice_transform = slice_transform
        self.num_slices = num_slices

        self.volume_files = sorted([
            f for f in self.volume_dir.rglob("*")
            if f.is_file() and f.name.lower().endswith((".npy", ".npz", ".nii", ".nii.gz"))
        ])

        self.labels: dict[str, int] = {}
        if isinstance(labels, (str, Path)):
            with open(labels) as f:
                self.labels = json.load(f)
        elif isinstance(labels, dict):
            self.labels = labels

    def __len__(self) -> int:
        return len(self.volume_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        vol_path = self.volume_files[index]

        lower_name = vol_path.name.lower()
        if lower_name.endswith(".npy"):
            volume = np.load(vol_path)
        elif lower_name.endswith(".npz"):
            with np.load(vol_path) as data:
                if not data.files:
                    raise ValueError(f"Empty NPZ volume: {vol_path}")
                volume = data[data.files[0]]
        else:
            import nibabel as nib
            volume = np.asarray(nib.load(str(vol_path)).dataobj)

        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D OCT volume, got shape {volume.shape} from {vol_path}"
            )

        if self.num_slices and volume.shape[0] > self.num_slices:
            indices = np.linspace(0, volume.shape[0] - 1, self.num_slices, dtype=int)
            volume = volume[indices]

        if self.slice_transform:
            slices = []
            for s in range(volume.shape[0]):
                sl = volume[s]
                if sl.ndim == 2:
                    sl = np.stack([sl] * 3, axis=-1)
                augmented = self.slice_transform(image=sl)
                slices.append(augmented["image"])
            volume = torch.stack(slices, dim=0)

        result: dict[str, Any] = {
            "volume": volume,
            "path": str(vol_path),
        }
        label_key = vol_path.name[:-7] if lower_name.endswith(".nii.gz") else vol_path.stem
        if label_key in self.labels:
            result["label"] = self.labels[label_key]
        return result


# ── Dataset adapters for specific public datasets ──────────────────────────

class KermanyDataset(OCTClassificationDataset):
    """Adapter for the Kermany et al. dataset structure."""

    def __init__(self, root: str | Path, split: str = "train", **kwargs):
        root = Path(root)
        split_dir = root / split
        if not split_dir.exists():
            for candidate in root.iterdir():
                if candidate.is_dir() and split in candidate.name.lower():
                    split_dir = candidate
                    break
        super().__init__(
            root=split_dir,
            class_names=["CNV", "DME", "DRUSEN", "NORMAL"],
            **kwargs,
        )


class RETOUCHDataset(OCTSegmentationDataset):
    """Adapter for the RETOUCH challenge dataset."""

    FLUID_CLASSES = ["Background", "IRF", "SRF", "PED"]

    def __init__(self, root: str | Path, device: str = "all", **kwargs):
        root = Path(root)
        images_dir = root / "images"
        masks_dir = root / "masks"
        super().__init__(
            images_dir=images_dir,
            masks_dir=masks_dir,
            num_classes=4,
            **kwargs,
        )
