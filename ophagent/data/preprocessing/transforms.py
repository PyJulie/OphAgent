"""
OCT-specific image transforms and augmentations.

Designed to respect OCT domain constraints:
- No aggressive color jittering (grayscale images)
- Careful with vertical flips (retinal layer order is meaningful)
- Speckle noise augmentation
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classification_transforms(
    image_size: int = 224,
    is_training: bool = True,
) -> A.Compose:
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=0, p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=0, p=0.2,
            ),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])


def get_segmentation_transforms(
    image_size: tuple[int, int] = (512, 512),
    is_training: bool = True,
) -> A.Compose:
    h, w = image_size
    if is_training:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(0.95, 1.05),
                rotate=(-5, 5),
                border_mode=0, p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.08, contrast_limit=0.08, p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.06), p=0.2),
            A.ElasticTransform(alpha=30, sigma=5, p=0.15),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])


def get_generation_transforms(
    image_size: int = 256,
    is_training: bool = True,
) -> A.Compose:
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])
