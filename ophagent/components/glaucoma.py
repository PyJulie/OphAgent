"""Inference-only architecture for the OphAgent glaucoma checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms


INPUT_SIZE = 392
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GlaucomaModel(nn.Module):
    """ViT-L DINOv2 backbone with referability and sign-specific heads."""

    def __init__(self, num_aux: int = 10, drop_path: float = 0.3):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=INPUT_SIZE,
            num_classes=0,
            drop_path_rate=drop_path,
        )
        feature_dim = self.backbone.embed_dim
        self.head_main = nn.Linear(feature_dim, 1)
        self.head_aux = nn.Linear(feature_dim, num_aux)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(image)
        if features.dim() == 3:
            if features.shape[1] > 1:
                features = features[:, 1:, :].mean(dim=1)
            else:
                features = features.squeeze(1)
        return self.head_main(features).squeeze(-1), self.head_aux(features)


def build_eval_transform(size: int = INPUT_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
