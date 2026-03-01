"""Trainer for Referable Glaucoma + Structural Features Detection."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class CFPGlaucomaTrainer(BaseTrainer):
    """
    Detects referable glaucoma and extracts structural disc features.
    Architecture: Swin-Transformer-Small with auxiliary structural feature head.
    """

    def build_model(self) -> nn.Module:
        class GlaucomaModel(nn.Module):
            def __init__(self, n_struct_features):
                super().__init__()
                self.backbone = timm.create_model(
                    "swin_small_patch4_window7_224", pretrained=True, num_classes=0
                )
                dim = self.backbone.num_features
                self.cls_head = nn.Linear(dim, 2)                      # Normal / Referable
                self.struct_head = nn.Linear(dim, n_struct_features)   # CDR, rim area, etc.

            def forward(self, x):
                feat = self.backbone(x)
                return {"cls": self.cls_head(feat), "struct": self.struct_head(feat)}

        return GlaucomaModel(n_struct_features=self.config.get("num_struct_features", 5))

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms, datasets
        from torch.utils.data import random_split

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        data_root = self.config.get("data_root", "data/cfp_glaucoma")
        dataset = datasets.ImageFolder(data_root, transform=transform)
        n_val = int(len(dataset) * 0.2)
        return random_split(dataset, [len(dataset) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            cls_loss = nn.CrossEntropyLoss()(outputs["cls"], targets[:, 0].long())
            struct_loss = nn.MSELoss()(outputs["struct"], targets[:, 1:].float())
            return cls_loss + 0.5 * struct_loss
        return nn.CrossEntropyLoss()(outputs, targets)
