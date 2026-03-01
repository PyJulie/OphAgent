"""Trainer for UWF Quality + Disease Classification."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class UWFQualityDiseaseTrainer(BaseTrainer):
    """
    Multi-task model for UWF images: quality assessment + disease classification.
    Shared ResNet-50 backbone with two separate heads.
    """

    def build_model(self) -> nn.Module:
        class MultiTaskModel(nn.Module):
            def __init__(self, n_quality, n_disease):
                super().__init__()
                self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
                dim = self.backbone.num_features
                self.quality_head = nn.Linear(dim, n_quality)
                self.disease_head = nn.Linear(dim, n_disease)

            def forward(self, x):
                feat = self.backbone(x)
                return {"quality": self.quality_head(feat), "disease": self.disease_head(feat)}

        return MultiTaskModel(
            n_quality=self.config.get("num_quality_classes", 3),
            n_disease=self.config.get("num_disease_classes", 5),
        )

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms, datasets
        from torch.utils.data import random_split

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        data_root = self.config.get("data_root", "data/uwf_quality_disease")
        dataset = datasets.ImageFolder(data_root, transform=transform)
        n_val = int(len(dataset) * 0.2)
        return random_split(dataset, [len(dataset) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            q_loss = nn.CrossEntropyLoss()(outputs["quality"], targets[:, 0])
            d_loss = nn.CrossEntropyLoss()(outputs["disease"], targets[:, 1])
            return q_loss + d_loss
        return nn.CrossEntropyLoss()(outputs, targets)
