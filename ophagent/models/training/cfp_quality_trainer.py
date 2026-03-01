"""Trainer for CFP Image Quality Reasoning Model."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class CFPQualityTrainer(BaseTrainer):
    """
    Trains a CFP image quality assessment model with multi-task output:
    quality score regression + quality label classification + reasoning generation.

    Architecture: EfficientNet-B4 backbone + dual-head (score + classification).
    """

    def build_model(self) -> nn.Module:
        backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
        feat_dim = backbone.num_features
        return nn.Sequential(
            backbone,
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.config.get("num_classes", 3)),  # Bad/Moderate/Good
        )

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms, datasets
        from torch.utils.data import random_split

        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        data_root = self.config.get("data_root", "data/cfp_quality")
        dataset = datasets.ImageFolder(data_root, transform=transform)
        n_val = int(len(dataset) * 0.2)
        return random_split(dataset, [len(dataset) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return nn.CrossEntropyLoss()(outputs, targets)

    def compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, float]:
        from ophagent.utils.metrics import accuracy, auc_roc

        preds = outputs.argmax(dim=-1).tolist()
        gt = targets.tolist()
        probs = outputs.softmax(dim=-1)
        metrics = {"accuracy": accuracy(gt, preds)}
        if probs.shape[1] == 2:
            metrics["auc"] = auc_roc(gt, probs[:, 1].tolist())
        return metrics
