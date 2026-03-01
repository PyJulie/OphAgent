"""Trainer for PDR Activity Grading."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class CFPPDRTrainer(BaseTrainer):
    """
    Grading model for Proliferative Diabetic Retinopathy (PDR) activity.
    Grades: Inactive / Low-activity / High-activity / Severe.
    Architecture: EfficientNet-B5 with ordinal regression head.
    """

    def build_model(self) -> nn.Module:
        n_classes = self.config.get("num_classes", 4)
        model = timm.create_model("efficientnet_b5", pretrained=True, num_classes=n_classes)
        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms, datasets
        from torch.utils.data import random_split

        transform = transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        data_root = self.config.get("data_root", "data/cfp_pdr")
        dataset = datasets.ImageFolder(data_root, transform=transform)
        n_val = int(len(dataset) * 0.2)
        return random_split(dataset, [len(dataset) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return nn.CrossEntropyLoss()(outputs, targets)

    def compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, float]:
        from ophagent.utils.metrics import kappa_score, accuracy

        preds = outputs.argmax(dim=-1).tolist()
        gt = targets.tolist()
        return {"accuracy": accuracy(gt, preds), "kappa": kappa_score(gt, preds)}
