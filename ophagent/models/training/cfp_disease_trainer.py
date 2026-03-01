"""Trainer for Multiple Retinal Disease Classification (CFP)."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class CFPDiseaseTrainer(BaseTrainer):
    """
    Multi-label retinal disease classification trainer.
    Architecture: ConvNeXt-Base with sigmoid multi-label head.
    Diseases: DR, AMD, glaucoma, RVO, myopia, others.
    """

    def build_model(self) -> nn.Module:
        n_classes = self.config.get("num_classes", 8)
        backbone = timm.create_model("convnext_base", pretrained=True, num_classes=n_classes)
        return backbone

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms
        from torch.utils.data import random_split, Dataset
        import json
        from PIL import Image

        class MultiLabelDataset(Dataset):
            def __init__(self, ann_file, transform):
                with open(ann_file) as f:
                    self.data = json.load(f)
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                img = Image.open(item["image_path"]).convert("RGB")
                label = torch.tensor(item["labels"], dtype=torch.float32)
                return self.transform(img), label

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ann_file = self.config.get("annotation_file", "data/cfp_disease/annotations.json")
        dataset = MultiLabelDataset(ann_file, transform)
        n_val = int(len(dataset) * 0.2)
        return torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, float]:
        from ophagent.utils.metrics import f1_score as f1

        preds = (outputs.sigmoid() > 0.5).int()
        gt = targets.int()
        # Per-sample F1
        return {"f1_macro": f1(gt.flatten().tolist(), preds.flatten().tolist(), average="macro")}
