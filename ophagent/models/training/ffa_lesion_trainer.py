"""Trainer for FFA Multi-Lesion Detection."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from ophagent.models.training.base_trainer import BaseTrainer


class FFALesionTrainer(BaseTrainer):
    """
    Object detection trainer for FFA multi-lesion detection.
    Uses torchvision Faster R-CNN with a ResNet-50-FPN backbone.
    Lesion types: microaneurysms, haemorrhages, exudates, neovascularisation.
    """

    def build_model(self) -> nn.Module:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        n_classes = self.config.get("num_classes", 5)  # background + 4 lesion types
        model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        return model

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms
        from torch.utils.data import Dataset, random_split
        import json
        from PIL import Image

        class FFADetDataset(Dataset):
            def __init__(self, ann_file, transforms=None):
                with open(ann_file) as f:
                    self.data = json.load(f)
                self.transforms = transforms

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                img = Image.open(item["image_path"]).convert("RGB")
                boxes = torch.tensor(item["boxes"], dtype=torch.float32)
                labels = torch.tensor(item["labels"], dtype=torch.int64)
                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": torch.tensor([idx]),
                }
                if self.transforms:
                    img = self.transforms(img)
                return img, target

        t = transforms.Compose([transforms.ToTensor()])
        ann = self.config.get("annotation_file", "data/ffa_lesion/annotations.json")
        ds = FFADetDataset(ann, transforms=t)
        n_val = int(len(ds) * 0.2)
        return random_split(ds, [len(ds) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        # Faster R-CNN returns dict of losses during training
        if isinstance(outputs, dict):
            return sum(outputs.values())
        return outputs

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for images, targets in loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)
