"""Trainer for Optic Disc and Fovea Localisation."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class DiscFoveaTrainer(BaseTrainer):
    """
    Landmark regression model for optic disc centre + fovea centre localisation.
    Architecture: HRNet-W32 (high-resolution) backbone + regression head.
    Output: [disc_x, disc_y, disc_r, fovea_x, fovea_y] normalised coordinates.
    """

    def build_model(self) -> nn.Module:
        class LandmarkModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model("hrnet_w32", pretrained=True, num_classes=0)
                self.regressor = nn.Sequential(
                    nn.Linear(self.backbone.num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 5),  # disc_x, disc_y, disc_r, fov_x, fov_y (normalised)
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.regressor(self.backbone(x))

        return LandmarkModel()

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms
        from torch.utils.data import Dataset, random_split
        import json
        from PIL import Image

        class LandmarkDataset(Dataset):
            def __init__(self, ann_file, transform):
                with open(ann_file) as f:
                    self.data = json.load(f)
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                img = Image.open(item["image_path"]).convert("RGB")
                w, h = img.size
                coords = torch.tensor([
                    item["disc_x"] / w, item["disc_y"] / h,
                    item["disc_r"] / max(w, h),
                    item["fovea_x"] / w, item["fovea_y"] / h,
                ], dtype=torch.float32)
                return self.transform(img), coords

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ann = self.config.get("annotation_file", "data/disc_fovea/annotations.json")
        ds = LandmarkDataset(ann, transform)
        n_val = int(len(ds) * 0.2)
        return random_split(ds, [len(ds) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return nn.MSELoss()(outputs, targets)

    def compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, float]:
        mse = nn.MSELoss()(outputs, targets).item()
        return {"mse": mse, "accuracy": max(0.0, 1.0 - mse)}
