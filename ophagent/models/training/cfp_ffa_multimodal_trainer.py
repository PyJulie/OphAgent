"""Trainer for Multimodal CFP + FFA Classification."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import timm
from ophagent.models.training.base_trainer import BaseTrainer


class CFPFFAMultimodalTrainer(BaseTrainer):
    """
    Dual-stream multimodal classifier for CFP + FFA image pairs.
    Two ViT-Small encoders (one per modality) + cross-attention fusion.
    """

    def build_model(self) -> nn.Module:
        class DualStreamModel(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.cfp_enc = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
                self.ffa_enc = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
                dim = self.cfp_enc.num_features
                self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
                self.classifier = nn.Sequential(
                    nn.LayerNorm(dim * 2),
                    nn.Linear(dim * 2, n_classes),
                )

            def forward(self, x):
                cfp, ffa = x["cfp"], x["ffa"]
                cfp_feat = self.cfp_enc(cfp).unsqueeze(1)
                ffa_feat = self.ffa_enc(ffa).unsqueeze(1)
                fused, _ = self.cross_attn(cfp_feat, ffa_feat, ffa_feat)
                out = torch.cat([fused.squeeze(1), cfp_feat.squeeze(1)], dim=-1)
                return self.classifier(out)

        return DualStreamModel(self.config.get("num_classes", 6))

    def build_datasets(self) -> Tuple[Any, Any]:
        from torchvision import transforms
        from torch.utils.data import Dataset, random_split
        import json
        from PIL import Image

        class MultimodalDataset(Dataset):
            def __init__(self, ann_file, transform):
                with open(ann_file) as f:
                    self.data = json.load(f)
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                cfp = self.transform(Image.open(item["cfp_path"]).convert("RGB"))
                ffa = self.transform(Image.open(item["ffa_path"]).convert("RGB"))
                label = torch.tensor(item["label"], dtype=torch.long)
                return {"cfp": cfp, "ffa": ffa}, label

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ann = self.config.get("annotation_file", "data/cfp_ffa/annotations.json")
        ds = MultimodalDataset(ann, transform)
        n_val = int(len(ds) * 0.2)
        return random_split(ds, [len(ds) - n_val, n_val])

    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        return nn.CrossEntropyLoss()(outputs, targets)

    def _unpack_batch(self, batch):
        inputs, targets = batch
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs, targets.to(self.device)
