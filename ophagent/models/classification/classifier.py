"""
Classification models for OCT disease diagnosis.

Supports multi-class classification across various OCT pathologies:
AMD, DME, CNV, Drusen, Glaucoma, ERM, MH, DR, etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.encoder import OCTBackbone


class OCTClassifier(nn.Module):
    """General-purpose OCT image classifier.

    Architecture: backbone encoder → global pooling → projection → classifier
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.backbone = OCTBackbone(backbone, pretrained=pretrained)
        embed_dim = self.backbone.embed_dim

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "features": features,
        }

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            output["predicted_class"] = output["logits"].argmax(dim=-1)
        return output


class OCTQualityAssessor(nn.Module):
    """Assesses OCT image quality: usable / low-quality / unusable.

    Used as a gate before running other analysis models.
    """

    QUALITY_LABELS = ["high", "medium", "low"]

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.backbone = OCTBackbone(backbone, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return {
            "logits": logits,
            "quality_score": F.softmax(logits, dim=-1),
            "quality_label": logits.argmax(dim=-1),
        }


class MultiModalClassifier(nn.Module):
    """Classifier that fuses OCT and fundus image features.

    Fusion strategies: early (concat), late (ensemble), intermediate (cross-attention).
    """

    def __init__(
        self,
        oct_backbone: str = "resnet50",
        fundus_backbone: str = "resnet50",
        num_classes: int = 9,
        fusion: str = "concat",
        pretrained: bool = True,
    ):
        super().__init__()
        self.oct_encoder = OCTBackbone(oct_backbone, pretrained=pretrained)
        self.fundus_encoder = OCTBackbone(fundus_backbone, pretrained=pretrained)
        self.fusion = fusion

        oct_dim = self.oct_encoder.embed_dim
        fundus_dim = self.fundus_encoder.embed_dim

        if fusion == "concat":
            self.head = nn.Sequential(
                nn.Linear(oct_dim + fundus_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        elif fusion == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=oct_dim, num_heads=8, batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(oct_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

    def forward(
        self, oct_image: torch.Tensor, fundus_image: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        oct_feat = self.oct_encoder(oct_image)
        fundus_feat = self.fundus_encoder(fundus_image)

        if self.fusion == "concat":
            fused = torch.cat([oct_feat, fundus_feat], dim=-1)
        elif self.fusion == "cross_attention":
            oct_q = oct_feat.unsqueeze(1)
            fundus_kv = fundus_feat.unsqueeze(1)
            attn_out, _ = self.cross_attn(oct_q, fundus_kv, fundus_kv)
            fused = attn_out.squeeze(1) + oct_feat

        logits = self.head(fused)
        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "oct_features": oct_feat,
            "fundus_features": fundus_feat,
        }
