"""
U-Net variants for OCT segmentation tasks.

Supports:
- Standard U-Net with ResNet/ConvNeXt encoder
- Swin-UNet (Transformer-based)
- Hybrid CNN-Transformer U-Net

Tasks: retinal layer segmentation, fluid segmentation, lesion segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.encoder import OCTBackbone


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class OCTUNet(nn.Module):
    """U-Net with timm encoder backbone and attention gates."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 9,
        pretrained: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.encoder = OCTBackbone(
            backbone, pretrained=pretrained, features_only=True
        )
        enc_channels = self.encoder.feature_dims
        self.use_attention = use_attention

        dec_channels = [256, 128, 64, 32]

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if use_attention else None

        for i in range(len(enc_channels) - 1):
            up_in = enc_channels[-(i + 1)] if i == 0 else dec_channels[i - 1]
            skip_ch = enc_channels[-(i + 2)]
            out_ch = dec_channels[i]

            self.upconvs.append(
                nn.ConvTranspose2d(up_in, out_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(ConvBlock(out_ch + skip_ch, out_ch))

            if use_attention:
                self.attn_gates.append(
                    AttentionGate(up_in, skip_ch, skip_ch // 2)
                )

        final_ch = dec_channels[len(enc_channels) - 2] if len(enc_channels) > 1 else dec_channels[0]
        self.final_up = nn.ConvTranspose2d(final_ch, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_features = self.encoder(x)

        d = enc_features[-1]
        for i in range(len(self.upconvs)):
            skip = enc_features[-(i + 2)]
            if self.use_attention and self.attn_gates:
                skip = self.attn_gates[i](d, skip)
            d = self.upconvs[i](d)
            d = F.interpolate(d, size=skip.shape[2:], mode="bilinear", align_corners=False)
            d = torch.cat([d, skip], dim=1)
            d = self.dec_blocks[i](d)

        d = self.final_up(d)
        d = F.interpolate(d, size=x.shape[2:], mode="bilinear", align_corners=False)
        logits = self.final_conv(d)

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "prediction": logits.argmax(dim=1),
        }


class OCTFluidSegmentor(OCTUNet):
    """Specialized for fluid segmentation (IRF, SRF, PED)."""

    FLUID_CLASSES = ["Background", "IRF", "SRF", "PED"]

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__(
            backbone=backbone,
            num_classes=4,
            pretrained=pretrained,
            use_attention=True,
        )


class OCTLayerSegmentor(OCTUNet):
    """Specialized for retinal layer boundary segmentation."""

    LAYER_NAMES = [
        "ILM", "NFL-GCL", "GCL-IPL", "IPL-INL",
        "INL-OPL", "OPL-ONL", "ELM", "IS-OS",
        "OS-RPE", "BM",
    ]

    def __init__(
        self,
        backbone: str = "resnet50",
        num_layers: int = 9,
        pretrained: bool = True,
    ):
        super().__init__(
            backbone=backbone,
            num_classes=num_layers,
            pretrained=pretrained,
            use_attention=True,
        )
