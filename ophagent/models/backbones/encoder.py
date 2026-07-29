"""
Shared backbone encoders for OCT analysis.

Supports: ResNet, ConvNeXt, ViT, Swin Transformer via timm.
Can be initialized with ImageNet pretrained weights or from
OCT-specific SSL pretraining checkpoints.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


class OCTBackbone(nn.Module):
    """Unified backbone wrapper around timm models."""

    SUPPORTED = {
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b3": "efficientnet_b3",
    }

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
        features_only: bool = False,
    ):
        super().__init__()
        model_name = self.SUPPORTED.get(name, name)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=features_only,
            num_classes=0,
        )
        self.name = name
        self.features_only = features_only

        if features_only:
            self.feature_dims = self.model.feature_info.channels()
        else:
            self.embed_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        return self.model(x)

    def load_ssl_weights(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        if "model" in state:
            state = state["model"]
        # strip prefix if any
        state = {k.replace("encoder.", "").replace("backbone.", ""): v for k, v in state.items()}
        msg = self.model.load_state_dict(state, strict=False)
        print(f"Loaded SSL weights: {msg}")


class MAEEncoder(nn.Module):
    """Masked Autoencoder encoder for OCT self-supervised pretraining.

    Uses ViT backbone with random patch masking.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
            in_chans=3,
        )
        self.embed_dim = embed_dim

        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2)
        self.decoder_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim // 2,
                nhead=num_heads // 2,
                dim_feedforward=embed_dim * 2,
                batch_first=True,
            ),
            num_layers=4,
        )
        self.decoder_pred = nn.Linear(
            embed_dim // 2, patch_size * patch_size * 3
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim // 2))
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        num_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_embed = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patch_embed], dim=1)
        tokens = tokens + self.encoder.pos_embed

        patch_tokens = tokens[:, 1:]
        cls_tok = tokens[:, :1]

        visible, mask, ids_restore = self.random_masking(patch_tokens)
        visible = torch.cat([cls_tok, visible], dim=1)

        for blk in self.encoder.blocks:
            visible = blk(visible)
        visible = self.encoder.norm(visible)

        latent = visible[:, 1:]
        latent = self.decoder_embed(latent)

        B, N_vis, D_dec = latent.shape
        N_mask = self.num_patches - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        full = torch.cat([latent, mask_tokens], dim=1)

        ids_unshuffle = ids_restore.unsqueeze(-1).expand(-1, -1, D_dec)
        full = torch.gather(full, 1, ids_unshuffle)

        decoded = self.decoder_blocks(full)
        pred = self.decoder_pred(decoded)

        return {"pred": pred, "mask": mask, "latent": visible[:, 0]}
