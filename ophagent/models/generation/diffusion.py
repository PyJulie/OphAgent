"""
Diffusion-based models for OCT image generation and restoration.

Tasks:
- Denoising (speckle noise removal)
- Super-resolution
- Cross-modality translation (OCT → OCTA)
- Synthetic OCT generation (data augmentation for rare pathologies)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class DenoisingUNet(nn.Module):
    """U-Net noise predictor for diffusion models.

    Predicts the noise ε given noisy image x_t and timestep t.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        time_dim: int = 256,
        use_attention_at: tuple[int, ...] = (2, 3),
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_channels
        channels_list = [ch]

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch, out_ch, time_dim),
                SelfAttention(out_ch) if i in use_attention_at else nn.Identity(),
            ]))
            ch = out_ch
            channels_list.append(ch)
            if i < len(channel_mults) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())

        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = channels_list[-(i + 1)]
            in_ch = ch + skip_ch if i > 0 else ch + ch
            self.up_blocks.append(nn.ModuleList([
                ResBlock(in_ch, out_ch, time_dim),
                SelfAttention(out_ch) if (len(channel_mults) - 1 - i) in use_attention_at else nn.Identity(),
            ]))
            ch = out_ch
            if i < len(channel_mults) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())

        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)

        skips = [h]
        for (res, attn), down in zip(self.down_blocks, self.down_samples):
            h = res(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for (res, attn), up in zip(self.up_blocks, self.up_samples):
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, s], dim=1)
            h = res(h, t_emb)
            h = attn(h)
            h = up(h)

        return self.final(h)


class OCTDiffusionModel(nn.Module):
    """Complete diffusion model pipeline for OCT images.

    Implements DDPM-style training and inference.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 256,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.denoiser = DenoisingUNet(
            in_channels=image_channels,
            out_channels=image_channels,
        )
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def forward_diffusion(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device)
        x_t, noise = self.forward_diffusion(x_0, t)
        pred_noise = self.denoiser(x_t, t.float())
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.float)
            pred_noise = self.denoiser(x, t_batch)
            alpha = self.alphas[t]
            alpha_cum = self.alphas_cumprod[t]
            beta = self.betas[t]
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cum)) * pred_noise
            )
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        return x


class OCTDenoiser(nn.Module):
    """Specialized denoiser for OCT speckle noise removal.

    Trained on pairs of noisy/clean OCT images, or self-supervised
    with noise2noise-style training on unpaired noisy data.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            *[nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
            ) for _ in range(6)],
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(self, noisy: torch.Tensor) -> dict[str, torch.Tensor]:
        clean = self.net(noisy)
        return {"denoised": clean, "residual": noisy - clean}


class OCTSuperResolver(nn.Module):
    """Super-resolution model for low-resolution portable-device OCT."""

    def __init__(self, scale_factor: int = 2, in_channels: int = 1):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            *[nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
            ) for _ in range(8)],
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale_factor ** 2, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(self, lr: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_extract(lr)
        sr = self.upsample(features)
        return {"super_resolved": sr}
