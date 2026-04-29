from __future__ import annotations

import math

import torch
from torch import nn


def build_2d_sincos_pos_embed(
    hidden_size: int,
    grid_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if hidden_size % 4 != 0:
        raise ValueError(f"hidden_size must be divisible by 4, got {hidden_size}.")

    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    omega = torch.arange(hidden_size // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / max(hidden_size // 4, 1)))

    out_y = torch.einsum("m,d->md", grid_y.reshape(-1), omega)
    out_x = torch.einsum("m,d->md", grid_x.reshape(-1), omega)
    return torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half, 1)
    freqs = exponent.exp()
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(timesteps, self.frequency_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_prob > 0.0:
            drop_mask = torch.rand(labels.shape, device=labels.device) < self.dropout_prob
            labels = labels.masked_fill(drop_mask, self.num_classes)
        return self.embedding(labels)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x)
        return tokens.flatten(2).transpose(1, 2)
