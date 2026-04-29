from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.model.blocks import AdaLNZeroBlock, modulate
from src.model.embeddings import LabelEmbedder, PatchEmbed, TimestepEmbedder, build_2d_sincos_pos_embed


@dataclass(frozen=True)
class DiTSpec:
    depth: int
    hidden_size: int
    num_heads: int
    mlp_ratio: float


MODEL_SPECS: dict[str, DiTSpec] = {
    "dit_plain_small": DiTSpec(depth=8, hidden_size=192, num_heads=4, mlp_ratio=4.0),
    "dit_plain_base": DiTSpec(depth=12, hidden_size=768, num_heads=12, mlp_ratio=4.0),
}


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PixelDiT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        class_dropout_prob: float,
        spec: DiTSpec,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = spec.hidden_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        self.patch_embed = PatchEmbed(in_channels, spec.hidden_size, patch_size)
        pos_embed = build_2d_sincos_pos_embed(spec.hidden_size, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.time_embed = TimestepEmbedder(spec.hidden_size)
        self.label_embed = LabelEmbedder(num_classes, spec.hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(spec.hidden_size, spec.num_heads, spec.mlp_ratio) for _ in range(spec.depth)]
        )
        self.final_layer = FinalLayer(spec.hidden_size, patch_size, in_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
        nn.init.zeros_(self.final_layer.ada_ln[-1].weight)
        nn.init.zeros_(self.final_layer.ada_ln[-1].bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patch_dim = self.patch_size * self.patch_size * self.in_channels
        x = x.view(batch_size, self.grid_size, self.grid_size, patch_dim)
        x = x.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch_size, self.in_channels, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
        condition = self.time_embed(timesteps) + self.label_embed(labels)
        for block in self.blocks:
            tokens = block(tokens, condition)
        output = self.final_layer(tokens, condition)
        return self.unpatchify(output)


def build_model(config: dict[str, object]) -> PixelDiT:
    model_name = str(config["name"])
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model name: {model_name}")
    if config.get("pos_embed_type", "2d_sincos") != "2d_sincos":
        raise ValueError("Only pos_embed_type='2d_sincos' is implemented.")
    spec = MODEL_SPECS[model_name]
    return PixelDiT(
        image_size=int(config["image_size"]),
        patch_size=int(config["patch_size"]),
        in_channels=int(config["in_channels"]),
        num_classes=int(config["num_classes"]),
        class_dropout_prob=float(config.get("class_dropout_prob", 0.0)),
        spec=spec,
    )
