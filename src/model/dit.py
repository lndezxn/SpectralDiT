from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.blocks import AdaLNZeroBlock, FreqResidualGatingConfig, modulate
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
        freq_residual_gating: FreqResidualGatingConfig | None = None,
        activation_checkpointing: bool = False,
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
        self.freq_residual_gating = freq_residual_gating or FreqResidualGatingConfig()
        self.activation_checkpointing = activation_checkpointing

        self.patch_embed = PatchEmbed(in_channels, spec.hidden_size, patch_size)
        pos_embed = build_2d_sincos_pos_embed(spec.hidden_size, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.time_embed = TimestepEmbedder(spec.hidden_size)
        self.label_embed = LabelEmbedder(num_classes, spec.hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    spec.hidden_size,
                    spec.num_heads,
                    spec.mlp_ratio,
                    freq_residual_gating=self.freq_residual_gating,
                )
                for _ in range(spec.depth)
            ]
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
            if block.freq_gate is not None:
                nn.init.zeros_(block.freq_gate[-1].weight)
                nn.init.zeros_(block.freq_gate[-1].bias)
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

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        debug_collector: Any | None = None,
    ) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
        condition = self.time_embed(timesteps) + self.label_embed(labels)
        if debug_collector is None:
            for block in self.blocks:
                if self.training and self.activation_checkpointing:
                    def run_block(block_tokens: torch.Tensor, block_condition: torch.Tensor, *, current_block: nn.Module = block) -> torch.Tensor:
                        return current_block(
                            block_tokens,
                            block_condition,
                            token_grid_size=self.grid_size,
                        )

                    tokens = checkpoint(
                        run_block,
                        tokens,
                        condition,
                        use_reentrant=False,
                    )
                else:
                    tokens = block(tokens, condition, token_grid_size=self.grid_size)
        else:
            for block in self.blocks:
                tokens, debug_tensors = block(
                    tokens,
                    condition,
                    return_debug_tensors=True,
                    token_grid_size=self.grid_size,
                )
                debug_collector.record_block(
                    attn_residual=debug_tensors.attn_residual,
                    freq_gate_low=debug_tensors.freq_gate_low,
                    freq_gate_high=debug_tensors.freq_gate_high,
                    mlp_residual_pre_freq_gate=debug_tensors.mlp_residual_pre_freq_gate,
                    mlp_residual_low_pre_gate=debug_tensors.mlp_residual_low_pre_gate,
                    mlp_residual_high_pre_gate=debug_tensors.mlp_residual_high_pre_gate,
                    mlp_residual_low_correction=debug_tensors.mlp_residual_low_correction,
                    mlp_residual_high_correction=debug_tensors.mlp_residual_high_correction,
                    mlp_residual=debug_tensors.mlp_residual,
                    block_output_tokens=debug_tensors.block_output_tokens,
                )
            debug_collector.set_step_output_tokens(tokens)
        output = self.final_layer(tokens, condition)
        return self.unpatchify(output)


def build_model(config: dict[str, object]) -> PixelDiT:
    model_name = str(config["name"])
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model name: {model_name}")
    if config.get("pos_embed_type", "2d_sincos") != "2d_sincos":
        raise ValueError("Only pos_embed_type='2d_sincos' is implemented.")
    spec = MODEL_SPECS[model_name]
    freq_gate_config = config.get("freq_residual_gating")
    if freq_gate_config is None:
        freq_residual_gating = FreqResidualGatingConfig(enabled=False, gate_scale=None)
    else:
        freq_gate_config = dict(freq_gate_config)
        if "enabled" not in freq_gate_config:
            raise ValueError("model.freq_residual_gating.enabled must be provided when freq_residual_gating is configured.")
        if bool(freq_gate_config["enabled"]) and "gate_scale" not in freq_gate_config:
            raise ValueError("model.freq_residual_gating.gate_scale must be provided when freq_residual_gating.enabled=true.")
        freq_residual_gating = FreqResidualGatingConfig(
            enabled=bool(freq_gate_config["enabled"]),
            gate_scale=float(freq_gate_config["gate_scale"]) if "gate_scale" in freq_gate_config else None,
        )
    activation_checkpointing = bool(config.get("activation_checkpointing", False))
    return PixelDiT(
        image_size=int(config["image_size"]),
        patch_size=int(config["patch_size"]),
        in_channels=int(config["in_channels"]),
        num_classes=int(config["num_classes"]),
        class_dropout_prob=float(config.get("class_dropout_prob", 0.0)),
        spec=spec,
        freq_residual_gating=freq_residual_gating,
        activation_checkpointing=activation_checkpointing,
    )
