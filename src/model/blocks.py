from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass(frozen=True)
class BlockDebugTensors:
    attn_residual: torch.Tensor
    freq_gate_low: torch.Tensor
    freq_gate_high: torch.Tensor
    mlp_residual_pre_freq_gate: torch.Tensor
    mlp_residual_low_pre_gate: torch.Tensor
    mlp_residual_high_pre_gate: torch.Tensor
    mlp_residual_low_correction: torch.Tensor
    mlp_residual_high_correction: torch.Tensor
    mlp_residual: torch.Tensor
    block_output_tokens: torch.Tensor


@dataclass(frozen=True)
class FreqResidualGatingConfig:
    enabled: bool = False
    gate_scale: float | None = None


def split_token_frequency(token_tensor: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    low_frequency = low_pass_token_frequency(token_tensor, grid_size)
    high_frequency = token_tensor - low_frequency
    return low_frequency, high_frequency


def low_pass_token_frequency(token_tensor: torch.Tensor, grid_size: int) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = token_tensor.shape
    if sequence_length != grid_size * grid_size:
        raise ValueError(f"Expected {grid_size * grid_size} tokens for grid_size={grid_size}, got {sequence_length}.")

    token_grid = token_tensor.transpose(1, 2).contiguous().view(batch_size, hidden_size, grid_size, grid_size)
    kernel_1d = token_tensor.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, 5, 5).expand(hidden_size, 1, 5, 5)
    padded_grid = F.pad(token_grid, (2, 2, 2, 2), mode="replicate")
    low_frequency_grid = F.conv2d(padded_grid, kernel, groups=hidden_size)
    return low_frequency_grid.view(batch_size, hidden_size, sequence_length).transpose(1, 2).contiguous()


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_size = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def _apply_rms_norm(self, x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
        weight = norm.weight
        if weight is not None:
            weight = weight.to(dtype=x.dtype)
        return F.rms_norm(x, (self.head_dim,), weight=weight, eps=norm.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self._apply_rms_norm(q, self.q_norm)
        k = self._apply_rms_norm(k, self.k_norm)
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.proj(attn_output)


class AdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        freq_residual_gating: FreqResidualGatingConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.freq_residual_gating = freq_residual_gating or FreqResidualGatingConfig()
        self.freq_gate: nn.Sequential | None = None
        if self.freq_residual_gating.enabled:
            gate_hidden_size = hidden_size // 4
            self.freq_gate = nn.Sequential(
                nn.Linear(hidden_size, gate_hidden_size),
                nn.SiLU(),
                nn.Linear(gate_hidden_size, 2),
            )

    def _compute_freq_gates(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.freq_gate is None:
            raise ValueError("freq_gate must be initialized when frequency residual gating is enabled.")
        if self.freq_residual_gating.gate_scale is None:
            raise ValueError("freq_residual_gating.gate_scale must be provided when frequency residual gating is enabled.")
        gate_logits = self.freq_gate(condition)
        freq_gates = torch.tanh(gate_logits) * self.freq_residual_gating.gate_scale
        gate_low = freq_gates[:, 0].view(-1, 1, 1)
        gate_high = freq_gates[:, 1].view(-1, 1, 1)
        return gate_low, gate_high

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        return_debug_tensors: bool = False,
        token_grid_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, BlockDebugTensors]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(condition).chunk(6, dim=-1)

        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(attn_input)
        attn_residual = gate_msa.unsqueeze(1) * attn_output
        x = x + attn_residual

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_raw = self.mlp(mlp_input)
        mlp_residual_pre_freq_gate = gate_mlp.unsqueeze(1) * mlp_raw
        mlp_residual = mlp_residual_pre_freq_gate
        gate_low: torch.Tensor | None = None
        gate_high: torch.Tensor | None = None
        if self.freq_residual_gating.enabled:
            if token_grid_size is None:
                raise ValueError("token_grid_size must be provided when frequency residual gating is enabled.")
            gate_low, gate_high = self._compute_freq_gates(condition)
            if return_debug_tensors:
                mlp_residual_low_pre_gate, mlp_residual_high_pre_gate = split_token_frequency(
                    mlp_residual_pre_freq_gate,
                    grid_size=token_grid_size,
                )
                mlp_residual_low_correction = gate_low * mlp_residual_low_pre_gate
                mlp_residual_high_correction = gate_high * mlp_residual_high_pre_gate
                mlp_residual = (
                    mlp_residual_pre_freq_gate + mlp_residual_low_correction + mlp_residual_high_correction
                )
            else:
                low_frequency = low_pass_token_frequency(
                    mlp_residual_pre_freq_gate,
                    grid_size=token_grid_size,
                )
                mlp_residual = (
                    mlp_residual_pre_freq_gate
                    + (gate_high * mlp_residual_pre_freq_gate)
                    + ((gate_low - gate_high) * low_frequency)
                )
        x = x + mlp_residual
        if not return_debug_tensors:
            return x
        if token_grid_size is None:
            raise ValueError("token_grid_size must be provided when return_debug_tensors=True.")
        mlp_residual_low_pre_gate: torch.Tensor | None = None
        mlp_residual_high_pre_gate: torch.Tensor | None = None
        mlp_residual_low_correction: torch.Tensor | None = None
        mlp_residual_high_correction: torch.Tensor | None = None
        if self.freq_residual_gating.enabled:
            if gate_low is None or gate_high is None:
                gate_low, gate_high = self._compute_freq_gates(condition)
            mlp_residual_low_pre_gate, mlp_residual_high_pre_gate = split_token_frequency(
                mlp_residual_pre_freq_gate,
                grid_size=token_grid_size,
            )
            mlp_residual_low_correction = gate_low * mlp_residual_low_pre_gate
            mlp_residual_high_correction = gate_high * mlp_residual_high_pre_gate
        else:
            mlp_residual_low_pre_gate, mlp_residual_high_pre_gate = split_token_frequency(
                mlp_residual_pre_freq_gate,
                grid_size=token_grid_size,
            )
            mlp_residual_low_correction = torch.zeros_like(mlp_residual_pre_freq_gate)
            mlp_residual_high_correction = torch.zeros_like(mlp_residual_pre_freq_gate)
        return x, BlockDebugTensors(
            attn_residual=attn_residual,
            freq_gate_low=gate_low.view(-1) if gate_low is not None else torch.zeros_like(condition[:, 0]),
            freq_gate_high=gate_high.view(-1) if gate_high is not None else torch.zeros_like(condition[:, 0]),
            mlp_residual_pre_freq_gate=mlp_residual_pre_freq_gate,
            mlp_residual_low_pre_gate=mlp_residual_low_pre_gate,
            mlp_residual_high_pre_gate=mlp_residual_high_pre_gate,
            mlp_residual_low_correction=mlp_residual_low_correction,
            mlp_residual_high_correction=mlp_residual_high_correction,
            mlp_residual=mlp_residual,
            block_output_tokens=x,
        )
