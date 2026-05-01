from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.config import ensure_dir


def resolve_debug_config(sample_config: dict[str, Any]) -> dict[str, Any]:
    debug_config = dict(sample_config.get("debug", {}))
    debug_config.setdefault("enabled", False)
    debug_config.setdefault("output_subdir", "debug_tokens")
    debug_config.setdefault("save_dtype", "float32")
    return debug_config


def resolve_save_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported debug save dtype: {name}")


class SamplingDebugCollector:
    def __init__(
        self,
        output_dir: str | Path,
        save_dtype: torch.dtype,
        labels: torch.Tensor,
        meta: dict[str, Any],
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.save_dtype = save_dtype
        self.labels = labels.detach().cpu()
        self.meta = dict(meta)
        self._attn_residuals: list[torch.Tensor] = []
        self._mlp_residuals: list[torch.Tensor] = []
        self._mlp_residuals_low: list[torch.Tensor] = []
        self._mlp_residuals_high: list[torch.Tensor] = []
        self._block_outputs: list[torch.Tensor] = []
        self._step_output_tokens: torch.Tensor | None = None
        self._step_xt_pixels: torch.Tensor | None = None
        self._step_prediction_pixels: torch.Tensor | None = None

    def record_block(
        self,
        attn_residual: torch.Tensor,
        mlp_residual: torch.Tensor,
        mlp_residual_low: torch.Tensor,
        mlp_residual_high: torch.Tensor,
        block_output_tokens: torch.Tensor,
    ) -> None:
        self._attn_residuals.append(self._prepare_tensor(attn_residual))
        self._mlp_residuals.append(self._prepare_tensor(mlp_residual))
        self._mlp_residuals_low.append(self._prepare_tensor(mlp_residual_low))
        self._mlp_residuals_high.append(self._prepare_tensor(mlp_residual_high))
        self._block_outputs.append(self._prepare_tensor(block_output_tokens))

    def set_step_output_tokens(self, step_output_tokens: torch.Tensor) -> None:
        self._step_output_tokens = self._prepare_tensor(step_output_tokens)

    def set_step_xt_pixels(self, step_xt_pixels: torch.Tensor) -> None:
        self._step_xt_pixels = self._prepare_tensor(step_xt_pixels)

    def set_step_prediction_pixels(self, step_prediction_pixels: torch.Tensor) -> None:
        self._step_prediction_pixels = self._prepare_tensor(step_prediction_pixels)

    def flush_step(self, step_index: int, timestep_value: float) -> None:
        if (
            not self._attn_residuals
            or not self._mlp_residuals
            or not self._mlp_residuals_low
            or not self._mlp_residuals_high
            or not self._block_outputs
        ):
            raise ValueError("Cannot flush debug step before any block outputs were recorded.")
        if self._step_output_tokens is None:
            raise ValueError("Cannot flush debug step before step output tokens were recorded.")
        if self._step_xt_pixels is None:
            raise ValueError("Cannot flush debug step before step x_t pixels were recorded.")
        if self._step_prediction_pixels is None:
            raise ValueError("Cannot flush debug step before step pixel predictions were recorded.")

        payload = {
            "step_index": step_index,
            "timestep_value": timestep_value,
            "attn_residual": torch.stack(self._attn_residuals, dim=0),
            "mlp_residual": torch.stack(self._mlp_residuals, dim=0),
            "mlp_residual_low": torch.stack(self._mlp_residuals_low, dim=0),
            "mlp_residual_high": torch.stack(self._mlp_residuals_high, dim=0),
            "block_output_tokens": torch.stack(self._block_outputs, dim=0),
            "step_output_tokens": self._step_output_tokens,
            "step_xt_pixels": self._step_xt_pixels,
            "step_prediction_pixels": self._step_prediction_pixels,
            "labels": self.labels,
            "meta": self.meta,
        }
        torch.save(payload, self.output_dir / f"sample_step_{step_index:04d}.pt")
        self._reset_step()

    def _reset_step(self) -> None:
        self._attn_residuals.clear()
        self._mlp_residuals.clear()
        self._mlp_residuals_low.clear()
        self._mlp_residuals_high.clear()
        self._block_outputs.clear()
        self._step_output_tokens = None
        self._step_xt_pixels = None
        self._step_prediction_pixels = None

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu", dtype=self.save_dtype)
