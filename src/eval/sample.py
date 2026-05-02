from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from src.eval.debug import SamplingDebugCollector, resolve_debug_config, resolve_save_dtype


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    num_samples: int,
    image_size: int,
    in_channels: int,
    labels: torch.Tensor,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    debug_output_dir: str | Path | None = None,
    debug_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    x = torch.randn(num_samples, in_channels, image_size, image_size, device=device, dtype=dtype)
    step_size = 1.0 / num_steps
    collector: SamplingDebugCollector | None = None
    resolved_debug_config = resolve_debug_config({} if debug_config is None else {"debug": debug_config})
    if resolved_debug_config["enabled"]:
        if debug_output_dir is None:
            raise ValueError("debug_output_dir must be provided when sample debug is enabled.")
        unwrapped_model = model.module if hasattr(model, "module") else model
        collector = SamplingDebugCollector(
            output_dir=debug_output_dir,
            save_dtype=resolve_save_dtype(str(resolved_debug_config["save_dtype"])),
            labels=labels,
            meta={
                "image_size": image_size,
                "in_channels": in_channels,
                "num_patches": int(unwrapped_model.num_patches),
                "grid_size": int(unwrapped_model.grid_size),
                "patch_size": int(unwrapped_model.patch_size),
                "hidden_size": int(unwrapped_model.hidden_size),
                "depth": len(unwrapped_model.blocks),
                "freq_residual_gating_enabled": bool(unwrapped_model.freq_residual_gating.enabled),
                "freq_residual_gating_scale": float(unwrapped_model.freq_residual_gating.gate_scale),
            },
        )
    for step in range(num_steps):
        timestep_value = step / num_steps
        t = torch.full((num_samples,), timestep_value, device=device, dtype=dtype)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=dtype)
            if device.type in {"cuda", "cpu"} and dtype in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with autocast_context:
            velocity = model(x, t, labels, debug_collector=collector)
        if collector is not None:
            collector.set_step_xt_pixels(x)
            collector.set_step_prediction_pixels(velocity)
        x = x + step_size * velocity
        if collector is not None:
            collector.flush_step(step_index=step, timestep_value=timestep_value)
    return x.clamp(-1.0, 1.0)


def make_label_batch(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    repeats = math.ceil(num_samples / num_classes)
    labels = torch.arange(num_classes, device=device).repeat(repeats)
    return labels[:num_samples]
