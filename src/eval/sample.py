from __future__ import annotations

import math
from contextlib import nullcontext

import torch


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
) -> torch.Tensor:
    x = torch.randn(num_samples, in_channels, image_size, image_size, device=device, dtype=dtype)
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((num_samples,), step / num_steps, device=device, dtype=dtype)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=dtype)
            if device.type in {"cuda", "cpu"} and dtype in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with autocast_context:
            velocity = model(x, t, labels)
        x = x + step_size * velocity
    return x.clamp(-1.0, 1.0)


def make_label_batch(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    repeats = math.ceil(num_samples / num_classes)
    labels = torch.arange(num_classes, device=device).repeat(repeats)
    return labels[:num_samples]
