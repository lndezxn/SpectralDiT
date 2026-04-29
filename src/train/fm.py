from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_flow_matching_inputs(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = images.shape[0]
    noise = torch.randn_like(images)
    timesteps = torch.rand(batch_size, device=images.device, dtype=images.dtype)
    view_shape = (batch_size,) + (1,) * (images.ndim - 1)
    x_t = (1 - timesteps.view(view_shape)) * noise + timesteps.view(view_shape) * images
    target_velocity = images - noise
    return x_t, timesteps, target_velocity


def flow_matching_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)
