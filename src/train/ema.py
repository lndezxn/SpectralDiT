from __future__ import annotations

from copy import deepcopy

import torch


def create_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_tensor in ema_state.items():
        model_tensor = model_state[name]
        if torch.is_floating_point(ema_tensor):
            ema_tensor.mul_(decay).add_(model_tensor, alpha=1.0 - decay)
        else:
            ema_tensor.copy_(model_tensor)
