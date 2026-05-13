from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def load_vae(path: str | Path, device: torch.device | str) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(path).to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_latents(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    scaling_factor: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = latents.device
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if device.type in {"cuda", "cpu"} and dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )
    with autocast_context:
        images = vae.decode(latents / scaling_factor).sample
    return images.float().clamp(-1.0, 1.0)
