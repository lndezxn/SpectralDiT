from __future__ import annotations

from collections.abc import Iterable

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def to_uint8_images(images: torch.Tensor) -> torch.Tensor:
    images = images.clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).round()
    return images.to(torch.uint8)


class GenerativeMetrics:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False).to(device)
        self.inception_score = InceptionScore(normalize=False).to(device)
        self._real_ready = False

    @torch.no_grad()
    def update_real(self, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if self._real_ready:
            return
        for images, _ in loader:
            self.fid.update(to_uint8_images(images).to(self.device), real=True)
        self._real_ready = True

    @torch.no_grad()
    def update_fake(self, fake_images: torch.Tensor) -> None:
        fake_uint8 = to_uint8_images(fake_images).to(self.device)
        self.fid.update(fake_uint8, real=False)
        self.inception_score.update(fake_uint8)

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        fid_value = float(self.fid.compute().item())
        inception_mean, inception_std = self.inception_score.compute()
        self.fid.reset()
        self.inception_score.reset()
        return {
            "fid": fid_value,
            "inception_score_mean": float(inception_mean.item()),
            "inception_score_std": float(inception_std.item()),
        }
