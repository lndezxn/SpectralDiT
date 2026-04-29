from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_dataset(root: str | Path, train: bool) -> datasets.CIFAR10:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return datasets.CIFAR10(root=str(root), train=train, download=False, transform=transform)


def build_dataloader(
    root: str | Path,
    train: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = build_cifar10_dataset(root=root, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
        persistent_workers=num_workers > 0,
    )
