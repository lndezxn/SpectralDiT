from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader, Dataset, Subset


class Imagenet100LatentDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.latents = payload["latents"]
        self.labels = payload["labels"].long()
        if self.latents.ndim != 4:
            raise ValueError(f"Expected latents shaped [N, C, H, W], got {tuple(self.latents.shape)}.")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.latents.shape[0]:
            raise ValueError("Latent labels must be shaped [N] and match the latent count.")

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[index].float(), self.labels[index]


class Imagenet100ImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: str | Path, split: str) -> None:
        dataset = load_from_disk(root)
        if split == "all":
            dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
        else:
            if split not in dataset:
                raise KeyError(f"Unknown ImageNet-100 split: {split}")
            dataset = dataset[split]
        self.dataset = dataset.with_format("torch")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.dataset[index]
        return example["image"].float(), example["label"].long()


def build_imagenet100_latent_dataloader(
    path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = Imagenet100LatentDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_imagenet100_image_dataloader(
    root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    limit: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]] = Imagenet100ImageDataset(root=root, split=split)
    if limit is not None:
        if limit > len(dataset):
            raise ValueError(f"limit={limit} exceeds ImageNet-100 {split} split size {len(dataset)}.")
        dataset = Subset(dataset, range(limit))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def resolve_imagenet100_latent_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(str(config["latent_root"]))
    return root / "train.pt", root / "validation.pt"
