from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_console, setup_logger

LOGGER = setup_logger(__name__)
LATENT_CHANNELS = 4
LATENT_SIZE = 32
IMAGE_CHANNELS = 3
IMAGE_SIZE = 256


class Imagenet100SplitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, split: Any, limit: int | None) -> None:
        self.split = split.with_format("torch")
        self.length = len(split) if limit is None else min(limit, len(split))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.split[index]
        return example["image"], example["label"]


def parse_args() -> argparse.Namespace:
    default_input = REPO_ROOT / "datasets" / "imagenet-100" / "processed" / "imagenet-100_256"
    default_output = REPO_ROOT / "datasets" / "imagenet-100" / "processed" / "imagenet-100_vae_latents_32"
    parser = argparse.ArgumentParser(description="Encode preprocessed ImageNet-100 images into SD VAE latents.")
    parser.add_argument("--input", type=Path, default=default_input, help="Input Hugging Face dataset directory.")
    parser.add_argument("--vae", type=Path, default=REPO_ROOT / "datasets" / "vae", help="Local AutoencoderKL directory.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output directory for .pt latent files.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Dataset splits to encode.")
    parser.add_argument("--batch-size", type=int, default=128, help="VAE encode batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--prefetch-factor", type=int, default=1, help="DataLoader prefetch factor when workers are enabled.")
    parser.add_argument("--device", type=str, default="cuda", help="Encoding device. Defaults to cuda.")
    parser.add_argument("--scaling-factor", type=float, default=0.18215, help="Latent scaling factor.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Stored latent dtype.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Encode only the first N examples per split.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def prepare_output_dir(output: Path, overwrite: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
    shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if args.scaling_factor <= 0.0:
        raise ValueError("--scaling-factor must be positive.")


def validate_image_batch(images: torch.Tensor, split_name: str, start_index: int) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"{split_name}[{start_index}:] expected images shaped [N, 3, 256, 256], got {tuple(images.shape)}."
        )
    image_min = float(images.min().item())
    image_max = float(images.max().item())
    if image_min < -1.00001 or image_max > 1.00001:
        raise ValueError(
            f"{split_name}[{start_index}:] expected image values in [-1, 1], got [{image_min}, {image_max}]."
        )


def encode_split(
    split_name: str,
    split: Any,
    vae: AutoencoderKL,
    args: argparse.Namespace,
    save_dtype: torch.dtype,
) -> dict[str, Any]:
    dataset = Imagenet100SplitDataset(split=split, limit=args.limit)
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        **loader_kwargs,
    )

    latents = torch.empty((len(dataset), LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=save_dtype)
    labels = torch.empty((len(dataset),), dtype=torch.long)
    write_index = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    )
    with progress:
        task = progress.add_task(f"Encoding {split_name}", total=len(loader))
        for images, batch_labels in loader:
            validate_image_batch(images, split_name=split_name, start_index=write_index)
            batch_size = images.shape[0]
            images = images.to(device=args.device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                encoded = vae.encode(images).latent_dist.mean * args.scaling_factor
            latents[write_index : write_index + batch_size] = encoded.to(dtype=save_dtype, device="cpu")
            labels[write_index : write_index + batch_size] = batch_labels.to(dtype=torch.long, device="cpu")
            write_index += batch_size
            progress.advance(task)

    if write_index != len(dataset):
        raise RuntimeError(f"{split_name}: wrote {write_index} rows, expected {len(dataset)}.")

    return {
        "latents": latents,
        "labels": labels,
        "meta": {
            "source": str(args.input),
            "vae": str(args.vae),
            "split": split_name,
            "num_examples": len(dataset),
            "image_shape": [IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
            "latent_shape": [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE],
            "posterior": "mean",
            "scaling_factor": args.scaling_factor,
            "dtype": args.dtype,
            "device": args.device,
            "limit": args.limit,
        },
    }


def main() -> None:
    from datasets import load_from_disk

    args = parse_args()
    validate_args(args)
    save_dtype = resolve_dtype(args.dtype)

    LOGGER.info("Loading processed ImageNet-100 from %s", args.input)
    dataset = load_from_disk(args.input)
    missing_splits = [split_name for split_name in args.splits if split_name not in dataset]
    if missing_splits:
        raise KeyError(f"Missing requested splits: {missing_splits}")

    LOGGER.info("Loading VAE from %s", args.vae)
    vae = AutoencoderKL.from_pretrained(args.vae).to(args.device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    prepare_output_dir(args.output, overwrite=args.overwrite)

    LOGGER.info(
        "Encoding splits=%s | batch_size=%d | dtype=%s | scaling_factor=%.5f",
        args.splits,
        args.batch_size,
        args.dtype,
        args.scaling_factor,
    )
    for split_name in args.splits:
        encoded_split = encode_split(
            split_name=split_name,
            split=dataset[split_name],
            vae=vae,
            args=args,
            save_dtype=save_dtype,
        )
        output_path = args.output / f"{split_name}.pt"
        LOGGER.info("Saving %s", output_path)
        torch.save(encoded_split, output_path)

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
