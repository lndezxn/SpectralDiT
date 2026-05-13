from __future__ import annotations

import argparse
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize

if TYPE_CHECKING:
    from datasets import DatasetDict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import setup_logger

LOGGER = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    default_root = REPO_ROOT / "datasets" / "imagenet-100"
    parser = argparse.ArgumentParser(
        description="Preprocess ImageNet-100 to square images by resizing the shorter side and center-cropping."
    )
    parser.add_argument("--data-root", type=Path, default=default_root, help="ImageNet-100 dataset directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "processed" / "imagenet-100_256",
        help="Output directory for the processed Hugging Face dataset.",
    )
    parser.add_argument("--size", type=int, default=256, help="Output image size.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size passed to datasets.map.")
    parser.add_argument("--num-proc", type=int, default=8, help="Number of processes for datasets.map/save_to_disk.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it already exists.")
    return parser.parse_args()


def load_imagenet100(data_root: Path) -> "DatasetDict":
    from datasets import Image, load_dataset

    data_dir = data_root / "data"
    train_files = sorted(data_dir.glob("train-*.parquet"))
    validation_files = sorted(data_dir.glob("validation-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No train parquet files found under {data_dir}.")
    if not validation_files:
        raise FileNotFoundError(f"No validation parquet files found under {data_dir}.")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": [str(path) for path in train_files],
            "validation": [str(path) for path in validation_files],
        },
    )
    return dataset.cast_column("image", Image(decode=True))


def resize_crop_normalize_batch(examples: dict[str, list[Any]], size: int) -> dict[str, list[Any]]:
    images: list[np.ndarray] = []
    for image in examples["image"]:
        image = image.convert("RGB")
        image = resize(image, size, interpolation=InterpolationMode.BICUBIC)
        image = center_crop(image, [size, size])
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        image_array = image_array / 127.5 - 1.0
        images.append(image_array)
    examples["image"] = images
    return examples


def prepare_output_dir(output: Path, overwrite: bool) -> None:
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
    shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from datasets import Array3D, Features

    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be positive.")

    LOGGER.info("Loading ImageNet-100 from %s", args.data_root)
    dataset = load_imagenet100(args.data_root)
    prepare_output_dir(args.output, args.overwrite)

    features = Features(
        {
            "image": Array3D(shape=(3, args.size, args.size), dtype="float32"),
            "label": dataset["train"].features["label"],
        }
    )

    LOGGER.info(
        "Resizing shorter side to %d, center-cropping to %dx%d, and normalizing to [-1, 1]",
        args.size,
        args.size,
        args.size,
    )
    processed = dataset.map(
        partial(resize_crop_normalize_batch, size=args.size),
        batched=True,
        batch_size=args.batch_size,
        features=features,
        num_proc=args.num_proc,
        desc=f"Preprocessing ImageNet-100 to {args.size}x{args.size}",
    )

    LOGGER.info("Saving processed dataset to %s", args.output)
    processed.save_to_disk(args.output, num_proc=args.num_proc)
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
