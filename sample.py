from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from src.eval.sample import make_label_batch, sample_euler
from src.model.dit import build_model
from src.train.ema import create_ema_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained pixel-space DiT.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a checkpoint file.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config["model"])
    ema_model = create_ema_model(model)
    load_checkpoint(args.ckpt, model, ema_model=ema_model)
    ema_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema_model.to(device)

    num_samples = args.num_samples or int(config["sample"]["num_samples"])
    labels = make_label_batch(num_samples, int(config["model"]["num_classes"]), device)
    samples = sample_euler(
        model=ema_model,
        num_samples=num_samples,
        image_size=int(config["model"]["image_size"]),
        in_channels=int(config["model"]["in_channels"]),
        labels=labels,
        num_steps=int(config["sample"]["num_steps"]),
        device=device,
        dtype=torch.float32,
    )

    output_dir = ensure_dir(Path(config["train"]["output_dir"]) / "manual_samples")
    grid = make_grid(samples.cpu(), nrow=min(8, num_samples), normalize=True, value_range=(-1, 1))
    image = (grid.clamp(0.0, 1.0) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(image).save(output_dir / "sample_grid.png")


if __name__ == "__main__":
    main()
