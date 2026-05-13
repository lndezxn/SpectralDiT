from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import Progress

from src.eval.sample import make_label_batch, sample_euler
from src.model.dit import build_model
from src.train.ema import create_ema_model
from src.utils.checkpoint import load_checkpoint, resolve_run_directory
from src.utils.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample several checkpoints from their resolved configs and render one compact row per checkpoint."
    )
    parser.add_argument("--ckpt", type=str, nargs="+", required=True, help="Checkpoint paths, one per output row.")
    parser.add_argument("--row-labels", type=str, nargs="+", required=True, help="Left-side labels, one per checkpoint.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count per checkpoint.")
    parser.add_argument("--class-label", type=int, default=None, help="Optional class label to use for every sample.")
    parser.add_argument("--seed", type=int, default=42, help="Seed reused before sampling every checkpoint.")
    parser.add_argument("--padding", type=int, default=2, help="Pixels between sampled images and rows.")
    parser.add_argument("--label-width", type=int, default=72, help="Width of the left label column in pixels.")
    parser.add_argument("--font-size", type=int, default=16, help="Label font size.")
    parser.add_argument("--font-path", type=str, default=None, help="Optional TrueType font path for row labels.")
    parser.add_argument("--output", type=str, default=None, help="Output image path.")
    return parser.parse_args()


def load_label_font(font_path: str | None, font_size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(font_path, font_size)
    for candidate in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_condition_labels(
    num_samples: int,
    num_classes: int,
    device: torch.device,
    class_label: int | None,
) -> torch.Tensor:
    if class_label is None:
        return make_label_batch(num_samples, num_classes, device)
    return torch.full((num_samples,), class_label, device=device, dtype=torch.long)


def load_resolved_config(checkpoint_path: str | Path) -> dict[str, Any]:
    config_path = resolve_run_directory(checkpoint_path) / "config_resolved.yaml"
    if not config_path.exists():
        raise ValueError(f"Resolved config does not exist for checkpoint {checkpoint_path}: {config_path}")
    return load_config(config_path)


def samples_to_uint8_tiles(samples: torch.Tensor) -> torch.Tensor:
    images = (samples.cpu().clamp(-1.0, 1.0) + 1.0) * 127.5
    return images.round().byte().permute(0, 2, 3, 1).contiguous()


def paste_sample_row(
    canvas: Image.Image,
    tiles: torch.Tensor,
    label: str,
    y_offset: int,
    label_width: int,
    padding: int,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    _, image_size, _, channels = tiles.shape
    if channels not in {1, 3}:
        raise ValueError(f"Expected 1 or 3 image channels, got {channels}.")

    x_offset = label_width + padding
    for tile_index in range(tiles.shape[0]):
        tile = tiles[tile_index]
        if channels == 1:
            image = Image.fromarray(tile.squeeze(-1).numpy(), mode="L").convert("RGB")
        else:
            image = Image.fromarray(tile.numpy(), mode="RGB")
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image_size + padding

    text_lines = list(label)
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in text_lines]
    total_text_height = sum(line_heights)
    text_y = y_offset + max(0, (image_size - total_text_height) // 2)
    for line, line_height in zip(text_lines, line_heights, strict=True):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = max(0, (label_width - text_width) // 2)
        draw.text((text_x, text_y), line, fill=(20, 20, 20), font=font)
        text_y += line_height


def render_checkpoint_rows(
    rows: list[tuple[str, torch.Tensor]],
    output_path: Path,
    label_width: int,
    padding: int,
    font: ImageFont.ImageFont,
) -> None:
    if not rows:
        raise ValueError("At least one row is required.")

    first_tiles = rows[0][1]
    _, image_size, _, _ = first_tiles.shape
    num_samples = first_tiles.shape[0]
    for _, tiles in rows:
        if tiles.shape != first_tiles.shape:
            raise ValueError("All checkpoint rows must have the same sampled image shape.")

    canvas_width = label_width + padding + num_samples * image_size + max(0, num_samples - 1) * padding
    canvas_height = len(rows) * image_size + max(0, len(rows) - 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

    y_offset = 0
    for label, tiles in rows:
        paste_sample_row(
            canvas=canvas,
            tiles=tiles,
            label=label,
            y_offset=y_offset,
            label_width=label_width,
            padding=padding,
            font=font,
        )
        y_offset += image_size + padding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if len(args.ckpt) != len(args.row_labels):
        raise ValueError("--ckpt and --row-labels must contain the same number of values.")
    if args.padding < 0:
        raise ValueError("--padding must be non-negative.")
    if args.label_width <= 0:
        raise ValueError("--label-width must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()

    rows: list[tuple[str, torch.Tensor]] = []
    first_config: dict | None = None
    expected_num_samples: int | None = None
    with Progress(console=console) as progress:
        task_id = progress.add_task("sampling checkpoints", total=len(args.ckpt))
        for checkpoint_path, row_label in zip(args.ckpt, args.row_labels, strict=True):
            config = load_resolved_config(checkpoint_path)
            if first_config is None:
                first_config = config

            model = build_model(config["model"])
            ema_model = create_ema_model(model)
            load_checkpoint(checkpoint_path, model, ema_model=ema_model)
            ema_model.eval()
            ema_model.to(device)

            num_samples = args.num_samples or int(config["sample"]["num_samples"])
            if expected_num_samples is None:
                expected_num_samples = num_samples
            elif num_samples != expected_num_samples:
                raise ValueError("All checkpoint configs must use the same sample.num_samples or pass --num-samples.")
            labels = make_condition_labels(
                num_samples=num_samples,
                num_classes=int(config["model"]["num_classes"]),
                device=device,
                class_label=args.class_label,
            )

            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
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
            rows.append((row_label, samples_to_uint8_tiles(samples)))
            progress.advance(task_id)

    if first_config is None:
        raise ValueError("At least one checkpoint is required.")
    output_path = (
        Path(args.output)
        if args.output is not None
        else ensure_dir(Path(first_config["train"]["output_dir"]) / "manual_samples") / "checkpoint_rows.png"
    )
    render_checkpoint_rows(
        rows=rows,
        output_path=output_path,
        label_width=args.label_width,
        padding=args.padding,
        font=load_label_font(args.font_path, args.font_size),
    )
    console.print(f"Saved checkpoint row visualization to {output_path}")


if __name__ == "__main__":
    main()
