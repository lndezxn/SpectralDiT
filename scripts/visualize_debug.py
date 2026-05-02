from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torchvision.utils import make_grid

from src.utils.config import ensure_dir
from src.utils.logging import get_console, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DiT sampling debug dumps.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing sample_step_*.pt files.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for PNG visualizations. Defaults to <input>/viz.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional override for rendered image size. Defaults to the dump metadata image_size.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional max number of samples to render from each step dump.",
    )
    return parser.parse_args()


def list_dump_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("sample_step_*.pt"))


def project_token_tensor_to_rgb(
    token_tensor: torch.Tensor,
    grid_size: int,
    patch_size: int,
    image_size: int,
) -> torch.Tensor:
    grouped = torch.stack([token_tensor[:, offset::3].mean(dim=-1) for offset in range(3)], dim=0)
    rgb = grouped.view(3, grid_size, grid_size)
    if patch_size > 1:
        rgb = rgb.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    if rgb.shape[-1] != image_size or rgb.shape[-2] != image_size:
        rgb = F.interpolate(rgb.unsqueeze(0), size=(image_size, image_size), mode="nearest").squeeze(0)
    return rgb


def project_pixel_tensor_to_rgb(pixel_tensor: torch.Tensor) -> torch.Tensor:
    if pixel_tensor.shape[0] == 3:
        return pixel_tensor
    if pixel_tensor.shape[0] == 1:
        return pixel_tensor.repeat(3, 1, 1)
    return torch.stack([pixel_tensor[offset::3].mean(dim=0) for offset in range(3)], dim=0)


def normalize_rgb_stack(rgb_stack: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    if symmetric:
        scale = rgb_stack.abs().amax()
        if float(scale) == 0.0:
            return torch.full_like(rgb_stack, 0.5)
        return (rgb_stack / (2.0 * scale)) + 0.5

    min_value = rgb_stack.amin()
    max_value = rgb_stack.amax()
    if float(max_value - min_value) == 0.0:
        return torch.zeros_like(rgb_stack)
    return (rgb_stack - min_value) / (max_value - min_value)


def save_rgb_image(rgb_tensor: torch.Tensor, output_path: Path) -> None:
    image = (rgb_tensor.clamp(0.0, 1.0) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(image).save(output_path)


def save_block_grid(
    token_blocks: torch.Tensor,
    output_path: Path,
    grid_size: int,
    patch_size: int,
    image_size: int,
) -> None:
    rendered_blocks = torch.stack(
        [project_token_tensor_to_rgb(block_tokens, grid_size, patch_size, image_size) for block_tokens in token_blocks],
        dim=0,
    )
    rendered_blocks = normalize_rgb_stack(rendered_blocks, symmetric=True)
    nrow = math.ceil(math.sqrt(rendered_blocks.shape[0]))
    grid = make_grid(rendered_blocks, nrow=nrow, padding=2)
    save_rgb_image(grid, output_path)


def save_single_token_map(
    token_tensor: torch.Tensor,
    output_path: Path,
    grid_size: int,
    patch_size: int,
    image_size: int,
) -> None:
    rgb = project_token_tensor_to_rgb(token_tensor, grid_size, patch_size, image_size)
    save_rgb_image(normalize_rgb_stack(rgb, symmetric=True), output_path)


def save_pixel_prediction(pixel_tensor: torch.Tensor, output_path: Path) -> None:
    rgb = project_pixel_tensor_to_rgb(pixel_tensor)
    save_rgb_image(normalize_rgb_stack(rgb, symmetric=True), output_path)


def compute_pred_x0_pixels(step_xt_pixels: torch.Tensor, step_prediction_pixels: torch.Tensor, timestep_value: float) -> torch.Tensor:
    return step_xt_pixels + (1.0 - timestep_value) * step_prediction_pixels


def save_block_flow_overview(
    attn_blocks: torch.Tensor,
    mlp_pre_blocks: torch.Tensor,
    mlp_low_correction_blocks: torch.Tensor,
    mlp_high_correction_blocks: torch.Tensor,
    mlp_final_blocks: torch.Tensor,
    output_blocks: torch.Tensor,
    output_path: Path,
    grid_size: int,
    patch_size: int,
    image_size: int,
) -> None:
    num_blocks = int(attn_blocks.shape[0])
    tile_size = max(image_size, 48)
    tile_padding = 4
    label_width = 26
    header_height = 14
    row_gap = 4
    width = label_width + (6 * tile_size) + (5 * tile_padding)
    height = header_height + (num_blocks * tile_size) + (max(num_blocks - 1, 0) * row_gap)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    column_labels = ("attn", "pre", "low_corr", "high_corr", "mlp", "x")
    for column_index, label in enumerate(column_labels):
        x_offset = label_width + column_index * (tile_size + tile_padding)
        draw.text((x_offset, 1), label, fill=(0, 0, 0))

    for block_index in range(num_blocks):
        row_y = header_height + block_index * (tile_size + row_gap)
        draw.text((1, row_y + 1), f"b{block_index}", fill=(0, 0, 0))
        block_images = (
            normalize_rgb_stack(
                project_token_tensor_to_rgb(attn_blocks[block_index], grid_size, patch_size, image_size),
                symmetric=True,
            ),
            normalize_rgb_stack(
                project_token_tensor_to_rgb(mlp_pre_blocks[block_index], grid_size, patch_size, image_size),
                symmetric=True,
            ),
            normalize_rgb_stack(
                project_token_tensor_to_rgb(mlp_low_correction_blocks[block_index], grid_size, patch_size, image_size),
                symmetric=True,
            ),
            normalize_rgb_stack(
                project_token_tensor_to_rgb(mlp_high_correction_blocks[block_index], grid_size, patch_size, image_size),
                symmetric=True,
            ),
            normalize_rgb_stack(
                project_token_tensor_to_rgb(mlp_final_blocks[block_index], grid_size, patch_size, image_size),
                symmetric=True,
            ),
            normalize_rgb_stack(
                project_token_tensor_to_rgb(output_blocks[block_index], grid_size, patch_size, image_size),
                symmetric=True,
            ),
        )
        for column_index, block_image in enumerate(block_images):
            x_offset = label_width + column_index * (tile_size + tile_padding)
            tile = Image.fromarray((block_image.clamp(0.0, 1.0) * 255).round().byte().permute(1, 2, 0).numpy())
            if tile_size != image_size:
                tile = tile.resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)
            canvas.paste(tile, (x_offset, row_y))

    canvas.save(output_path)


def visualize_dump_file(dump_path: Path, input_root: Path, output_root: Path, image_size_override: int | None, sample_limit: int | None) -> None:
    payload = torch.load(dump_path, map_location="cpu")
    if "step_xt_pixels" not in payload:
        raise ValueError(f"Dump file is missing step_xt_pixels and must be regenerated: {dump_path}")
    required_keys = (
        "mlp_residual_pre_freq_gate",
        "mlp_residual_low_pre_gate",
        "mlp_residual_high_pre_gate",
        "mlp_residual_low_correction",
        "mlp_residual_high_correction",
        "mlp_residual",
    )
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"Dump file is missing required keys {missing_keys} and must be regenerated: {dump_path}")

    meta = payload["meta"]
    grid_size = int(meta["grid_size"])
    patch_size = int(meta["patch_size"])
    image_size = image_size_override or int(meta["image_size"])
    batch_size = int(payload["step_output_tokens"].shape[0])
    max_samples = batch_size if sample_limit is None else min(batch_size, sample_limit)
    timestep_value = float(payload["timestep_value"])

    relative_parent = dump_path.parent.relative_to(input_root)
    step_dir = ensure_dir(output_root / relative_parent / dump_path.stem)
    for sample_index in range(max_samples):
        sample_dir = ensure_dir(step_dir / f"sample_{sample_index:03d}")
        save_block_grid(
            payload["attn_residual"][:, sample_index],
            sample_dir / "attn_residual_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["mlp_residual"][:, sample_index],
            sample_dir / "mlp_residual_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["mlp_residual_pre_freq_gate"][:, sample_index],
            sample_dir / "mlp_residual_pre_freq_gate_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["mlp_residual_low_pre_gate"][:, sample_index],
            sample_dir / "mlp_residual_low_pre_gate_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["mlp_residual_high_pre_gate"][:, sample_index],
            sample_dir / "mlp_residual_high_pre_gate_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["mlp_residual_low_correction"][:, sample_index],
            sample_dir / "mlp_residual_low_correction_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["mlp_residual_high_correction"][:, sample_index],
            sample_dir / "mlp_residual_high_correction_blocks.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_grid(
            payload["block_output_tokens"][:, sample_index],
            sample_dir / "block_output_tokens.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_block_flow_overview(
            payload["attn_residual"][:, sample_index],
            payload["mlp_residual_pre_freq_gate"][:, sample_index],
            payload["mlp_residual_low_correction"][:, sample_index],
            payload["mlp_residual_high_correction"][:, sample_index],
            payload["mlp_residual"][:, sample_index],
            payload["block_output_tokens"][:, sample_index],
            sample_dir / "block_flow_overview.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_single_token_map(
            payload["step_output_tokens"][sample_index],
            sample_dir / "step_output_tokens.png",
            grid_size=grid_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        save_pixel_prediction(
            payload["step_xt_pixels"][sample_index],
            sample_dir / "step_xt_pixels.png",
        )
        save_pixel_prediction(
            payload["step_prediction_pixels"][sample_index],
            sample_dir / "step_prediction_pixels.png",
        )
        save_pixel_prediction(
            compute_pred_x0_pixels(
                payload["step_xt_pixels"][sample_index],
                payload["step_prediction_pixels"][sample_index],
                timestep_value=timestep_value,
            ),
            sample_dir / "step_pred_x0_pixels.png",
        )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir = ensure_dir(Path(args.output) if args.output is not None else input_dir / "viz")
    dump_files = list_dump_files(input_dir)
    if not dump_files:
        raise ValueError(f"No sample_step_*.pt files found under: {input_dir}")

    logger = setup_logger("spectraldit.visualize_debug")
    logger.info("Found %d debug dump files under %s", len(dump_files), input_dir)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=get_console(),
    )
    with progress:
        task_id = progress.add_task("visualizing", total=len(dump_files))
        for dump_path in dump_files:
            visualize_dump_file(
                dump_path=dump_path,
                input_root=input_dir,
                output_root=output_dir,
                image_size_override=args.image_size,
                sample_limit=args.sample_limit,
            )
            progress.update(task_id, advance=1)

    logger.info("Saved visualizations to %s", output_dir)


if __name__ == "__main__":
    main()
