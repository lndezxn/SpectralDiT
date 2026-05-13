from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.eval.sample import make_label_batch
from src.model.dit import build_model
from src.train.ema import create_ema_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ensure_dir, load_config
from src.utils.logging import get_console, setup_logger


STEP_DIR_PATTERN = re.compile(r"^step_(\d+)$")


class GateCollector:
    def __init__(self) -> None:
        self.low_gates: list[torch.Tensor] = []
        self.high_gates: list[torch.Tensor] = []

    def record_block(
        self,
        attn_residual: torch.Tensor,
        freq_gate_low_logit: torch.Tensor,
        freq_gate_high_logit: torch.Tensor,
        freq_gate_low: torch.Tensor,
        freq_gate_high: torch.Tensor,
        mlp_residual_pre_freq_gate: torch.Tensor,
        mlp_residual_low_pre_gate: torch.Tensor,
        mlp_residual_high_pre_gate: torch.Tensor,
        mlp_residual_low_correction: torch.Tensor,
        mlp_residual_high_correction: torch.Tensor,
        mlp_residual: torch.Tensor,
        block_output_tokens: torch.Tensor,
    ) -> None:
        del (
            attn_residual,
            freq_gate_low_logit,
            freq_gate_high_logit,
            mlp_residual_pre_freq_gate,
            mlp_residual_low_pre_gate,
            mlp_residual_high_pre_gate,
            mlp_residual_low_correction,
            mlp_residual_high_correction,
            mlp_residual,
            block_output_tokens,
        )
        self.low_gates.append(freq_gate_low.detach().float().mean().cpu())
        self.high_gates.append(freq_gate_high.detach().float().mean().cpu())

    def set_step_output_tokens(self, step_output_tokens: torch.Tensor) -> None:
        del step_output_tokens

    def flush(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.low_gates or not self.high_gates:
            raise ValueError("No gate tensors were recorded for the current sample step.")
        low_gates = torch.stack(self.low_gates, dim=0)
        high_gates = torch.stack(self.high_gates, dim=0)
        self.low_gates.clear()
        self.high_gates.clear()
        return low_gates, high_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize frequency gate evolution across training checkpoints.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory containing checkpoints/.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path. Defaults to <run-dir>/config_resolved.yaml.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for cache and PNGs. Defaults to <run-dir>/gate_evolution.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Gate cache path. Defaults to <output>/gate_evolution.pt.",
    )
    parser.add_argument("--plot-only", action="store_true", help="Only redraw PNGs from an existing cache.")
    parser.add_argument("--refresh-cache", action="store_true", help="Overwrite an existing cache by rerunning inference.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count for gate collection.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override Euler sample steps for gate collection.")
    parser.add_argument("--label", type=int, default=None, help="Optional class label to use for every sample.")
    parser.add_argument("--seed", type=int, default=42, help="Seed reused for every checkpoint's initial noise.")
    parser.add_argument("--t-min", type=float, default=None, help="Minimum sampling t to include when plotting.")
    parser.add_argument("--t-max", type=float, default=None, help="Maximum sampling t to include when plotting.")
    parser.add_argument("--step-min", type=int, default=None, help="Minimum training step to include when plotting.")
    parser.add_argument("--step-max", type=int, default=None, help="Maximum training step to include when plotting.")
    parser.add_argument("--max-x-ticks", type=int, default=8, help="Maximum number of training-step labels to show.")
    parser.add_argument("--max-y-ticks", type=int, default=8, help="Maximum number of sampling-t labels to show.")
    return parser.parse_args()


def resolve_step(checkpoint_path: Path) -> int:
    match = STEP_DIR_PATTERN.match(checkpoint_path.parent.name)
    if match is None:
        raise ValueError(f"Checkpoint parent directory must be named step_XXXXXXX: {checkpoint_path}")
    return int(match.group(1))


def list_checkpoints(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    checkpoint_paths = sorted(
        checkpoint_dir.glob("step_*/checkpoint.pt"),
        key=resolve_step,
    )
    if not checkpoint_paths:
        raise ValueError(f"No checkpoint.pt files found under: {checkpoint_dir}")
    return checkpoint_paths


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_checkpoint_gates(
    model: torch.nn.Module,
    num_samples: int,
    image_size: int,
    in_channels: int,
    labels: torch.Tensor,
    num_steps: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    seed_all(seed)
    x = torch.randn(num_samples, in_channels, image_size, image_size, device=device, dtype=torch.float32)
    step_size = 1.0 / num_steps
    low_steps: list[torch.Tensor] = []
    high_steps: list[torch.Tensor] = []
    timesteps: list[float] = []

    with torch.no_grad():
        for step in range(num_steps):
            timestep_value = step / num_steps
            timesteps.append(timestep_value)
            t = torch.full((num_samples,), timestep_value, device=device, dtype=torch.float32)
            collector = GateCollector()
            velocity = model(x, t, labels, debug_collector=collector)
            low_gates, high_gates = collector.flush()
            low_steps.append(low_gates)
            high_steps.append(high_gates)
            x = x + step_size * velocity

    return torch.stack(low_steps, dim=0), torch.stack(high_steps, dim=0), timesteps


def collect_gate_cache(
    run_dir: Path,
    config: dict[str, Any],
    checkpoint_paths: list[Path],
    num_samples_override: int | None,
    num_steps_override: int | None,
    label: int | None,
    seed: int,
) -> dict[str, Any]:
    model_config = config["model"]
    if not bool(model_config["freq_residual_gating"]["enabled"]):
        raise ValueError("model.freq_residual_gating.enabled must be true to visualize gate evolution.")

    model = build_model(model_config)
    ema_model = create_ema_model(model)
    ema_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema_model.to(device)

    sample_config = config["sample"]
    num_samples = num_samples_override or int(sample_config["num_samples"])
    num_steps = num_steps_override or int(sample_config["num_steps"])
    if label is None:
        labels = make_label_batch(num_samples, int(model_config["num_classes"]), device)
    else:
        labels = torch.full((num_samples,), label, device=device, dtype=torch.long)

    logger = setup_logger("spectraldit.gate_evolution")
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=get_console(),
    )
    train_steps: list[int] = []
    low_by_checkpoint: list[torch.Tensor] = []
    high_by_checkpoint: list[torch.Tensor] = []
    timesteps: list[float] | None = None

    with progress:
        task_id = progress.add_task("collecting gates", total=len(checkpoint_paths))
        for checkpoint_path in checkpoint_paths:
            checkpoint = load_checkpoint(checkpoint_path, model, ema_model=ema_model)
            ema_model.eval()
            checkpoint_step = int(checkpoint.get("step", resolve_step(checkpoint_path)))
            low_gates, high_gates, current_timesteps = collect_checkpoint_gates(
                model=ema_model,
                num_samples=num_samples,
                image_size=int(model_config["image_size"]),
                in_channels=int(model_config["in_channels"]),
                labels=labels,
                num_steps=num_steps,
                device=device,
                seed=seed,
            )
            if timesteps is None:
                timesteps = current_timesteps
            elif timesteps != current_timesteps:
                raise ValueError("Collected timestep grids differ across checkpoints.")
            train_steps.append(checkpoint_step)
            low_by_checkpoint.append(low_gates)
            high_by_checkpoint.append(high_gates)
            progress.update(task_id, advance=1)

    if timesteps is None:
        raise ValueError("No gate values were collected.")

    logger.info("Collected gate evolution from %d checkpoints under %s", len(checkpoint_paths), run_dir)
    return {
        "train_steps": train_steps,
        "timesteps": timesteps,
        "low_gates": torch.stack(low_by_checkpoint, dim=0),
        "high_gates": torch.stack(high_by_checkpoint, dim=0),
        "metadata": {
            "run_dir": str(run_dir),
            "num_samples": num_samples,
            "num_steps": num_steps,
            "label": label,
            "seed": seed,
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "freq_residual_gating": dict(model_config["freq_residual_gating"]),
        },
    }


def sparse_tick_labels(labels: list[str], max_ticks: int, argument_name: str) -> list[str]:
    if max_ticks <= 0:
        raise ValueError(f"{argument_name} must be positive.")
    if len(labels) <= max_ticks:
        return labels
    if max_ticks == 1:
        return [label if index == len(labels) - 1 else "" for index, label in enumerate(labels)]
    stride = max(1, (len(labels) - 1) // (max_ticks - 1))
    visible_indices = set(range(0, len(labels), stride))
    visible_indices.add(len(labels) - 1)
    return [label if index in visible_indices else "" for index, label in enumerate(labels)]


def filter_t_range(
    timesteps: list[float],
    low_gates: torch.Tensor,
    high_gates: torch.Tensor,
    t_min: float | None,
    t_max: float | None,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    if t_min is not None and t_max is not None and t_min > t_max:
        raise ValueError("--t-min must be less than or equal to --t-max.")
    selected_indices = [
        index
        for index, timestep in enumerate(timesteps)
        if (t_min is None or timestep >= t_min) and (t_max is None or timestep <= t_max)
    ]
    if not selected_indices:
        raise ValueError("No cached sampling timesteps fall within the requested t range.")
    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected_timesteps = [timesteps[index] for index in selected_indices]
    return selected_timesteps, low_gates.index_select(1, index_tensor), high_gates.index_select(1, index_tensor)


def filter_train_step_range(
    train_steps: list[int],
    low_gates: torch.Tensor,
    high_gates: torch.Tensor,
    step_min: int | None,
    step_max: int | None,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    if step_min is not None and step_max is not None and step_min > step_max:
        raise ValueError("--step-min must be less than or equal to --step-max.")
    selected_indices = [
        index
        for index, train_step in enumerate(train_steps)
        if (step_min is None or train_step >= step_min) and (step_max is None or train_step <= step_max)
    ]
    if not selected_indices:
        raise ValueError("No cached training steps fall within the requested step range.")
    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    selected_train_steps = [train_steps[index] for index in selected_indices]
    return selected_train_steps, low_gates.index_select(0, index_tensor), high_gates.index_select(0, index_tensor)


def draw_gate_heatmap(
    axis: plt.Axes,
    values: torch.Tensor,
    cmap: Any,
    norm: mcolors.Normalize,
    x_labels: list[str] | None,
    y_labels: list[str] | None,
) -> Any:
    image = axis.imshow(
        values.numpy(),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    if x_labels is None:
        axis.set_xticks([])
    else:
        axis.set_xticks(range(len(x_labels)))
        axis.set_xticklabels(x_labels)
    if y_labels is None:
        axis.set_yticks([])
    else:
        axis.set_yticks(range(len(y_labels)))
        axis.set_yticklabels(y_labels)
    axis.tick_params(axis="x", labelrotation=90)
    axis.tick_params(axis="y", labelrotation=0)
    return image


def plot_gate_heatmaps(
    cache: dict[str, Any],
    output_dir: Path,
    t_min: float | None,
    t_max: float | None,
    step_min: int | None,
    step_max: int | None,
    max_x_ticks: int,
    max_y_ticks: int,
) -> None:
    train_steps = [int(step) for step in cache["train_steps"]]
    timesteps = [float(timestep) for timestep in cache["timesteps"]]
    low_gates = cache["low_gates"].float()
    high_gates = cache["high_gates"].float()
    if low_gates.shape != high_gates.shape:
        raise ValueError("low_gates and high_gates must have the same shape.")
    if low_gates.ndim != 3:
        raise ValueError("Expected cached gate tensors with shape [num_train_steps, num_sample_steps, num_blocks].")
    train_steps, low_gates, high_gates = filter_train_step_range(
        train_steps,
        low_gates,
        high_gates,
        step_min,
        step_max,
    )
    timesteps, low_gates, high_gates = filter_t_range(timesteps, low_gates, high_gates, t_min, t_max)

    plot_dir = ensure_dir(output_dir / "plots")
    sns.set_theme(style="white")
    max_abs = float(torch.stack([low_gates.abs().amax(), high_gates.abs().amax()]).amax().item())
    if max_abs == 0.0:
        max_abs = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    num_blocks = int(low_gates.shape[2])
    x_labels = sparse_tick_labels([str(step) for step in train_steps], max_x_ticks, "--max-x-ticks")
    y_labels = sparse_tick_labels([f"{timestep:.2f}" for timestep in timesteps], max_y_ticks, "--max-y-ticks")
    block_cmap = sns.color_palette("vlag", as_cmap=True)
    overview_cmap = "coolwarm"
    for block_index in range(num_blocks):
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
        panels = (
            ("Low Gate", axes[0], low_gates[:, :, block_index].T),
            ("High Gate", axes[1], high_gates[:, :, block_index].T),
        )
        for title, axis, values in panels:
            image = draw_gate_heatmap(
                axis=axis,
                values=values,
                cmap=block_cmap,
                norm=norm,
                x_labels=x_labels,
                y_labels=y_labels,
            )
            figure.colorbar(image, ax=axis, label="gate value")
            axis.set_title(title)
            axis.set_xlabel("training step")
            axis.set_ylabel("sampling t")
        figure.suptitle(f"Block {block_index} Gate Evolution")
        figure.savefig(plot_dir / f"block_{block_index:03d}_gate_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    overview_columns = 4
    overview_low_rows = (num_blocks + overview_columns - 1) // overview_columns
    overview_rows = 2 * overview_low_rows
    overview_figure, overview_axes = plt.subplots(
        overview_rows,
        4,
        figsize=(12, max(2.6 * overview_rows, 7.5)),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    overview_figure.subplots_adjust(left=0.08, right=0.9, bottom=0.14, top=0.9, wspace=0.08, hspace=0.35)
    colorbar_axis = overview_figure.add_axes((0.94, 0.22, 0.012, 0.58))
    overview_axes = overview_axes.reshape(overview_rows, overview_columns)
    overview_image = None
    for block_index in range(num_blocks):
        block_row = block_index // overview_columns
        block_column = block_index % overview_columns
        panels = (
            ("Low", overview_axes[block_row, block_column], low_gates[:, :, block_index].T),
            (
                "High",
                overview_axes[overview_low_rows + block_row, block_column],
                high_gates[:, :, block_index].T,
            ),
        )
        for row_name, axis, values in panels:
            is_high_row = row_name == "High"
            show_x_labels = is_high_row and block_row == overview_low_rows - 1
            overview_image = draw_gate_heatmap(
                axis=axis,
                values=values,
                cmap=overview_cmap,
                norm=norm,
                x_labels=x_labels if show_x_labels else None,
                y_labels=y_labels if block_column == 0 else None,
            )
            axis.set_title(f"{row_name} Block {block_index}")
            axis.set_xlabel("training step" if show_x_labels else "")
            axis.set_ylabel("sampling t" if block_column == 0 else "")
    for empty_block_index in range(num_blocks, overview_low_rows * overview_columns):
        empty_row = empty_block_index // overview_columns
        empty_column = empty_block_index % overview_columns
        overview_axes[empty_row, empty_column].axis("off")
        overview_axes[overview_low_rows + empty_row, empty_column].axis("off")
    if overview_image is None:
        raise ValueError("No overview heatmaps were rendered.")
    overview_figure.colorbar(overview_image, cax=colorbar_axis, label="gate value")
    overview_figure.savefig(plot_dir / "all_blocks_gate_heatmap.png", dpi=180, bbox_inches="tight")
    overview_figure.savefig(plot_dir / "all_blocks_gate_heatmap.pdf", dpi=180, bbox_inches="tight")

    plt.close(overview_figure)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = ensure_dir(Path(args.output) if args.output is not None else run_dir / "gate_evolution")
    cache_path = Path(args.cache) if args.cache is not None else output_dir / "gate_evolution.pt"
    logger = setup_logger("spectraldit.gate_evolution")

    if args.plot_only:
        if not cache_path.exists():
            raise ValueError(f"Cache does not exist for --plot-only: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        plot_gate_heatmaps(
            cache,
            output_dir,
            t_min=args.t_min,
            t_max=args.t_max,
            step_min=args.step_min,
            step_max=args.step_max,
            max_x_ticks=int(args.max_x_ticks),
            max_y_ticks=int(args.max_y_ticks),
        )
        logger.info("Saved gate heatmaps to %s", output_dir / "plots")
        return

    if cache_path.exists() and not args.refresh_cache:
        raise ValueError(f"Cache already exists: {cache_path}. Use --plot-only or --refresh-cache.")

    config_path = Path(args.config) if args.config is not None else run_dir / "config_resolved.yaml"
    config = load_config(config_path)
    checkpoint_paths = list_checkpoints(run_dir)
    cache = collect_gate_cache(
        run_dir=run_dir,
        config=config,
        checkpoint_paths=checkpoint_paths,
        num_samples_override=args.num_samples,
        num_steps_override=args.num_steps,
        label=args.label,
        seed=int(args.seed),
    )
    ensure_dir(cache_path.parent)
    torch.save(cache, cache_path)
    logger.info("Saved gate cache to %s", cache_path)
    plot_gate_heatmaps(
        cache,
        output_dir,
        t_min=args.t_min,
        t_max=args.t_max,
        step_min=args.step_min,
        step_max=args.step_max,
        max_x_ticks=int(args.max_x_ticks),
        max_y_ticks=int(args.max_y_ticks),
    )
    logger.info("Saved gate heatmaps to %s", output_dir / "plots")


if __name__ == "__main__":
    main()
