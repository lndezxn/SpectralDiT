from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.metric_prc import prc_features_to_metric
from torch_fidelity.utils import create_feature_extractor

from src.data.cifar10 import build_cifar10_dataset
from src.eval.metrics import to_uint8_images
from src.eval.paper_metrics import PaperMetricState, compare_paper_metrics, finalize_state, initialize_state, update_state
from src.eval.sample import sample_euler
from src.model.dit import build_model
from src.train.ema import create_ema_model
from src.utils.checkpoint import load_checkpoint, resolve_run_directory
from src.utils.config import ensure_dir, load_config
from src.utils.logging import get_console, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pixel-space DiT with paper metrics.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a checkpoint.pt file.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path. Defaults to config_resolved.yaml in the checkpoint case directory.",
    )
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of generated samples.")
    parser.add_argument("--num-real", type=int, default=10000, help="Number of real reference images.")
    parser.add_argument("--real-split", type=str, choices=("train", "test"), default="test", help="Real CIFAR-10 split.")
    parser.add_argument("--label", type=int, default=None, help="Optional class label to use for every generated sample.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--hf-threshold", type=float, default=0.5, help="High-frequency threshold as a radial fraction.")
    parser.add_argument("--pr-neighborhood", type=int, default=3, help="Neighborhood size for precision/recall.")
    parser.add_argument("--pr-batch-size", type=int, default=10000, help="Distance batch size for precision/recall.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def resolve_config_path(checkpoint_path: str | Path, config_path: str | None) -> Path:
    if config_path is not None:
        return Path(config_path)
    run_directory = resolve_run_directory(checkpoint_path)
    resolved_config_path = run_directory / "config_resolved.yaml"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"Could not find resolved config at {resolved_config_path}. "
            "Pass --config explicitly if this checkpoint does not use the standard case directory layout."
        )
    return resolved_config_path


def make_label_batch(offset: int, num_samples: int, num_classes: int, device: torch.device, label: int | None) -> torch.Tensor:
    if label is not None:
        return torch.full((num_samples,), label, device=device, dtype=torch.long)
    labels = torch.arange(offset, offset + num_samples, device=device, dtype=torch.long)
    return labels.remainder(num_classes)


def build_real_loader(config: dict[str, Any], split: str, num_real: int, batch_size: int) -> DataLoader[tuple[torch.Tensor, int]]:
    dataset = build_cifar10_dataset(root=config["data"]["root"], train=split == "train")
    if num_real > len(dataset):
        raise ValueError(f"num_real={num_real} exceeds CIFAR-10 {split} split size {len(dataset)}.")
    subset = Subset(dataset, range(num_real))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        drop_last=False,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )


def render_metrics_table(metrics: dict[str, float]) -> Table:
    table = Table(title="DiT Paper Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for name, value in metrics.items():
        table.add_row(name, f"{value:.6f}")
    return table


def build_radial_spectrum_payload(
    real_state: PaperMetricState,
    generated_state: PaperMetricState,
    eps: float = 1e-12,
) -> dict[str, list[float]]:
    real_spectrum = finalize_state(real_state)["radial_spectrum"].detach().cpu()
    generated_spectrum = finalize_state(generated_state)["radial_spectrum"].detach().cpu()
    radius_bins = torch.arange(real_spectrum.shape[0], dtype=torch.float32)
    normalized_radius = radius_bins / radius_bins.max().clamp_min(1.0)
    return {
        "radius_bins": [float(value) for value in radius_bins.tolist()],
        "normalized_radius": [float(value) for value in normalized_radius.tolist()],
        "real": [float(value) for value in real_spectrum.tolist()],
        "generated": [float(value) for value in generated_spectrum.tolist()],
        "log_real": [float(value) for value in torch.log(real_spectrum.clamp_min(eps)).tolist()],
        "log_generated": [float(value) for value in torch.log(generated_spectrum.clamp_min(eps)).tolist()],
    }


def save_metrics(
    metrics: dict[str, float],
    radial_spectrum: dict[str, list[float]],
    args: argparse.Namespace,
    config_path: Path,
    path: str | Path,
) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    payload = {
        "config": str(config_path),
        "checkpoint": args.ckpt,
        "num_samples": args.num_samples,
        "num_real": args.num_real,
        "real_split": args.real_split,
        "hf_threshold": args.hf_threshold,
        "pr_neighborhood": args.pr_neighborhood,
        "pr_batch_size": args.pr_batch_size,
        "metrics": metrics,
        "radial_spectrum": radial_spectrum,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logger = setup_logger("spectraldit.evaluate")
    console = get_console()
    config_path = resolve_config_path(args.ckpt, args.config)
    logger.info("Using config: %s", config_path)
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or int(config["eval"]["batch_size"])
    image_size = int(config["model"]["image_size"])
    in_channels = int(config["model"]["in_channels"])
    num_classes = int(config["model"]["num_classes"])

    model = build_model(config["model"])
    ema_model = create_ema_model(model)
    load_checkpoint(args.ckpt, model, ema_model=ema_model)
    ema_model.to(device)
    ema_model.eval()

    real_loader = build_real_loader(config, args.real_split, args.num_real, batch_size)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    pr_feature_extractor = create_feature_extractor(
        "inception-v3-compat",
        ["2048", "logits_unbiased"],
        cuda=device.type == "cuda",
        verbose=False,
    )
    real_state = initialize_state(image_size, device)
    generated_state = initialize_state(image_size, device)
    real_pr_features: list[torch.Tensor] = []
    generated_pr_features: list[torch.Tensor] = []
    generated_isc_logits: list[torch.Tensor] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        real_task = progress.add_task("Processing real images", total=args.num_real)
        for images, _ in real_loader:
            images = images.to(device, non_blocking=True)
            real_uint8 = to_uint8_images(images)
            fid.update(real_uint8, real=True)
            real_pr_features.append(pr_feature_extractor(real_uint8)[0].detach().cpu())
            update_state(real_state, images, args.hf_threshold)
            progress.advance(real_task, int(images.shape[0]))

        generated_task = progress.add_task("Generating images", total=args.num_samples)
        for offset in range(0, args.num_samples, batch_size):
            current_batch_size = min(batch_size, args.num_samples - offset)
            labels = make_label_batch(offset, current_batch_size, num_classes, device, args.label)
            samples = sample_euler(
                model=ema_model,
                num_samples=current_batch_size,
                image_size=image_size,
                in_channels=in_channels,
                labels=labels,
                num_steps=int(config["sample"]["num_steps"]),
                device=device,
                dtype=torch.float32,
            )
            generated_uint8 = to_uint8_images(samples)
            fid.update(generated_uint8, real=False)
            generated_features, generated_logits = pr_feature_extractor(generated_uint8)
            generated_pr_features.append(generated_features.detach().cpu())
            generated_isc_logits.append(generated_logits.detach().cpu())
            update_state(generated_state, samples, args.hf_threshold)
            progress.advance(generated_task, current_batch_size)

    metrics = {"fid": float(fid.compute().item())}
    metrics.update(
        isc_features_to_metric(
            torch.cat(generated_isc_logits, dim=0),
            splits=min(10, args.num_samples),
            shuffle=True,
            rng_seed=2020,
        )
    )
    metrics.update(
        prc_features_to_metric(
            torch.cat(generated_pr_features, dim=0),
            torch.cat(real_pr_features, dim=0),
            prc_neighborhood=args.pr_neighborhood,
            prc_batch_size=args.pr_batch_size,
            save_cpu_ram=True,
            verbose=False,
        )
    )
    metrics.update(compare_paper_metrics(real_state, generated_state))
    console.print(render_metrics_table(metrics))
    if args.output is not None:
        radial_spectrum = build_radial_spectrum_payload(real_state, generated_state)
        save_metrics(metrics, radial_spectrum, args, config_path, args.output)
        logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
