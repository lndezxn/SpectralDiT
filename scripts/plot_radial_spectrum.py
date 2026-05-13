from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from rich.table import Table

from src.utils.config import ensure_dir
from src.utils.logging import get_console, setup_logger

Y_LIMIT_FREQUENCY_RANGE = (0.3, 1.0)
Y_LIMIT_PADDING_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot radial Fourier spectra from evaluate_dit JSON files.")
    parser.add_argument("json_files", nargs="+", type=str, help="Metric JSON files produced by scripts/evaluate_dit.py.")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for generated spectra. Must match the number of JSON files.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Plot raw spectrum values instead of log spectrum values.",
    )
    parser.add_argument(
        "--show-real-all",
        action="store_true",
        help="Plot the real spectrum from every JSON instead of only the first one.",
    )
    return parser.parse_args()


def load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "radial_spectrum" not in payload:
        raise ValueError(f"Missing radial_spectrum field. Regenerate metrics with the current evaluate script: {path}")
    return payload


def resolve_labels(json_files: list[str], labels: list[str] | None) -> list[str]:
    if labels is None:
        return [Path(path).stem for path in json_files]
    if len(labels) != len(json_files):
        raise ValueError(f"Expected {len(json_files)} labels, got {len(labels)}.")
    return labels


def collect_y_values_in_frequency_range(x_values: list[float], y_values: list[float]) -> list[float]:
    lower, upper = Y_LIMIT_FREQUENCY_RANGE
    return [y for x, y in zip(x_values, y_values, strict=True) if lower <= x <= upper]


def apply_frequency_range_ylim(axis: plt.Axes, y_values: list[float]) -> None:
    if not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    y_span = y_max - y_min
    padding = Y_LIMIT_PADDING_FRACTION * y_span if y_span > 0 else max(abs(y_min), 1.0) * Y_LIMIT_PADDING_FRACTION
    axis.set_ylim(y_min - padding, y_max + padding)


def plot_spectra(payloads: list[dict[str, Any]], labels: list[str], output_path: str | Path, linear: bool, show_real_all: bool) -> None:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    generated_key = "generated" if linear else "log_generated"
    real_key = "real" if linear else "log_real"
    y_label = "Radial Fourier spectrum" if linear else "Log radial Fourier spectrum"
    palette = sns.color_palette("husl", len(payloads))
    ylim_values: list[float] = []

    first_spectrum = payloads[0]["radial_spectrum"]
    first_x_values = first_spectrum.get("normalized_radius", first_spectrum["radius_bins"])
    ylim_values.extend(collect_y_values_in_frequency_range(first_x_values, first_spectrum[real_key]))
    sns.lineplot(
        x=first_x_values,
        y=first_spectrum[real_key],
        ax=axis,
        color="grey",
        linewidth=1.4,
        linestyle="--",
        label="real",
    )

    for index, (payload, label) in enumerate(zip(payloads, labels, strict=True)):
        spectrum = payload["radial_spectrum"]
        x_values = spectrum.get("normalized_radius", spectrum["radius_bins"])
        if show_real_all and index > 0:
            ylim_values.extend(collect_y_values_in_frequency_range(x_values, spectrum[real_key]))
            sns.lineplot(
                x=x_values,
                y=spectrum[real_key],
                ax=axis,
                color=palette[index],
                linewidth=1.5,
                linestyle=":",
                label=f"{label} real",
            )
        ylim_values.extend(collect_y_values_in_frequency_range(x_values, spectrum[generated_key]))
        sns.lineplot(
            x=x_values,
            y=spectrum[generated_key],
            ax=axis,
            color=palette[index],
            linewidth=2.0,
            label=f"{label}",
        )

    axis.set_xlabel("Normalized radial frequency")
    axis.set_ylabel(y_label)
    axis.set_xlim(*Y_LIMIT_FREQUENCY_RANGE)
    apply_frequency_range_ylim(axis, ylim_values)
    axis.legend(frameon=False)
    output = Path(output_path)
    ensure_dir(output.parent)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_input_table(json_files: list[str], labels: list[str]) -> Table:
    table = Table(title="Radial Spectrum Inputs")
    table.add_column("Label", style="cyan")
    table.add_column("JSON")
    for path, label in zip(json_files, labels, strict=True):
        table.add_row(label, path)
    return table


def main() -> None:
    args = parse_args()
    logger = setup_logger("spectraldit.plot_radial_spectrum")
    console = get_console()
    labels = resolve_labels(args.json_files, args.labels)
    payloads = [load_payload(path) for path in args.json_files]
    plot_spectra(payloads, labels, args.output, args.linear, args.show_real_all)
    console.print(render_input_table(args.json_files, labels))
    logger.info("Saved radial spectrum plot to %s", args.output)


if __name__ == "__main__":
    main()
