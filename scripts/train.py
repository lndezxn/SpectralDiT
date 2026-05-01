from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.train.trainer import run_training
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pixel-space flow matching DiT on CIFAR-10.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["_meta"] = {
        "config_path": args.config,
        "command": " ".join(sys.argv),
    }
    set_seed(int(config["train"]["seed"]))
    run_training(config)


if __name__ == "__main__":
    main()
