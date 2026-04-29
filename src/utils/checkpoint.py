from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator


@dataclass
class TrainingProgress:
    global_step: int = 0
    train_epoch: int = 0
    batches_seen_in_epoch: int = 0
    output_dir: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "train_epoch": self.train_epoch,
            "batches_seen_in_epoch": self.batches_seen_in_epoch,
            "output_dir": self.output_dir,
            "config": self.config,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.global_step = int(state_dict["global_step"])
        self.train_epoch = int(state_dict["train_epoch"])
        self.batches_seen_in_epoch = int(state_dict["batches_seen_in_epoch"])
        self.output_dir = str(state_dict["output_dir"])
        self.config = dict(state_dict["config"])


def normalize_resume_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: value
        for key, value in config.items()
        if key != "_meta"
    }
    if "train" in normalized:
        normalized["train"] = dict(normalized["train"])
        normalized["train"].pop("resume_from", None)
        normalized["train"].pop("output_dir", None)
    return normalized


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    step: int,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    checkpoint_dir = Path(output_dir) / f"step_{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(output_dir=str(checkpoint_dir), safe_serialization=False)

    sample_checkpoint_path = checkpoint_dir / "checkpoint.pt"
    sample_state = {
        "model": accelerator.get_state_dict(model),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "step": step,
        "config": config,
    }
    accelerator.save(sample_state, sample_checkpoint_path)
    return checkpoint_dir


def resolve_resume_directory(path: str | Path) -> Path:
    resume_path = Path(path)
    if resume_path.is_dir():
        return resume_path
    if resume_path.name != "checkpoint.pt":
        raise ValueError(f"Resume path must be a checkpoint directory or checkpoint.pt, got: {resume_path}")
    return resume_path.parent


def resolve_run_directory(path: str | Path) -> Path:
    checkpoint_dir = resolve_resume_directory(path)
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError(f"Could not infer run directory from checkpoint path: {path}")
    return checkpoint_dir.parent.parent


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if ema_model is not None and checkpoint.get("ema_model") is not None:
        ema_model.load_state_dict(checkpoint["ema_model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
