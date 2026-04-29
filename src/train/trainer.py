from __future__ import annotations
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision

from src.data.cifar10 import build_dataloader
from src.eval.metrics import GenerativeMetrics
from src.eval.sample import make_label_batch, sample_euler
from src.model.dit import build_model
from src.train.ema import create_ema_model, update_ema
from src.train.fm import flow_matching_loss, sample_flow_matching_inputs
from src.utils.checkpoint import (
    TrainingProgress,
    normalize_resume_config,
    resolve_run_directory,
    resolve_resume_directory,
    save_checkpoint,
)
from src.utils.config import ensure_dir, load_config, save_config
from src.utils.logging import get_console, setup_logger


def _resolve_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_lr_lambda(warmup_steps: int, max_steps: int, min_lr_ratio: float) -> callable:
    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if max_steps <= warmup_steps:
            return min_lr_ratio

        progress = (current_step - warmup_steps) / float(max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = setup_logger("spectraldit.train")
        train_config = config["train"]
        self.resume_path = train_config.get("resume_from")
        self.resume_state_dir = resolve_resume_directory(self.resume_path) if self.resume_path else None
        self.normalized_config = normalize_resume_config(config)
        self.accelerator = Accelerator(
            mixed_precision=str(train_config["mixed_precision"]),
            gradient_accumulation_steps=int(train_config["grad_accum_steps"]),
            dataloader_config=DataLoaderConfiguration(
                use_seedable_sampler=True,
                data_seed=int(train_config["seed"]),
            ),
        )
        self.output_dir = self._build_output_dir(str(train_config["output_dir"]), self.resume_state_dir)
        self.config["train"]["output_dir"] = str(self.output_dir)
        self.logger = setup_logger("spectraldit.train", self.output_dir / "train.log")
        self.sample_dir = ensure_dir(self.output_dir / "samples")
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard")) if self.accelerator.is_main_process else None
        self.device = self.accelerator.device

        data_config = config["data"]
        self.train_loader = build_dataloader(
            root=data_config["root"],
            train=True,
            batch_size=int(data_config["batch_size"]),
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
        )
        self.eval_loader = build_dataloader(
            root=data_config["root"],
            train=False,
            batch_size=int(config["eval"]["batch_size"]),
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
        )
        self.metric_loader = build_dataloader(
            root=data_config["root"],
            train=False,
            batch_size=int(config["eval"]["batch_size"]),
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
        )

        self.model = build_model(config["model"])
        self.ema_model = create_ema_model(self.model)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            betas=(float(train_config["beta1"]), float(train_config["beta2"])),
        )
        min_lr = float(train_config["min_learning_rate"])
        base_lr = float(train_config["learning_rate"])
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=_build_lr_lambda(
                warmup_steps=int(train_config["warmup_steps"]),
                max_steps=int(train_config["max_steps"]),
                min_lr_ratio=min_lr / base_lr,
            ),
        )

        self.model, self.optimizer, self.train_loader, self.eval_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.scheduler,
        )
        self.ema_model.to(self.device)
        self.metrics = GenerativeMetrics(device=self.device)
        self.global_step = 0
        self.train_epoch = 0
        self.batches_seen_in_epoch = 0
        self.ema_decay = float(train_config["ema_decay"])
        self.train_batches_per_epoch = len(self.train_loader)
        self.progress_state = TrainingProgress(
            global_step=0,
            train_epoch=0,
            batches_seen_in_epoch=0,
            output_dir=str(self.output_dir),
            config=self.normalized_config,
        )
        self.accelerator.register_for_checkpointing(self.ema_model, self.progress_state)

        if self.resume_state_dir is not None:
            self.accelerator.load_state(str(self.resume_state_dir))
            self._restore_progress_state()
            self._validate_resume_config()

        if self.accelerator.is_main_process:
            self._save_run_artifacts()
            self.logger.info("Parameter count: %.2fM", _count_parameters(self.model) / 1e6)
            self.logger.info(
                "Training setup | output_dir=%s | ema_decay=%.4f | eval_uses_ema=true | resumed=%s",
                self.output_dir,
                self.ema_decay,
                self.resume_state_dir is not None,
            )
        self.accelerator.wait_for_everyone()

    def _build_output_dir(self, base_output_dir: str, resume_state_dir: Path | None) -> Path:
        if resume_state_dir is not None:
            return ensure_dir(resolve_run_directory(resume_state_dir))
        timestamp = _timestamp_string() if self.accelerator.is_main_process else ""
        if self.accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
            shared_timestamp = [timestamp]
            dist.broadcast_object_list(shared_timestamp, src=0)
            timestamp = str(shared_timestamp[0])
        return ensure_dir(Path(base_output_dir) / timestamp)

    def _restore_progress_state(self) -> None:
        self.global_step = int(self.progress_state.global_step)
        self.train_epoch = int(self.progress_state.train_epoch)
        self.batches_seen_in_epoch = int(self.progress_state.batches_seen_in_epoch)

    def _validate_resume_config(self) -> None:
        checkpoint_config = normalize_resume_config(self.progress_state.config)
        if checkpoint_config != self.normalized_config:
            raise ValueError("Current config does not match the checkpoint config after resume normalization.")

    def _current_train_sampler(self) -> Any:
        return self.train_loader.batch_sampler.sampler

    def _build_train_iterator(self) -> Any:
        sampler = self._current_train_sampler()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.train_epoch)
        return iter(self.train_loader)

    def _fast_forward_train_iterator(self, data_iterator: Any) -> Any:
        for _ in range(self.batches_seen_in_epoch):
            next(data_iterator)
        return data_iterator

    def _update_progress_state(self) -> None:
        self.progress_state.global_step = self.global_step
        self.progress_state.train_epoch = self.train_epoch
        self.progress_state.batches_seen_in_epoch = self.batches_seen_in_epoch
        self.progress_state.output_dir = str(self.output_dir)
        self.progress_state.config = self.normalized_config

    def _save_run_artifacts(self) -> None:
        meta_config = self.config.get("_meta", {})
        raw_config = {
            "config_path": meta_config.get("config_path"),
            "command": meta_config.get("command"),
        }
        save_config(raw_config, self.output_dir / "launch.yaml")
        config_path = meta_config.get("config_path")
        if config_path:
            save_config(load_config(str(config_path)), self.output_dir / "config_original.yaml")
        save_config(self.config, self.output_dir / "config_resolved.yaml")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_summary = {
            "model_repr": str(unwrapped_model),
            "parameter_count": _count_parameters(unwrapped_model),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in unwrapped_model.parameters() if parameter.requires_grad
            ),
        }
        with (self.output_dir / "model.txt").open("w", encoding="utf-8") as handle:
            handle.write(model_summary["model_repr"])
            handle.write("\n\n")
            handle.write(f"parameter_count: {model_summary['parameter_count']}\n")
            handle.write(f"trainable_parameter_count: {model_summary['trainable_parameter_count']}\n")

        run_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "accelerate_num_processes": self.accelerator.num_processes,
            "accelerate_mixed_precision": str(self.accelerator.mixed_precision),
            "accelerate_distributed_type": str(self.accelerator.distributed_type),
            "device": str(self.device),
            "seed": int(self.config["train"]["seed"]),
            "output_dir": str(self.output_dir),
            "resumed_from": str(self.resume_state_dir) if self.resume_state_dir is not None else None,
        }
        if torch.cuda.is_available():
            run_info["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        save_config(run_info, self.output_dir / "run_info.yaml")

    def train(self) -> None:
        train_config = self.config["train"]
        max_steps = int(train_config["max_steps"])
        log_every = int(train_config["log_every"])
        eval_every = int(train_config["eval_every"])
        save_every = int(train_config["save_every"])
        grad_clip = float(train_config["grad_clip"])

        console = get_console()
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=not self.accelerator.is_main_process,
        )

        data_iterator = self._build_train_iterator()
        if self.batches_seen_in_epoch > 0:
            data_iterator = self._fast_forward_train_iterator(data_iterator)
        with progress:
            task_id = progress.add_task("training", total=max_steps, completed=self.global_step, loss=float("nan"))
            while self.global_step < max_steps:
                try:
                    images, labels = next(data_iterator)
                except StopIteration:
                    data_iterator = self._build_train_iterator()
                    images, labels = next(data_iterator)

                self.batches_seen_in_epoch += 1
                if self.batches_seen_in_epoch == self.train_batches_per_epoch:
                    self.train_epoch += 1
                    self.batches_seen_in_epoch = 0

                with self.accelerator.accumulate(self.model):
                    x_t, timesteps, target_velocity = sample_flow_matching_inputs(images)
                    prediction = self.model(x_t, timesteps, labels)
                    loss = flow_matching_loss(prediction, target_velocity)
                    self.accelerator.backward(loss)

                    grad_norm = None
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if not self.accelerator.sync_gradients:
                    continue

                update_ema(self.ema_model, self.accelerator.unwrap_model(self.model), self.ema_decay)
                self.global_step += 1
                self._update_progress_state()
                progress.update(task_id, advance=1, loss=float(loss.detach().item()))

                if self.writer is not None and self.global_step % log_every == 0:
                    self.writer.add_scalar("train/loss", float(loss.item()), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                    if grad_norm is not None:
                        self.writer.add_scalar("train/grad_norm", float(grad_norm.item()), self.global_step)
                    self.logger.info(
                        "Train step %d | loss %.6f | lr %.8f | grad_norm %s",
                        self.global_step,
                        float(loss.item()),
                        self.optimizer.param_groups[0]["lr"],
                        f"{float(grad_norm.item()):.6f}" if grad_norm is not None else "n/a",
                    )

                if self.global_step % eval_every == 0:
                    self.evaluate()

                if self.global_step % save_every == 0:
                    self.accelerator.wait_for_everyone()
                    save_checkpoint(
                        accelerator=self.accelerator,
                        model=self.model,
                        ema_model=self.ema_model,
                        step=self.global_step,
                        config=self.config,
                        output_dir=self.checkpoint_dir,
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info("Saved checkpoint at step %d", self.global_step)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    @torch.no_grad()
    def evaluate(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.info("Running evaluation at step %d", self.global_step)

        self.accelerator.wait_for_everyone()
        self.model.eval()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            self.model.train()
            return

        eval_model = self.ema_model
        eval_model.eval()
        self.metrics.update_real(self.metric_loader)

        eval_config = self.config["eval"]
        model_config = self.config["model"]
        sample_config = self.config["sample"]
        total_samples = int(eval_config["fid_num_samples"])
        batch_size = int(eval_config["sample_batch_size"])
        dtype = _resolve_dtype(str(self.config["train"]["mixed_precision"]))
        preview_batch: torch.Tensor | None = None

        for batch_start in range(0, total_samples, batch_size):
            current_batch = min(batch_size, total_samples - batch_start)
            labels = make_label_batch(current_batch, int(model_config["num_classes"]), self.device)
            samples = sample_euler(
                model=eval_model,
                num_samples=current_batch,
                image_size=int(model_config["image_size"]),
                in_channels=int(model_config["in_channels"]),
                labels=labels,
                num_steps=int(sample_config["num_steps"]),
                device=self.device,
                dtype=dtype,
            )
            samples = samples.float()
            if preview_batch is None:
                preview_batch = samples[: min(64, samples.shape[0])].cpu()
            self.metrics.update_fake(samples)
            del samples

        results = self.metrics.compute()
        if self.writer is not None:
            for name, value in results.items():
                self.writer.add_scalar(f"eval/{name}", value, self.global_step)
            if preview_batch is not None:
                grid = make_grid(preview_batch, nrow=8, normalize=True, value_range=(-1, 1))
                self.writer.add_image("eval/samples", grid, self.global_step)
                image_path = Path(self.sample_dir) / f"step_{self.global_step:07d}.png"
                grid_uint8 = ((grid.clamp(0.0, 1.0) * 255).round().byte().permute(1, 2, 0).numpy())
                from PIL import Image

                Image.fromarray(grid_uint8).save(image_path)
        self.logger.info(
            "Eval step %d | FID %.4f | IS %.4f +/- %.4f",
            self.global_step,
            results["fid"],
            results["inception_score_mean"],
            results["inception_score_std"],
        )
        self.accelerator.wait_for_everyone()
        self.model.train()


def run_training(config: dict[str, Any]) -> None:
    Trainer(config).train()
