"""
Main trainer for Expera AI.

Provides:
- Single GPU training
- Mixed precision training (bf16/fp16)
- Gradient accumulation and clipping
- EMA
- Checkpointing
- Validation
- Multiple loggers
"""

import time
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, IterableDataset

from .loss import LanguageModelingLoss


@dataclass
class TrainerConfig:
    """Configuration for trainer."""
    # Required
    model: nn.Module = None
    train_dataloader: Any = None
    optimizer: torch.optim.Optimizer = None

    # Optional components
    scheduler: Any = None
    loss_fn: Optional[nn.Module] = None

    # Device
    device: str = "cuda"

    # Training params
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16

    # Checkpointing
    save_every: int = 1000
    save_dir: str = "checkpoints"
    keep_last_n: int = 3

    # Validation
    val_dataloader: Optional[Any] = None
    val_every: int = 5000
    val_batches: Optional[int] = None

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Logging
    log_every: int = 100
    metrics_logger: Optional[Any] = None

    # Extra metadata embedded verbatim into checkpoints (e.g. source YAML paths,
    # model configuration). See CHECKPOINT_FORMAT.
    checkpoint_meta: Optional[Dict[str, Any]] = None


CHECKPOINT_FORMAT = "exp-coder-v1"


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
    ):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    Main trainer class for Expera AI.

    Supports:
    - Mixed precision training (bf16/fp16)
    - Gradient accumulation
    - Gradient clipping
    - EMA
    - Checkpointing
    - Validation
    - Multiple loggers
    """

    def __init__(self, config: TrainerConfig):
        self.config = config

        # Set device
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        # Move model to device
        self.model = config.model.to(self.device)
        self.model.train()

        # Optimizer and scheduler
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler

        # Loss function
        self.loss_fn = config.loss_fn or LanguageModelingLoss()

        # AMP scaler (CUDA only; CPU runs fp32)
        self.scaler = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()

        # EMA
        self.ema = None
        if config.use_ema:
            self.ema = ExponentialMovingAverage(
                self.model,
                decay=config.ema_decay,
            )

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Metrics tracking
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []

        # Timing
        self.step_times: List[float] = []
        self.last_step_time = time.time()

        # Create save directory
        self.save_dir = Path(config.save_dir)
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        self.model.train()
        dataloader = self.config.train_dataloader

        # Get iterator
        if isinstance(dataloader, IterableDataset):
            iterator = iter(dataloader)
        else:
            iterator = iter(dataloader)

        # Track start time
        start_time = time.time()

        # Training loop
        while self.global_step < self.config.max_steps:
            try:
                # Get batch
                batch = next(iterator)
            except StopIteration:
                # Restart iterator
                iterator = iter(dataloader)
                batch = next(iterator)
                self.epoch += 1

            # Train step
            metrics = self.train_step(batch)

            # Log
            if self.global_step % self.config.log_every == 0:
                self._log_metrics(metrics, "train")

            # Validate
            if self.config.val_dataloader and self.global_step % self.config.val_every == 0:
                val_metrics = self.validate()
                self._log_metrics(val_metrics, "val")
                self.val_metrics.append(val_metrics)

            # Save checkpoint
            if self.global_step % self.config.save_every == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")

        # Final validation
        if self.config.val_dataloader:
            val_metrics = self.validate()
            self._log_metrics(val_metrics, "val")

        # Save final checkpoint
        self.save_checkpoint("final")

        elapsed_time = time.time() - start_time

        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "elapsed_time": elapsed_time,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]

        # Handle gradient accumulation
        accumulation_steps = self.config.gradient_accumulation_steps

        # Forward pass (mixed precision only on CUDA; CPU always fp32)
        with self._amp_context():
            # Get inputs and labels
            if isinstance(batch, dict):
                inputs = batch.get("input_ids", batch.get("inputs"))
                labels = batch.get("labels", inputs)
            else:
                inputs, labels = batch[0], batch[1]

            # Forward
            model_output = self.model(inputs)

            # Handle both dict and tensor outputs
            if isinstance(model_output, dict):
                logits = model_output["logits"]
            else:
                logits = model_output

            loss = self.loss_fn(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update EMA
            if self.ema is not None:
                self.ema.update()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Compute metrics
        with torch.no_grad():
            tokens_per_sec = self._compute_tokens_per_sec(inputs)
            gpu_memory = self._get_gpu_memory()
            gpu_util = self._get_gpu_utilization()

        lr = self.optimizer.param_groups[0]["lr"]

        self.global_step += 1

        # Record step time
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time

        return {
            "loss": loss.item() * accumulation_steps,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
            "batch_time": step_time,
            "gpu_memory": gpu_memory,
            "gpu_util": gpu_util,
        }

    def _amp_context(self):
        """Mixed-precision autocast on CUDA; no-op (fp32) on CPU."""
        if self.config.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.config.amp_dtype)
        return nullcontext()

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        val_dataloader = self.config.val_dataloader
        val_batches = self.config.val_batches

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if val_batches and i >= val_batches:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                else:
                    batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]

                # Forward
                with self._amp_context():
                    if isinstance(batch, dict):
                        inputs = batch.get("input_ids", batch.get("inputs"))
                        labels = batch.get("labels", inputs)
                    else:
                        inputs, labels = batch[0], batch[1]

                    model_output = self.model(inputs)

                    # Handle both dict and tensor outputs
                    if isinstance(model_output, dict):
                        logits = model_output["logits"]
                    else:
                        logits = model_output

                    loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(1, num_batches)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint (Exp-Coder format v1)."""
        if self.save_dir is None:
            return

        checkpoint_path = self.save_dir / f"{path}.pt"

        # Build checkpoint dict
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "config": self.config.checkpoint_meta or {},
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.shadow

        # Save
        torch.save(checkpoint, checkpoint_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint produced by :meth:`save_checkpoint` (or a nearby
        compatible dict containing at least the core keys).

        Restores model, optimizer, scheduler, scaler, EMA and RNG state, so
        training can resume.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.save_dir / path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load EMA
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.shadow = checkpoint["ema_state_dict"]

        # Restore RNG state
        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

        self.global_step = checkpoint.get("global_step", checkpoint.get("step", 0))
        self.epoch = checkpoint.get("epoch", 0)

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints."""
        if self.save_dir is None or self.config.keep_last_n <= 0:
            return

        # List checkpoints
        checkpoints = sorted(self.save_dir.glob("step_*.pt"))

        # Remove old checkpoints
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()

    def _compute_tokens_per_sec(self, inputs: torch.Tensor) -> float:
        """Compute tokens per second."""
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        tokens = batch_size * seq_len

        if self.step_times:
            avg_step_time = sum(self.step_times[-100:]) / min(len(self.step_times), 100)
            if avg_step_time > 0:
                return tokens / avg_step_time

        return 0.0

    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization."""
        # Placeholder - real implementation would use nvidia-ml-py
        return 0.0

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train") -> None:
        """Log metrics."""
        # Print to console
        step = self.global_step
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"Step {step} [{prefix}]: {metrics_str}")

        # Log to logger
        if self.config.metrics_logger is not None:
            self.config.metrics_logger.log(metrics, step)

        # Store metrics
        if prefix == "train":
            self.train_metrics.append(metrics)
        else:
            self.val_metrics.append(metrics)