"""
Training callbacks for Expera AI.

Provides:
- Early stopping
- Model checkpointing
- Learning rate adjustment
- Training control
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class Callback(ABC):
    """
    Base callback class.

    Callbacks allow customizable training behavior.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called at start of training."""
        pass

    def on_train_end(self, trainer: Any, result: Dict[str, Any]) -> None:
        """Called at end of training."""
        pass

    def on_step_start(self, step: int) -> None:
        """Called at start of training step."""
        pass

    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Called at end of training step."""
        pass

    def on_val_start(self) -> None:
        """Called at start of validation."""
        pass

    def on_val_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Called at end of validation."""
        pass

    def on_checkpoint_save(self, step: int, path: str) -> None:
        """Called after saving checkpoint."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback.

    Stops training when metric stops improving.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.01,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode

        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_val_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Check if should stop."""
        if self.metric not in metrics:
            return

        current = metrics[self.metric]

        # Check improvement
        if self.best_value is None:
            self.best_value = current
            improved = True
        elif self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping: no improvement for {self.patience} validations")
            self.should_stop = True


class ModelCheckpointCallback(Callback):
    """
    Model checkpointing callback.

    Saves checkpoints based on metric improvement.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last

        self.best_value: Optional[float] = None

    def on_val_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Check if should save checkpoint."""
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        # Check if best
        if self.best_value is None:
            is_best = True
        elif self.mode == "min":
            is_best = current < self.best_value
        else:
            is_best = current > self.best_value

        if is_best:
            self.best_value = current

        # Save
        if is_best or not self.save_best_only:
            # Actual save handled by trainer
            pass


class LearningRateSchedulerCallback(Callback):
    """
    Learning rate scheduler callback.
    """

    def __init__(
        self,
        scheduler: Any,
    ):
        self.scheduler = scheduler

    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Step scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()


class GradientAccumulationCallback(Callback):
    """
    Adjust gradient accumulation dynamically.
    """

    def __init__(
        self,
        target_batch_size: int = 8192,
        min_accumulation: int = 1,
        max_accumulation: int = 8,
    ):
        self.target_batch_size = target_batch_size
        self.min_accumulation = min_accumulation
        self.max_accumulation = max_accumulation

    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Adjust accumulation."""
        # Adjust based on metrics (placeholder)
        pass


class CallbackList:
    """
    Container for multiple callbacks.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add callback."""
        self.callbacks.append(callback)

    def on_train_start(self, trainer: Any) -> None:
        """Train start."""
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any, result: Dict[str, Any]) -> None:
        """Train end."""
        for cb in self.callbacks:
            cb.on_train_end(trainer, result)

    def on_step_start(self, step: int) -> None:
        """Step start."""
        for cb in self.callbacks:
            cb.on_step_start(step)

    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Step end."""
        for cb in self.callbacks:
            cb.on_step_end(step, metrics)

    def on_val_start(self) -> None:
        """Validation start."""
        for cb in self.callbacks:
            cb.on_val_start()

    def on_val_end(self, step: int, metrics: Dict[str, float]) -> None:
        """Validation end."""
        for cb in self.callbacks:
            cb.on_val_end(step, metrics)