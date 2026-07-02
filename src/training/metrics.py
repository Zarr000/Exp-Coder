"""
Metrics collection for Expera AI training.

Provides:
- Training metrics tracking
- Metric aggregation
- GPU metrics
- Performance monitoring
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import time
import torch


@dataclass
class TrainingMetrics:
    """Metrics for training."""
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    tokens_per_sec: float = 0.0
    batch_time: float = 0.0
    gpu_memory: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class ValidationMetrics:
    """Metrics for validation."""
    step: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    accuracy: float = 0.0


class MetricsCollector:
    """
    Collect and aggregate training metrics.

    Features:
    - Rolling window averaging
    - GPU metrics
    - Throughput tracking
    - Export to various formats
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Metrics history
        self.train_metrics: deque = deque(maxlen=window_size)
        self.val_metrics: deque = deque(maxlen=window_size)

        # Cumulative metrics
        self.total_tokens = 0
        self.total_steps = 0
        self.start_time = time.time()

    def record(self, metrics: TrainingMetrics) -> None:
        """
        Record training metrics.

        Args:
            metrics: Training metrics to record
        """
        self.train_metrics.append(metrics)
        self.total_steps += 1

        # Update total tokens
        if metrics.tokens_per_sec > 0 and metrics.batch_time > 0:
            tokens_in_batch = metrics.tokens_per_sec * metrics.batch_time
            self.total_tokens += int(tokens_in_batch)

    def record_val(self, metrics: ValidationMetrics) -> None:
        """
        Record validation metrics.

        Args:
            metrics: Validation metrics to record
        """
        self.val_metrics.append(metrics)

    def get_average(self, window: int = 100) -> Dict[str, float]:
        """
        Get averaged metrics.

        Args:
            window: Window size for averaging

        Returns:
            Dictionary with averaged metrics
        """
        if not self.train_metrics:
            return {}

        # Get recent metrics
        recent = list(self.train_metrics)[-window:]

        if not recent:
            return {}

        # Compute averages
        avg_loss = sum(m.loss for m in recent) / len(recent)
        avg_lr = sum(m.lr for m in recent) / len(recent)
        avg_tokens_per_sec = sum(m.tokens_per_sec for m in recent) / len(recent)
        avg_batch_time = sum(m.batch_time for m in recent) / len(recent)
        avg_gpu_memory = sum(m.gpu_memory for m in recent) / len(recent)
        avg_gpu_util = sum(m.gpu_utilization for m in recent) / len(recent)

        # Compute throughput
        elapsed = time.time() - self.start_time
        overall_tokens_per_sec = self.total_tokens / max(1, elapsed)

        return {
            "loss": avg_loss,
            "lr": avg_lr,
            "tokens_per_sec": avg_tokens_per_sec,
            "batch_time": avg_batch_time,
            "gpu_memory": avg_gpu_memory,
            "gpu_utilization": avg_gpu_util,
            "overall_tokens_per_sec": overall_tokens_per_sec,
            "total_steps": self.total_steps,
        }

    def get_validation_average(self) -> Dict[str, float]:
        """Get validation metrics average."""
        if not self.val_metrics:
            return {}

        recent = list(self.val_metrics)
        avg_loss = sum(m.loss for m in recent) / len(recent)
        avg_perplexity = sum(m.perplexity for m in recent) / len(recent)

        return {
            "val_loss": avg_loss,
            "val_perplexity": avg_perplexity,
        }

    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        if not torch.cuda.is_available():
            return {}

        metrics = {}

        # Memory
        metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
        metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        metrics["gpu_memory_max_allocated"] = torch.cuda.max_memory_allocated() / 1e9

        # Utilization (if available)
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization"] = util.gpu

            pynvml.nvmlExit()
        except (ImportError, Exception):
            metrics["gpu_utilization"] = 0.0

        return metrics

    def reset(self) -> None:
        """Reset metrics."""
        self.train_metrics.clear()
        self.val_metrics.clear()
        self.total_tokens = 0
        self.total_steps = 0
        self.start_time = time.time()

    def export(self, format: str = "dict") -> Any:
        """
        Export metrics.

        Args:
            format: Export format ("dict", "list", "dataframe")

        Returns:
            Exported metrics
        """
        if format == "dict":
            return {
                "train": self.get_average(),
                "validation": self.get_validation_average(),
                "gpu": self.get_gpu_metrics(),
            }
        elif format == "list":
            return {
                "train": list(self.train_metrics),
                "validation": list(self.val_metrics),
            }
        else:
            return self.get_average()


class ThroughputTracker:
    """
    Track training throughput.
    """

    def __init__(self):
        self.batch_times: deque = deque(maxlen=1000)
        self.token_counts: deque = deque(maxlen=1000)
        self.start_time = time.time()

    def record(self, batch_time: float, num_tokens: int) -> None:
        """Record batch throughput."""
        self.batch_times.append(batch_time)
        self.token_counts.append(num_tokens)

    def get_tokens_per_sec(self, window: int = 100) -> float:
        """Get tokens per second."""
        if not self.batch_times:
            return 0.0

        recent_times = list(self.batch_times)[-window:]
        recent_tokens = list(self.token_counts)[-window:]

        if not recent_times or sum(recent_times) <= 0:
            return 0.0

        return sum(recent_tokens) / sum(recent_times)

    def get_batches_per_sec(self, window: int = 100) -> float:
        """Get batches per second."""
        if not self.batch_times:
            return 0.0

        recent = list(self.batch_times)[-window:]

        if not recent or sum(recent) <= 0:
            return 0.0

        return len(recent) / sum(recent)

    def get_average_batch_time(self, window: int = 100) -> float:
        """Get average batch time."""
        if not self.batch_times:
            return 0.0

        recent = list(self.batch_times)[-window:]
        return sum(recent) / len(recent)


def create_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """
    Create metrics collector from config.

    Args:
        config: Configuration dict

    Returns:
        MetricsCollector
    """
    window_size = config.get("metrics_window", 100)
    return MetricsCollector(window_size=window_size)