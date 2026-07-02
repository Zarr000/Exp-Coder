"""
TensorBoard logger for Expera AI.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

# Try to import tensorboard, fallback to dummy if not available
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None


class TensorBoardLogger:
    """
    TensorBoard logging.

    Provides:
    - Scalars (loss, learning rate, etc.)
    - Histograms of gradients/activations
    - Images
    - Text
    """

    def __init__(
        self,
        log_dir: str = "runs",
        flush_secs: int = 120,
    ):
        self.log_dir = Path(log_dir)
        self.flush_secs = flush_secs

        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=self.flush_secs,
            )
        else:
            self.writer = None

    def log(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """
        Log scalar value.

        Args:
            tag: Metric tag
            value: Metric value
            step: Training step
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        for tag, value in metrics.items():
            self.log(tag, value, step)

    def log_histogram(
        self,
        tag: str,
        values: List[float],
        step: int,
    ) -> None:
        """
        Log histogram of values.

        Args:
            tag: Metric tag
            values: List of values
            step: Training step
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_image(
        self,
        tag: str,
        image: Any,
        step: int,
    ) -> None:
        """
        Log image.

        Args:
            tag: Image tag
            image: Image tensor or PIL Image
            step: Training step
        """
        if self.writer is not None:
            self.writer.add_image(tag, image, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """
        Log text.

        Args:
            tag: Text tag
            text: Text content
            step: Training step
        """
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def log_graph(
        self,
        model: Any,
        input_to_model: Any,
    ) -> None:
        """
        Log model graph.

        Args:
            model: PyTorch model
            input_to_model: Example input
        """
        if self.writer is not None:
            self.writer.add_graph(model, input_to_model)

    def close(self) -> None:
        """Close writer."""
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DummyTensorBoardLogger:
    """Dummy TensorBoard logger when tensorboard is not available."""

    def __init__(self, log_dir: str = "runs", flush_secs: int = 120):
        pass

    def log(self, tag: str, value: float, step: int) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_histogram(self, tag: str, values: List[float], step: int) -> None:
        pass

    def close(self) -> None:
        pass