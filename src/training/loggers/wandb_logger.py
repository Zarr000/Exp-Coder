"""
Weights & Biases logger for Expera AI.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

# Try to import wandb, fallback to dummy if not available
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


class WandBLogger:
    """
    Weights & Biases logging.

    Provides:
    - Scalar logging
    - Summary statistics
    - Media logging
    - Tables
    """

    def __init__(
        self,
        project: str = "expera-ai",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.project = project
        self.name = name
        self.entity = entity
        self.tags = tags

        if HAS_WANDB:
            # Initialize wandb
            wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                tags=tags,
            )
            self.initialized = True
        else:
            self.initialized = False

    def log(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        if self.initialized:
            wandb.log(metrics, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        """
        Log metrics (alias for log).

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        self.log(metrics, step)

    def log_summary(
        self,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log summary metrics.

        Args:
            metrics: Summary metrics
        """
        if self.initialized:
            wandb.summary.update(metrics)

    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        """
        Log table.

        Args:
            name: Table name
            columns: Column names
            data: Table data
        """
        if self.initialized:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({name: table})

    def log_image(
        self,
        name: str,
        image: Any,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log image.

        Args:
            name: Image name
            image: Image
            caption: Optional caption
        """
        if self.initialized:
            wandb.log({name: wandb.Image(image, caption=caption)})

    def log_histogram(
        self,
        name: str,
        values: List[float],
    ) -> None:
        """
        Log histogram.

        Args:
            name: Histogram name
            values: Values
        """
        if self.initialized:
            wandb.log({name: wandb.Histogram(values)})

    def log_code(
        self,
        file_path: str,
    ) -> None:
        """
        Log code file.

        Args:
            file_path: Path to code file
        """
        if self.initialized:
            wandb.log({"code": wandb.Code(file_path)})

    def finish(self) -> None:
        """Finish logging."""
        if self.initialized:
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class DummyWandBLogger:
    """Dummy WandB logger when wandb is not available."""

    def __init__(
        self,
        project: str = "expera-ai",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        pass

    def log(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_summary(self, metrics: Dict[str, float]) -> None:
        pass

    def finish(self) -> None:
        pass