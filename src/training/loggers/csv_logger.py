"""
CSV logger for Expera AI.
"""

import csv
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


class CSVLogger:
    """
    CSV file logging.

    Provides:
    - Flexible column handling
    - Append mode
    - Header auto-generation
    """

    def __init__(
        self,
        path: str = "logs/metrics.csv",
        fieldnames: Optional[List[str]] = None,
    ):
        self.path = Path(path)
        self.fieldnames = fieldnames or ["timestamp", "step", "loss", "lr"]

        # Create directory
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists
        self.file_exists = self.path.exists()

    def log(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Log metrics to CSV.

        Args:
            metrics: Metrics dictionary
        """
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()

        # Determine fieldnames
        if not self.file_exists:
            self.fieldnames = list(metrics.keys())

        # Write to CSV
        mode = "a" if self.path.exists() else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)

            if not self.file_exists:
                writer.writeheader()

            # Filter to existing fieldnames
            row = {k: v for k, v in metrics.items() if k in self.fieldnames}
            writer.writerow(row)

        self.file_exists = True

    def log_metrics(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Log metrics (alias for log).

        Args:
            metrics: Metrics dictionary
        """
        self.log(metrics)

    def read(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Read metrics from CSV.

        Args:
            n: Number of rows to read (None for all)

        Returns:
            List of metric dictionaries
        """
        if not self.path.exists():
            return []

        with open(self.path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if n is not None:
            return rows[-n:]

        return rows

    def get_recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics.

        Args:
            n: Number of recent rows

        Returns:
            List of recent metrics
        """
        return self.read(n)

    def close(self) -> None:
        """Close logger (no-op for CSV)."""
        pass


class MetricsCSVLogger:
    """
    Specialized metrics CSV logger.
    """

    def __init__(
        self,
        path: str = "logs/metrics.csv",
    ):
        self.path = Path(path)
        self.fieldnames = [
            "timestamp",
            "step",
            "epoch",
            "phase",  # "train" or "val"
            "loss",
            "perplexity",
            "lr",
            "tokens_per_sec",
            "gpu_memory",
            "gpu_utilization",
        ]

        # Create directory
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists
        self.file_exists = self.path.exists()

    def log(
        self,
        step: int,
        phase: str,
        metrics: Dict[str, float],
        epoch: int = 0,
    ) -> None:
        """
        Log metrics.

        Args:
            step: Training step
            phase: "train" or "val"
            metrics: Metrics dict
            epoch: Epoch number
        """
        import time

        row = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "phase": phase,
            "loss": metrics.get("loss", 0.0),
            "perplexity": metrics.get("perplexity", 0.0),
            "lr": metrics.get("lr", 0.0),
            "tokens_per_sec": metrics.get("tokens_per_sec", 0.0),
            "gpu_memory": metrics.get("gpu_memory", 0.0),
            "gpu_utilization": metrics.get("gpu_utilization", 0.0),
        }

        # Write
        mode = "a" if self.path.exists() else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)

            if not self.file_exists:
                writer.writeheader()

            writer.writerow(row)

        self.file_exists = True

    def close(self) -> None:
        """Close logger."""
        pass