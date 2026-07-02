"""
Checkpoint management for Expera AI training.

Provides:
- Periodic saving
- Automatic cleanup
- Resume from latest
- Best model tracking
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    step: int
    epoch: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class CheckpointManager:
    """
    Manages checkpointing with automatic cleanup.

    Features:
    - Periodic saving
    - Automatic cleanup of old checkpoints
    - Resume from latest
    - Best model tracking
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        save_dir: str = "checkpoints",
        keep_last_n: int = 3,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n

        # Best model tracking
        self.best_metric = float("inf")
        self.best_checkpoint_path: Optional[Path] = None

        # Create save directory
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        self.metadata_file = self.save_dir / "metadata.json"

    def save(
        self,
        step: int,
        metrics: Dict[str, float],
        rank: int = 0,
    ) -> str:
        """
        Save checkpoint.

        Args:
            step: Current training step
            metrics: Metrics to save
            rank: Current process rank (for DDP)

        Returns:
            Path to saved checkpoint
        """
        # Build filename
        filename = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.save_dir / filename

        # Build checkpoint dict
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint (only rank 0 in DDP)
        if rank == 0:
            torch.save(checkpoint, checkpoint_path)

            # Update metadata
            metadata = CheckpointMetadata(
                step=step,
                epoch=0,
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
            )
            self._save_metadata(metadata)

        # Update best checkpoint
        if "val_loss" in metrics and metrics["val_loss"] < self.best_metric:
            self.best_metric = metrics["val_loss"]
            self.best_checkpoint_path = checkpoint_path
            self._save_best_marker(step, metrics)

        # Cleanup old checkpoints
        self.cleanup()

        return str(checkpoint_path)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint.

        Returns:
            Checkpoint dict or None if not found
        """
        if not self.save_dir.exists():
            return None

        # Find latest checkpoint
        checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        return self._load_checkpoint(latest)

    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint.

        Returns:
            Checkpoint dict or None if not found
        """
        if self.best_checkpoint_path is None or not self.best_checkpoint_path.exists():
            # Try to find best marker
            best_marker = self.save_dir / "best.txt"
            if best_marker.exists():
                with open(best_marker) as f:
                    step = int(f.read().strip())
                best_path = self.save_dir / f"checkpoint_step_{step}.pt"
                if best_path.exists():
                    return self._load_checkpoint(best_path)
            return None

        return self._load_checkpoint(self.best_checkpoint_path)

    def load(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint from path.

        Args:
            path: Checkpoint path (relative or absolute)

        Returns:
            Checkpoint dict
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.save_dir / path

        return self._load_checkpoint(checkpoint_path)

    def cleanup(self) -> None:
        """Remove old checkpoints."""
        if self.keep_last_n <= 0:
            return

        # List checkpoints
        checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))

        # Remove old checkpoints (keep best)
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                # Don't remove best checkpoint
                if self.best_checkpoint_path is None or ckpt != self.best_checkpoint_path:
                    ckpt.unlink()

    def get_latest_step(self) -> int:
        """
        Get latest checkpoint step.

        Returns:
            Step number or -1 if no checkpoints
        """
        if not self.save_dir.exists():
            return -1

        checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return -1

        # Extract step from filename
        latest = checkpoints[-1]
        step = int(latest.stem.split("_")[-1])
        return step

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with metadata.

        Returns:
            List of checkpoint info
        """
        if not self.save_dir.exists():
            return []

        checkpoints = []
        for ckpt in sorted(self.save_dir.glob("checkpoint_step_*.pt")):
            step = int(ckpt.stem.split("_")[-1])
            metrics = {}
            tags = []

            # Check if best
            if self.best_checkpoint_path == ckpt:
                tags.append("best")

            checkpoints.append({
                "path": str(ckpt),
                "step": step,
                "metrics": metrics,
                "tags": tags,
            })

        return checkpoints

    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint from file."""
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint

    def _save_metadata(self, metadata: CheckpointMetadata) -> None:
        """Save checkpoint metadata."""
        # Load existing metadata
        all_metadata: Dict[str, Any] = {}
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                all_metadata = json.load(f)

        # Add new metadata
        all_metadata[f"step_{metadata.step}"] = {
            "step": metadata.step,
            "epoch": metadata.epoch,
            "timestamp": metadata.timestamp,
            "metrics": metadata.metrics,
            "tags": metadata.tags,
        }

        # Save
        with open(self.metadata_file, "w") as f:
            json.dump(all_metadata, f, indent=2)

    def _save_best_marker(self, step: int, metrics: Dict[str, float]) -> None:
        """Save best checkpoint marker."""
        best_marker = self.save_dir / "best.txt"
        with open(best_marker, "w") as f:
            f.write(str(step))

        # Save best metrics
        best_metrics = self.save_dir / "best_metrics.json"
        with open(best_metrics, "w") as f:
            json.dump(metrics, f, indent=2)