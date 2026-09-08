"""
Exp-Coder checkpoint management — canonical format.

Checkpoint format ``exp-coder-v1`` (single source of truth):

    {
        "format": "exp-coder-v1",
        "step": int,
        "epoch": int,
        "metrics": {...},
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "scheduler_state_dict": ... (optional),
        "scaler_state_dict": ... (optional),
        "rng_state": {"python": ..., "torch": ..., "cuda": ...},
        "config": {...},               # model/training config metadata
        "tokenizer_metadata": {...},   # vocab size / special tokens
    }

``src.training.trainer.Trainer.save_checkpoint`` produces the same format, so
checkpoints written here can be loaded by the Trainer (and vice versa) when
the containing keys match.

Features:
- periodic saves with cleanup
- resume from latest / best
- task metadata embedded in the checkpoint
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


CHECKPOINT_FORMAT = "exp-coder-v1"


@dataclass
class CheckpointMetadata:
    """JSON-serializable metadata for a checkpoint."""
    step: int
    epoch: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    format: str = CHECKPOINT_FORMAT


class CheckpointManager:
    """
    Manages checkpointing (directory-per-checkpoint, Exp-Coder v1 format).

    Features:
    - Periodic saving
    - Automatic cleanup of old checkpoints
    - Resume from latest / best
    - RNG + config + tokenizer metadata embedded for faithful resume
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

        self.best_metric = float("inf")
        self.best_checkpoint_path: Optional[Path] = None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.save_dir / "metadata.json"

    def save(
        self,
        step: int,
        metrics: Dict[str, float],
        epoch: int = 0,
        rank: int = 0,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Save checkpoint in the canonical Exp-Coder v1 format.

        Args:
            step: current training step
            metrics: metrics dict
            epoch: current epoch
            rank: process rank (only rank 0 writes with DDP)
            config: model/training configuration (embedded in checkpoint)
            tokenizer: tokenizer whose metadata is embedded
        """
        filename = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.save_dir / filename

        tokenizer_meta: Dict[str, Any] = {}
        if tokenizer is not None:
            tokenizer_meta = {
                "vocab_size": getattr(tokenizer, "vocab_size", None),
                "actual_vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
                "special_tokens": dict(getattr(tokenizer, "special_tokens", {}) or {}),
            }

        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
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
            "config": config or {},
            "tokenizer_metadata": tokenizer_meta,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if rank == 0:
            torch.save(checkpoint, checkpoint_path)

            metadata = CheckpointMetadata(
                step=step,
                epoch=epoch,
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
            )
            self._save_metadata(metadata)

        if "val_loss" in metrics and metrics["val_loss"] < self.best_metric:
            self.best_metric = metrics["val_loss"]
            self.best_checkpoint_path = checkpoint_path
            self._save_best_marker(step, metrics)

        self.cleanup()
        return str(checkpoint_path)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint (``None`` if none exists)."""
        if not self.save_dir.exists():
            return None
        checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        return self._load_checkpoint(checkpoints[-1])

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint by recorded metric."""
        if self.best_checkpoint_path is not None and self.best_checkpoint_path.exists():
            return self._load_checkpoint(self.best_checkpoint_path)
        best_marker = self.save_dir / "best.txt"
        if best_marker.exists():
            step = int(best_marker.read_text().strip())
            best_path = self.save_dir / f"checkpoint_step_{step}.pt"
            if best_path.exists():
                return self._load_checkpoint(best_path)
        return None

    def load(self, path: str) -> Dict[str, Any]:
        """Load checkpoint from explicit path."""
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.save_dir / path
        return self._load_checkpoint(checkpoint_path)

    def resume_from(self, checkpoint: Dict[str, Any]) -> None:
        """Apply a loaded checkpoint to model/optimizer/scheduler + RNG."""
        def _pop(key, required=False):
            if key not in checkpoint:
                if required:
                    raise KeyError(f"checkpoint missing required key {key!r}")
                return None
            return checkpoint[key]

        self.model.load_state_dict(_pop("model_state_dict", required=True))
        opt_state = _pop("optimizer_state_dict")
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

        if self.scheduler is not None:
            sched_state = _pop("scheduler_state_dict")
            if sched_state is not None:
                self.scheduler.load_state_dict(sched_state)

        rng = _pop("rng_state")
        if rng is not None:
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

    def cleanup(self) -> None:
        """Remove old checkpoints beyond ``keep_last_n``."""
        if self.keep_last_n <= 0:
            return
        checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                if self.best_checkpoint_path is None or ckpt != self.best_checkpoint_path:
                    ckpt.unlink()

    def get_latest_step(self) -> int:
        """Return the latest checkpoint step (``-1`` if none)."""
        if not self.save_dir.exists():
            return -1
        checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return -1
        return int(checkpoints[-1].stem.split("_")[-1])

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List checkpoints with metadata."""
        if not self.save_dir.exists():
            return []
        out = []
        for ckpt in sorted(self.save_dir.glob("checkpoint_step_*.pt")):
            step = int(ckpt.stem.split("_")[-1])
            tags = ["best"] if self.best_checkpoint_path == ckpt else []
            out.append({"path": str(ckpt), "step": step, "tags": tags})
        return out

    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu")

    def _save_metadata(self, metadata: CheckpointMetadata) -> None:
        all_metadata: Dict[str, Any] = {}
        if self.metadata_file.exists():
            all_metadata = json.loads(self.metadata_file.read_text())
        all_metadata[f"step_{metadata.step}"] = {
            "step": metadata.step,
            "epoch": metadata.epoch,
            "timestamp": metadata.timestamp,
            "metrics": metadata.metrics,
            "tags": metadata.tags,
        }
        self.metadata_file.write_text(json.dumps(all_metadata, indent=2))

    def _save_best_marker(self, step: int, metrics: Dict[str, float]) -> None:
        (self.save_dir / "best.txt").write_text(str(step))
        (self.save_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))