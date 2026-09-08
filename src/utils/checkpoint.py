"""
DEPRECATED checkpoint management utilities.

This module is legacy. The canonical Exp-Coder checkpoint format is
``exp-coder-v1`` and lives in:
    src/training/checkpoint_manager.py (CheckpointManager)
and is also emitted by:
    src/training/trainer.py (Trainer.save_checkpoint)

Both write and read the same dict format. Keep this module only for
backward compatibility with historical scripts; do not use it in new code.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import torch


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 5,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.best_checkpoint = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Save checkpoint."""
        if name is None:
            name = f"checkpoint-{step}"
            
        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics or {},
        }
        
        torch.save(state_dict, checkpoint_path / "model.pt")
        
        # Track best model
        if metrics and self.metric_for_best_model in metrics:
            current_metric = metrics[self.metric_for_best_model]
            if (self.greater_is_better and current_metric > self.best_metric) or \
               (not self.greater_is_better and current_metric < self.best_metric):
                self.best_metric = current_metric
                self.best_checkpoint = name
                
                # Save best model separately
                torch.save(state_dict, self.output_dir / "best_model.pt")
                
        # Enforce save limit
        self._enforce_save_limit()
        
        return str(checkpoint_path)
    
    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        if load_best:
            checkpoint_path = self.output_dir / "best_model.pt"
        elif checkpoint_path is None:
            # Load latest
            checkpoints = sorted(self.output_dir.glob("checkpoint-*"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1] / "model.pt"
        else:
            checkpoint_path = Path(checkpoint_path) / "model.pt"
            
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(state_dict["model_state_dict"])
        
        if optimizer and state_dict.get("optimizer_state_dict"):
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            
        if scheduler and state_dict.get("scheduler_state_dict"):
            scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            
        return {
            "step": state_dict["step"],
            "epoch": state_dict["epoch"],
            "metrics": state_dict["metrics"],
        }
    
    def _enforce_save_limit(self):
        """Remove old checkpoints if over limit."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda d: d.stat().st_mtime,
        )
        
        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)