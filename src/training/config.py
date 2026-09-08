"""
Training Config for Expera AI.

Unified training configuration:
- Model configs
- Dataset configs
- Optimizer configs

Usage:
    config = TrainingConfig()
    config.model.hidden_size = 768
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "expera-coder-120m"
    hidden_size: int = 768
    intermediate_size: int = 1536
    num_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 200
    context_length: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: str = "data/test"
    batch_size: int = 8
    seq_length: int = 2048
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    clip_grad_norm: float = 1.0
    use_fused: bool = True


@dataclass
class SchedulerConfig:
    """Learning rate scheduler."""

    name: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    min_lr: float = 1e-5
    max_lr: float = 1e-3
    num_cycles: float = 0.5


@dataclass
class TrainingConfig:
    """Full training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Training parameters
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 100
    seed: int = 42
    precision: str = "fp32"

    # Hardware
    device: str = "cuda"
    num_gpus: int = 1
    distributed: bool = False

    def save(self, path: str) -> None:
        """Save config to file."""
        data = {
            "model": self.model.__dict__,
            "dataset": self.dataset.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "training": {
                "num_epochs": self.num_epochs,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "eval_steps": self.eval_steps,
                "save_steps": self.save_steps,
                "log_steps": self.log_steps,
                "seed": self.seed,
                "precision": self.precision,
                "device": self.device,
                "num_gpus": self.num_gpus,
                "distributed": self.distributed,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load config from file."""
        with open(path) as f:
            data = json.load(f)

        self.model = ModelConfig(**data.get("model", {}))
        self.dataset = DatasetConfig(**data.get("dataset", {}))
        self.optimizer = OptimizerConfig(**data.get("optimizer", {}))
        self.scheduler = SchedulerConfig(**data.get("scheduler", {}))
        training = data.get("training", {})

        self.num_epochs = training.get("num_epochs", 3)
        self.gradient_accumulation_steps = training.get("gradient_accumulation_steps", 1)
        self.eval_steps = training.get("eval_steps", 500)
        self.save_steps = training.get("save_steps", 1000)
        self.log_steps = training.get("log_steps", 100)
        self.seed = training.get("seed", 42)
        self.precision = training.get("precision", "fp32")
        self.device = training.get("device", "cuda")
        self.num_gpus = training.get("num_gpus", 1)
        self.distributed = training.get("distributed", False)


# Export
__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "DatasetConfig",
    "OptimizerConfig",
    "SchedulerConfig",
]