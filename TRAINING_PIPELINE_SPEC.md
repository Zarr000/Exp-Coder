# Phase 3: Distributed Pretraining Pipeline

## Overview

Build a production-quality large-scale pretraining system for Expera AI supporting:
- Multi-GPU and multi-node training
- Mixed precision training (bf16/fp16)
- Gradient accumulation and checkpointing
- Elastic and fault-tolerant training
- Comprehensive logging and monitoring

## Goals

1. Support billion-token datasets
2. Enable efficient multi-GPU/multi-node scaling
3. Provide comprehensive monitoring
4. Support resumption from checkpoints
5. Production-quality reliability

## Architecture

### Module Structure

```
src/training/
├── __init__.py
├── loss.py              # Existing: Loss functions
├── optimizer.py         # Existing: Optimizer factory
├── scheduler.py        # Existing: Learning rate schedulers
├── trainer.py          # Core trainer
├── distributed_trainer.py  # DDP/FSDP wrapper
├── checkpoint_manager.py  # Checkpoint management
├── evaluator.py       # Validation/evaluation
├── metrics.py        # Metrics collection
├── callbacks.py      # Training callbacks
├── loggers/
│   ├── __init__.py
│   ├── tensorboard_logger.py
│   ├── wandb_logger.py
│   ├── csv_logger.py
│   └── json_logger.py
├── benchmarks/
│   ├── __init__.py
│   └── training_benchmark.py
└── configs/
    └── __init__.py
    └── trainer_config.py
```

## Implementation Details

### trainer.py

```python
@dataclass
class TrainerConfig:
    """Configuration for trainer."""
    model: torch.nn.Module
    train_dataloader: IterableDataset
    val_dataloader: Optional[IterableDataset] = None
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any] = None
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
    val_every: int = 5000

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Logging
    log_every: int = 100
    metrics_logger: Optional[Any] = None


class Trainer:
    """
    Main trainer class for Expera AI.

    Supports:
    - Mixed precision training (bf16/fp16)
    - Gradient accumulation
    - Gradient clipping
    - EMA
    - checkpointing
    - Validation
    - Multiple loggers
    """

    def __init__(self, config: TrainerConfig) -> None:
        ...

    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        ...

    def train_step(self, batch: Batch) -> Dict[str, float]:
        """Single training step."""
        ...

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        ...
```

### checkpoint_manager.py

```python
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
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        save_dir: str = "checkpoints",
        keep_last_n: int = 3,
    ) -> None:
        ...

    def save(
        self,
        step: int,
        metrics: Dict[str, float],
        rank: int = 0,
    ) -> str:
        """Save checkpoint."""
        ...

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        ...

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint."""
        ...

    def cleanup(self) -> None:
        """Remove old checkpoints."""
        ...
```

### distributed_trainer.py

```python
class DistributedTrainer:
    """
    Distributed trainer supporting DDP/FSDP.

    Features:
    - DDP (DistributedDataParallel)
    - FSDP (FullyShardedDataParallel)
    - DeepSpeed integration
    - Multi-node support
    - Elastic training
    """

    def __init__(
        self,
        config: TrainerConfig,
        strategy: str = "ddp",  # "ddp", "fsdp", "deepspeed"
        world_size: int = 1,
    ) -> None:
        ...

    def setup_distributed(self) -> None:
        """Initialize distributed training."""
        ...

    def train(self) -> Dict[str, Any]:
        """Run distributed training."""
        ...
```

### evaluator.py

```python
class Evaluator:
    """
    Evaluation for pretraining.

    Features:
    - Validation loop
    - Test set evaluation
    - Perplexity computation
    - Zero-shot evaluation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: IterableDataset,
        device: str = "cuda",
    ) -> None:
        ...

    def evaluate(
        self,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run evaluation."""
        ...

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss."""
        ...
```

### metrics.py

```python
@dataclass
class TrainingMetrics:
    """Metrics for training."""
    step: int
    loss: float
    lr: float
   _tokens_per_sec: float
    batch_time: float
    gpu_memory: float
    gpu_utilization: float


class MetricsCollector:
    """Collect and aggregate training metrics."""

    def __init__(self) -> None:
        ...

    def record(self, metrics: TrainingMetrics) -> None:
        """Record metrics."""
        ...

    def get_average(self, window: int = 100) -> Dict[str, float]:
        """Get averaged metrics."""
        ...
```

### loggers/

**tensorboard_logger.py**

```python
class TensorBoardLogger:
    """TensorBoard logging."""

    def __init__(self, log_dir: str) -> None:
        ...

    def log(self, tag: str, value: float, step: int) -> None:
        ...

    def log_histogram(self, tag: str, values: List[float], step: int) -> None:
        ...
```

**wandb_logger.py**

```python
class WandBLogger:
    """Weights & Biases logging."""

    def __init__(self, project: str, name: Optional[str] = None) -> None:
        ...

    def log(self, metrics: Dict[str, float], step: int) -> None:
        ...
```

**csv_logger.py**

```python
class CSVLogger:
    """CSV file logging."""

    def __init__(self, path: str) -> None:
        ...

    def log(self, metrics: Dict[str, Any]) -> None:
        ...
```

**json_logger.py**

```python
class JSONLogger:
    """JSON file logging."""

    def __init__(self, path: str) -> None:
        ...

    def log(self, metrics: Dict[str, Any]) -> None:
        ...
```

### callbacks.py

```python
class Callback:
    """Base callback class."""

    def on_step_start(self, step: int) -> None:
        ...

    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        ...

    def on_val_start(self) -> None:
        ...

    def on_val_end(self, metrics: Dict[str, float]) -> None:
        ...


class EarlyStoppingCallback(Callback):
    """Early stopping."""

    def __init__(self, patience: int = 3, min_delta: float = 0.01) -> None:
        ...

class ModelCheckpointCallback(Callback):
    """Model checkpointing callback."""

    ...
```

## Performance Targets

| Metric | Target |
|--------|-------|
| Training throughput | 10K tokens/sec/GPU |
| Single GPU memory | < 40GB |
| Multi-GPU scaling | > 0.9 efficiency |
| Checkpoint save | < 30 seconds |
| Validation time | < 60 seconds |

## Tests

All modules require tests:

1. Basic functionality
2. Edge cases
3. Error handling
4. Integration with model/data
5. Multi-GPU scenarios (when available)

## Exit Criteria

1. All modules implemented with type hints and docstrings
2. All tests passing
3. Benchmarks complete
4. Documentation complete