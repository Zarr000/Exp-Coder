"""
Distributed trainer for Expera AI.

Provides:
- DDP (DistributedDataParallel)
- FSDP (FullyShardedDataParallel)
- DeepSpeed integration
- Multi-node support
- Elastic training
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .trainer import Trainer, TrainerConfig


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    strategy: str = "ddp"  # "ddp", "fsdp", "deepspeed"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # DDP settings
    find_unused_params: bool = False
    gradient_as_bucket_view: bool = True

    # FSDP settings
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    mixed_precision: bool = True

    # DeepSpeed
    deepspeed_config: Optional[Dict[str, Any]] = None

    # Multi-node
    master_addr: str = "localhost"
    master_port: int = 29500

    # Elastic
    elastic_training: bool = False


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
        strategy: str = "ddp",
        world_size: int = 1,
    ):
        self.config = config
        self.strategy = strategy
        self.world_size = world_size
        self.rank = 0
        self.local_rank = 0

        # Model wrapper
        self.model_wrapped: Optional[nn.Module] = None

        # State
        self.is_distributed = False
        self.is_initialized = False

    def setup_distributed(self) -> None:
        """Initialize distributed training."""
        # Get rank info from environment
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        if self.world_size > 1:
            self._init_process_group()

        # Wrap model
        if self.strategy == "ddp":
            self._setup_ddp()
        elif self.strategy == "fsdp":
            self._setup_fsdp()
        elif self.strategy == "deepspeed":
            self._setup_deepspeed()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_distributed = True
        self.is_initialized = True

    def _init_process_group(self) -> None:
        """Initialize process group."""
        local_rank = self.local_rank
        rank = self.rank

        # Initialize distributed
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=self.world_size,
            rank=rank,
        )

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    def _setup_ddp(self) -> None:
        """Setup DDP."""
        self.model_wrapped = nn.parallel.DistributedDataParallel(
            self.config.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
        )

    def _setup_fsdp(self) -> None:
        """Setup FSDP."""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                MixedPrecision,
                CPUOffload,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        except ImportError:
            print("FSDP not available, falling back to DDP")
            self.strategy = "ddp"
            self._setup_ddp()
            return

        # Sharding strategy
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_strategy_map.get(
            self.config.get("sharding_strategy", "FULL_SHARD"),
            ShardingStrategy.FULL_SHARD,
        )

        # Mixed precision
        mixed_precision = None
        if self.config.get("mixed_precision", True):
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        # CPU offload
        cpu_offload = None
        if self.config.get("cpu_offload", False):
            cpu_offload = CPUOffload(
                param_offload=True,
                buffer_offload=True,
            )

        # Wrap model
        self.model_wrapped = FSDP(
            self.config.model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
        )

    def _setup_deepspeed(self) -> None:
        """Setup DeepSpeed."""
        try:
            import deepspeed
        except ImportError:
            print("DeepSpeed not available, falling back to DDP")
            self.strategy = "ddp"
            self._setup_ddp()
            return

        # Initialize DeepSpeed
        deepspeed_config = self.config.get("deepspeed_config", {})
        self.model_wrapped, self.optimizer, _, _ = deepspeed.initialize(
            model=self.config.model,
            optimizer=self.config.optimizer,
            config=deepspeed_config,
        )

    def train(self) -> Dict[str, Any]:
        """Run distributed training."""
        if not self.is_initialized:
            self.setup_distributed()

        # Get wrapped model
        model = self.model_wrapped or self.config.model

        # Create new config
        distributed_config = TrainerConfig(
            model=model,
            train_dataloader=self.config.train_dataloader,
            optimizer=self.config.optimizer,
            scheduler=self.config.scheduler,
            loss_fn=self.config.loss_fn,
            device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu",
            max_steps=self.config.max_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            use_amp=self.config.use_amp,
            amp_dtype=self.config.amp_dtype,
            save_every=self.config.save_every,
            save_dir=self.config.save_dir,
            keep_last_n=self.config.keep_last_n,
            val_dataloader=self.config.val_dataloader,
            val_every=self.config.val_every,
            val_batches=self.config.val_batches,
            use_ema=self.config.use_ema,
            ema_decay=self.config.ema_decay,
            log_every=self.config.log_every,
            metrics_logger=self.config.metrics_logger,
        )

        # Create trainer
        trainer = Trainer(distributed_config)
        trainer.global_step = self.config.get("resume_step", 0)

        # Run training
        result = trainer.train()

        # Cleanup
        self.cleanup()

        return result

    def cleanup(self) -> None:
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    def get_rank(self) -> int:
        """Get process rank."""
        return self.rank

    def get_world_size(self) -> int:
        """Get world size."""
        return self.world_size


def setup_distributed(
    strategy: str = "ddp",
    backend: str = "nccl",
) -> tuple:
    """
    Setup distributed training environment.

    Args:
        strategy: Training strategy ("ddp", "fsdp")
        backend: Communication backend

    Returns:
        (rank, world_size, local_rank)
    """
    if "RANK" not in os.environ:
        # Single GPU
        return 0, 1, 0

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()