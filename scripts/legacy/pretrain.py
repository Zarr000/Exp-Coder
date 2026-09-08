#!/usr/bin/env python3
"""
Pre-training Entry Point for Expera AI.

Main training script supporting:
- Single/multi-GPU training
- Gradient accumulation
- BF16/FP16 mixed precision
- Checkpointing with resumable training
- TensorBoard/WandB logging
- LoRA fine-tuning

Usage:
    # Single GPU
    python scripts/pretrain.py --config config/training.yaml

    # Multi-GPU
    python -m torch.distributed.run --nproc_per_node=4 scripts/pretrain.py --config config/training.yaml

    # Resume training
    python scripts/pretrain.py --config config/training.yaml --resume checkpoints/step_1000/

    # LoRA fine-tuning
    python scripts/pretrain.py --config config/lora_training.yaml --lora --lora-rank 16
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml

from src.model.architecture import ExperaModel
from src.tokenizer import SentencePieceTokenizer
from src.training import (
    get_optimizer,
    get_scheduler,
    CheckpointManager,
    LanguageModelingLoss,
    TrainerConfig,
    Trainer,
)
from src.data import get_dataloader, DataLoaderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    return 0, 1, 0  # Single GPU


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_config(config: Dict[str, Any], variant: str = "base") -> Dict[str, Any]:
    """Get model configuration from training config."""
    model_config = config.get("model", {})
    variants = config.get("variants", {})

    if variant in variants:
        base = model_config.copy()
        base.update(variants[variant])
        return base

    return model_config


def create_model(
    config: Dict[str, Any],
    device: torch.device,
    resume_path: Optional[str] = None,
) -> nn.Module:
    """Create and initialize model."""
    model_config = config.get("model", {})

    model = ExperaModel(
        vocab_size=model_config.get("vocab_size", 50304),
        hidden_size=model_config.get("hidden_size", 512),
        num_layers=model_config.get("num_layers", 8),
        num_heads=model_config.get("num_heads", 8),
        intermediate_size=model_config.get("intermediate_size", 2048),
        max_position_embeddings=model_config.get("max_position_embeddings", 4096),
        activation=model_config.get("activation", "gelu"),
        dropout=model_config.get("dropout", 0.1),
        use_rope=model_config.get("use_rope", True),
        rope_theta=model_config.get("rope_theta", 10000.0),
        use_gated_activations=model_config.get("use_gated_activations", False),
    ).to(device)

    # Load from checkpoint if resuming
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {resume_path}")

    return model


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Create optimizer."""
    optimizer_config = config.get("training", {}).get("optimizer", {})
    return get_optimizer(model.parameters(), optimizer_config)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    scheduler_config = config.get("training", {}).get("scheduler", {})
    return get_scheduler(optimizer, scheduler_config, num_training_steps)


def create_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: Dict[str, Any],
    device: torch.device,
) -> Trainer:
    """Create trainer."""
    train_config = config.get("training", {})
    trainer_config = TrainerConfig(
        max_steps=train_config.get("duration", {}).get("max_steps", 10000),
        gradient_accumulation_steps=train_config.get("strategy", {}).get(
            "gradient_accumulation_steps", 1
        ),
        max_grad_norm=train_config.get("strategy", {}).get("max_grad_norm", 1.0),
        logging_steps=train_config.get("logging", {}).get("logging_steps", 100),
        save_steps=train_config.get("checkpointing", {}).get("save_steps", 1000),
        eval_steps=train_config.get("evaluation", {}).get("eval_steps", 1000),
        warmup_steps=train_config.get("scheduler", {}).get("warmup_steps", 500),
        bf16=train_config.get("precision", {}).get("bf16", False),
        fp16=train_config.get("precision", {}).get("fp16", False),
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        device=device,
    )


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    scaler: Optional[GradScaler],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute single training step."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch.get("labels", input_ids).to(device)

    use_amp = config.get("training", {}).get("precision", {}).get("bf16", False) or \
              config.get("training", {}).get("precision", {}).get("fp16", False)

    if use_amp and scaler is not None:
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            loss = criterion(outputs["logits"], labels)

        scaler.scale(loss).backward()
        return {"loss": loss.item()}

    else:
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        return {"loss": loss.item()}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pre-train Expera AI")

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/training.yaml",
        help="Training configuration file"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Model configuration file (overrides config)"
    )

    # Model variants
    parser.add_argument(
        "--variant", "-v",
        type=str,
        default="base",
        choices=["tiny", "small", "medium", "base", "large"],
        help="Model variant"
    )

    # Resume training
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Checkpoint to resume from"
    )

    # LoRA
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA fine-tuning"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Tokenized data directory"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path or directory"
    )

    # Mixed precision
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision"
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output/checkpoint directory"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="expera-ai",
        help="Project name for logging"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    # Load configuration
    config = load_config(args.config)
    if args.model_config:
        model_config = load_config(args.model_config)
        config["model"] = model_config.get("model", {})

    # Override variant
    config["model"].update(get_model_config(config, args.variant))

    # Device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        logger.info("Starting Expera AI pre-training")
        logger.info(f"Config: {args.config}")
        logger.info(f"Variant: {args.variant}")
        logger.info(f"Device: {device}")

    # Create model
    model = create_model(config, device, args.resume)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {num_params:,} params, {num_trainable:,} trainable")

    # Wrap in DDP for multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if is_main:
            logger.info(f"Distributed training with {world_size} GPUs")

    # Create optimizer and scheduler
    num_steps = config.get("training", {}).get("duration", {}).get("max_steps", 10000)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, num_steps)

    # Mixed precision scaler
    scaler = None
    if args.bf16 or config.get("training", {}).get("precision", {}).get("bf16"):
        scaler = GradScaler()
    elif args.fp16 or config.get("training", {}).get("precision", {}).get("fp16"):
        scaler = GradScaler()

    # Loss function
    criterion = LanguageModelingLoss(
        label_smoothing=config.get("training", {}).get("regularization", {}).get(
            "label_smoothing", 0.0
        ),
    )

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=args.output_dir,
        save_total_limit=config.get("training", {}).get("checkpointing", {}).get(
            "save_total_limit", 5
        ),
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        checkpoint_data = checkpoint_manager.load(model, optimizer, scheduler)
        start_step = checkpoint_data.get("step", 0)
        if is_main:
            logger.info(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    global_step = start_step
    total_loss = 0.0

    # Dummy data for demo (replace with real dataloader)
    batch_size = config.get("training", {}).get("batch", {}).get(
        "per_device_train_batch_size", 4
    )
    seq_len = config.get("model", {}).get("max_position_embeddings", 4096)
    vocab_size = config.get("model", {}).get("vocab_size", 50304)

    max_steps = config.get("training", {}).get("duration", {}).get("max_steps", 10000)
    grad_accum_steps = config.get("training", {}).get("strategy", {}).get(
        "gradient_accumulation_steps", 1
    )

    try:
        for step in range(max_steps):
            # Dummy batch (replace with real data)
            input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, min(seq_len, 512)))
            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            # Forward pass
            with autocast(enabled=scaler is not None):
                outputs = model(input_ids, use_cache=False)
                loss = criterion(outputs["logits"], labels)
                loss = loss / grad_accum_steps

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * grad_accum_steps

            # Optimizer step
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                max_grad_norm = config.get("training", {}).get("strategy", {}).get(
                    "max_grad_norm", 1.0
                )
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer and scheduler step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if is_main and global_step % 100 == 0:
                    avg_loss = total_loss / 100
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                    total_loss = 0.0

                # Checkpointing
                if is_main and global_step % 1000 == 0:
                    checkpoint_path = checkpoint_manager.save(
                        model,
                        optimizer,
                        scheduler,
                        step=global_step,
                    )
                    logger.info(f"Saved checkpoint: {checkpoint_path}")

    except KeyboardInterrupt:
        if is_main:
            logger.info("Training interrupted")

    finally:
        # Save final checkpoint
        if is_main:
            checkpoint_path = checkpoint_manager.save(
                model,
                optimizer,
                scheduler,
                step=global_step,
                name="final",
            )
            logger.info(f"Training complete! Final checkpoint: {checkpoint_path}")

    # Cleanup
    cleanup_distributed()

    return global_step


if __name__ == "__main__":
    main()