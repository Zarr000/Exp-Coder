"""
Training script for Expera AI.

Usage:
    python scripts/train.py --config config/training_config.yaml

This script handles:
- Loading and parsing configuration
- Initializing model, optimizer, scheduler
- Training loop with logging and checkpointing
- Evaluation and metrics tracking
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.model.architecture import ExperaModel
from src.training.loss import LanguageModelingLoss
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import setup_logger
from src.utils.metrics import compute_metrics


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Expera AI")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def get_model_config(model_config_path: str, variant: str = "base") -> dict:
    """Load model configuration."""
    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get variant configuration
    variant_config = config.get("model", {}).get("architecture", {})
    
    # Override with specific variant if available
    variants = config.get("variants", {})
    if variant in variants:
        variant_config.update(variants[variant])
    
    return variant_config


def train():
    """Main training function."""
    args = parse_args()
    
    # Load configurations
    train_config = load_config(args.config)
    model_config = get_model_config(args.model_config, variant="tiny")  # Start small
    
    # Setup logging
    logger = setup_logger(
        name="expera",
        log_level=train_config.get("training", {}).get("logging", {}).get("level", "INFO"),
        log_file=os.path.join(
            train_config.get("training", {}).get("logging_dir", "./logs"),
            "training.log",
        ),
    )
    logger.info("Starting Expera AI training")
    logger.info(f"Model config: {model_config}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = ExperaModel(
        vocab_size=model_config.get("vocab_size", 50304),
        hidden_size=model_config.get("hidden_size", 256),
        num_layers=model_config.get("num_layers", 4),
        num_heads=model_config.get("num_heads", 4),
        intermediate_size=model_config.get("intermediate_size", 1024),
        max_position_embeddings=model_config.get("max_position_embeddings", 1024),
        activation=model_config.get("activation", "gelu"),
        dropout=model_config.get("dropout", 0.1),
        use_rope=model_config.get("advanced", {}).get("use_rope", True),
        rope_theta=model_config.get("advanced", {}).get("rope_theta", 10000.0),
    ).to(device)
    
    logger.info(f"Model initialized with {model.get_num_params()['total']:,} total parameters")
    logger.info(f"Trainable parameters: {model.get_num_params()['trainable']:,}")
    
    # Initialize optimizer
    optimizer = get_optimizer(
        model.named_parameters(),
        train_config.get("training", {}).get("optimizer", {}),
    )
    
    # Initialize scheduler
    scheduler = get_scheduler(
        optimizer,
        train_config.get("training", {}).get("scheduler", {}),
    )
    
    # Initialize loss
    criterion = LanguageModelingLoss(
        label_smoothing=train_config.get("training", {}).get("regularization", {}).get("label_smoothing", 0.0),
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=train_config.get("training", {}).get("output_dir", "./checkpoints"),
        save_total_limit=train_config.get("training", {}).get("checkpointing", {}).get("save_total_limit", 5),
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    start_epoch = 0
    if args.resume:
        checkpoint_data = checkpoint_manager.load(model, optimizer, scheduler, args.resume)
        start_step = checkpoint_data["step"]
        start_epoch = checkpoint_data["epoch"]
        logger.info(f"Resumed from checkpoint at step {start_step}, epoch {start_epoch}")
    
    # Training configuration
    batch_size = train_config.get("training", {}).get("batch", {}).get("per_device_train_batch_size", 4)
    max_steps = train_config.get("training", {}).get("duration", {}).get("max_steps", -1)
    num_epochs = train_config.get("training", {}).get("duration", {}).get("num_train_epochs", 3)
    grad_accum_steps = train_config.get("training", {}).get("strategy", {}).get("gradient_accumulation_steps", 1)
    max_grad_norm = train_config.get("training", {}).get("strategy", {}).get("max_grad_norm", 1.0)
    log_steps = train_config.get("training", {}).get("logging", {}).get("logging_steps", 100)
    save_steps = train_config.get("training", {}).get("checkpointing", {}).get("save_steps", 5000)
    
    logger.info("Starting training loop")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"Effective batch size: {batch_size * grad_accum_steps}")
    
    # Training loop
    model.train()
    global_step = start_step
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Create dummy data for testing
    vocab_size = model_config.get("vocab_size", 50304)
    seq_len = model_config.get("max_position_embeddings", 1024)
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx in range(100):  # Placeholder: replace with actual dataloader
            # Dummy input for testing
            input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, min(seq_len, 512))).to(device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids, use_cache=False, return_dict=True)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % log_steps == 0:
                    avg_loss = total_loss / log_steps
                    metrics = compute_metrics(logits.detach(), labels, avg_loss)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step} | Loss: {avg_loss:.4f} | "
                        f"PPL: {metrics['perplexity']:.2f} | "
                        f"Acc: {metrics['accuracy']:.2%} | "
                        f"LR: {lr:.2e}"
                    )
                    total_loss = 0.0
                
                # Checkpointing
                if global_step % save_steps == 0:
                    checkpoint_path = checkpoint_manager.save(
                        model, optimizer, scheduler,
                        step=global_step,
                        epoch=epoch,
                        metrics={"loss": avg_loss},
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Check max steps
                if max_steps > 0 and global_step >= max_steps:
                    logger.info(f"Reached max steps {max_steps}")
                    break
                    
        # End of epoch
        logger.info(f"Completed epoch {epoch + 1}")
        
        if max_steps > 0 and global_step >= max_steps:
            break
    
    # Save final checkpoint
    checkpoint_path = checkpoint_manager.save(
        model, optimizer, scheduler,
        step=global_step,
        epoch=num_epochs,
        metrics={"loss": total_loss / max(1, global_step)},
        name="final",
    )
    logger.info(f"Training complete! Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()