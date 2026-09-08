#!/usr/bin/env python3
"""Chat training script for Expera."""

import sys
import os
import json
import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.architecture.expera_model import ExperaModel


@dataclass
class ChatTrainingConfig:
    """Training configuration - Phase 13 RTX 4050 setup."""
    # Model
    model_path: str = "checkpoints/expera_coder_120m"
    output_dir: str = "release/expera_chat_120m"

    # Training (RTX 4050 config)
    epochs: int = 10
    batch_size: int = 2
    grad_accum: int = 8
    seq_len: int = 512
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Mixed precision
    fp16: bool = True
    max_batches: int = 100000  # Full training

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Resume
    resume: bool = True  # Resume from checkpoint if exists

    # Logging
    log_every: int = 10
    save_every: int = 1


class ChatDataset(Dataset):
    """Chat dataset."""

    def __init__(self, file_path: str, tokenizer, seq_len: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize
        ids = self.tokenizer.encode(text)
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]

        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Collate variable-length sequences."""
    # Getmax length
    max_len = max(len(x) for x in batch)
    max_len = min(max_len, 512)  # Cap at seq_len

    # Pad
    padded = []
    for x in batch:
        if len(x) < max_len:
            x = torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)])
        elif len(x) > max_len:
            x = x[:max_len]
        padded.append(x)

    return torch.stack(padded)


def main():
    """Main training function."""
    import sentencepiece as spm

    config = ChatTrainingConfig()

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(Path(config.model_path) / "tokenizer.model"))
    print(f"Loaded tokenizer: {tokenizer.GetPieceSize()} tokens")

    # Load config
    with open(Path(config.model_path) / "config.json") as f:
        model_config = json.load(f)

    # Create model
    model = ExperaModel(
        vocab_size=model_config.get("vocab_size", 200),
        hidden_size=model_config.get("hidden_size", 768),
        num_layers=model_config.get("num_hidden_layers", 12),
        num_heads=model_config.get("num_attention_heads", 8),
        num_kv_heads=model_config.get("num_key_value_heads", 4),
    )
    print(f"Created model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        for layer in model.layers:
            layer.gradient_checkpointing = True
        print("Enabled gradient checkpointing")

    # Load pretrained weights
    if (Path(config.model_path) / "model.pt").exists():
        state = torch.load(Path(config.model_path) / "model.pt", map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # Fix keys
        new_state = {}
        for k, v in state.items():
            key = k[9:] if k.startswith("_orig_mod.") else k
            new_state[key] = v
        model.load_state_dict(new_state, strict=False)
        print("Loaded pretrained weights")

    model.train()

    # Datasets
    train_dataset = ChatDataset(
        "data/processed/chat/train.jsonl",
        tokenizer,
        config.seq_len
    )
    valid_dataset = ChatDataset(
        "data/processed/chat/valid.jsonl",
        tokenizer,
        config.seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler with warmup
    total_steps = len(train_loader) * config.epochs // config.grad_accum
    warmup_steps = config.warmup_steps

    # Custom scheduler with warmup + cosine decay
    class WarmupCosineScheduler:
        def __init__(self, optimizer, warmup_steps, total_steps):
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]
            self.current_step = 0

        def step(self, step_num=None):
            if step_num is None:
                step_num = self.current_step
            else:
                self.current_step = step_num

            if step_num < self.warmup_steps:
                # Linear warmup
                scale = step_num / max(1, self.warmup_steps)
            else:
                # Cosine decay
                progress = (step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                scale = 0.5 * (1 + math.cos(math.pi * progress))

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = base_lr * scale

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.fp16 and torch.cuda.is_available() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    print(f"Using {device}")

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    if config.resume:
        checkpoint_files = sorted(output_dir.glob("chat_epoch_*.pt"))
        if checkpoint_files:
            latest_ckpt = checkpoint_files[-1]
            ckpt = torch.load(latest_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            step = ckpt.get("step", 0)
            print(f"Resumed from epoch {start_epoch}, step {step}")
        else:
            # Try to resume from final.pt
            final_path = output_dir / "final.pt"
            if final_path.exists():
                ckpt = torch.load(final_path, map_location="cpu")
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    print("Resumed from final.pt")
    best_loss = float("inf")
    step = start_epoch * len(train_loader) // config.grad_accum

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config.max_batches:
                break
            batch = batch.to(device)

            # Forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(batch, return_dict=True)
                    logits = outputs["logits"]
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch.view(-1),
                        ignore_index=0
                    )
            else:
                outputs = model(batch, return_dict=True)
                logits = outputs["logits"]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch.view(-1),
                    ignore_index=0
                )

            # Scale loss
            loss = loss / config.grad_accum

            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * config.grad_accum

            # Update weights
            if (batch_idx + 1) % config.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % config.log_every == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {epoch_loss / config.log_every:.4f}")
                    epoch_loss = 0

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = output_dir / f"chat_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved {ckpt_path}")

            # Save best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = output_dir / "best.pt"
                torch.save({"model_state_dict": model.state_dict()}, best_path)
                print(f"Saved best: {best_loss:.4f}")

    # Save final
    final_path = output_dir / "final.pt"
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    print(f"Training complete! Saved to {final_path}")


if __name__ == "__main__":
    main()