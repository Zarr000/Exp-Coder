"""
Pretraining Script for Expera 1B Model.

Usage:
    torchrun --nproc_per_node=8 scripts/train/train_1b.py --data data/mixture --output checkpoints/expera-1b --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class GPT1BConfig:
    """Configuration for 1B model."""

    vocab_size: int = 32000
    embedding_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    hidden_dim: int = 8192
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class GPT1B(nn.Module):
    """1B parameter GPT model."""

    def __init__(self, config: GPT1BConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_dim)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.output_norm = nn.RMSNorm(config.embedding_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, position_ids=None) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

        for block in self.blocks:
            x = block(x, attention_mask)

        return self.lm_head(self.output_norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPT1BConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.embedding_dim, config.num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.norm1 = nn.RMSNorm(config.embedding_dim, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.embedding_dim, eps=config.rms_norm_eps)

    def forward(self, x, attention_mask=None) -> torch.Tensor:
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x


class TextDataset(Dataset):
    def __init__(self, data_path: Path, seq_length: int = 4096, vocab_size: int = 32000):
        self.data_path = data_path
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data: list[int] = []
        self._load_data()

    def _load_data(self) -> None:
        all_tokens = []
        if self.data_path.exists():
            for file in self.data_path.glob("*.jsonl"):
                try:
                    with open(file, "r") as f:
                        for line in f:
                            item = json.loads(line)
                            text = item.get("content", item.get("code", ""))
                            tokens = [ord(c) % self.vocab_size for c in text[:2000]]
                            all_tokens.extend(tokens)
                except:
                    continue

        self.data = all_tokens if all_tokens else [i % self.vocab_size for i in range(1000)]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.data[idx:idx + self.seq_length]
        labels = self.data[idx + 1:idx + self.seq_length + 1]

        if len(input_ids) < self.seq_length:
            input_ids = input_ids + [0] * (self.seq_length - len(input_ids))
        if len(labels) < self.seq_length:
            labels = labels + [-100] * (self.seq_length - len(labels))

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}


@dataclass
class TrainingConfig:
    data_path: Path = Path("data/mixture")
    output_path: Path = Path("checkpoints/expera-1b")
    seq_length: int = 4096
    vocab_size: int = 32000

    epochs: int = 3
    batch_size: int = 32
    gradient_accumulation: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    precision: str = "bf16"
    use_gradient_checkpointing: bool = True

    log_interval: int = 10
    save_interval: int = 500
    resume_from: Optional[Path] = None


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    mask = labels != -100
    labels = labels[mask]
    logits = logits[mask]
    if len(labels) == 0:
        return logits.sum()
    return nn.functional.cross_entropy(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mixture")
    parser.add_argument("--output", default="checkpoints/expera-1b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_length=args.seq_length,
        precision=args.precision,
        resume_from=Path(args.resume) if args.resume else None,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model = GPT1B(GPT1BConfig())
    model = model.to(torch.cuda.current_device())

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()])

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    scaler = GradScaler() if config.precision in ("bf16", "fp16") else None

    dataset = TextDataset(config.data_path, config.seq_length, config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        step = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(torch.cuda.current_device(), non_blocking=True)
            labels = batch["labels"].to(torch.cuda.current_device(), non_blocking=True)

            if config.precision == "bf16":
                with autocast(dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = compute_loss(logits, labels) / config.gradient_accumulation
            else:
                logits = model(input_ids)
                loss = compute_loss(logits, labels) / config.gradient_accumulation

            if config.precision in ("bf16", "fp16"):
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.gradient_accumulation == 0:
                if config.precision in ("bf16", "fp16"):
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                if config.precision in ("bf16", "fp16"):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            total_loss += loss.item() * config.gradient_accumulation
            step += 1

            if step % config.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch} Step {step} Loss: {loss.item() * config.gradient_accumulation:.4f} LR: {lr:.2e}")

            if step % config.save_interval == 0:
                path = config.output_path / f"checkpoint_{epoch}_{step}.pt"
                torch.save({"epoch": epoch, "step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, path)
                logger.info(f"Saved: {path}")

        scheduler.step()
        logger.info(f"Epoch {epoch} completed. Avg Loss: {total_loss / step:.4f}")

    logger.info("Training complete!")


if __name__ == "__main__":
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
    main()