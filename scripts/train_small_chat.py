#!/usr/bin/env python3
"""
Simple Chat Fine-tuning Script for Expera AI.

Fine-tunes the model on chat/instruction datasets.

Usage:
    python scripts/train_small_chat.py --model checkpoints/expera_coder_120m --data datasets/chat/basic_chat.json
    python scripts/train_small_chat.py --model checkpoints/expera_coder_120m --data datasets/chat/basic_chat.json --lora
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    """Chat dataset for instruction-response pairs."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """Load dataset."""
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSON
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item."""
        item = self.data[idx]

        # Format as conversation
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        # Build prompt: "User: {instruction}\nAssistant: {response}"
        text = f"User: {instruction}\nAssistant: {response}"

        # Tokenize
        ids = self.tokenizer.encode(text)

        # Truncate
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        return {
            "ids": ids,
            "text": text,
        }

    def collate_fn(self, batch):
        """Collate batch - simple padding to max length."""
        # Pad sequences to same length
        max_len = max(len(x["ids"]) for x in batch)

        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            ids = item["ids"]
            pad_len = max_len - len(ids)
            pad_token = self.tokenizer.pad_id() if self.tokenizer.pad_id() >= 0 else 0

            # Pad with pad_token (same for input and labels)
            input_ids.append(ids + [pad_token] * pad_len)
            labels.append(ids + [pad_token] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }


class ExperaTrainer:
    """Simple trainer for Expera AI."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        lr: float = 2e-4,
        epochs: int = 20,
        gradient_accum: int = 1,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        device: str = "cuda",
    ):
        """Initialize trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accum = gradient_accum
        self.device = device

        # Prepare dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=val_dataset.collate_fn,
            )

        # Setup LoRA if requested
        if use_lora:
            self._setup_lora(lora_rank, lora_alpha)
        else:
            # Full finetune - unfreeze all
            for param in model.parameters():
                param.requires_grad = True

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs * len(self.train_loader),
            eta_min=lr * 0.1,
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Checkpoint dir
        self.checkpoint_dir = "release"

    def _setup_lora(self, rank: int, alpha: int):
        """Setup LoRA."""
        logger.info(f"Setting up LoRA: rank={rank}, alpha={alpha}")

        # Freeze base model
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Add LoRA to attention layers
        for name, module in self.model.named_modules():
            if hasattr(module, "attention") or "attention" in name.lower():
                # Add LoRA adapters
                if hasattr(module, "q_proj"):
                    in_dim = module.q_proj.in_features
                    out_dim = module.q_proj.out_features
                    module.q_lora_A = nn.Linear(in_dim, rank, bias=False)
                    module.q_lora_B = nn.Linear(rank, out_dim, bias=False)
                    for p in module.q_lora_A.parameters():
                        p.requires_grad = True
                    for p in module.q_lora_B.parameters():
                        p.requires_grad = True

                if hasattr(module, "v_proj"):
                    in_dim = module.v_proj.in_features
                    out_dim = module.v_proj.out_features
                    module.v_lora_A = nn.Linear(in_dim, rank, bias=False)
                    module.v_lora_B = nn.Linear(rank, out_dim, bias=False)
                    for p in module.v_lora_A.parameters():
                        p.requires_grad = True
                    for p in module.v_lora_B.parameters():
                        p.requires_grad = True

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch - simplified batch handling."""
        self.model.train()
        total_loss = 0

        # Get all samples manually to avoid DataLoader issues
        all_samples = []
        for batch in self.train_loader:
            # Each batch is already processed - extract tensors
            all_samples.append(batch)

        num_batches = len(all_samples)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(all_samples):
            # Ensure 2D tensors
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Ensure proper shape (batch, seq)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            # Forward - model creates its own causal mask
            outputs = self.model(
                input_ids=input_ids,
            )

            # Simple causal language modeling loss
            logits = outputs["logits"]  # (batch, seq, vocab)
            batch_size, seq_len, vocab_size = logits.shape

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Simple flatten
            loss = self.loss_fn(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Scale by gradient accumulation
            loss = loss / self.gradient_accum

            # Backward
            loss.backward()

            # Update every gradient_accum steps
            if (batch_idx + 1) % self.gradient_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accum

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"loss={loss.item() * self.gradient_accum:.4f}"
                )

            self.scheduler.step()

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Validate."""
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids)
            logits = outputs["logits"]
            batch_size, seq_len, vocab_size = logits.shape

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = self.loss_fn(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
            total_loss += loss.item()

        num_batches = len(self.val_loader) if len(self.val_loader) > 0 else 1
        return total_loss / num_batches

    def train(self, save_every: int = 5):
        """Train full loop."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            logger.info(
                f"Epoch {epoch}/{self.epochs} - "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            # Save checkpoint
            if epoch % save_every == 0 or val_loss < best_loss:
                save_path = f"{self.checkpoint_dir}/expera_chat_{epoch}.pt"
                self.save_checkpoint(save_path)
                logger.info(f"Saved to {save_path}")
                best_loss = min(best_loss, val_loss)

        # Final save
        final_path = f"{self.checkpoint_dir}/expera_chat_final.pt"
        self.save_checkpoint(final_path)
        logger.info(f"Training complete! Saved to {final_path}")

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)


def train(
    model_path: str,
    data_path: str,
    output_dir: str = "release",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 2e-4,
    max_length: int = 512,
    use_lora: bool = False,
    gradient_accum: int = 1,
) -> None:
    """Main training function."""
    import sentencepiece as spm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer_path = Path(model_path) / "tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = "tokenizer/tokenizer.model"

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    # Load model
    from src.model.architecture import ExperaModel

    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model = ExperaModel(
        vocab_size=config.get("vocab_size", 200),
        hidden_size=config.get("hidden_size", 768),
        num_layers=config.get("num_hidden_layers", 12),
        num_heads=config.get("num_attention_heads", 8),
        num_kv_heads=config.get("num_key_value_heads", 4),
        max_position_embeddings=config.get("max_position_embeddings", 2048),
    )

    # Load checkpoint
    ckpt_path = Path(model_path) / "model.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle key prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[9:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    logger.info(f"Loaded model from {ckpt_path}")

    # Create dataset
    train_dataset = ChatDataset(data_path, tokenizer, max_length)

    # Split for validation
    val_size = min(10, len(train_dataset) // 5)
    val_dataset = None

    # Trainer
    trainer = ExperaTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        gradient_accum=gradient_accum,
        use_lora=use_lora,
        device=device,
    )

    trainer.checkpoint_dir = output_dir

    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    trainer.train(save_every=max(1, epochs // 4))


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Chat Fine-tuning")

    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data", type=str, required=True, help="Training data JSON")
    parser.add_argument("--output_dir", type=str, default="release", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--gradient_accum", type=int, default=1, help="Gradient accumulation")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    train(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        use_lora=args.lora,
        gradient_accum=args.gradient_accum,
    )


if __name__ == "__main__":
    main()