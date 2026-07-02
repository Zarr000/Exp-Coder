#!/usr/bin/env python3
"""
SFT Training Script for Expera AI.

Fine-tunes a pre-trained model on chat/instruction data using our internal trainer.
Designed for consumer GPUs (RTX 4050).

Usage:
    python scripts/train_sft.py --model checkpoints/final/ --data data/processed/chat/train.jsonl
    python scripts/train_sft.py --model checkpoints/expera_small --data data/processed/chat/ --output checkpoints/chat_sft
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.architecture.expera_model import ExperaModel
from src.training.trainer import Trainer, TrainerConfig
from src.training.loss import LanguageModelingLoss
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
# Use a simple SentencePiece wrapper
import sentencepiece as spm


class SimpleTokenizer:
    """Simple wrapper for SentencePiece tokenizer."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self._vocab_size = self.sp.GetPieceSize()

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer from path."""
        # Check for model file
        model_path = Path(path)
        if not model_path.exists():
            # Try parent
            model_path = model_path.parent

        for mp in [model_path / "tokenizer.model", model_path]:
            if mp.exists() and mp.is_file():
                return cls(str(mp))

        # Find in checkpoints
        for ckpt_dir in Path("checkpoints").glob("*/"):
            mp = ckpt_dir / "tokenizer.model"
            if mp.exists():
                return cls(str(mp))

        raise FileNotFoundError(f"No tokenizer found at {path}")

    def encode(self, text: str) -> List[int]:
        """Encode text to ids."""
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        """Decode ids to text."""
        return self.sp.DecodeIds(ids)

    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size

    def eos_id(self) -> int:
        """Get EOS token id."""
        return self.sp.eos_id()

    def pad_id(self) -> int:
        """Get PAD token id."""
        return self.sp.pad_id()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Chat template (simple format without Jinja2)
CHAT_TEMPLATE = "System: You are Expera AI, a helpful AI assistant.\n\n"


class ChatDataset(Dataset):
    """Chat dataset for SFT training."""

    def __init__(
        self,
        file_path: Path,
        tokenizer: SimpleTokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = self._load_records(file_path)
        logger.info(f"Loaded {len(self.records)} chat records")

    def _load_records(self, file_path: Path) -> List[Dict]:
        """Load records from JSONL."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item."""
        record = self.records[idx]

        # Format as chat
        text = self._format_chat(record)

        # Tokenize
        ids = self.tokenizer.encode(text)

        # Truncate
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        # Create labels (for SFT: labels = input_ids, mask user parts)
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _format_chat(self, record: Dict) -> str:
        """Format conversation as text."""
        conversations = record.get("conversations", [])

        output = "System: You are Expera AI, a helpful AI assistant.\n\n"

        for msg in conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if not content:
                continue

            if role == "user":
                output += f"User: {content}\n\n"
            elif role == "assistant":
                output += f"Assistant: {content}\n\n"

        return output.strip()


def load_model(
    model_path: str,
    device: str = "cuda",
) -> nn.Module:
    """Load pre-trained model."""
    model_path = Path(model_path)

    # Try to load from checkpoint
    config_path = model_path / "config.json"
    if not config_path:
        config_path = model_path / ".." / "config.json"

    # Default config for small model
    if not config_path.exists():
        config_path = None

    # Load config
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default small config
        config = {
            "vocab_size": 200,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 1024,
        }

    # Create model
    model = ExperaModel(
        vocab_size=config.get("vocab_size", 200),
        hidden_size=config.get("hidden_size", 768),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 12),
        intermediate_size=config.get("intermediate_size", 3072),
        max_position_embeddings=config.get("max_position_embeddings", 1024),
    )

    # Try to load weights
    checkpoint_path = model_path / "pytorch_model.bin"
    if not checkpoint_path.exists():
        checkpoint_path = model_path / "model.safetensors"
    if not checkpoint_path.exists():
        checkpoint_path = model_path / ".." / "pytorch_model.bin"

    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    return model


def load_tokenizer(tokenizer_path: str) -> SimpleTokenizer:
    """Load tokenizer."""
    tokenizer_path = Path(tokenizer_path)

    # Try loading from directory with config.json + vocab.json
    config_file = tokenizer_path / "config.json"
    vocab_file = tokenizer_path / "vocab.json"

    if config_file.exists() and vocab_file.exists():
        return SimpleTokenizer.load(str(tokenizer_path))

    # Try SentencePiece-style model file
    model_file = tokenizer_path / "tokenizer.model"
    if model_file.exists():
        return SimpleTokenizer.load(str(tokenizer_path))

    # Fallback - try finding in checkpoints
    for ckpt_dir in Path("checkpoints").glob("*/"):
        config_file = ckpt_dir / "config.json"
        vocab_file = ckpt_dir / "vocab.json"
        if config_file.exists() and vocab_file.exists():
            logger.info(f"Using tokenizer from {ckpt_dir}")
            return SimpleTokenizer.load(str(ckpt_dir))

    # Try SentencePiece-style
    for ckpt_dir in Path("checkpoints").glob("*/"):
        model_file = ckpt_dir / "tokenizer.model"
        if model_file.exists():
            logger.info(f"Using tokenizer from {model_file}")
            return SimpleTokenizer.load(str(ckpt_dir))

    raise FileNotFoundError("No tokenizer found")


def create_dataloader(
    file_path: Path,
    tokenizer: SimpleTokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader."""
    dataset = ChatDataset(file_path, tokenizer, max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def train(
    model_path: str,
    train_file: str,
    val_file: Optional[str] = None,
    output_dir: str = "checkpoints/sft",
    tokenizer_path: Optional[str] = None,
    max_length: int = 1024,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    max_steps: int = 10000,
    warmup_steps: int = 500,
    max_grad_norm: float = 1.0,
    save_every: int = 1000,
    log_every: int = 25,
) -> None:
    """Run SFT training."""
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer
    if tokenizer_path:
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer = load_tokenizer(model_path)

    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size()}")

    # Load model
    model = load_model(model_path, device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create dataloaders
    train_loader = create_dataloader(
        Path(train_file),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )
    logger.info(f"Train batches: {len(train_loader)}")

    val_loader = None
    if val_file:
        val_loader = create_dataloader(
            Path(val_file),
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
        )
        logger.info(f"Val batches: {len(val_loader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=learning_rate * 0.1,
    )

    # Loss
    loss_fn = LanguageModelingLoss()

    # Create trainer config
    config = TrainerConfig(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_amp=True,
        val_dataloader=val_loader,
        save_dir=output_dir,
        save_every=save_every,
        log_every=log_every,
    )

    # Train
    trainer = Trainer(config)
    trainer.train()

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "pytorch_model.bin")

    # Save config
    model_config = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(model_config, f)

    logger.info(f"Model saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="SFT Training for Expera AI")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/final",
        help="Pre-trained model path"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (optional)"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Training data JSONL file"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        help="Validation data JSONL file"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/sft",
        help="Output directory"
    )

    # Training params
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Max training steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save every N steps"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=25,
        help="Log every N steps"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    train(
        model_path=args.model,
        train_file=args.data,
        val_file=args.val_data,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        save_every=args.save_every,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()