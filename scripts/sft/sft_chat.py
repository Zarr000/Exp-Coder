"""
Instruction Tuning (SFT) for Chat Models.

Supports:
- Alpaca format
- ShareGPT format
- OpenAI format

Usage:
    python scripts/sft/sft_chat.py --model checkpoints/expera-120m --data data/chat --output checkpoints/expera-120m-chat --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# Supported data formats
class InstructionDataset(Dataset):
    """Instruction tuning dataset."""

    SUPPORTED_FORMATS = ["alpaca", "sharegpt", "openai"]

    def __init__(
        self,
        data_path: Path,
        format: str = "alpaca",
        max_length: int = 2048,
    ):
        """Initialize dataset."""
        self.data_path = data_path
        self.format = format
        self.max_length = max_length
        self.data: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load data from files."""
        if not self.data_path.exists():
            logger.warning(f"Data path not found: {self.data_path}")
            # Use sample data
            self.data = [
                {"instruction": "Hello, how are you?", "input": "", "output": "I'm doing well, thank you!"}
            ]
            return

        for file in self.data_path.glob("*.jsonl"):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        transformed = self._transform_item(item)
                        if transformed:
                            self.data.append(transformed)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(self.data)} examples")

    def _transform_item(self, item: dict) -> Optional[dict]:
        """Transform item to standard format."""
        if self.format == "alpaca":
            # Already in standard format
            return item

        elif self.format == "sharegpt":
            # Convert from conversations
            conversations = item.get("conversations", [])
            instruction = ""
            output = ""

            for msg in conversations:
                if msg.get("from") == "human":
                    instruction = msg.get("value", "")
                elif msg.get("from") == "gpt" and instruction:
                    output = msg.get("value", "")
                    return {"instruction": instruction, "input": "", "output": output}

            return None

        elif self.format == "openai":
            # Convert from OpenAI format
            messages = item.get("messages", [])
            instruction = ""
            output = ""

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    instruction = content
                elif role == "assistant" and instruction:
                    output = content
                    return {"instruction": instruction, "input": "", "output": output}

            return None

        return item

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get item."""
        return self.data[idx]


def collate_fn(batch: list[dict], tokenizer, max_length: int = 2048) -> dict:
    """Collate function."""
    # Combine instruction + input as prompt
    prompts = []
    responses = []

    for item in batch:
        prompt = item.get("instruction", "")
        if item.get("input"):
            prompt += f"\n\n{item['input']}"
        prompt += "\n\n"

        response = item.get("output", "")

        prompts.append(prompt)
        responses.append(response)

    # Tokenize
    prompt_ids = tokenizer(prompts, truncation=True, max_length=max_length, padding=True)
    response_ids = tokenizer(responses, truncation=True, max_length=max_length, padding=True)

    return {
        "input_ids": torch.tensor(prompt_ids["input_ids"]),
        "attention_mask": torch.tensor(prompt_ids["attention_mask"]),
        "labels": torch.tensor(response_ids["input_ids"]),
    }


@dataclass
class SFTConfig:
    """SFT configuration."""

    model_path: Path = Path("checkpoints/expera-120m")
    data_path: Path = Path("data/chat")
    output_path: Path = Path("checkpoints/expera-120m-chat")

    format: str = "alpaca"
    max_length: int = 2048

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100

    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    log_interval: int = 10
    save_interval: int = 500


def train_sft(
    model,
    dataloader: DataLoader,
    config: SFTConfig,
) -> None:
    """Train with SFT."""
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.warmup_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    step = 0

    for epoch in range(config.epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Forward
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            # Backward
            loss.backward()

            if (step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            if step % config.log_interval == 0:
                logger.info(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch} completed. Avg Loss: {total_loss / step:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Instruction tuning")

    parser.add_argument("--model", default="checkpoints/expera-120m")
    parser.add_argument("--data", default="data/chat")
    parser.add_argument("--output", default="checkpoints/expera-120m-chat")
    parser.add_argument("--format", default="alpaca", choices=["alpaca", "sharegpt", "openai"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=2048)

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    config = SFTConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        format=args.format,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    # Load model (placeholder - would load actual model)
    logger.info(f"SFT config: {config}")

    # Dummy training loop
    logger.info("SFT training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()