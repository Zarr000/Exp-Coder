"""
Python Finetuning for Expera AI.

Specialized finetuning for Python code generation.

Usage:
    python scripts/finetune/finetune_python.py --model checkpoints/expera-350m --data data/python --output checkpoints/expera-350m-python
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class PythonDataset(Dataset):
    """Python code dataset."""

    def __init__(self, data_path: Path, max_length: int = 2048):
        self.data_path = data_path
        self.max_length = max_length
        self.data: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            self.data = [
                {
                    "instruction": "Write a function to calculate factorial.",
                    "input": "",
                    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
                },
                {
                    "instruction": "Write a function to reverse a string.",
                    "input": "",
                    "output": "def reverse_string(s):\n    return s[::-1]"
                },
            ]
            return

        for file in self.data_path.glob("*.jsonl"):
            with open(file, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except:
                        continue

        logger.info(f"Loaded {len(self.data)} Python examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def collate_fn(self, batch, tokenizer) -> dict:
        """Collate batch."""
        prompts = [item["instruction"] + "\n\n" + item.get("input", "") for item in batch]
        outputs = [item["output"] for item in batch]

        prompt_ids = tokenizer(prompts, truncation=True, max_length=self.max_length, padding=True)
        output_ids = tokenizer(outputs, truncation=True, max_length=self.max_length // 2, padding=True)

        return {
            "input_ids": torch.tensor(prompt_ids["input_ids"]),
            "attention_mask": torch.tensor(prompt_ids["attention_mask"]),
            "labels": torch.tensor(output_ids["input_ids"]),
        }


@dataclass
class PythonFinetuneConfig:
    model_path: Path = Path("checkpoints/expera-350m")
    data_path: Path = Path("data/python")
    output_path: Path = Path("checkpoints/expera-350m-python")

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-6
    max_length: int = 2048

    # Python-specific
    include_types: bool = True
    include_docs: bool = True


class PythonFinetuner:
    """Python-specific finetuner."""

    def __init__(self, model, config: PythonFinetuneConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Forward
            outputs = self.model(batch["input_ids"])
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )

            # Backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return {"avg_loss": total_loss / len(dataloader)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python finetuning")
    parser.add_argument("--model", default="checkpoints/expera-350m")
    parser.add_argument("--data", default="data/python")
    parser.add_argument("--output", default="checkpoints/expera-350m-python")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PythonFinetuneConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    dataset = PythonDataset(config.data_path, config.max_length)
    logger.info(f"Loaded {len(dataset)} Python examples")

    logger.info("Python finetuning complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()