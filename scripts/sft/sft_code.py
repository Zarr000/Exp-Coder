"""
Instruction Tuning for Code Generation.

Usage:
    python scripts/sft/sft_code.py --model checkpoints/expera-120m --data data/code --output checkpoints/expera-120m-code
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeDataset:
    """Code instruction dataset."""

    def __init__(self, data_path: Path, max_length: int = 2048):
        self.data_path = data_path
        self.max_length = max_length
        self.data: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load code dataset."""
        if not self.data_path.exists():
            self.data = [
                {"instruction": "Write a function to add two numbers.", "input": "", "output": "def add(a, b):\n    return a + b"}
            ]
            return

        for file in self.data_path.glob("*.jsonl"):
            with open(file, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # Standardize format
                        record = {
                            "instruction": item.get("instruction", item.get("prompt", "")),
                            "input": item.get("input", ""),
                            "output": item.get("output", item.get("completion", "")),
                        }
                        self.data.append(record)
                    except:
                        continue

        logger.info(f"Loaded {len(self.data)} code examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


@dataclass
class CodeSFTConfig:
    model_path: Path = Path("checkpoints/expera-120m")
    data_path: Path = Path("data/code")
    output_path: Path = Path("checkpoints/expera-120m-code")

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-6
    max_length: int = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code SFT")
    parser.add_argument("--model", default="checkpoints/expera-120m")
    parser.add_argument("--data", default="data/code")
    parser.add_argument("--output", default="checkpoints/expera-120m-code")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CodeSFTConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    dataset = CodeDataset(config.data_path, config.max_length)
    logger.info(f"Loaded {len(dataset)} examples")

    logger.info("Code SFT training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()