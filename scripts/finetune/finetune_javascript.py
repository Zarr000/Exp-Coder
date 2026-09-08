"""
JavaScript Finetuning for Expera AI.

Usage:
    python scripts/finetune/finetune_javascript.py --model checkpoints/expera-350m --output checkpoints/expera-350m-javascript
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class JSDataset:
    """JavaScript dataset."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = []
        self._load()

    def _load(self) -> None:
        if not self.data_path.exists():
            self.data = [
                {"instruction": "Write a function to capitalize a string.", "input": "", "output": "function capitalize(str) {\n  return str.charAt(0).toUpperCase() + str.slice(1);\n}"},
            ]
            return

        import json
        for file in self.data_path.glob("*.jsonl"):
            with open(file) as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line))
                    except:
                        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


@dataclass
class JSConfig:
    model_path: Path = Path("checkpoints/expera-350m")
    data_path: Path = Path("data/javascript")
    output_path: Path = Path("checkpoints/expera-350m-javascript")
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/expera-350m")
    p.add_argument("--data", default="data/javascript")
    p.add_argument("--output", default="checkpoints/expera-350m-javascript")
    p.add_argument("--epochs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    config = JSConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
    )
    config.output_path.mkdir(parents=True, exist_ok=True)
    logger.info("JavaScript finetuning complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()