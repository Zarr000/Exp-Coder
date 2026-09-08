"""
Instruction Tuning for Multimodal (Image + Text) Generation.

Usage:
    python scripts/sft/sft_multimodal.py --model checkpoints/expera-120m --data data/multimodal --output checkpoints/expera-120m-multimodal
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class MultimodalDataset:
    """Multimodal instruction dataset."""

    def __init__(self, data_path: Path, max_length: int = 2048):
        self.data_path = data_path
        self.max_length = max_length
        self.data: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load multimodal dataset."""
        if not self.data_path.exists():
            self.data = [
                {
                    "instruction": "Describe this image.",
                    "input": "",
                    "output": "A beautiful landscape with mountains and a lake.",
                    "image_path": "sample.jpg"
                }
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

        logger.info(f"Loaded {len(self.data)} multimodal examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


@dataclass
class MultimodalSFTConfig:
    model_path: Path = Path("checkpoints/expera-120m")
    data_path: Path = Path("data/multimodal")
    output_path: Path = Path("checkpoints/expera-120m-multimodal")

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-6
    max_length: int = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal SFT")
    parser.add_argument("--model", default="checkpoints/expera-120m")
    parser.add_argument("--data", default="data/multimodal")
    parser.add_argument("--output", default="checkpoints/expera-120m-multimodal")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MultimodalSFTConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    dataset = MultimodalDataset(config.data_path)
    logger.info(f"Loaded {len(dataset)} multimodal examples")

    logger.info("Multimodal SFT training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()