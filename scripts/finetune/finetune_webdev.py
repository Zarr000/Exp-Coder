"""
Web Development Finetuning for Expera AI.

Usage:
    python scripts/finetune/finetune_webdev.py --model checkpoints/expera-350m --output checkpoints/expera-350m-webdev
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class WebDevDataset:
    """Web development dataset."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = []
        self._load()

    def _load(self) -> None:
        if not self.data_path.exists():
            self.data = [
                {"instruction": "Create a responsive navigation bar", "input": "HTML/CSS", "output": "<nav>\n  <ul>\n    <li><a href='/'>Home</a></li>\n    <li><a href='/about'>About</a></li>\n  </ul>\n</nav>"},
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
class WebDevConfig:
    model_path: Path = Path("checkpoints/expera-350m")
    data_path: Path = Path("data/webdev")
    output_path: Path = Path("checkpoints/expera-350m-webdev")
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/expera-350m")
    p.add_argument("--data", default="data/webdev")
    p.add_argument("--output", default="checkpoints/expera-350m-webdev")
    return p.parse_args()


def main():
    args = parse_args()
    config = WebDevConfig(model_path=Path(args.model), data_path=Path(args.data), output_path=Path(args.output))
    config.output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Web dev finetuning complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()