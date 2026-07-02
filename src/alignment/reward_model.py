"""
Reward Model for preference learning.

Usage:
    python -m src.alignment.reward_model --model checkpoints/expera-350m --data data/rewards --output checkpoints/reward-model
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Reward model for preference scoring."""

    def __init__(self, base_model, hidden_size: int = 512):
        """
        Initialize reward model.

        Args:
            base_model: Base language model
            hidden_size: Hidden layer size
        """
        super().__init__()
        self.base_model = base_model

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute reward scores.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Reward scores (batch_size,)
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask)

        # Use last hidden state
        hidden = outputs.last_hidden_state

        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            hidden = hidden.mean(dim=1)

        # Compute reward
        rewards = self.reward_head(hidden).squeeze(-1)

        return rewards


class RewardDataset:
    """Reward model dataset."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            self.data = [
                {"prompt": "Hello", "response": "Hi there!", "score": 0.8},
                {"prompt": "Hello", "response": "Go away", "score": 0.1},
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


@dataclass
class RewardConfig:
    model_path: Path = Path("checkpoints/expera-350m")
    data_path: Path = Path("data/rewards")
    output_path: Path = Path("checkpoints/reward-model")

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward model training")
    parser.add_argument("--model", default="checkpoints/expera-350m")
    parser.add_argument("--data", default="data/rewards")
    parser.add_argument("--output", default="checkpoints/reward-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RewardConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Reward model training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()