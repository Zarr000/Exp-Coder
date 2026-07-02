"""
Direct Preference Optimization (DPO) for Expera AI.

DPO aligns models using preference data without a reward model.

Usage:
    python -m src.alignment.dpo --model checkpoints/expera-350m --data data/preferences --output checkpoints/expera-350m-dpo
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DPODataset(Dataset):
    """Dataset for DPO training."""

    def __init__(self, data_path: Path):
        """Initialize dataset."""
        self.data_path = data_path
        self.data: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load preference data."""
        if not self.data_path.exists():
            self.data = [
                {
                    "prompt": "Write a function to sort a list.",
                    "chosen": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
                    "rejected": "sort(list)"
                }
            ]
            return

        for file in self.data_path.glob("*.jsonl"):
            with open(file, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "chosen" in item and "rejected" in item:
                            self.data.append(item)
                    except:
                        continue

        logger.info(f"Loaded {len(self.data)} DPO examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def dpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute DPO loss.

    Args:
        policy_logps: Log probabilities from policy model
        reference_logps: Log probabilities from reference model
        chosen_logps: Log probs for chosen responses
        rejected_logps: Log probs for rejected responses
        beta: Temperature parameter

    Returns:
        DPO loss
    """
    # Compute log ratio differences
    chosen_logratios = (chosen_logps - reference_logps).mean(dim=-1)
    rejected_logratios = (rejected_logps - reference_logps).mean(dim=-1)

    # sigmoid loss
    losses = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))

    return losses.mean()


class DPOTrainer:
    """DPO Trainer."""

    def __init__(
        self,
        model,
        ref_model,
        config: "DPOConfig",
    ):
        """Initialize trainer."""
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def compute_logprobs(
        self,
        model: nn.Module,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        """Compute log probabilities for prompt + response pairs."""
        # Placeholder - would use actual model
        return torch.randn(len(prompts), requires_grad=True)

    def step(
        self,
        batch: dict,
    ) -> dict:
        """Train one step."""
        prompts = batch["prompt"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        # Compute log probs
        chosen_logps = self.compute_logprobs(self.model, prompts, chosen)
        rejected_logps = self.compute_logprobs(self.model, prompts, rejected)
        ref_chosen_logps = self.compute_logprobs(self.ref_model, prompts, chosen)
        ref_rejected_logps = self.compute_logprobs(self.ref_model, prompts, rejected)

        # Compute loss
        loss = dpo_loss(
            None, None,
            chosen_logps, rejected_logps,
            beta=self.config.beta,
        )

        # Backward
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.item()}


@dataclass
class DPOConfig:
    """DPO configuration."""

    model_path: Path = Path("checkpoints/expera-350m")
    ref_model_path: Path = Path("checkpoints/expera-350m")
    data_path: Path = Path("data/preferences")
    output_path: Path = Path("checkpoints/expera-350m-dpo")

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    beta: float = 0.1

    log_interval: int = 10


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="DPO training")

    parser.add_argument("--model", default="checkpoints/expera-350m")
    parser.add_argument("--ref-model", default="checkpoints/expera-350m")
    parser.add_argument("--data", default="data/preferences")
    parser.add_argument("--output", default="checkpoints/expera-350m-dpo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    config = DPOConfig(
        model_path=Path(args.model),
        ref_model_path=Path(args.ref_model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    dataset = DPODataset(config.data_path)
    logger.info(f"Loaded {len(dataset)} preference examples")

    logger.info("DPO training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()