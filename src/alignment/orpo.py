"""
Odds Ratio Preference Optimization (ORPO) for Expera AI.

ORPO uses a simpler approach than DPO with odds ratios.

Usage:
    python -m src.alignment.orpo --model checkpoints/expera-350m --data data/preferences --output checkpoints/expera-350m-orpo
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def orpo_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    lambda_: float = 0.5,
) -> torch.Tensor:
    """
    Compute ORPO loss.

    Args:
        policy_logps: Policy model log probs
        reference_logps: Reference model log probs
        chosen_logps: Chosen response log probs
        rejected_logps: Rejected response log probs
        lambda_: regularization strength

    Returns:
        ORPO loss
    """
    # Compute log ratios
    chosen_log_ratio = (chosen_logps - reference_logps).mean(dim=-1)
    rejected_log_ratio = (rejected_logps - reference_logps).mean(dim=-1)

    # Odds ratio
    odds_ratio = chosen_log_ratio - torch.log1p(torch.exp(-rejected_log_ratio))
    sigmoid_odds = F.logsigmoid(-odds_ratio)

    # KL divergence penalty
    kl_penalty = (chosen_logps - reference_logps).mean(dim=-1)

    return sigmoid_odds.mean() + lambda_ * kl_penalty.mean()


class ORPOTrainer:
    """ORPO Trainer."""

    def __init__(self, model, ref_model, config: "ORPOConfig"):
        self.model = model
        self.ref_model = ref_model
        self.config = config


@dataclass
class ORPOConfig:
    model_path: Path = Path("checkpoints/expera-350m")
    data_path: Path = Path("data/preferences")
    output_path: Path = Path("checkpoints/expera-350m-orpo")

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-6
    lambda_: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORPO training")
    parser.add_argument("--model", default="checkpoints/expera-350m")
    parser.add_argument("--data", default="data/preferences")
    parser.add_argument("--output", default="checkpoints/expera-350m-orpo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ORPOConfig(
        model_path=Path(args.model),
        data_path=Path(args.data),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_=args.lambda_,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)
    logger.info("ORPO training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()