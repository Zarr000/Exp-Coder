"""
Synthetic Preference Generation for Expera AI.

Generates preference data from model outputs and heuristics.

Usage:
    python -m src.alignment.synthetic_preferences --model checkpoints/expera-350m --prompts data/prompts --output data/preferences
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Preference scoring heuristics
def score_correctness(response: str, expected: str) -> float:
    """Score correctness of code."""
    # Simple heuristic: check if key elements present
    if expected.strip() == response.strip():
        return 1.0

    # Check for key patterns
    score = 0.5
    for token in expected.split()[:5]:
        if token in response:
            score += 0.1

    return min(1.0, score)


def score_safety(response: str) -> float:
    """Score safety of response."""
    dangerous = ["delete", "drop", "rm -rf", "exec(", "eval(", "system("]
    for word in dangerous:
        if word in response.lower():
            return 0.0
    return 1.0


def score_helpfulness(response: str) -> float:
    """Score helpfulness."""
    if len(response) < 10:
        return 0.0
    if response.startswith("I don't know") or response.startswith("Sorry"):
        return 0.3
    return min(1.0, len(response) / 500)


def generate_preferences(
    model,
    prompts: list[str],
    num_samples: int = 2,
) -> list[dict]:
    """
    Generate synthetic preference data.

    Args:
        model: Language model
        prompts: List of prompts
        num_samples: Number of samples per prompt

    Returns:
        List of preference pairs
    """
    preferences = []

    for prompt in prompts:
        # Generate responses (placeholder)
        responses = [f"Response {i} for: {prompt}" for i in range(num_samples)]

        # Score responses
        scores = [0.5 + random.random() * 0.5 for _ in responses]

        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        if scores[sorted_indices[0]] > scores[sorted_indices[-1]]:
            preferences.append({
                "prompt": prompt,
                "chosen": responses[sorted_indices[0]],
                "rejected": responses[sorted_indices[-1]],
            })

    return preferences


def self_play_preferences(
    model,
    prompts: list[str],
    num_generations: int = 4,
) -> list[dict]:
    """
    Generate preference data via self-play.

    Generate multiple responses, pick best and worst based on heuristics.
    """
    preferences = []

    for prompt in prompts:
        # Generate multiple responses
        generations = []
        for _ in range(num_generations):
            response = f"Generated response for: {prompt}"
            generations.append(response)

        # Score each
        scores = [
            score_correctness(gen, "") * 0.4 +
            score_safety(gen) * 0.3 +
            score_helpfulness(gen) * 0.3
            for gen in generations
        ]

        # Find best and worst
        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))

        preferences.append({
            "prompt": prompt,
            "chosen": generations[best_idx],
            "rejected": generations[worst_idx],
        })

    return preferences


@dataclass
class SyntheticConfig:
    model_path: Path = Path("checkpoints/expera-350m")
    prompts_path: Path = Path("data/prompts")
    output_path: Path = Path("data/preferences")

    num_samples: int = 4
    method: str = "self-play"  # self-play, generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic preferences")
    parser.add_argument("--model", default="checkpoints/expera-350m")
    parser.add_argument("--prompts", default="data/prompts")
    parser.add_argument("--output", default="data/preferences")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--method", default="self-play", choices=["self-play", "generate"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SyntheticConfig(
        model_path=Path(args.model),
        prompts_path=Path(args.prompts),
        output_path=Path(args.output),
        num_samples=args.samples,
        method=args.method,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = []
    if config.prompts_path.exists():
        for file in config.prompts_path.glob("*.jsonl"):
            with open(file, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        prompt = item.get("prompt", item.get("instruction", ""))
                        prompts.append(prompt)
                    except:
                        continue

    if not prompts:
        prompts = ["Write a function to add two numbers.", "Hello, how are you?"]

    # Generate preferences
    preferences = []
    for prompt in prompts[:100]:
        prefs = self_play_preferences(None, [prompt], config.num_samples)
        preferences.extend(prefs)

    # Save
    output_file = config.output_path / "preferences.jsonl"
    with open(output_file, "w") as f:
        for pref in preferences:
            f.write(json.dumps(pref) + "\n")

    logger.info(f"Saved {len(preferences)} preferences to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()