"""
Download Chat/Instruction Datasets for Expera AI Training.

Supports:
- OpenHermes
- UltraChat
- Capybara
- SlimOrca

Usage:
    python scripts/datasets/download_chat_datasets.py --dataset openhermes --output data/datasets/chat
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ChatDatasetConfig:
    """Configuration for chat dataset download."""

    dataset: str
    output_dir: Path
    max_samples: int = 100000
    format: str = "alpaca"  # alpaca, sharegpt, openai


class ChatDatasetDownloader:
    """Downloads and processes instruction datasets."""

    DATASET_REPOS = {
        "openhermes": "openchat/openhermes-2.5",
        "ultrachat": "OpenChat/ultrachat",
        "capybara": "openchat/openchat-capybara-5k",
        "slimorca": "OpenChat/slimorca",
        "codealpaca": "sev Rockwell/code_alpaca",
        " EvolInstruction": "microsoft/ EvolInstruction",
    }

    def __init__(self, output_dir: Path):
        """Initialize downloader."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_openhermes(
        self,
        max_samples: int = 100000,
        output_format: str = "alpaca",
    ) -> Path:
        """Download OpenHermes dataset."""
        logger.info("Downloading OpenHermes-2.5...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "openchat/openhermes-2.5",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "openhermes.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc="Processing OpenHermes"):
                # Convert to Alpaca format
                if output_format == "alpaca":
                    converted = {
                        "instruction": item.get("system_prompt", "") + "\n\n" + item.get("question", ""),
                        "input": "",
                        "output": item.get("answer", ""),
                    }
                else:
                    converted = item

                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_ultrachat(
        self,
        max_samples: int = 100000,
    ) -> Path:
        """Download UltraChat dataset."""
        logger.info("Downloading UltraChat...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "OpenChat/ultrachat",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "ultrachat.jsonl"

        # UltraChat has multi-turn conversations
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc="Processing UltraChat"):
                # Convert to single-turn Alpaca format
                messages = item.get("messages", [])
                if len(messages) < 2:
                    continue

                # Get first user-assistant pair
                for i in range(0, len(messages) - 1, 2):
                    if messages[i].get("role") == "user":
                        converted = {
                            "instruction": messages[i].get("content", ""),
                            "input": "",
                            "output": messages[i + 1].get("content", "") if i + 1 < len(messages) else "",
                        }
                        f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_codealpaca(
        self,
        max_samples: int = 100000,
    ) -> Path:
        """Download Code Alpaca dataset."""
        logger.info("Downloading Code Alpaca...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "sev Rockwell/code_alpaca",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "codealpaca.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc="Processing Code Alpaca"):
                converted = {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                }
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_slimorca(
        self,
        max_samples: int = 100000,
    ) -> Path:
        """Download SlimOrca dataset."""
        logger.info("Downloading SlimOrca...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "OpenChat/slimorca",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "slimorca.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc="Processing SlimOrca"):
                # Convert from conversations format
                conversations = item.get("conversations", [])
                if len(conversations) < 2:
                    continue

                instruction = ""
                output = ""

                for i, msg in enumerate(conversations):
                    if msg.get("from") == "human":
                        instruction = msg.get("value", "")
                    elif msg.get("from") == "gpt" and instruction:
                        output = msg.get("value", "")
                        converted = {
                            "instruction": instruction,
                            "input": "",
                            "output": output,
                        }
                        f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        count += 1
                        break

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download chat/instruction datasets for Expera AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="openhermes",
        choices=["openhermes", "ultrachat", "codealpaca", "slimorca"],
        help="Dataset to download",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/chat",
        help="Output directory",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum number of samples",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt", "openai"],
        help="Output format",
    )

    return parser.parse_args()


async def main() -> None:
    """Main function."""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = ChatDatasetDownloader(output_dir)

    if args.dataset == "openhermes":
        downloader.download_openhermes(args.max_samples, args.format)
    elif args.dataset == "ultrachat":
        downloader.download_ultrachat(args.max_samples)
    elif args.dataset == "codealpaca":
        downloader.download_codealpaca(args.max_samples)
    elif args.dataset == "slimorca":
        downloader.download_slimorca(args.max_samples)

    logger.info("Download complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())