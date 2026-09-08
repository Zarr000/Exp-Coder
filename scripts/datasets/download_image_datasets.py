"""
Download Image-Text Datasets for Expera AI Training.

NOTE: This script prepares metadata for image generation training/conditioning.
 Actual image downloading requires significant storage.

Supports:
- LAION-Aesthetics
- JourneyDB
- DiffusionDB
- Pix2Struct

Usage:
    python scripts/datasets/download_image_datasets.py --dataset journeydb
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageDatasetConfig:
    """Configuration for image dataset."""

    dataset: str
    output_dir: Path
    max_samples: int = 100000
    min_resolution: int = 256


class ImageDatasetDownloader:
    """Downloads metadata for image-text datasets."""

    DATASET_REPOS = {
        "laion-aesthetics": "laion/laion-aesthetic",
        "journeydb": "JourneyDB/JourneyDB",
        "diffusiondb": "poloclub/diffusiondb",
        "pix2struct": "google/pix2struct",
    }

    def __init__(self, output_dir: Path):
        """Initialize downloader."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_journeydb(
        self,
        max_samples: int = 100000,
    ) -> Path:
        """Download JourneyDB metadata (prompts + generations info)."""
        logger.info("Downloading JourneyDB metadata...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "JourneyDB/JourneyDB",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "journeydb.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                # Extract prompt and metadata
                record = {
                    "prompt": item.get("prompt", ""),
                    "prompt_enhanced": item.get("prompt_enhanced", ""),
                    "model": item.get("model", ""),
                    "resolution": item.get("resolution", ""),
                    "seed": item.get("seed", 0),
                    "steps": item.get("steps", 0),
                    "cfg_scale": item.get("cfg_scale", 0),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_diffusiondb(
        self,
        max_samples: int = 100000,
    ) -> Path:
        """Download DiffusionDB prompts."""
        logger.info("Downloading DiffusionDB metadata...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "poloclub/diffusiondb",
            "2m_samples",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "diffusiondb.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                record = {
                    "prompt": item.get("prompt", ""),
                    "seed": item.get("seed", 0),
                    "steps": item.get("steps", 0),
                    "cfg_scale": item.get("cfg_scale", 0),
                    "width": item.get("width", 512),
                    "height": item.get("height", 512),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_laion_aesthetics(
        self,
        max_samples: int = 100000,
    ) -> Path:
        """Download LAION aesthetic scores (subset of LAION with aesthetic ratings)."""
        logger.info("Downloading LAION-Aesthetics metadata...")

        # Note: LAION is very large, we download metadata only
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        try:
            ds = load_dataset(
                "laion/laion-aesthetic",
                "laion_aesthetic_12m",
                split="train",
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Could not load full dataset: {e}")
            # Try alternative
            ds = load_dataset(
                "laion/laion-aesthetic",
                split="train",
                trust_remote_code=True,
                streaming=True,
            )
            ds = list(ds.take(max_samples))

        output_file = self.output_dir / "laion_aesthetics.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                record = {
                    "url": item.get("URL", ""),
                    "text": item.get("text", ""),
                    "aesthetic_score": item.get("aesthetic_score", 0),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_pix2struct(
        self,
        max_samples: int = 50000,
    ) -> Path:
        """Download Pix2Struct for image-to-text training."""
        logger.info("Downloading Pix2Struct...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "google/pix2struct",
            "textcaps",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "pix2struct.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                record = {
                    "image_id": item.get("image_id", ""),
                    "text": item.get("text", ""),
                    "cap_id": item.get("cap_id", ""),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_samples:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download image-text datasets metadata",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="journeydb",
        choices=["journeydb", "diffusiondb", "laion-aesthetics", "pix2struct"],
        help="Dataset to download",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/image",
        help="Output directory",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum samples",
    )

    return parser.parse_args()


async def main() -> None:
    """Main function."""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = ImageDatasetDownloader(output_dir)

    if args.dataset == "journeydb":
        downloader.download_journeydb(args.max_samples)
    elif args.dataset == "diffusiondb":
        downloader.download_diffusiondb(args.max_samples)
    elif args.dataset == "laion-aesthetics":
        downloader.download_laion_aesthetics(args.max_samples)
    elif args.dataset == "pix2struct":
        downloader.download_pix2struct(args.max_samples)

    logger.info("Download complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())