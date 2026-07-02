"""
Download Code Datasets for Expera AI Training.

Supports:
- The Stack
- StarCoderData
- CodeSearchNet

Usage:
    python scripts/datasets/download_code_datasets.py --dataset stack --languages python javascript
    python scripts/datasets/download_code_datasets.py --dataset starcoder --output data/starcoder
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset download."""

    dataset: str
    output_dir: Path
    languages: list[str] = field(default_factory=lambda: ["python", "javascript", "typescript", "rust", "go"])
    max_size_gb: float = 100.0
    deduplicate: bool = True
    quality_filter: bool = True


class CodeDatasetDownloader:
    """Downloads and processes code datasets."""

    # Dataset URLs (using HuggingFace datasets)
    DATASET_REPOS = {
        "stack": "bigcode/the-stack",
        "starcoder": "bigcode/starcoderdata",
        "codesearchnet": "codeparrot/codesearchnet",
    }

    def __init__(self, output_dir: Path):
        """Initialize downloader."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def download_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        quality_filter: bool = True,
    ) -> Path:
        """Download dataset using HuggingFace datasets library."""
        try:
            from datasets import load_dataset

            logger.info(f"Downloading {dataset_name} from HuggingFace...")

            # Map languages to file extensions
            lang_map = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "rust": ".rs",
                "go": ".go",
                "java": ".java",
                "c": ".c",
                "cpp": ".cpp",
            }

            # Load dataset config
            repo = self.DATASET_REPOS.get(dataset_name, dataset_name)

            try:
                ds = load_dataset(repo, split=split, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Could not load full dataset: {e}")
                logger.info("Trying subset...")
                ds = load_dataset(repo, "python", split=split, trust_remote_code=True)

            output_file = self.output_dir / f"{dataset_name}_{split}.jsonl"

            # Filter and save
            filtered_count = 0
            with open(output_file, "w", encoding="utf-8") as f:
                for item in tqdm(ds, desc=f"Processing {dataset_name}"):
                    if quality_filter:
                        # Basic quality filters
                        content = item.get("content", item.get("code", ""))
                        if len(content) < 50 or len(content) > 10000:
                            continue
                        if "<|file" in content or "<|repository" in content:
                            continue

                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    filtered_count += 1

                    if filtered_count >= 100000:  # Limit for initial download
                        break

            logger.info(f"Saved {filtered_count} items to {output_file}")
            return output_file

        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            raise

    def download_the_stack(
        self,
        languages: list[str],
        max_files: int = 50000,
    ) -> dict[str, Path]:
        """Download The Stack dataset."""
        logger.info(f"Downloading The Stack for languages: {languages}")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        output_files = {}

        for lang in languages:
            logger.info(f"Downloading {lang}...")

            try:
                ds = load_dataset(
                    "bigcode/the-stack",
                    lang=lang,
                    split="train",
                    trust_remote_code=True,
                )

                output_file = self.output_dir / f"the_stack_{lang}.jsonl"

                # Filter and save
                count = 0
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in tqdm(ds, desc=f"Processing {lang}"):
                        # Quality filter
                        content = item.get("content", "")
                        if not content or len(content) < 50:
                            continue
                        if len(content) > 50000:  # Skip very large files
                            continue

                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        count += 1

                        if count >= max_files:
                            break

                output_files[lang] = output_file
                logger.info(f"Saved {count} {lang} files to {output_file}")

            except Exception as e:
                logger.warning(f"Could not download {lang}: {e}")
                continue

        return output_files

    def download_starcoderdata(
        self,
        max_files: int = 50000,
    ) -> Path:
        """Download StarCoderData dataset."""
        logger.info("Downloading StarCoderData...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "bigcode/starcoderdata",
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / "starcoderdata.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc="Processing"):
                content = item.get("content", "")
                if not content or len(content) < 50:
                    continue
                if len(content) > 50000:
                    continue

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_files:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file

    def download_codesearchnet(
        self,
        language: str = "python",
        max_files: int = 30000,
    ) -> Path:
        """Download CodeSearchNet dataset."""
        logger.info(f"Downloading CodeSearchNet ({language})...")

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed")
            raise

        ds = load_dataset(
            "codeparrot/codesearchnet",
            language,
            split="train",
            trust_remote_code=True,
        )

        output_file = self.output_dir / f"codesearchnet_{language}.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc=f"Processing {language}"):
                code = item.get("code", "")
                if not code or len(code) < 50:
                    continue

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

                if count >= max_files:
                    break

        logger.info(f"Saved {count} items to {output_file}")
        return output_file


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download code datasets for Expera AI training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_code_datasets.py --dataset stack --languages python javascript
  python download_code_datasets.py --dataset starcoder
  python download_code_datasets.py --dataset codesearchnet --language python
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="stack",
        choices=["stack", "starcoder", "codesearchnet"],
        help="Dataset to download",
    )

    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["python", "javascript", "typescript", "rust", "go"],
        help="Programming languages to download",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Language for CodeSearchNet",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/code",
        help="Output directory",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=50000,
        help="Maximum number of files per language",
    )

    parser.add_argument(
        "--quality-filter",
        action="store_true",
        default=True,
        help="Apply quality filters",
    )

    parser.add_argument(
        "--no-quality-filter",
        dest="quality_filter",
        action="store_false",
        help="Disable quality filters",
    )

    return parser.parse_args()


async def main() -> None:
    """Main function."""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = CodeDatasetDownloader(output_dir)

    if args.dataset == "stack":
        downloader.download_the_stack(args.languages, args.max_files)

    elif args.dataset == "starcoder":
        downloader.download_starcoderdata(args.max_files)

    elif args.dataset == "codesearchnet":
        downloader.download_codesearchnet(args.language, args.max_files)

    logger.info("Download complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())