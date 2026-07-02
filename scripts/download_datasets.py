#!/usr/bin/env python3
"""
Dataset Downloader for Expera AI.

Downloads datasets from various sources:
- HuggingFace datasets
- GitHub repositories
- URLs

Supports:
- Resume downloads
- Validation
- Progress tracking

Usage:
    python scripts/download_datasets.py --dataset the-stack --output data/raw/
    python scripts/download_datasets.py code-search-net --output data/raw/
"""

import argparse
import subprocess
import urllib.request
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import json
import hashlib
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Supported datasets
DATASET_REGISTRY = {
    # Code datasets
    "the-stack": {
        "source": "hf",
        "repo": "bigcode/the-stack",
        "split": "train",
        "columns": ["content", "language", "repo_name", "license"],
    },
    "code-search-net": {
        "source": "hf",
        "repo": "CodeSearchNet",
        "split": "train",
    },
    "project-codenet": {
        "source": "hf",
        "repo": "bigcode/project-codenet",
        "split": "train",
    },
    # Text datasets
    "fineweb": {
        "source": "hf",
        "repo": "HuggingFaceFW/fineweb",
        "split": "train",
    },
    "redpajama": {
        "source": "hf",
        "repo": "togethercomputer/RedPajama-Data-v2",
        "split": "train",
    },
    "wikipedia": {
        "source": "hf",
        "repo": "wikipedia",
        "language": "en",
    },
    # Instruction datasets
    "openassistant": {
        "source": "hf",
        "repo": "OpenAssistant/oasst1",
        "split": "train",
    },
    "ultrachat": {
        "source": "hf",
        "repo": "openbmb/UltraChat",
        "split": "train",
    },
    "sharegpt": {
        "source": "hf",
        "repo": "Salesforce/dialogstudio",
        "subset": "ShareGPT",
    },
    "alpaca": {
        "source": "hf",
        "repo": "tatsu-lab/alpaca",
        "split": "train",
    },
    "code-alpaca": {
        "source": "hf",
        "repo": "sahil2801/CodeAlpaca-20k",
        "split": "train",
    },
    # Vision-language
    "llava": {
        "source": "hf",
        "repo": "liuhaotian/LLaVA-Instruct-150K",
        "split": "train",
    },
}


def download_huggingface(
    dataset_name: str,
    output_dir: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Download dataset from HuggingFace."""
    from datasets import load_dataset, load_dataset_builder

    config = DATASET_REGISTRY[dataset_name]
    repo_id = config["repo"]

    logger.info(f"Downloading {dataset_name} from HuggingFace...")

    # Determine split
    dataset_split = config.get("split", split)

    # Load dataset info first
    try:
        builder = load_dataset_builder(repo_id)
        logger.info(f"Dataset info: {builder.info}")
    except Exception as e:
        logger.warning(f"Could not load builder info: {e}")

    # Load with streaming for large datasets
    try:
        ds = load_dataset(
            repo_id,
            split=dataset_split,
            streaming=True,
            trust_remote_code=True,
        )

        output_path = output_dir / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Download with progress
        count = 0
        records = []

        for record in ds:
            records.append(record)
            count += 1

            if max_samples and count >= max_samples:
                break

            if count % 10000 == 0:
                logger.info(f"Downloaded {count} records")

        # Save to JSONL
        output_file = output_path / f"{dataset_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Downloaded {count} records to {output_file}")

        return {
            "dataset": dataset_name,
            "records": count,
            "output": str(output_file),
        }

    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        return {"error": str(e)}


def download_from_url(url: str, output_path: Path, resume: bool = True) -> Dict[str, Any]:
    """Download from URL with resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for resume
    if resume and output_path.exists():
        downloaded = output_path.stat().st_size
        logger.info(f"Resuming from {downloaded} bytes")
    else:
        downloaded = 0

    # Download
    try:
        request = urllib.request.Request(url)
        if downloaded > 0:
            request.add_header("Range", f"bytes={downloaded}-")

        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            mode = "ab" if downloaded > 0 else "wb"

            with open(output_path, mode) as f:
                while True:
                    chunk = response.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) == 0:
                            logger.info(f"Progress: {progress:.1f}%")

        return {
            "url": url,
            "size": downloaded,
            "output": str(output_path),
        }

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return {"error": str(e)}


def clone_github(repo: str, output_dir: Path) -> Dict[str, Any]:
    """Clone GitHub repository."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(output_dir)],
            check=True,
            capture_output=True,
        )
        return {"repo": repo, "output": str(output_dir)}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}


def download_dataset(
    dataset: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Download a dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset not in DATASET_REGISTRY:
        return {"error": f"Unknown dataset: {dataset}"}

    config = DATASET_REGISTRY[dataset]
    source = config.get("source", "hf")

    if source == "hf":
        return download_huggingface(
            dataset,
            output_path,
            max_samples=max_samples,
            resume=resume,
        )
    elif source == "url":
        return download_from_url(config["url"], output_path / config["filename"], resume)
    elif source == "github":
        return clone_github(config["repo"], output_path / config.get("subdir", ""))

    return {"error": f"Unknown source: {source}"}


def list_available_datasets() -> List[str]:
    """List available datasets."""
    return list(DATASET_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for Expera AI")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset name")
    parser.add_argument("--output", "-o", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--max-samples", "-m", type=int, help="Maximum samples to download")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume downloads")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available datasets:")
        for name in list_available_datasets():
            print(f"  - {name}")
        return

    if not args.dataset:
        logger.error("No dataset specified. Use --list to see available datasets.")
        return

    logger.info(f"Downloading {args.dataset} to {args.output}")
    result = download_dataset(
        args.dataset,
        args.output,
        max_samples=args.max_samples,
        resume=args.resume,
    )

    if "error" in result:
        logger.error(f"Failed: {result['error']}")
    else:
        logger.info(f"Complete: {result}")


if __name__ == "__main__":
    main()