#!/usr/bin/env python3
"""
Dataset Preparation Script for Expera AI.

Prepares raw datasets for training:
1. Validates input format (JSONL/Parquet)
2. Applies quality filtering
3. Deduplicates
4. Generates manifest

Usage:
    python scripts/prepare_dataset.py --input data/raw/my_dataset.jsonl --output data/processed/
    python scripts/prepare_dataset.py --input data/raw/ --output data/processed/ --workers 4
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import ray

from src.data import (
    ExperaDataset,
    DatasetManifest,
    DatasetStage,
    ManifestManager,
    DatasetValidator,
    QualityFilter,
    Deduplicator,
    ShardWriter,
    ShardInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default configurations
DEFAULT_QUALITY_CONFIG = {
    "enabled_filters": ["length", "repeat", "language"],
    "min_length": 50,
    "max_length": 100000,
    "repeat_ngram_threshold": 0.3,
    "language_whitelist": ["en", "code"],  # English + code
}

DEFAULT_DEDUP_CONFIG = {
    "enabled": True,
    "method": "exact",
    "minhash": False,
    "num_hashes": 128,
    "threshold": 0.8,
}


def load_raw_dataset(input_path: str, format_hint: Optional[str] = None) -> List[dict]:
    """Load raw dataset from file or directory."""
    input_path = Path(input_path)
    logger.info(f"Loading data from {input_path}")

    if input_path.is_file():
        file_format = format_hint or input_path.suffix.lstrip(".")
        if file_format == "jsonl":
            with open(input_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
        elif file_format == "json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif file_format == "parquet":
            import pandas as pd
            df = pd.read_parquet(input_path)
            data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        logger.info(f"Loaded {len(data)} records from {input_path.name}")
        return data

    # Directory - load all files
    data = []
    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lstrip(".") in ("jsonl", "json", "parquet"):
            try:
                subdata = load_raw_dataset(str(file_path))
                data.extend(subdata)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
    logger.info(f"Loaded {len(data)} total records")
    return data


def process_chunk(args: tuple) -> List[dict]:
    """Process a chunk of data with quality filters."""
    chunk, quality_config, dedup_enabled = args
    filtered = []

    # Quality filtering
    if quality_config.get("enabled", True):
        qfilter = QualityFilter(quality_config)
        for item in chunk:
            result = qfilter.filter(item)
            if result.is_valid:
                filtered.append(item)
    else:
        filtered = chunk

    # Basic deduplication in chunk
    if dedup_enabled:
        seen = set()
        for item in filtered:
            key = item.get("text", "")[:1000]  # Use first 1000 chars as key
            if key not in seen:
                seen.add(key)
                yield item

    yield from []


def prepare_dataset(
    input_path: str,
    output_dir: str,
    quality_config: Optional[dict] = None,
    dedup_config: Optional[dict] = None,
    workers: int = 4,
    shard_size: int = 10000,
    name: str = "dataset",
) -> dict:
    """
    Prepare dataset: filter, dedup, shard.

    Args:
        input_path: Input file or directory
        output_dir: Output directory
        quality_config: Quality filtering config
        dedup_config: Deduplication config
        workers: Number of parallel workers
        shard_size: Records per shard
        name: Dataset name

    Returns:
        Statistics dictionary
    """
    quality_config = quality_config or DEFAULT_QUALITY_CONFIG
    dedup_config = dedup_config or DEFAULT_DEDUP_CONFIG

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_raw_dataset(input_path)
    total_input = len(data)
    logger.info(f"Processing {total_input} records with {workers} workers")

    # Quality filter
    if quality_config.get("enabled", True):
        logger.info("Applying quality filters...")
        qfilter = QualityFilter(quality_config)
        filtered_data = []
        for i, item in enumerate(data):
            result = qfilter.filter(item)
            if result.is_valid:
                filtered_data.append(item)
            if (i + 1) % 10000 == 0:
                logger.info(f"Quality filtered: {i+1}/{total_input}")
        data = filtered_data
        logger.info(f"After quality: {len(data)} records")

    # Deduplicate
    after_dedup = len(data)
    if dedup_config.get("enabled", True):
        logger.info("Deduplicating...")
        deduplicator = Deduplicator(dedup_config)
        result = deduplicator.deduplicate(data)
        data = result.unique_documents
        logger.info(f"Removed {after_dedup - len(data)} duplicates")

    # Shard and write
    logger.info(f"Writing shards to {output_path}")
    manifest = DatasetManifest(
        name=name,
        stage=DatasetStage.PROCESSED,
        num_examples=len(data),
    )

    shard_writer = ShardWriter(output_path, shard_size=shard_size)
    for i, record in enumerate(data):
        shard_writer.write(record)

    shard_info = shard_writer.close()

    # Update manifest
    manifest.metadata["shards"] = len(shard_info)
    manifest.metadata["shard_size"] = shard_size
    manifest.metadata["input_count"] = total_input
    manifest.metadata["output_count"] = len(data)

    # Save manifest
    manifest_manager = ManifestManager(output_path)
    manifest_manager.save(manifest)

    stats = {
        "name": name,
        "input_count": total_input,
        "output_count": len(data),
        "duplicates_removed": total_input - len(data),
        "shards": len(shard_info),
        "output_dir": str(output_path),
    }

    logger.info(f"Dataset prepared: {stats}")
    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Expera AI training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file, directory, or manifest"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML config file"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="dataset",
        help="Dataset name"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Records per shard"
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Only apply quality filters, skip dedup"
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality filtering"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Use Ray for distributed processing"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build configs
    quality_config = DEFAULT_QUALITY_CONFIG.copy()
    dedup_config = DEFAULT_DEDUP_CONFIG.copy()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            quality_config.update(config.get("quality", {}))
            dedup_config.update(config.get("dedup", {}))

    if args.no_quality:
        quality_config["enabled"] = False
    if args.no_dedup:
        dedup_config["enabled"] = False

    # Run preparation
    logger.info("Starting dataset preparation")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    stats = prepare_dataset(
        input_path=args.input,
        output_dir=args.output,
        quality_config=quality_config,
        dedup_config=dedup_config,
        workers=args.workers,
        shard_size=args.shard_size,
        name=args.name,
    )

    logger.info(f"Dataset preparation complete!")
    logger.info(f"  Input:  {stats['input_count']:,} records")
    logger.info(f"  Output: {stats['output_count']:,} records")
    logger.info(f"  Removed: {stats['duplicates_removed']:,} duplicates")
    logger.info(f"  Shards: {stats['shards']}")
    logger.info(f"  Dir:    {stats['output_dir']}")

    return stats


if __name__ == "__main__":
    main()