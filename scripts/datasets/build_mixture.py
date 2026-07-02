"""
Build Dataset Mixtures for Expera AI Training.

Creates balanced dataset mixtures from multiple sources:
- Code datasets
- Chat/instruction datasets
- Image datasets

Features:
- Configurable ratios
- Sharding
- Statistics generation

Usage:
    python scripts/datasets/build_mixture.py --config configs/dataset_mixture.yaml --output data/mixture
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatasetSource:
    """A single dataset source."""

    name: str
    path: Path
    weight: float = 1.0
    split: str = "train"
    shuffle: bool = True
    sampling: str = "random"  # random, sequential, weighted


@dataclass
class MixtureConfig:
    """Configuration for dataset mixture."""

    sources: list[DatasetSource] = field(default_factory=list)
    output_dir: Path = field(default_factory=Path)
    total_samples: int = 100000
    num_shards: int = 10
    seed: int = 42
    format: str = "jsonl"  # jsonl, arrow, parquet


class MixtureBuilder:
    """Builds dataset mixtures."""

    def __init__(self, config: MixtureConfig):
        """Initialize builder."""
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(config.seed)

    def load_source(self, source: DatasetSource) -> list[dict]:
        """Load a single source."""
        logger.info(f"Loading {source.name} from {source.path}")

        records = []
        with open(source.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    record["__source"] = source.name
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(records)} records from {source.name}")
        return records

    def sample_records(
        self,
        records: list[dict],
        weight: float,
        target_count: int,
        method: str = "random",
    ) -> list[dict]:
        """Sample records from a source."""
        if len(records) == 0:
            return []

        if method == "random":
            # Random sampling
            count = min(target_count, len(records))
            return random.sample(records, count)

        elif method == "sequential":
            # Sequential (first N)
            return records[:target_count]

        elif method == "weighted":
            # Weighted by some field
            weights = [r.get("weight", 1.0) for r in records]
            total_weight = sum(weights)
            count = min(target_count, len(records))

            # Rejection sampling
            result = []
            for _ in range(count * 2):  # Allow extra attempts
                if len(result) >= count:
                    break
                idx = random.randint(0, len(records) - 1)
                if random.random() < weights[idx] / total_weight:
                    result.append(records[idx])

            return result[:count]

        return records[:target_count]

    def build_mixture(
        self,
        target_total: Optional[int] = None,
    ) -> tuple[list[dict], dict]:
        """Build the dataset mixture."""
        target_total = target_total or self.config.total_samples

        # Load all sources
        all_records: list[dict] = []
        source_stats: dict[str, dict] = {}

        for source in self.config.sources:
            if not source.path.exists():
                logger.warning(f"Source not found: {source.path}")
                continue

            records = self.load_source(source)

            # Calculate target count based on weight
            target_count = int(target_total * source.weight)
            target_count = min(target_count, len(records))

            # Sample
            sampled = self.sample_records(
                records,
                source.weight,
                target_count,
                source.sampling,
            )

            source_stats[source.name] = {
                "original": len(records),
                "sampled": len(sampled),
                "weight": source.weight,
            }

            all_records.extend(sampled)

        # Shuffle
        random.shuffle(all_records)

        stats = {
            "total": len(all_records),
            "sources": source_stats,
            "format_version": "1.0",
        }

        logger.info(f"Built mixture with {len(all_records)} total records")
        return all_records, stats

    def create_shards(
        self,
        records: list[dict],
        num_shards: Optional[int] = None,
    ) -> list[Path]:
        """Create dataset shards."""
        num_shards = num_shards or self.config.num_shards

        if len(records) == 0:
            return []

        shard_size = len(records) // num_shards
        output_paths: list[Path] = []

        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else len(records)

            shard_records = records[start:end]
            output_path = self.config.output_dir / f"shard_{i:04d}.jsonl"

            with open(output_path, "w", encoding="utf-8") as f:
                for record in shard_records:
                    # Remove internal field for final output
                    clean_record = {k: v for k, v in record.items() if not k.startswith("__")}
                    f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

            output_paths.append(output_path)

        logger.info(f"Created {len(output_paths)} shards")
        return output_paths

    def save_mixture(
        self,
        records: list[dict],
        output_name: str = "train",
    ) -> Path:
        """Save mixture to file."""
        output_path = self.config.output_dir / f"{output_name}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                clean_record = {k: v for k, v in record.items() if not k.startswith("__")}
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

        logger.info(f"Saved mixture to {output_path}")
        return output_path

    def generate_stats(
        self,
        records: list[dict],
        stats: dict,
    ) -> dict:
        """Generate detailed statistics."""
        # Source distribution
        source_counts = Counter(r.get("__source", "unknown") for r in records)

        # Language distribution (for code)
        language_counts = Counter()
        length_counts = Counter()

        for r in records:
            content = r.get("content", r.get("code", r.get("text", "")))
            if content:
                # Simple language detection
                if "def " in content or "import " in content:
                    language_counts["python"] += 1
                elif "function " in content or "const " in content:
                    language_counts["javascript"] += 1
                elif "fn " in content:
                    language_counts["rust"] += 1
                else:
                    language_counts["other"] += 1

                # Length buckets
                length = len(content)
                if length < 100:
                    length_counts["<100"] += 1
                elif length < 500:
                    length_counts["100-500"] += 1
                elif length < 1000:
                    length_counts["500-1000"] += 1
                elif length < 5000:
                    length_counts["1000-5000"] += 1
                else:
                    length_counts[">5000"] += 1

        detailed_stats = {
            **stats,
            "source_distribution": dict(source_counts),
            "language_distribution": dict(language_counts),
            "length_distribution": dict(length_counts),
            "num_records": len(records),
        }

        return detailed_stats

    def build(
        self,
        name: str = "train",
        create_shards: bool = True,
    ) -> Path:
        """Build complete mixture."""
        # Build
        records, stats = self.build_mixture()

        # Save
        output_path = self.save_mixture(records, name)

        # Shards
        if create_shards:
            self.create_shards(records)

        # Stats
        detailed_stats = self.generate_stats(records, stats)
        stats_path = self.config.output_dir / "stats.json"

        with open(stats_path, "w") as f:
            json.dump(detailed_stats, f, indent=2)

        logger.info(f"Mixture built: {output_path}")
        return output_path


def load_config(config_path: Path) -> MixtureConfig:
    """Load mixture configuration from YAML."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    sources = []
    for s in config_dict.get("sources", []):
        sources.append(
            DatasetSource(
                name=s["name"],
                path=Path(s["path"]),
                weight=s.get("weight", 1.0),
                split=s.get("split", "train"),
                shuffle=s.get("shuffle", True),
                sampling=s.get("sampling", "random"),
            )
        )

    return MixtureConfig(
        sources=sources,
        output_dir=Path(config_dict.get("output_dir", "data/mixture")),
        total_samples=config_dict.get("total_samples", 100000),
        num_shards=config_dict.get("num_shards", 10),
        seed=config_dict.get("seed", 42),
        format=config_dict.get("format", "jsonl"),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build dataset mixture")

    parser.add_argument(
        "--config",
        type=str,
        help="YAML config file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/mixture",
        help="Output directory",
    )

    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        help="Source files (format: path:weight)",
    )

    parser.add_argument(
        "--total-samples",
        type=int,
        default=100000,
        help="Total samples",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of shards",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()

    # Build config
    if args.config:
        config = load_config(Path(args.config))
    else:
        # Build from command line
        sources = []
        if args.sources:
            for s in args.sources:
                parts = s.split(":")
                path = Path(parts[0])
                weight = float(parts[1]) if len(parts) > 1 else 1.0
                sources.append(DatasetSource(path.stem, path, weight))

        config = MixtureConfig(
            sources=sources,
            output_dir=Path(args.output),
            total_samples=args.total_samples,
            num_shards=args.num_shards,
            seed=args.seed,
        )

    # Build
    builder = MixtureBuilder(config)
    builder.build()

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()