#!/usr/bin/env python3
"""
Near Deduplication for Datasets.

Removes near-duplicate content using:
- MinHash LSH for fast similarity
- Exact dedup with checksums
- Substring matching

Usage:
    python scripts/near_dedup.py --input data/filtered/text/ --output data/deduped/
    python scripts/near_dedup.py --input data/filtered/code/ --similarity 0.8 --method minhash
"""

import argparse
import json
import logging
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Number of permutations for MinHash
NUM_PERMUTATIONS = 128
# Number of bands for LSH
NUM_BANDS = 16
# Rows per band
ROWS_PER_BAND = NUM_PERMUTATIONS // NUM_BANDS


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Generate hash of text."""
    if algorithm == "md5":
        return hashlib.md5(text.encode()).hexdigest()[:16]
    elif algorithm == "sha1":
        return hashlib.sha1(text.encode()).hexdigest()[:16]
    else:
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def get_shingles(text: str, k: int = 5) -> Set[str]:
    """Get k-shingles (k-grams) from text."""
    # Tokenize
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    shingles = set()
    words = text.split()

    if len(words) < k:
        return {text}

    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i + k])
        shingles.add(shingle)

    return shingles


def minhash(shingles: Set[str], seed: int = 0) -> tuple:
    """Compute MinHash signature."""
    import random

    # Generate random hash functions
    random.seed(seed)
    max_hash = 2 ** 32 - 1

    hashes = []
    for i in range(NUM_PERMUTATIONS):
        min_val = max_hash

        for shingle in shingles:
            # Simple hash
            h = hash((i, hash(shingle)))
            if h < min_val:
                min_val = h

        hashes.append(min_val)

    return tuple(hashes)


def lsh_buckets(signature: tuple, num_bands: int = NUM_BANDS) -> List[str]:
    """Get LSH bucket keys."""
    rows_per_band = len(signature) // num_bands
    buckets = []

    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band
        band = signature[start:end]

        # Create bucket key
        bucket_key = f"band{band_idx}_" + '_'.join(str(h) for h in band)
        buckets.append(bucket_key)

    return buckets


def compute_jaccard(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity."""
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def compute_signature_similarity(sig1: tuple, sig2: tuple) -> float:
    """Compute similarity between MinHash signatures."""
    if len(sig1) != len(sig2):
        return 0.0

    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def exact_deduplicate(
    input_path: Path,
    output_path: Path,
) -> Dict:
    """Remove exact duplicates."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    stats = {"total": 0, "unique": 0, "duplicates": 0}

    output_file = output_path / "exact_dedup.jsonl"

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get content
            content = (
                record.get("text", "") or
                record.get("content", "") or
                record.get("code", "")
            )

            if not content:
                continue

            # Compute hash
            content_hash = hash_text(content)

            if content_hash in seen_hashes:
                stats["duplicates"] += 1
                continue

            seen_hashes.add(content_hash)
            record["_hash"] = content_hash
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["unique"] += 1

            if stats["total"] % 10000 == 0:
                logger.info(f"Processed: {stats['total']}, unique: {stats['unique']}")

    logger.info(f"Exact dedup complete: {stats}")
    return stats


def ngram_deduplicate(
    input_path: Path,
    output_path: Path,
    similarity_threshold: float = 0.8,
    k: int = 5,
) -> Dict:
    """Remove near-duplicates using n-gram similarity."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Store signatures
    signatures: List[tuple] = []
    records: List[Dict] = []
    kept_indices: Set[int] = set()

    stats = {"total": 0, "unique": 0, "duplicates": 0, "buckets": Counter()}

    # First pass: compute signatures
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get content
            content = (
                record.get("text", "") or
                record.get("content", "") or
                record.get("code", "")
            )

            if not content:
                continue

            # Get shingles
            shingles = get_shingles(content, k)

            if not shingles:
                continue

            # Compute MinHash signature
            signature = minhash(shingles)
            signatures.append(signature)

            # Get LSH buckets
            buckets = lsh_buckets(signature)
            for bucket in buckets:
                stats["buckets"][bucket] += 1

            records.append(record)

    # Second pass: find near-duplicates
    for i, (sig, record) in enumerate(zip(signatures, records)):
        is_duplicate = False

        # Compare with previous signatures in same bucket
        for j in kept_indices:
            prev_sig = signatures[j]

            # Quick check: compute signature similarity
            sim = compute_signature_similarity(sig, prev_sig)

            if sim >= similarity_threshold:
                # Check actual Jaccard
                content_i = record.get("text", "") or record.get("content", "") or record.get("code", "")
                content_j = records[j].get("text", "") or records[j].get("content", "") or records[j].get("code", "")

                shingles_i = get_shingles(content_i, k)
                shingles_j = get_shingles(content_j, k)

                jaccard = compute_jaccard(shingles_i, shingles_j)

                if jaccard >= similarity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept_indices.add(i)
            stats["unique"] += 1
        else:
            stats["duplicates"] += 1

    # Write unique records
    output_file = output_path / "near_dedup.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for i in kept_indices:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    logger.info(f"Near dedup complete: {stats}")
    return stats


def substring_deduplicate(
    input_path: Path,
    output_path: Path,
    min_substring_len: int = 200,
    max_substring_ratio: float = 0.5,
) -> Dict:
    """Remove records that are substrings of other records."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract shorter substrings from longer texts
    all_substrings: Dict[str, List[int]] = {}
    records: List[Dict] = []

    stats = {"total": 0, "unique": 0, "substrings_removed": 0}

    # Load records
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    stats["total"] = len(records)

    # Sort by length (longer first)
    records_with_len = []
    for i, record in enumerate(records):
        content = record.get("text", "") or record.get("content", "") or record.get("code", "")
        records_with_len.append((len(content), i, record))

    records_with_len.sort(reverse=True)

    kept_indices: Set[int] = set()

    for length, idx, record in records_with_len:
        content = record.get("text", "") or record.get("content", "") or record.get("code", "")

        # Check if this is a substring of a kept record
        is_substring = False

        for kept_idx in kept_indices:
            kept_content = records[kept_idx].get("text", "") or records[kept_idx].get("content", "") or records[kept_idx].get("code", "")

            if len(content) < len(kept_content):
                # Check if content is substring
                if content in kept_content:
                    # Check length ratio
                    if len(content) / len(kept_content) < max_substring_ratio:
                        is_substring = True
                        break

        if not is_substring:
            kept_indices.add(idx)
            stats["unique"] += 1
        else:
            stats["substrings_removed"] += 1

    # Write unique records
    output_file = output_path / "substring_dedup.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx in sorted(kept_indices):
            f.write(json.dumps(records[idx], ensure_ascii=False) + "\n")

    logger.info(f"Substring dedup complete: {stats}")
    return stats


def combined_deduplicate(
    input_path: Path,
    output_path: Path,
    similarity: float = 0.8,
    method: str = "all",
) -> Dict:
    """Apply combined deduplication."""
    # First: exact dedup
    temp_path = output_path / "temp_exact.jsonl"

    exact_stats = exact_deduplicate(input_path, temp_path)

    if method in ["minhash", "all"]:
        # Second: near dedup
        temp_path2 = output_path / "temp_near.jsonl"
        ngram_stats = ngram_deduplicate(temp_path, temp_path2, similarity)
        temp_path = temp_path2

    if method in ["substring", "all"]:
        # Third: substring dedup
        final_path = output_path / "final_dedup.jsonl"
        substring_stats = substring_deduplicate(temp_path, final_path)

    # Clean up temp files
    for f in output_path.glob("temp_*"):
        f.unlink()

    logger.info("Combined dedup complete")
    return {"exact": exact_stats}


def generate_report(input_path: Path) -> Dict:
    """Generate deduplication report."""
    stats = {"total": 0, "unique": 0, "duplicates": 0}

    for file_path in Path(input_path).glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                stats["total"] += 1

                if "_hash" in line:
                    stats["unique"] += 1

    stats["duplicates"] = stats["total"] - stats["unique"]

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Near deduplication")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--similarity", type=float, default=0.8, help="Similarity threshold")
    parser.add_argument("--method", choices=["exact", "minhash", "substring", "all"], default="all", help="Dedup method")
    parser.add_argument("--k", type=int, default=5, help="N-gram size")
    parser.add_argument("--report", action="store_true", help="Generate report")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Dedup report: {stats}")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.method == "exact":
        stats = exact_deduplicate(input_path, output_path)
    elif args.method == "minhash":
        stats = ngram_deduplicate(input_path, output_path, args.similarity, args.k)
    elif args.method == "substring":
        stats = substring_deduplicate(input_path, output_path)
    else:
        stats = combined_deduplicate(input_path, output_path, args.similarity, args.method)

    logger.info(f"Complete: {stats}")


if __name__ == "__main__":
    main()