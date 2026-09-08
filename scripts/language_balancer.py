#!/usr/bin/env python3
"""
Language Balancer for Datasets.

Balances datasets across languages:
- Language distribution
- Topic balancing
- Source balancing

Usage:
    python scripts/language_balancer.py --input data/deduped/ --output data/balanced/ --target-langs en:0.7 es:0.15 fr:0.15
    python scripts/language_balancer.py --input data/merged/ --output data/balanced/ --strategy downsample
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Simple language detection."""
    if not text:
        return "unknown"

    # Common words for detection
    lang_words = {
        "en": {"the", "is", "are", "was", "were", "have", "has", "be", "been", "being", "and", "or", "but", "if", "then", "this", "that", "these", "those"},
        "es": {"el", "la", "los", "las", "un", "una", "de", "en", "que", "es", "por", "para", "con", "una", "su", "al", "se", "le", "lo"},
        "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "en", "que", "qui", "est", "pas", "sur", "une", "pour", "avec", "plus", "ce", "mais"},
        "de": {"der", "die", "das", "und", "in", "zu", "den", "das", "nicht", "mit", "sie", "es", "von", "sie", "auch", "auf", "ein", "eine"},
        "pt": {"o", "a", "os", "as", "um", "uma", "de", "em", "que", "e", "para", "com", "nao", "por", "mais", "como"},
        "it": {"il", "la", "di", "che", "e", "in", "un", "una", "per", "non", "sono", "con", "da", "io", "lei"},
        "zh": {"的", "是", "在", "有", "和", "了", "我", "你", "他", "她", "它", "们", "这", "那", "来", "去"},
        "ja": {"の", "は", "が", "を", "に", "と", "で", "も", "な", "か", "から", "まで", "から", "より"},
    }

    text_lower = text.lower()
    words = set(text_lower.split())

    best_lang = "en"
    best_score = 0

    for lang, indicators in lang_words.items():
        score = len(words & indicators)
        if score > best_score:
            best_score = score
            best_lang = lang

    return best_lang


def categorize_by_language(
    input_path: Path,
) -> Dict[str, List[Dict]]:
    """Categorize records by language."""
    categories: Dict[str, List[Dict]] = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get text
            text = record.get("text", "") or record.get("content", "")
            if not text:
                continue

            # Detect language
            lang = detect_language(text)

            record["_language"] = lang
            categories[lang].append(record)

    return categories


def upsample(
    records: List[Dict],
    target_count: int,
    target_lang: str,
) -> List[Dict]:
    """Upsample records for a language."""
    if not records:
        return []

    if len(records) >= target_count:
        return records[:target_count]

    # Repeat records
    result = records.copy()
    while len(result) < target_count:
        # Random record
        idx = random.randint(0, len(records) - 1)
        result.append(records[idx].copy())

    return result[:target_count]


def downsample(
    records: List[Dict],
    target_count: int,
    strategy: str = "random",
) -> List[Dict]:
    """Downsample records for a language."""
    if not records:
        return []

    if len(records) <= target_count:
        return records

    if strategy == "random":
        return random.sample(records, target_count)
    elif strategy == "first":
        return records[:target_count]
    elif strategy == "last":
        return records[-target_count:]

    return random.sample(records, target_count)


def balance_by_language(
    input_path: Path,
    output_path: Path,
    target_distribution: Dict[str, float],
    max_examples: Optional[int] = None,
    strategy: str = "downsample",
) -> Dict:
    """Balance dataset by language distribution."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Categorize by language
    categories = categorize_by_language(input_path)

    logger.info(f"Found languages: {dict(Counter(categories.keys()))}")

    # Calculate target counts
    total = sum(len(records) for records in categories.values())

    if max_examples is None:
        max_examples = total

    result_records: List[Dict] = []

    for lang, ratio in target_distribution.items():
        target_count = int(max_examples * ratio)

        # Get records for this language
        records = categories.get(lang, [])

        if strategy == "downsample":
            balanced = downsample(records, target_count)
        else:
            balanced = upsample(records, target_count, lang)

        result_records.extend(balanced)

        logger.info(f"{lang}: {len(records)} -> {len(balanced)}")

    # Shuffle
    random.shuffle(result_records)

    # Write output
    output_file = output_path / "balanced.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in result_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total": len(result_records),
        "languages": dict(Counter(r.get("_language", "unknown") for r in result_records)),
    }


def resample_by_source(
    input_path: Path,
    output_path: Path,
    target_sources: Dict[str, float],
    max_examples: Optional[int] = None,
) -> Dict:
    """Resample by source."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by source
    by_source: Dict[str, List[Dict]] = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                source = record.get("source", "unknown")
                by_source[source].append(record)
            except json.JSONDecodeError:
                continue

    # Calculate targets
    if max_examples is None:
        max_examples = sum(len(records) for records in by_source.values())

    result = []
    for source, ratio in target_sources.items():
        target_count = int(max_examples * ratio)
        records = by_source.get(source, [])

        if len(records) > target_count:
            balanced = downsample(records, target_count)
        else:
            balanced = upsample(records, target_count, source)

        result.extend(balanced)

    random.shuffle(result)

    output_file = output_path / "balanced.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in result:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"total": len(result)}


def merge_and_balance(
    input_paths: List[Path],
    output_path: Path,
    target_distribution: Dict[str, float],
    max_examples: Optional[int] = None,
) -> Dict:
    """Merge multiple datasets and balance."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = []

    for input_path in input_paths:
        input_path = Path(input_path)
        if not input_path.exists():
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(line)

    # Load as JSON
    records = []
    for line in all_records:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Re-balance by language
    categories: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        text = record.get("text", "") or record.get("content", "")
        if text:
            lang = record.get("_language", detect_language(text))
            record["_language"] = lang
            categories[lang].append(record)

    if max_examples is None:
        max_examples = len(records)

    result = []
    for lang, ratio in target_distribution.items():
        target_count = int(max_examples * ratio)
        records_lang = categories.get(lang, [])

        if len(records_lang) > target_count:
            balanced = downsample(records_lang, target_count)
        else:
            balanced = upsample(records_lang, target_count, lang)

        result.extend(balanced)

    random.shuffle(result)

    output_file = output_path / "balanced.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in result:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"total": len(result)}


def generate_report(input_path: Path) -> Dict:
    """Generate language distribution report."""
    stats = {"total": 0, "languages": Counter()}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1
            try:
                record = json.loads(line)
                lang = record.get("_language", detect_language(record.get("text", "")))
                stats["languages"][lang] += 1
            except json.JSONDecodeError:
                continue

    return dict(stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Balance dataset languages")
    parser.add_argument("--input", "-i", type=str, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--target-langs", type=str, help="Target distribution (e.g., en:0.7 es:0.15)")
    parser.add_argument("--source", type=str, help="Source distribution")
    parser.add_argument("--max-examples", type=int, help="Maximum examples total")
    parser.add_argument("--strategy", choices=["upsample", "downsample"], default="downsample")
    parser.add_argument("--merge", nargs="+", help="Merge multiple datasets")
    parser.add_argument("--report", action="store_true", help="Generate report")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Language distribution: {stats}")
        return

    # Parse target distribution
    target_distribution = {}
    if args.target_langs:
        for item in args.target_langs.split():
            lang, ratio = item.split(":")
            target_distribution[lang] = float(ratio)
    elif args.source:
        for item in args.source.split():
            source, ratio = item.split(":")
            target_distribution[source] = float(ratio)

    output_path = Path(args.output)

    if args.merge:
        input_paths = [Path(p) for p in args.merge]
        stats = merge_and_balance(input_paths, output_path, target_distribution, args.max_examples)
    else:
        input_path = Path(args.input)
        if args.source:
            stats = resample_by_source(input_path, output_path, target_distribution, args.max_examples)
        else:
            stats = balance_by_language(input_path, output_path, target_distribution, args.max_examples, args.strategy)

    logger.info(f"Complete: {stats}")


if __name__ == "__main__":
    main()