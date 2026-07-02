#!/usr/bin/env python3
"""
Toxicity Filter for Datasets.

Filters harmful content:
- Hate speech detection
- PII detection
- NSFW content
- Personal attacks

Usage:
    python scripts/toxicity_filter.py --input data/curated/text/ --output data/filtered/
    python scripts/toxicity_filter.py --input data/curated/code/ --filters hate pii nsfw --threshold 0.5
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Hate speech keywords (curated list)
HATE_KEYWORDS = [
    "hate", "kill", "die", "attack", "fight",
    # Extremist references
    "terrorist", "extremist", "radical",
    # Slur proxies (word patterns that indicate hate)
    "supremacy", "deportation", "genocide",
]

# Harassment patterns
HARASSMENT_PATTERNS = [
    r'\bstupid\b',
    r'\bidiot\b',
    r'\bmoron\b',
    r'\bdumb\b',
    r'\b(shut up|go away|die)\b',
]

# PII patterns
PII_PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone": r'(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}',
    "ssn": r'\d{3}-\d{2}-\d{4}',
    "ip": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
}

# NSFW patterns
NSFW_PATTERNS = [
    r'\bxxx\b',
    r'\bporn\b',
    r'\bnsfw\b',
    r'\badult\s+only',
    # Common explicit content indicators
    r'explicit\s+content',
    r'age\s+restricted',
]

# Safe topics for medical/health contexts
SAFE_CONTEXT = [
    "medical", "health", "doctor", "hospital",
    "patient", "treatment", "diagnosis",
    "symptom", "medicine", "pharmacy",
]

# Banned content categories
CATEGORIES = {
    "hate": {
        "keywords": HATE_KEYWORDS,
        "patterns": HARASSMENT_PATTERNS,
        "threshold": 0.3,
    },
    "pii": {
        "patterns": PII_PATTERNS,
        "threshold": 0.1,
    },
    "nsfw": {
        "patterns": NSFW_PATTERNS,
        "threshold": 0.5,
    },
}


def safe_context_check(text: str) -> bool:
    """Check if text is in safe context (medical, educational, etc.)."""
    text_lower = text.lower()

    for safe_word in SAFE_CONTEXT:
        if safe_word in text_lower:
            # Check for surrounding context
            idx = text_lower.find(safe_word)
            context = text_lower[max(0, idx-50):min(len(text), idx+50)]

            # Medical context - likely safe
            if any(w in context for w in ["medical", "health", "doctor", "patient"]):
                return True

    return False


def check_hate_speech(text: str, threshold: float = 0.3) -> Dict:
    """Check for hate speech."""
    text_lower = text.lower()

    # Check keywords
    keyword_count = sum(1 for kw in HATE_KEYWORDS if kw in text_lower)
    word_count = len(text.split())

    keyword_score = keyword_count / max(word_count / 100, 1)

    # Check patterns
    pattern_matches = 0
    for pattern in HARASSMENT_PATTERNS:
        if re.search(pattern, text_lower):
            pattern_matches += 1

    pattern_score = min(1.0, pattern_matches / 3)

    # Combined score
    score = (keyword_score + pattern_score) / 2

    # Safe context check
    if safe_context_check(text):
        score *= 0.5

    return {
        "has_hate": score >= threshold,
        "score": score,
        "keyword_count": keyword_count,
        "pattern_matches": pattern_matches,
    }


def check_pii(text: str, threshold: float = 0.1) -> Dict:
    """Check for PII."""
    matches = {}

    for pii_type, pattern in PII_PATTERNS.items():
        found = re.findall(pattern, text)
        matches[pii_type] = len(found)

    total_pii = sum(matches.values())

    # Normalize score
    word_count = len(text.split())
    score = total_pii / max(word_count / 500, 1)

    return {
        "has_pii": score >= threshold,
        "score": min(1.0, score),
        "matches": matches,
    }


def check_nsfw(text: str, threshold: float = 0.5) -> Dict:
    """Check for NSFW content."""
    text_lower = text.lower()

    matches = 0
    for pattern in NSFW_PATTERNS:
        if re.search(pattern, text_lower):
            matches += 1

    score = matches / len(NSFW_PATTERNS)

    # Safe context check
    if safe_context_check(text):
        score *= 0.3

    return {
        "has_nsfw": score >= threshold,
        "score": score,
        "matches": matches,
    }


def check_personal_attack(text: str) -> Dict:
    """Check for personal attacks."""
    text_lower = text.lower()

    # You-focused attacks
    you_patterns = [
        r'\byou\b.{0,30}(stupid|idiot|dumb|moron|worthless)',
        r'\byour\b.{0,30}(fault|problem|choice)',
    ]

    matches = []
    for pattern in you_patterns:
        if re.search(pattern, text_lower):
            matches.append(pattern)

    score = min(1.0, len(matches) / 2)

    return {
        "has_attack": score > 0.3,
        "score": score,
    }


def evaluate_toxicity(
    text: str,
    filters: List[str] = ["hate", "pii", "nsfw"],
) -> Dict:
    """Evaluate content toxicity."""
    results = {}

    if "hate" in filters:
        results["hate"] = check_hate_speech(text)

    if "pii" in filters:
        results["pii"] = check_pii(text)

    if "nsfw" in filters:
        results["nsfw"] = check_nsfw(text)

    if "attack" in filters:
        results["attack"] = check_personal_attack(text)

    # Overall toxicity
    scores = [r.get("score", 0) for r in results.values()]
    results["overall"] = {
        "toxic": any(r.get("has_hate", r.get("has_pii", r.get("has_nsfw", False))) for r in results.values()),
        "max_score": max(scores) if scores else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
    }

    return results


def filter_dataset(
    input_path: Path,
    output_path: Path,
    filters: List[str] = ["hate", "pii", "nsfw"],
    threshold: float = 0.5,
) -> Dict:
    """Filter dataset for toxicity."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "passed": 0,
        "filtered": 0,
        "reasons": Counter(),
    }

    output_file = output_path / "filtered.jsonl"

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
                stats["filtered"] += 1
                stats["reasons"]["json_error"] += 1
                continue

            # Get content
            content = (
                record.get("text", "") or
                record.get("content", "") or
                record.get("code", "") or
                record.get("output", "")
            )

            if not content:
                stats["filtered"] += 1
                stats["reasons"]["empty"] += 1
                continue

            # Evaluate
            results = evaluate_toxicity(content, filters)

            # Check if toxic
            is_toxic = results.get("overall", {}).get("toxic", False)
            score = results.get("overall", {}).get("max_score", 0)

            if not is_toxic or score < threshold:
                # Add toxicity info
                record["_toxicity"] = results

                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["passed"] += 1
            else:
                # Determine filter reason
                for filter_name in filters:
                    if filter_name in results and results[filter_name].get(f"has_{filter_name}", False):
                        stats["reasons"][filter_name] += 1

                stats["filtered"] += 1

            if stats["total"] % 10000 == 0:
                logger.info(f"Processed: {stats['total']}, passed: {stats['passed']}")

    logger.info(f"Completed: {stats}")
    return stats


def remove_pii(
    input_path: Path,
    output_path: Path,
) -> Dict:
    """Remove PII from dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "removed": 0, "pii_types": Counter()}
    output_file = output_path / "cleaned.jsonl"

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
                record.get("content", "")
            )

            # Check PII
            pii_results = check_pii(content, threshold=0)

            content_clean = content
            pii_removed = False

            for pii_type, matches in pii_results.get("matches", {}).items():
                if matches > 0:
                    pattern = PII_PATTERNS.get(pii_type)
                    if pattern:
                        # Replace with placeholder
                        content_clean = re.sub(pattern, f"[{pii_type.upper()}]", content_clean)
                        stats["pii_types"][pii_type] += matches
                        pii_removed = True

            if pii_removed:
                stats["removed"] += 1

                # Update record
                if "text" in record:
                    record["text"] = content_clean
                elif "content" in record:
                    record["content"] = content_clean

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Completed: {stats}")
    return stats


def generate_report(input_path: Path) -> Dict:
    """Generate toxicity report."""
    stats = {
        "total": 0,
        "passed": 0,
        "filtered": 0,
        "reasons": Counter(),
    }

    for file_path in Path(input_path).glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    stats["total"] += 1

                    toxicity = record.get("_toxicity", {})
                    if toxicity.get("overall", {}).get("toxic", False):
                        stats["filtered"] += 1
                    else:
                        stats["passed"] += 1

                except json.JSONDecodeError:
                    continue

    return dict(stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Filter toxic content")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--filters", nargs="+", default=["hate", "pii", "nsfw"], help="Filters to apply")
    parser.add_argument("--threshold", type=float, default=0.5, help="Toxicity threshold")
    parser.add_argument("--remove-pii", action="store_true", help="Remove PII instead of filtering")
    parser.add_argument("--report", action="store_true", help="Generate report")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Toxicity report: {stats}")
        return

    if not args.output:
        logger.error("Please specify --output")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.remove_pii:
        stats = remove_pii(input_path, output_path)
    else:
        stats = filter_dataset(input_path, output_path, args.filters, args.threshold)

    logger.info(f"Complete: {stats}")


if __name__ == "__main__":
    main()