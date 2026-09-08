#!/usr/bin/env python3
"""
Text Dataset Preparation.

Prepares text/web datasets for training:
- Filters short and low-quality content
- Extracts text from HTML/markdown
- Removes personal information
- Generates metadata
- Language detection

Usage:
    python scripts/prepare_text_dataset.py --input data/raw/fineweb/ --output data/processed/text/
    python scripts/prepare_text_dataset.py --input data/raw/wikipedia/ --output data/processed/wikipedia/
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import hashlib
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Language detection patterns
LANGUAGE_PATTERNS = {
    "en": r'\b(the|is|are|was|were|have|has|be|been|being|that|which|with|from|they|this|for|but|not|you|all|can|had|her|were|she|there|their|what|so|up|out|if|about|who|get|which|go|me)\b',
    "es": r'\b(el|la|los|las|un|una|de|en|que|es|por|para|con|una|su|al|se|le|lo|mas|ya|o|este|esta|pero|sus|como|mas|muy|todo)\b',
    "fr": r'\b(le|la|les|un|une|des|de|du|en|que|qui|est|pas|sur|une|pour|avec|plus|ce|mais|comme|tout|ont|dans|ces|ses|faire|vous)\b',
    "de": r'\b(der|die|das|und|in|zu|den|das|nicht|mit|sie|es|von|sie|auch|auf|ein|eine|als|aus|nur|uber|habe|hat|werden|worden)\b',
    "pt": r'\b(o|a|os|as|um|uma|de|em|que|e|para|com|nao|por|mais|como|seu|sua|seus|suas|dos|das|ao|aos|no|na)\b',
    "it": r'\b(il|la|di|che|e|in|un|una|per|non|sono|con|da|io|lei|gli|le|si|questo|questa|quello|quella|anche|pero)\b',
    "zh": r'[\u4e00-\u9fff]',
    "ja": r'[\u3040-\u309f\u30a0-\u30ff]',
    "ko": r'[\uac00-\ud7af]',
    "ar": r'[\u0600-\u06ff]',
    "ru": r'[\u0400-\u04ff]',
}


# Personal information patterns to remove
PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
    (r'\b\d{9}\b', '[ID]'),  # ID numbers
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),  # Phone
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]'),  # IP addresses
]


# Patterns indicating low quality
LOW_QUALITY_PATTERNS = [
    r'^.{0,20}$|^.{10000,}$',  # Too short or too long single line
    r'<script[^>]*>.*?</script>',  # JavaScript
    r'<style[^>]*>.*?</style>',  # CSS
    r'google_ad_slot|_ads|advertisement',  # Ads
    r'click here to|subscribe now|buy now|limited offer',  # Spam
    r'^(?=.*(.)\1{20,})',  # Repeating characters
]


def detect_language(text: str) -> Optional[str]:
    """Detect language from text."""
    text_lower = text.lower()

    best_lang = None
    best_score = 0

    for lang, pattern in LANGUAGE_PATTERNS.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches > best_score:
            best_score = matches
            best_lang = lang

    return best_lang if best_score > 5 else None


def remove_pii(text: str) -> str:
    """Remove personal information."""
    text = text.copy() if isinstance(text, str) else text

    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)

    return text


def clean_html(text: str) -> str:
    """Remove HTML tags and clean text."""
    # Remove scripts and styles
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def extract_paragraphs(text: str) -> List[str]:
    """Extract paragraphs from text."""
    # Split by common paragraph delimiters
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def calculate_text_quality(text: str) -> Dict:
    """Calculate text quality metrics."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    if not text.strip():
        return {"quality": 0, "has_content": False}

    # Basic quality metrics
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

    # Check for content variety
    unique_words = len(set(words))
    vocabulary_richness = unique_words / max(word_count, 1)

    # Sentence quality
    avg_sentence_len = word_count / max(sentence_count, 1)

    # Penalize low quality patterns
    quality = 1.0
    for pattern in LOW_QUALITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            quality -= 0.2

    quality *= vocabulary_richness
    quality = max(0, min(1, quality))

    return {
        "quality": quality,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_len,
        "vocabulary_richness": vocabulary_richness,
        "avg_sentence_length": avg_sentence_len,
        "has_content": word_count >= 50,
    }


def validate_text(
    text: str,
    min_words: int = 50,
    max_words: int = 100000,
    min_quality: float = 0.3,
    target_language: Optional[str] = None,
) -> Dict:
    """Validate and score text."""
    if not text.strip():
        return {"valid": False, "reason": "Empty text"}

    words = text.split()
    word_count = len(words)

    if word_count < min_words:
        return {"valid": False, "reason": f"Too few words: {word_count}"}

    if word_count > max_words:
        return {"valid": False, "reason": f"Too many words: {word_count}"}

    # Language detection
    detected_lang = detect_language(text)
    if target_language and detected_lang != target_language:
        return {"valid": False, "reason": f"Wrong language: {detected_lang}"}

    # Quality check
    quality = calculate_text_quality(text)
    if quality["quality"] < min_quality:
        return {"valid": False, "reason": f"Low quality: {quality['quality']:.2f}"}

    if not quality["has_content"]:
        return {"valid": False, "reason": "No substantial content"}

    return {
        "valid": True,
        "language": detected_lang,
        "quality": quality,
    }


def process_record(record: Dict, target_language: Optional[str] = None) -> Optional[Dict]:
    """Process a single text record."""
    # Extract text field
    content = record.get("text", record.get("content", record.get("html", "")))
    if not content:
        return None

    # Clean HTML if needed
    if record.get("html"):
        content = clean_html(content)

    # Remove PII
    content = remove_pii(content)

    # Validate
    validation = validate_text(content, target_language=target_language)
    if not validation.get("valid"):
        return None

    return {
        "text": content,
        "language": validation.get("language"),
        "word_count": validation["quality"]["word_count"],
        "sentence_count": validation["quality"]["sentence_count"],
        "quality": validation["quality"]["quality"],
        "vocabulary_richness": validation["quality"]["vocabulary_richness"],
        "checksum": hashlib.sha256(content.encode()).hexdigest()[:16],
    }


def process_jsonl(
    input_path: Path,
    output_path: Path,
    target_language: Optional[str] = None,
    min_words: int = 50,
    max_words: int = 100000,
) -> Dict:
    """Process JSONL text dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "valid": 0, "invalid": 0, "languages": Counter()}

    output_file = output_path / f"{input_path.stem}_processed.jsonl"

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            stats["processed"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue

            processed = process_record(record, target_language)
            if not processed:
                stats["invalid"] += 1
                continue

            # Add metadata
            record.update(processed)

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["valid"] += 1
            stats["languages"][processed.get("language", "unknown")] += 1

            if stats["processed"] % 10000 == 0:
                logger.info(f"Processed {stats['processed']}, valid: {stats['valid']}")

    logger.info(f"Completed: {stats}")
    return stats


def process_wikipedia(input_path: Path, output_path: Path) -> Dict:
    """Process Wikipedia dump."""
    from mwxml import Dump

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "valid": 0, "invalid": 0, "languages": Counter()}

    output_file = output_path / "wikipedia_processed.jsonl"

    with open(output_file, "w", encoding="utf-8") as outfile:
        dump = Dump.from_file(input_path)

        for page in dump:
            for revision in page:
                stats["processed"] += 1

                text = revision.text
                if not text:
                    continue

                # Skip redirects and special pages
                if page.is_special or page.is_redirect:
                    stats["invalid"] += 1
                    continue

                # Process content
                validation = validate_text(text, target_language="en")
                if not validation.get("valid"):
                    stats["invalid"] += 1
                    continue

                page_title = page.title or ""

                record = {
                    "title": page_title,
                    "text": text,
                    "language": "en",
                    "word_count": validation["quality"]["word_count"],
                    "quality": validation["quality"]["quality"],
                    "checksum": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "source": "wikipedia",
                }

                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["valid"] += 1
                stats["languages"]["en"] += 1

                if stats["processed"] % 10000 == 0:
                    logger.info(f"Processed {stats['processed']}, valid: {stats['valid']}")

    logger.info(f"Completed: {stats}")
    return stats


def split_train_val(
    input_path: Path,
    output_dir: Path,
    val_split: float = 0.01,
) -> Dict:
    """Split dataset into train and validation."""
    import random

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(line)

    random.shuffle(records)

    val_size = int(len(records) * val_split)
    val_records = records[:val_size]
    train_records = records[val_size:]

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_records))

    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_records))

    return {
        "train": len(train_records),
        "val": len(val_records),
    }


def generate_report(input_path: Path) -> Dict:
    """Generate text dataset statistics."""
    stats = {"total_docs": 0, "languages": Counter(), "total_words": 0, "avg_quality": 0}
    quality_scores = []
    word_counts = []

    for file_path in Path(input_path).glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    stats["total_docs"] += 1
                    stats["languages"][record.get("language", "unknown")] += 1
                    stats["total_words"] += record.get("word_count", 0)
                    quality_scores.append(record.get("quality", 0))
                    word_counts.append(record.get("word_count", 0))
                except json.JSONDecodeError:
                    continue

    if quality_scores:
        stats["avg_quality"] = sum(quality_scores) / len(quality_scores)

    if word_counts:
        stats["avg_words"] = sum(word_counts) / len(word_counts)
        stats["median_words"] = sorted(word_counts)[len(word_counts) // 2]

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare text datasets for training")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--language", type=str, help="Target language filter")
    parser.add_argument("--min-words", type=int, default=50, help="Minimum words")
    parser.add_argument("--max-words", type=int, default=100000, help="Maximum words")
    parser.add_argument("--format", choices=["jsonl", "wikipedia"], default="jsonl", help="Input format")
    parser.add_argument("--split", action="store_true", help="Split into train/val")
    parser.add_argument("--val-split", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Dataset stats: {stats}")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.format == "wikipedia":
        stats = process_wikipedia(input_path, output_path)
    else:
        stats = process_jsonl(
            input_path,
            output_path,
            target_language=args.language,
            min_words=args.min_words,
            max_words=args.max_words,
        )

    if args.split:
        split_stats = split_train_val(
            output_path / f"{input_path.stem}_processed.jsonl",
            output_path,
            args.val_split,
        )
        logger.info(f"Split: {split_stats}")

    logger.info(f"Processing complete: {stats}")


if __name__ == "__main__":
    main()