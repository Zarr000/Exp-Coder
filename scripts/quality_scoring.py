#!/usr/bin/env python3
"""
Quality Scoring for Datasets.

Scores text/code quality using multiple signals:
- Text quality (vocabulary, structure)
- Code quality (syntax, patterns)
- Content quality (repetition, coherence)
- Perplexity-based scoring

Usage:
    python scripts/quality_scoring.py --input data/processed/text/ --output data/curated/
    python scripts/quality_scoring.py --input data/processed/code/ --scorers text code --threshold 0.6
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import math
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Common programming keywords for code detection
CODE_KEYWORDS = {
    "python": {"def ", "class ", "import ", "from ", "if __name__"},
    "javascript": {"function", "const ", "let ", "var ", "import ", "export "},
    "java": {"public class", "private ", "import java", "System.out"},
    "go": {"func ", "import ", "package ", "go "},
}


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def calculate_vocabulary_richness(text: str) -> float:
    """Calculate vocabulary richness (Type-Token Ratio)."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    unique = len(set(tokens))
    return unique / len(tokens)


def calculate Repetition_score(text: str, n: int = 5) -> float:
    """Calculate repetition penalty."""
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0.0

    # Count n-grams
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n)]
    if not ngrams:
        return 0.0

    ngram_counts = Counter(ngrams)
    max_repeat = max(ngram_counts.values()) if ngram_counts else 1

    return max(0, 1 - (max_repeat / len(ngrams)))


def calculate_repetition_penalty(text: str) -> float:
    """Calculate overall repetition penalty."""
    scores = []
    for n in [3, 5, 10]:
        score = calculate_repetition_score(text, n)
        if score > 0:
            scores.append(score)

    return sum(scores) / len(scores) if scores else 1.0


def calculate_sentence_variation(text: str) -> float:
    """Calculate sentence structure variation."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)

    if avg_length < 1:
        return 0.0

    # Variation coefficient
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(variance) / avg_length if avg_length > 0 else 0

    return min(1.0, cv)


def detect_language(text: str) -> str:
    """Detect language (code vs natural)."""
    tokens = tokenize(text)

    # Check for code keywords
    for lang, keywords in CODE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return lang

    # Check for programming symbols
    code_symbols = ['{', '}', '();', '[];', '//', '/*', '*/', '=>']
    code_count = sum(1 for s in code_symbols if s in text)

    if code_count >= 2:
        return "code"

    return "text"


def calculate Code_quality(text: str) -> Dict:
    """Calculate code quality metrics."""
    # Basic checks
    lines = text.split('\n')

    # Check for common patterns
    has_import = any('import' in line for line in lines)
    has_function = any(('def ' in line or 'function' in line) for line in lines)
    has_class = 'class ' in text

    # Check balanced braces
    open_braces = text.count('{') + text.count('(') + text.count('[')
    close_braces = text.count('}') + text.count(')') + text.count(']')
    balanced = abs(open_braces - close_braces) <= 1

    # Score
    score = 0.3
    if has_import:
        score += 0.15
    if has_function:
        score += 0.2
    if has_class:
        score += 0.15
    if balanced:
        score += 0.2

    return {
        "code_score": min(1.0, score),
        "has_import": has_import,
        "has_function": has_function,
        "balanced": balanced,
    }


def calculate_text_quality(text: str) -> Dict:
    """Calculate natural text quality."""
    words = tokenize(text)

    if not words:
        return {"text_score": 0.0}

    # Vocabulary richness
    vtr = calculate_vocabulary_richness(text)

    # Repetition
    rep_penalty = calculate_repetition_penalty(text)

    # Sentence variation
    sent_var = calculate_sentence_variation(text)

    # Word frequency diversity
    word_counts = Counter(words)
    top_word_ratio = word_counts.most_common(1)[0][1] / len(words) if words else 0
    diversity = 1 - top_word_ratio

    # Combine scores
    score = (vtr * 0.3) + (rep_penalty * 0.3) + (sent_var * 0.2) + (diversity * 0.2)

    return {
        "text_score": min(1.0, score),
        "vocabulary_richness": vtr,
        "repetition_penalty": rep_penalty,
        "sentence_variation": sent_var,
        "diversity": diversity,
    }


def calculate content_score(text: str) -> float:
    """Calculate content-based quality signals."""
    # Check for meaningful content
    if len(text) < 50:
        return 0.0

    # Check for common spam patterns
    spam_patterns = [
        r'click here',
        r'buy now',
        r'limited offer',
        r'subscribe now',
        r'visit our website',
    ]

    for pattern in spam_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return 0.0

    # Check for link patterns
    link_ratio = len(re.findall(r'https?://|www\.', text)) / max(len(text.split()), 1)
    if link_ratio > 0.1:
        return 0.3

    return 0.7


def calculate_perplexity_score(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
) -> float:
    """Calculate perplexity-based quality (requires model)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tokenizer is None:
            return 0.5

        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())

        # Normalize: lower perplexity = better quality
        # Clamp to 0-1 range
        score = max(0, min(1, 100 / perplexity))
        return score

    except Exception as e:
        logger.warning(f"Perplexity calculation failed: {e}")
        return 0.5


def detect_quality_issues(text: str) -> List[str]:
    """Detect quality issues."""
    issues = []

    # Empty or too short
    if len(text.strip()) < 10:
        issues.append("too_short")

    # All uppercase
    if text.isupper() and len(text) > 50:
        issues.append("all_uppercase")

    # Check for random characters
    if len(re.findall(r'[^a-zA-Z0-9\s.,!?]', text)) > len(text) * 0.3:
        issues.append("too_many_symbols")

    # Check for copy-paste patterns
    if text.count('\n\n\n') > 3:
        issues.append("excessive_newlines")

    return issues


def score_record(
    record: Dict,
    scorers: List[str] = ["text", "code"],
) -> Optional[Dict]:
    """Score a single record."""
    # Get content
    content = (
        record.get("text", "") or
        record.get("content", "") or
        record.get("code", "") or
        record.get("output", "")
    )

    if not content or len(content) < 10:
        return None

    # Detect type
    content_type = detect_language(content)

    # Calculate scores
    scores = {}

    if "text" in scorers:
        text_quality = calculate_text_quality(content)
        scores.update(text_quality)

    if "code" in scorers:
        code_quality = calculate_code_quality(content)
        scores.update(code_quality)

    # Content score
    scores["content_score"] = calculate_content_score(content)

    # Issues
    scores["issues"] = detect_quality_issues(content)

    # Overall quality
    quality_scores = [
        scores.get("text_score", 0.5),
        scores.get("code_score", 0.5),
        scores.get("content_score", 0.5),
    ]
    scores["overall_quality"] = sum(quality_scores) / len(quality_scores)

    # Type
    scores["content_type"] = content_type

    return scores


def score_dataset(
    input_path: Path,
    output_path: Path,
    scorers: List[str] = ["text", "code"],
    threshold: float = 0.3,
) -> Dict:
    """Score entire dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "avg_quality": 0,
    }
    quality_scores = []

    output_file = output_path / "scored.jsonl"

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
                stats["failed"] += 1
                continue

            # Score
            scores = score_record(record, scorers)
            if not scores:
                stats["failed"] += 1
                continue

            # Add scores to record
            record["_quality"] = scores

            # Filter by threshold
            if scores["overall_quality"] >= threshold:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["passed"] += 1
                quality_scores.append(scores["overall_quality"])
            else:
                stats["failed"] += 1

            if stats["total"] % 10000 == 0:
                logger.info(f"Scored: {stats['total']}, passed: {stats['passed']}")

    if quality_scores:
        stats["avg_quality"] = sum(quality_scores) / len(quality_scores)

    logger.info(f"Completed: {stats}")
    return stats


def filter_by_score(
    input_path: Path,
    output_path: Path,
    scores_to_use: List[str] = ["overall_quality"],
    threshold: float = 0.3,
) -> Dict:
    """Filter dataset by score threshold."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "passed": 0, "failed": 0}

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
                stats["failed"] += 1
                continue

            quality = record.get("_quality", {})
            score = quality.get(scores_to_use[0], 0) if scores_to_use else quality.get("overall_quality", 0)

            if score >= threshold:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["passed"] += 1
            else:
                stats["failed"] += 1

    logger.info(f"Filtered: {stats}")
    return stats


def generate_quality_report(
    input_path: Path,
) -> Dict:
    """Generate quality report."""
    stats = {
        "total": 0,
        "avg_quality": 0,
        "avg_text_score": 0,
        "avg_code_score": 0,
        "avg_content_score": 0,
    }
    quality_scores = []
    text_scores = []
    code_scores = []
    content_scores = []
    issues = Counter()

    for file_path in Path(input_path).glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    stats["total"] += 1

                    quality = record.get("_quality", {})
                    quality_scores.append(quality.get("overall_quality", 0))
                    text_scores.append(quality.get("text_score", 0))
                    code_scores.append(quality.get("code_score", 0))
                    content_scores.append(quality.get("content_score", 0))

                    for issue in quality.get("issues", []):
                        issues[issue] += 1

                except json.JSONDecodeError:
                    continue

    if quality_scores:
        stats["avg_quality"] = sum(quality_scores) / len(quality_scores)
    if text_scores:
        stats["avg_text_score"] = sum(text_scores) / len(text_scores)
    if code_scores:
        stats["avg_code_score"] = sum(code_scores) / len(code_scores)
    if content_scores:
        stats["avg_content_score"] = sum(content_scores) / len(content_scores)

    stats["issues"] = dict(issues)

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Score dataset quality")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--scorers", nargs="+", default=["text", "code"], help="Scorers to use")
    parser.add_argument("--threshold", type=float, default=0.3, help="Quality threshold")
    parser.add_argument("--filter", action="store_true", help="Filter by threshold")
    parser.add_argument("--score-field", default="overall_quality", help="Score field to filter by")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_quality_report(Path(args.input))
        logger.info(f"Quality report: {stats}")
        return

    if not args.output:
        logger.error("Please specify --output")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.filter:
        stats = filter_by_score(
            input_path,
            output_path,
            scores_to_use=[args.score_field],
            threshold=args.threshold,
        )
    else:
        stats = score_dataset(
            input_path,
            output_path,
            scorers=args.scorers,
            threshold=args.threshold,
        )

    logger.info(f"Complete: {stats}")


if __name__ == "__main__":
    main()