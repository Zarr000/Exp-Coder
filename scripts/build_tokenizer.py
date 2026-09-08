#!/usr/bin/env python3
"""
Build Custom Tokenizer for Expera AI.

Trains a custom BPE tokenizer from training data:
- Collects text corpus from files or datasets
- Trains SentencePiece model
- Saves tokenizer files

Usage:
    python scripts/build_tokenizer.py --input data/processed/ --output data/tokenized/
    python scripts/build_tokenizer.py --input data/raw/*.jsonl --vocab-size 50304
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import logging
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_tokenizer(
    input_path: str,
    output_dir: str,
    vocab_size: int = 50304,
    model_type: str = "unigram",  # unigram, bpe, word, char
    character_coverage: float = 0.9995,
    split_digits: bool = True,
    allow_new_tokens: bool = True,
    byte_fallback: bool = True,
    # Training parameters
    min_frequency: int = 2,
    seed_size: int = 100000,
    # Processing
    max_sentence_length: int = 16384,
    add_dummy_prefix: bool = True,
    remove_extra_whitespaces: bool = True,
) -> dict:
    """
    Train SentencePiece tokenizer.

    Args:
        input_path: Input text file or directory
        output_dir: Output directory for tokenizer files
        vocab_size: Target vocabulary size
        model_type: Model type (unigram, bpe, word, char)
        character_coverage: Character coverage ratio
        split_digits: Split digits into separate tokens
        allow_new_tokens: Allow unknown tokens
        byte_fallback: Use byte fallback for unknown chars
        min_frequency: Minimum token frequency
        seed_size: Seed sentences for training
        max_sentence_length: Max sentence length
        add_dummy_prefix: Add dummy prefix
        remove_extra_whitespaces: Clean whitespace

    Returns:
        Training statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create temporary corpus file
    logger.info("Building training corpus...")

    if Path(input_path).is_dir():
        # Collect all text from directory
        text_files = list(Path(input_path).rglob("*.txt"))
        if not text_files:
            # Try JSONL
            text_files = list(Path(input_path).rglob("*.jsonl"))

        if not text_files:
            # Process existing data format
            import json
            text_lines = []

            for jsonl_file in Path(input_path).rglob("*.jsonl"):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                text = data.get("text", data.get("content", ""))
                                if text:
                                    text_lines.append(text)
                            except:
                                pass

            if not text_lines:
                logger.error("No text found in input directory")
                return {"error": "No input data"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            corpus_path = tmp_file.name
            for text in text_lines:
                tmp_file.write(text + "\n")
    else:
        # Single file - use directly
        corpus_path = input_path

    # Training arguments
    train_args = [
        f"--input={corpus_path}",
        f"--model_prefix={output_path / 'tokenizer'}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--min_count={min_frequency}",
        f"--split_digits={split_digits}",
        f"--allow_new_tokens={allow_new_tokens}",
        f"--byte_fallback={byte_fallback}",
        f"--max_sentence_length={max_sentence_length}",
        f"--add_dummy_prefix={add_dummy_prefix}",
        f"--remove_extra_whitespaces={remove_extra_whitespaces}",
    ]

    logger.info(f"Training tokenizer ( vocab_size={vocab_size}, model_type={model_type} )")
    logger.info(f"Output: {output_path}")

    # Train
    spm.SentencePieceTrainer.train(" ".join(train_args))

    # Verify output
    model_file = output_path / "tokenizer.model"
    if not model_file.exists():
        logger.error("Tokenizer training failed")
        return {"error": "Training failed"}

    # Load and verify
    sp_handler = spm.SentencePieceProcessor()
    sp_handler.load(str(model_file))

    stats = {
        "vocab_size": sp_handler.get_piece_size(),
        "model_file": str(model_file),
        "output_dir": str(output_path),
        "config": {
            "vocab_size": vocab_size,
            "model_type": model_type,
            "character_coverage": character_coverage,
        }
    }

    logger.info(f"Tokenizer trained: vocab_size={stats['vocab_size']}")
    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build custom tokenizer for Expera AI"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input text file, JSONL, or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=50304,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "word", "char"],
        help="Tokenizer model type"
    )
    parser.add_argument(
        "--character-coverage", "-c",
        type=float,
        default=0.9995,
        help="Character coverage ratio"
    )
    parser.add_argument(
        "--byte-fallback",
        action="store_true",
        default=True,
        help="Enable byte fallback"
    )
    parser.add_argument(
        "--split-digits",
        action="store_true",
        default=True,
        help="Split digits into separate tokens"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("Building tokenizer...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Vocab size: {args.vocab_size}")

    stats = train_tokenizer(
        input_path=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        split_digits=args.split_digits,
        byte_fallback=args.byte_fallback,
    )

    if "error" in stats:
        logger.error(f"Tokenizer training failed: {stats['error']}")
        sys.exit(1)

    logger.info(f"Tokenizer built successfully!")
    logger.info(f"  Vocab size: {stats['vocab_size']}")
    logger.info(f"  Model: {stats['model_file']}")

    return stats


if __name__ == "__main__":
    main()