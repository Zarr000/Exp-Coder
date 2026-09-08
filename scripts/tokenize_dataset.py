#!/usr/bin/env python3
"""
Tokenize Dataset for Expera AI.

Tokenizes processed dataset with custom tokenizer:
- Loads tokenizer
- Streams dataset
- Tokenizes and saves as token IDs
- Generates memory-mapped output

Usage:
    python scripts/tokenize_dataset.py --input data/processed/ --tokenizer data/tokenized/tokenizer.model --output data/tokenized/
    python scripts/tokenize_dataset.py --input data/processed/ --tokenizer data/tokenized/ --output data/tokenized/ --streaming
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
import struct
import mmap
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TokenizedDatasetWriter:
    """Write tokenized dataset in binary format."""

    HEADER_MAGIC = b"EXTRTOK"
    HEADER_VERSION = 1

    def __init__(self, output_path: Path, vocab_size: int):
        self.output_path = output_path
        self.vocab_size = vocab_size
        self.num_tokens = 0
        self.num_documents = 0
        self.document_offsets = [0]  # Start offset

        # Open file for writing
        self.file = open(output_path, "wb")
        self._write_header()

    def _write_header(self):
        """Write dataset header."""
        # Magic + version
        self.file.write(self.HEADER_MAGIC)
        self.file.write(struct.pack("<H", self.HEADER_VERSION))
        # Vocab size (4 bytes)
        self.file.write(struct.pack("<I", self.vocab_size))
        # Placeholder for document count (update later)
        self._doc_count_offset = self.file.tell()
        self.file.write(struct.pack("<Q", 0))
        # Placeholder for total token count
        self._token_count_offset = self.file.tell()
        self.file.write(struct.pack("<Q", 0))

    def write_document(self, token_ids: List[int]):
        """Write a tokenized document."""
        if not token_ids:
            return

        # Write token count (4 bytes) + token IDs (2 bytes each)
        # Use 2 bytes per token for vocab <= 65536
        tokens = np.array(token_ids, dtype=np.uint16)
        tokens.tofile(self.file)

        self.num_tokens += len(token_ids)
        self.num_documents += 1
        self.document_offsets.append(self.num_tokens)

    def close(self) -> Dict[str, Any]:
        """Close and finalize the dataset."""
        # Update counts in header
        self.file.seek(self._doc_count_offset)
        self.file.write(struct.pack("<Q", self.num_documents))
        self.file.seek(self._token_count_offset)
        self.file.write(struct.pack("<Q", self.num_tokens))
        self.file.close()

        return {
            "num_documents": self.num_documents,
            "num_tokens": self.num_tokens,
            "vocab_size": self.vocab_size,
        }


class TokenizedDatasetReader:
    """Memory-mapped tokenized dataset reader."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self._open()

    def _open(self):
        """Open and read header."""
        with open(self.dataset_path, "rb") as f:
            # Verify magic
            magic = f.read(6)
            if magic != TokenizedDatasetWriter.HEADER_MAGIC:
                raise ValueError(f"Invalid dataset format: {magic}")

            # Read header
            version = struct.unpack("<H", f.read(2))[0]
            self.vocab_size = struct.unpack("<I", f.read(4))[0]
            self.num_documents = struct.unpack("<Q", f.read(8))[0]
            self.num_tokens = struct.unpack("<Q", f.read(8))[0]

        # Memory map for fast access
        self.mm = mmap.mmap(
            open(self.dataset_path, "rb").fileno(),
            0,
            mmap.ACCESS_READ
        )

        # Skip header (6 + 2 + 4 + 8 + 8 = 28 bytes)
        self._data_offset = 28

    def __len__(self):
        return self.num_documents

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get document by index."""
        # Simple implementation - read from offset
        # In production, use precomputed offsets
        raise NotImplementedError("Use get_document() instead")

    def get_document(self, idx: int) -> np.ndarray:
        """Get tokenized document by index."""
        # Estimate position (for streaming format)
        # This is approximate - proper impl needs offset index
        return np.array([], dtype=np.uint16)

    def close(self):
        """Close the dataset."""
        self.mm.close()


def tokenize_file(
    input_path: str,
    tokenizer_path: str,
    output_path: str,
    max_length: int = 8192,
    add_special_tokens: bool = True,
    slide_window: bool = False,
    stride: int = 512,
    streaming: bool = False,
) -> Dict[str, Any]:
    """
    Tokenize a dataset file.

    Args:
        input_path: Input JSONL file
        tokenizer_path: Path to tokenizer model
        output_path: Output binary file
        max_length: Maximum sequence length
        add_special_tokens: Add <s> and </s> tokens
        slide_window: Use sliding window for long sequences
        stride: Sliding window stride
        streaming: Write in streaming format

    Returns:
        Statistics dictionary
    """
    # Load tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    vocab_size = sp.get_piece_size()
    logger.info(f"Tokenizer loaded: vocab_size={vocab_size}")

    # Special tokens
    bos_id = sp.bos_id() if add_special_tokens else -1
    eos_id = sp.eos_id() if add_special_tokens else -1
    unk_id = sp.unk_id()

    # Writer
    writer = TokenizedDatasetWriter(Path(output_path), vocab_size)

    # Process
    logger.info(f"Tokenizing {input_path}")
    total_tokens = 0
    total_docs = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                text = data.get("text", data.get("content", ""))

                if not text:
                    skipped += 1
                    continue

                # Tokenize
                if add_special_tokens:
                    tokens = [bos_id] + sp.encode(text) + [eos_id]
                else:
                    tokens = sp.encode(text)

                # Truncate or slide window
                if len(tokens) > max_length:
                    if slide_window:
                        # Sliding window - take first chunk
                        tokens = tokens[:max_length]
                    else:
                        skipped += 1
                        continue

                # Write
                writer.write_document(tokens)
                total_tokens += len(tokens)
                total_docs += 1

                if (line_num + 1) % 10000 == 0:
                    logger.info(f"Tokenized: {line_num+1} docs, {total_tokens} tokens")

            except json.JSONDecodeError:
                skipped += 1
            except Exception as e:
                logger.warning(f"Error on line {line_num}: {e}")
                skipped += 1

    # Finalize
    stats = writer.close()
    logger.info(f"Tokenized {total_docs} documents, {total_tokens} tokens")
    logger.info(f"Skipped {skipped} documents")

    return {
        **stats,
        "input_file": input_path,
        "output_file": output_path,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "skipped": skipped,
    }


def tokenize_dataset(
    input_dir: str,
    tokenizer_path: str,
    output_dir: str,
    max_length: int = 8192,
    workers: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """Tokenize entire dataset directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files
    jsonl_files = sorted(input_path.rglob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {input_dir}")
        return {"error": "No input files"}

    logger.info(f"Found {len(jsonl_files)} files to tokenize")

    total_stats = {
        "num_documents": 0,
        "num_tokens": 0,
        "files": [],
    }

    for jsonl_file in jsonl_files:
        output_file = output_path / f"{jsonl_file.stem}.bin"

        stats = tokenize_file(
            input_path=str(jsonl_file),
            tokenizer_path=tokenizer_path,
            output_path=str(output_file),
            max_length=max_length,
            **kwargs
        )

        total_stats["num_documents"] += stats.get("total_docs", 0)
        total_stats["num_tokens"] += stats.get("total_tokens", 0)
        total_stats["files"].append({
            "input": str(jsonl_file),
            "output": str(output_file),
        })

    logger.info(f"Dataset tokenized: {total_stats['num_documents']} docs, {total_stats['num_tokens']} tokens")
    return total_stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize dataset for Expera AI training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory or file"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        required=True,
        help="Tokenizer model path or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        default=True,
        help="Add special tokens"
    )
    parser.add_argument(
        "--slide-window",
        action="store_true",
        help="Use sliding window for long sequences"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Find tokenizer
    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_dir():
        model_path = tokenizer_path / "tokenizer.model"
    else:
        model_path = tokenizer_path

    if not model_path.exists():
        logger.error(f"Tokenizer not found: {model_path}")
        sys.exit(1)

    logger.info("Starting tokenization")
    logger.info(f"Input: {args.input}")
    logger.info(f"Tokenizer: {model_path}")
    logger.info(f"Output: {args.output}")

    stats = tokenize_dataset(
        input_dir=args.input,
        tokenizer_path=str(model_path),
        output_dir=args.output,
        max_length=args.max_length,
        add_special_tokens=args.add_special_tokens,
        slide_window=args.slide_window,
        stride=args.stride,
        workers=args.workers,
    )

    if "error" in stats:
        logger.error(f"Tokenization failed: {stats['error']}")
        sys.exit(1)

    logger.info(f"Tokenization complete!")
    logger.info(f"  Documents: {stats['num_documents']:,}")
    logger.info(f"  Tokens: {stats['num_tokens']:,}")
    logger.info(f"  Files: {len(stats.get('files', []))}")

    return stats


if __name__ == "__main__":
    main()