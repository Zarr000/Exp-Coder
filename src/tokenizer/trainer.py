"""
Tokenizer Trainer for Expera AI.

Provides high-level training interface for BPE tokenizers:
- Train from text lists
- Train from files
- Train from iterators (streaming)
- Progress tracking
- Checkpointing
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Iterator, Dict, Any
from pathlib import Path
import json

from .bpe_tokenizer import BPETokenizer


@dataclass
class TrainingProgress:
    """Training progress information."""
    current_merges: int
    total_merges: int
    documents_processed: int
    vocab_size: int
    elapsed_sec: float
    eta_sec: Optional[float] = None


class TokenizerTrainer:
    """
    Train BPE tokenizer on text corpora.

    Supports training from:
    - In-memory text lists
    - File paths (JSONL, text files)
    - Iterators (streaming for large corpora)

    Attributes:
        vocab_size: Target vocabulary size
        min_frequency: Minimum pair frequency for merge
        special_tokens: Special token definitions
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        special_tokens: Optional[Dict[str, str]] = None,
        min_frequency: int = 2,
        streaming: bool = False,
    ):
        """
        Initialize tokenizer trainer.

        Args:
            vocab_size: Target vocabulary size
            special_tokens: Special token definitions
            min_frequency: Minimum frequency for a merge to be kept
            streaming: Use streaming mode for large corpora
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.streaming = streaming
        self._tokenizer: Optional[BPETokenizer] = None
        self._progress: Optional[TrainingProgress] = None

        # Default special tokens
        if special_tokens is None:
            special_tokens = {
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
            }
        self.special_tokens = special_tokens

    def train(
        self,
        texts: List[str],
        num_merges: Optional[int] = None,
        verbose: bool = True,
    ) -> BPETokenizer:
        """
        Train tokenizer on a list of texts.

        Args:
            texts: List of training texts
            num_merges: Number of merge operations (None = auto)
            verbose: Print progress

        Returns:
            Trained BPETokenizer
        """
        if verbose:
            print(f"Training tokenizer on {len(texts)} texts...")

        start_time = time.perf_counter()

        # Create tokenizer
        self._tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
        )

        # Calculate merges
        if num_merges is None:
            base_size = 256 + len(self.special_tokens) + len(self._tokenizer.additional_special_tokens)
            num_merges = self.vocab_size - base_size

        # Train
        self._tokenizer.train(
            texts=texts,
            num_merges=num_merges,
            min_frequency=self.min_frequency,
            verbose=verbose,
        )

        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"Training completed in {elapsed:.2f}s")

        return self._tokenizer

    def train_from_files(
        self,
        file_paths: List[str],
        num_merges: Optional[int] = None,
        verbose: bool = True,
        encoding: str = "utf-8",
    ) -> BPETokenizer:
        """
        Train tokenizer from files.

        Supports:
        - Plain text files (one document per line)
        - JSONL files (one JSON object per line with 'text' field)

        Args:
            file_paths: List of file paths
            num_merges: Number of merge operations
            verbose: Print progress
            encoding: File encoding

        Returns:
            Trained BPETokenizer
        """
        if verbose:
            print(f"Reading files: {file_paths}")

        texts = []
        total_size = 0

        for path in file_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Detect format
            if path.suffix == ".jsonl":
                # JSONL format
                with open(path, "r", encoding=encoding) as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", obj.get("content", ""))
                            if text:
                                texts.append(text)
                                total_size += len(text)
                        except json.JSONDecodeError:
                            continue
            else:
                # Plain text - one document per line
                with open(path, "r", encoding=encoding) as f:
                    for line in f:
                        line = line.rstrip("\n\r")
                        if line:
                            texts.append(line)
                            total_size += len(line)

        if verbose:
            print(f"Loaded {len(texts)} documents, {total_size:,} characters")

        return self.train(texts, num_merges, verbose)

    def train_from_iterator(
        self,
        text_iterator: Iterator[str],
        num_merges: Optional[int] = None,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> BPETokenizer:
        """
        Train tokenizer from an iterator (streaming mode).

        This is memory-efficient for large corpora. It accumulates
        texts and performs incremental BPE training.

        Args:
            text_iterator: Iterator yielding text documents
            num_merges: Number of merge operations
            progress_callback: Optional callback for progress updates

        Returns:
            Trained BPETokenizer
        """
        # Calculate merges
        if num_merges is None:
            base_size = 256 + len(self.special_tokens)
            base_size += 22  # additional_special_tokens
            num_merges = self.vocab_size - base_size

        # Create tokenizer
        self._tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
        )

        # For streaming, we collect a subset then train
        # In production, would use StreamingTrainer
        buffer_size = 100000  # Hold 100K docs in memory
        texts = []

        start_time = time.perf_counter()
        doc_count = 0

        for text in text_iterator:
            texts.append(text)
            doc_count += 1

            # Train when buffer is full
            if len(texts) >= buffer_size:
                if progress_callback:
                    self._progress = TrainingProgress(
                        current_merges=0,
                        total_merges=num_merges,
                        documents_processed=doc_count,
                        vocab_size=0,
                        elapsed_sec=time.perf_counter() - start_time,
                    )
                    progress_callback(self._progress)

                # Train on buffer
                self._tokenizer.train(
                    texts=texts,
                    num_merges=num_merges,
                    min_frequency=self.min_frequency,
                    verbose=False,
                )
                texts = []  # Clear buffer

        # Train on remaining texts
        if texts:
            self._tokenizer.train(
                texts=texts,
                num_merges=num_merges,
                min_frequency=self.min_frequency,
                verbose=False,
            )

        elapsed = time.perf_counter() - start_time
        print(f"Streaming training completed: {doc_count} docs in {elapsed:.2f}s")

        return self._tokenizer

    def get_tokenizer(self) -> Optional[BPETokenizer]:
        """Get the trained tokenizer."""
        return self._tokenizer

    def get_progress(self) -> Optional[TrainingProgress]:
        """Get current training progress."""
        return self._progress


class TrainerConfig:
    """Configuration for tokenizer training."""

    def __init__(
        self,
        vocab_size: int = 50304,
        min_frequency: int = 2,
        max_texts: Optional[int] = None,
        streaming: bool = False,
        buffer_size: int = 100000,
        checkpoint_every: int = 100000,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_texts = max_texts
        self.streaming = streaming
        self.buffer_size = buffer_size
        self.checkpoint_every = checkpoint_every


def train_tokenizer(
    texts: List[str],
    vocab_size: int = 50304,
    special_tokens: Optional[Dict[str, str]] = None,
    min_frequency: int = 2,
    verbose: bool = True,
) -> BPETokenizer:
    """
    Convenience function to train a tokenizer.

    Args:
        texts: Training texts
        vocab_size: Target vocabulary size
        special_tokens: Special token definitions
        min_frequency: Minimum pair frequency
        verbose: Print progress

    Returns:
        Trained BPETokenizer
    """
    trainer = TokenizerTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
    )
    return trainer.train(texts, verbose=verbose)


def train_tokenizer_from_files(
    file_paths: List[str],
    vocab_size: int = 50304,
    special_tokens: Optional[Dict[str, str]] = None,
    min_frequency: int = 2,
    verbose: bool = True,
) -> BPETokenizer:
    """
    Convenience function to train tokenizer from files.

    Args:
        file_paths: File paths to train from
        vocab_size: Target vocabulary size
        special_tokens: Special token definitions
        min_frequency: Minimum pair frequency
        verbose: Print progress

    Returns:
        Trained BPETokenizer
    """
    trainer = TokenizerTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
    )
    return trainer.train_from_files(file_paths, verbose=verbose)