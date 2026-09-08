"""
Streaming Tokenizer Trainer for Expera AI.

Trains tokenizer on streaming data:
- Memory-efficient batch processing
- Progress tracking
- Checkpointing
- Resumability
"""

import time
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Callable, Dict, Any
from collections import Counter
import regex as re

from .bpe_tokenizer import BPETokenizer


@dataclass
class StreamTrainerState:
    """Streaming trainer state for checkpointing."""
    word_freqs: Dict[str, int] = field(default_factory=dict)
    pair_freqs: Dict[str, int] = field(default_factory=dict)
    documents_processed: int = 0
    merges_performed: int = 0


class StreamingTrainer:
    """
    Train tokenizer on streaming data.

    Memory-efficient training for large corpora:
    - Accumulates frequency counts
    - Performs periodic merges
    - Supports checkpointing
    - Resumable
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        min_frequency: int = 2,
        buffer_size: int = 100000,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize streaming trainer.

        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency
            buffer_size: Documents to buffer before training
            checkpoint_dir: Directory for checkpoints
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.buffer_size = buffer_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Tokenizer to build
        self._tokenizer: Optional[BPETokenizer] = None

        # Frequency counters
        self._word_freqs: Counter = Counter()
        self._pair_freqs: Counter = Counter()

        # State
        self._docs_processed = 0
        self._progress = 0.0

        # Pre-tokenization pattern
        self._pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )

        # Byte encoder from BPETokenizer
        self._byte_encoder = BPETokenizer().byte_encoder

    def feed(self, text: str) -> None:
        """
        Feed a single text document.

        Args:
            text: Text document
        """
        # Pre-tokenize
        words = re.findall(self._pat, text)

        # Count words
        for word in words:
            byte_encoded = ''.join(self._byte_encoder[b] for b in word.encode('utf-8'))
            self._word_freqs[byte_encoded] += 1

        self._docs_processed += 1

    def feed_batch(self, texts: List[str]) -> None:
        """
        Feed a batch of texts.

        Args:
            texts: List of text documents
        """
        for text in texts:
            self.feed(text)

    def train(
        self,
        num_merges: Optional[int] = None,
        verbose: bool = True,
    ) -> BPETokenizer:
        """
        Train tokenizer from accumulated frequencies.

        Args:
            num_merges: Number of merge operations
            verbose: Print progress

        Returns:
            Trained BPETokenizer
        """
        if verbose:
            print(f"Training on {self._docs_processed} documents...")

        # Calculate merges
        if num_merges is None:
            base_size = 256 + 4 + 22  # bytes + special + additional
            num_merges = self.vocab_size - base_size

        # Create splits
        splits = {word: tuple(word) for word in self._word_freqs.keys()}

        # Count pair frequencies from splits
        if verbose:
            print("Counting pair frequencies...")

        self._pair_freqs = Counter()
        for word, freq in self._word_freqs.items():
            pairs = self._get_pairs(splits[word])
            for pair in pairs:
                pair_key = tuple(pair)
                self._pair_freqs[pair_key] += freq

        # Create tokenizer
        tokenizer = self._create_tokenizer()

        if verbose:
            print(f"Performing {num_merges} merges...")

        # Perform merges
        for i in range(num_merges):
            if not self._pair_freqs:
                break

            best_pair = max(self._pair_freqs.items(), key=lambda x: x[1])
            if best_pair[1] < self.min_frequency:
                break

            best_pair = best_pair[0]

            # Merge
            new_splits = {}
            for word in splits:
                new_word = self._merge_pair(splits[word], best_pair)
                new_splits[word] = new_word
            splits = new_splits

            # Update pair frequencies
            new_pair_freqs = Counter()
            for pair, freq in self._pair_freqs.items():
                new_pair_freqs[pair] = freq
            self._pair_freqs = new_pair_freqs

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Merge {i + 1}/{num_merges}")

        self._tokenizer = tokenizer
        self._progress = 1.0

        if verbose:
            print(f"Training complete! Vocab size: {len(tokenizer.vocab)}")

        return tokenizer

    def _get_pairs(self, word: tuple) -> List[tuple]:
        """Get adjacent pairs from a word."""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i+1]))
        return pairs

    def _merge_pair(self, word: tuple, pair: tuple) -> tuple:
        """Merge all occurrences of a pair."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(''.join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _create_tokenizer(self) -> BPETokenizer:
        """Create tokenizer from merge operations."""
        tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            special_tokens={
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
            },
        )

        # Build vocab and merges from pair frequencies
        vocab = {chr(i): i for i in range(256)}

        # Add special tokens
        current_id = 256
        for token in tokenizer.special_tokens.values():
            vocab[token] = current_id
            current_id += 1

        for token in tokenizer.additional_special_tokens:
            vocab[token] = current_id
            current_id += 1

        # Apply top merges
        merges = sorted(self._pair_freqs.items(), key=lambda x: -x[1])
        for rank, (pair, _) in enumerate(merges):
            merged = ''.join(pair)
            if merged not in vocab:
                vocab[merged] = current_id
                current_id += 1
                tokenizer.merges[tuple(pair)] = merged
                tokenizer.merge_ranks[tuple(pair)] = rank

            if current_id >= self.vocab_size:
                break

        tokenizer.vocab = vocab
        tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}

        return tokenizer

    def get_progress(self) -> float:
        """
        Get training progress.

        Returns:
            Progress as fraction (0-1)
        """
        return self._progress

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        state = StreamTrainerState(
            word_freqs=dict(self._word_freqs),
            pair_freqs=dict(self._pair_freqs),
            documents_processed=self._docs_processed,
        )

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self._word_freqs = Counter(state.word_freqs)
        self._pair_freqs = Counter(state.pair_freqs)
        self._docs_processed = state.documents_processed

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "documents_processed": self._docs_processed,
            "unique_words": len(self._word_freqs),
            "unique_pairs": len(self._pair_freqs),
            "progress": self._progress,
        }


def train_tokenizer_streaming(
    text_iterator: Iterator[str],
    vocab_size: int = 50304,
    min_frequency: int = 2,
    buffer_size: int = 100000,
    verbose: bool = True,
) -> BPETokenizer:
    """
    Convenience function for streaming training.

    Args:
        text_iterator: Iterator of texts
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency
        buffer_size: Buffer size
        verbose: Print progress

    Returns:
        Trained BPETokenizer
    """
    trainer = StreamingTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        buffer_size=buffer_size,
    )

    # Feed texts
    texts = []
    for text in text_iterator:
        texts.append(text)
        if len(texts) >= buffer_size:
            trainer.feed_batch(texts)
            texts = []

        if trainer._docs_processed % 100000 == 0 and verbose:
            print(f"Processed {trainer._docs_processed} documents")

    # Feed remaining
    if texts:
        trainer.feed_batch(texts)

    # Train
    return trainer.train(verbose=verbose)