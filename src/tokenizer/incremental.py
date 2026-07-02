"""
Incremental Tokenizer Trainer for Expera AI.

Incrementally expands tokenizer vocabulary:
- Add new tokens from new texts
- Preserve existing tokens
- Merge new pair frequencies
"""

import regex as re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

from .bpe_tokenizer import BPETokenizer


class IncrementalTrainer:
    """
    Incrementally expand tokenizer vocabulary.

    Allows adding new tokens to an existing tokenizer
    without retraining from scratch.
    """

    def __init__(
        self,
        base_tokenizer: BPETokenizer,
        target_vocab_size: int,
        min_frequency: int = 2,
    ):
        """
        Initialize incremental trainer.

        Args:
            base_tokenizer: Existing tokenizer to expand
            target_vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency
        """
        self.base_tokenizer = base_tokenizer
        self.target_vocab_size = target_vocab_size
        self.min_frequency = min_frequency

        # New frequencies to track
        self._new_word_freqs: Counter = Counter()
        self._new_pair_freqs: Counter = Counter()

        # New tokens discovered
        self._new_tokens: Set[str] = set()

        # Pre-tokenization pattern
        self._pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )

    def add_texts(self, new_texts: List[str]) -> int:
        """
        Add texts to track for potential new tokens.

        Args:
            new_texts: New training texts

        Returns:
            Number of new unique words found
        """
        existing_vocab = set(self.base_tokenizer.vocab.keys())

        for text in new_texts:
            # Pre-tokenize
            words = re.findall(self._pat, text)

            for word in words:
                self._new_word_freqs[word] += 1

                # Track new words not in vocab
                if word not in existing_vocab:
                    self._new_tokens.add(word)

        # Update pair frequencies
        for word, freq in self._new_word_freqs.items():
            word_tuple = tuple(word)
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                self._new_pair_freqs[pair] += freq

        return len(self._new_tokens)

    def expand(
        self,
        num_merges: Optional[int] = None,
    ) -> BPETokenizer:
        """
        Expand tokenizer with new merges.

        Args:
            num_merges: Number of new merge operations

        Returns:
            Expanded BPETokenizer
        """
        if num_merges is None:
            num_merges = self.target_vocab_size - len(self.base_tokenizer.vocab)

        if num_merges <= 0:
            return self.base_tokenizer

        # Get new pairs sorted by frequency
        new_pairs = sorted(
            self._new_pair_freqs.items(),
            key=lambda x: -x[1]
        )

        # Filter to new pairs only
        existing_pairs = set(self.base_tokenizer.merges.keys())
        truly_new_pairs = [
            (pair, freq) for pair, freq in new_pairs
            if pair not in existing_pairs and freq >= self.min_frequency
        ]

        # Create new tokenizer with additional merges
        merges = dict(self.base_tokenizer.merges)
        merge_ranks = dict(self.base_tokenizer.merge_ranks)
        vocab = dict(self.base_tokenizer.vocab)

        current_id = max(vocab.values()) + 1

        for pair, freq in truly_new_pairs[:num_merges]:
            merged = ''.join(pair)
            if merged not in vocab:
                vocab[merged] = current_id
                merges[pair] = merged
                merge_ranks[pair] = len(merge_ranks)
                current_id += 1

        # Create new tokenizer
        new_tokenizer = BPETokenizer(
            vocab_size=self.target_vocab_size,
            special_tokens=self.base_tokenizer.special_tokens,
        )
        new_tokenizer.merges = merges
        new_tokenizer.merge_ranks = merge_ranks
        new_tokenizer.vocab = vocab
        new_tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}
        new_tokenizer.additional_special_tokens = self.base_tokenizer.additional_special_tokens

        return new_tokenizer

    def get_new_tokens(self) -> List[str]:
        """
        Get list of new tokens discovered.

        Returns:
            List of new tokens
        """
        return list(self._new_tokens)

    def get_potential_expansion(self) -> int:
        """
        Get potential vocabulary expansion.

        Returns:
            Number of new merges possible
        """
        return len(self._new_tokens)

    def reset(self) -> None:
        """Reset accumulated frequencies."""
        self._new_word_freqs.clear()
        self._new_pair_freqs.clear()
        self._new_tokens.clear()


def expand_tokenizer(
    base_tokenizer: BPETokenizer,
    new_texts: List[str],
    target_vocab_size: int,
) -> BPETokenizer:
    """
    Convenience function to expand tokenizer.

    Args:
        base_tokenizer: Base tokenizer
        new_texts: New training texts
        target_vocab_size: Target vocabulary size

    Returns:
        Expanded tokenizer
    """
    trainer = IncrementalTrainer(base_tokenizer, target_vocab_size)
    trainer.add_texts(new_texts)
    return trainer.expand()