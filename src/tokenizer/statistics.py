"""
Tokenizer Statistics for Expera AI.

Collects tokenizer statistics during training:
- Document counts
- Token counts
- Vocabulary statistics
- Merge operations
"""

import regex as re
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Optional
from collections import Counter

from .bpe_tokenizer import BPETokenizer


@dataclass
class TokenizerStats:
    """Tokenizer statistics."""
    total_documents: int = 0
    total_words: int = 0
    total_tokens_generated: int = 0
    total_bytes: int = 0
    avg_doc_length: float = 0.0
    avg_word_length: float = 0.0
    vocab_size: int = 0
    unique_words: int = 0
    unique_bigrams: int = 0
    merge_operations: int = 0


class StatsCollector:
    """
    Collect tokenizer statistics.

    Gathers statistics during training or evaluation
    to understand token distribution and corpus characteristics.
    """

    def __init__(self):
        """Initialize stats collector."""
        self._word_freqs: Counter = Counter()
        self._bigram_freqs: Counter = Counter()
        self._doc_lengths: List[int] = []
        self._total_chars = 0

    def collect(
        self,
        tokenizer: BPETokenizer,
        texts: Iterator[str],
    ) -> TokenizerStats:
        """
        Collect statistics from texts.

        Args:
            tokenizer: Tokenizer for encoding
            texts: Text iterator

        Returns:
            TokenizerStats with collected data
        """
        total_docs = 0
        total_words = 0
        total_tokens = 0

        for text in texts:
            total_docs += 1

            # Document length
            self._doc_lengths.append(len(text))
            self._total_chars += len(text)

            # Word count
            words = re.findall(r'\w+', text)
            total_words += len(words)
            for w in words:
                self._word_freqs[w.lower()] += 1

            # Bigrams
            for i in range(len(words) - 1):
                bigram = (words[i].lower(), words[i+1].lower())
                self._bigram_freqs[bigram] += 1

            # Token count
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)

        # Compute averages
        avg_doc_length = self._total_chars / max(1, total_docs)
        avg_word_length = sum(
            len(w) * f for w, f in self._word_freqs.items()
        ) / max(1, sum(self._word_freqs.values()))

        return TokenizerStats(
            total_documents=total_docs,
            total_words=total_words,
            total_tokens_generated=total_tokens,
            total_bytes=self._total_chars,
            avg_doc_length=avg_doc_length,
            avg_word_length=avg_word_length,
            vocab_size=len(tokenizer.vocab),
            unique_words=len(self._word_freqs),
            unique_bigrams=len(self._bigram_freqs),
            merge_operations=len(tokenizer.merges),
        )

    def collect_sample(
        self,
        tokenizer: BPETokenizer,
        texts: List[str],
        sample_size: int = 10000,
    ) -> TokenizerStats:
        """
        Collect statistics from a sample.

        Args:
            tokenizer: Tokenizer for encoding
            texts: List of texts
            sample_size: Maximum texts to process

        Returns:
            TokenizerStats
        """
        sample = texts[:sample_size]
        return self.collect(tokenizer, iter(sample))

    def get_word_frequencies(self) -> Dict[str, int]:
        """Get word frequency distribution."""
        return dict(self._word_freqs.most_common(1000))

    def get_bigram_frequencies(self) -> Dict[str, int]:
        """Get bigram frequency distribution."""
        return dict(self._bigram_freqs.most_common(1000))

    def get_document_length_stats(self) -> Dict[str, float]:
        """Get document length statistics."""
        if not self._doc_lengths:
            return {"mean": 0, "median": 0, "p95": 0, "p99": 0}

        sorted_lengths = sorted(self._doc_lengths)
        n = len(sorted_lengths)

        return {
            "mean": sum(sorted_lengths) / n,
            "median": sorted_lengths[n // 2],
            "p95": sorted_lengths[int(n * 0.95)],
            "p99": sorted_lengths[int(n * 0.99)],
            "min": sorted_lengths[0],
            "max": sorted_lengths[-1],
        }


def collect_stats(
    tokenizer: BPETokenizer,
    texts: List[str],
) -> TokenizerStats:
    """
    Convenience function to collect statistics.

    Args:
        tokenizer: Tokenizer
        texts: Sample texts

    Returns:
        TokenizerStats
    """
    collector = StatsCollector()
    return collector.collect_sample(tokenizer, texts)