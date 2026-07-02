"""
Vocabulary Pruner for Expera AI.

Prunes tokenizer vocabulary:
- Remove low-usage tokens
- Merge rare tokens back to sub-tokens
- Analyze usage patterns
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import regex as re

from .bpe_tokenizer import BPETokenizer


@dataclass
class PruneResult:
    """Results from vocabulary pruning."""
    original_vocab_size: int
    pruned_vocab_size: int
    tokens_removed: int
    tokens_merged: int


class VocabularyPruner:
    """
    Prune tokenizer vocabulary by removing low-usage tokens.

    Helps reduce vocabulary size while preserving
    important tokens.
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        min_usage: int = 10,
    ):
        """
        Initialize pruner.

        Args:
            tokenizer: Tokenizer to prune
            min_usage: Minimum token usage to preserve
        """
        self.tokenizer = tokenizer
        self.min_usage = min_usage
        self._usage_counts: Dict[str, int] = {}

    def analyze_usage(
        self,
        texts: List[str],
    ) -> Dict[str, int]:
        """
        Analyze token usage in texts.

        Args:
            texts: Sample texts

        Returns:
            Dictionary of token -> usage count
        """
        usage = Counter()

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for token_id in tokens:
                token = self.tokenizer.inverse_vocab.get(token_id)
                if token:
                    usage[token] += 1

        self._usage_counts = dict(usage)
        return self._usage_counts

    def prune(
        self,
        preserve_tokens: Optional[Set[str]] = None,
    ) -> BPETokenizer:
        """
        Prune vocabulary.

        Args:
            preserve_tokens: Additional tokens to preserve

        Returns:
            Pruned tokenizer
        """
        if not self._usage_counts:
            return self.tokenizer

        preserve_tokens = preserve_tokens or set()

        # Calculate which tokens to remove
        special = set(self.tokenizer.special_tokens.values())
        additional = set(self.tokenizer.additional_special_tokens)

        to_remove = set()
        for token, count in self._usage_counts.items():
            if count < self.min_usage and token not in special and token not in additional:
                if token not in preserve_tokens:
                    to_remove.add(token)

        # Create new tokenizer without removed tokens
        new_vocab = {}
        new_inverse = {}
        new_merges = {}
        new_merge_ranks = {}

        current_id = 0

        for token, old_id in self.tokenizer.vocab.items():
            if token not in to_remove:
                new_vocab[token] = current_id
                new_inverse[current_id] = token
                current_id += 1

        # Update merges
        for pair, merged in self.tokenizer.merges.items():
            if merged not in to_remove:
                new_merges[pair] = merged
                new_merge_ranks[pair] = len(new_merge_ranks)

        # Create new tokenizer
        new_tokenizer = BPETokenizer(
            vocab_size=len(new_vocab),
            special_tokens=self.tokenizer.special_tokens,
        )
        new_tokenizer.vocab = new_vocab
        new_tokenizer.inverse_vocab = new_inverse
        new_tokenizer.merges = new_merges
        new_tokenizer.merge_ranks = new_merge_ranks
        new_tokenizer.additional_special_tokens = self.tokenizer.additional_special_tokens

        return new_tokenizer

    def merge_low_freq(
        self,
        threshold: int = 5,
    ) -> Dict[str, str]:
        """
        Get mapping of low-frequency tokens to their decomposed forms.

        Args:
            threshold: Frequency threshold

        Returns:
            Dictionary of token -> decomposition
        """
        decompose = {}

        for token, count in self._usage_counts.items():
            if count < threshold and len(token) > 1:
                # Try to decompose
                decomp = self._decompose_token(token)
                if decomp != token:
                    decompose[token] = decomp

        return decompose

    def _decompose_token(self, token: str) -> str:
        """Decompose token using merge operations."""
        # Find splits using merge operations
        result = []
        for char in token:
            if char in self.tokenizer.vocab:
                result.append(char)

        # Note: Full decomposition would require BPE decoding
        # This is simplified
        return token

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        if not self._usage_counts:
            return {}

        counts = list(self._usage_counts.values())
        return {
            "total_tokens": sum(counts),
            "unique_tokens": len(counts),
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
            "median": sorted(counts)[len(counts) // 2],
        }


def prune_vocabulary(
    tokenizer: BPETokenizer,
    texts: List[str],
    min_usage: int = 10,
) -> BPETokenizer:
    """
    Convenience function to prune vocabulary.

    Args:
        tokenizer: Tokenizer to prune
        texts: Sample texts for analysis
        min_usage: Minimum usage

    Returns:
        Pruned tokenizer
    """
    pruner = VocabularyPruner(tokenizer, min_usage)
    pruner.analyze_usage(texts)
    return pruner.prune()