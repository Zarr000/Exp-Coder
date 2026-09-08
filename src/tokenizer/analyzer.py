"""
Vocabulary Analyzer for Expera AI.

Analyzes tokenizer vocabulary:
- Token statistics
- Coverage analysis
- Frequency distributions
- Compression metrics
"""

import regex as re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

from .bpe_tokenizer import BPETokenizer


@dataclass
class VocabAnalysis:
    """Vocabulary analysis results."""
    vocab_size: int
    avg_token_length: float
    max_token_length: int
    min_token_length: int
    coverage: Dict[str, float]  # language -> coverage
    compression_ratio: float
    token_distribution: Dict[str, float]  # token type -> count
    special_token_count: int
    byte_token_count: int
    merged_token_count: int


class VocabularyAnalyzer:
    """
    Analyze tokenizer vocabulary quality and coverage.

    Provides insights into vocabulary composition,
    coverage for different languages, and compression metrics.
    """

    def __init__(self, tokenizer: BPETokenizer):
        """
        Initialize analyzer.

        Args:
            tokenizer: BPETokenizer to analyze
        """
        self.tokenizer = tokenizer

    def analyze(self, texts: List[str]) -> VocabAnalysis:
        """
        Perform comprehensive vocabulary analysis.

        Args:
            texts: Sample texts for analysis

        Returns:
            VocabAnalysis with results
        """
        # Gather statistics
        vocab = self.tokenizer.vocab
        vocab_size = len(vocab)

        # Token lengths
        token_lengths = [len(t) for t in vocab.keys()]
        avg_token_length = sum(token_lengths) / max(1, len(token_lengths))
        max_token_length = max(token_lengths)
        min_token_length = min(token_lengths)

        # Token type counts
        special_tokens = set(self.tokenizer.special_tokens.values())
        additional_special = set(self.tokenizer.additional_special_tokens)

        special_count = 0
        byte_count = 0
        merged_count = 0

        for token in vocab.keys():
            if token in special_tokens or token in additional_special:
                special_count += 1
            elif len(token) == 1:
                byte_count += 1
            else:
                merged_count += 1

        # Coverage by language
        coverage = self.analyze_coverage(texts)

        # Compression ratio
        total_chars = sum(len(t) for t in texts)
        if total_chars > 0:
            # Estimate tokens per text
            total_tokens = 0
            for text in texts[:min(1000, len(texts))]:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
            avg_tokens = total_tokens / min(1000, len(texts))
            avg_chars_per_text = total_chars / min(1000, len(texts))
            compression_ratio = avg_chars_per_text / max(1, avg_tokens)
        else:
            compression_ratio = 0.0

        # Token distribution
        token_dist = self._get_token_distribution(texts)

        return VocabAnalysis(
            vocab_size=vocab_size,
            avg_token_length=avg_token_length,
            max_token_length=max_token_length,
            min_token_length=min_token_length,
            coverage=coverage,
            compression_ratio=compression_ratio,
            token_distribution=token_dist,
            special_token_count=special_count,
            byte_token_count=byte_count,
            merged_token_count=merged_count,
        )

    def analyze_coverage(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze vocabulary coverage for different text types.

        Args:
            texts: Sample texts

        Returns:
            Dictionary of coverage by text type
        """
        # Known code patterns
        code_patterns = {
            "python": r"(def |class |import |from |import|from|__\w+__|print\()|if __name__)",
            "javascript": r"(function |const |let |var |=>|require\(|module\.exports)",
            "java": r"(public |private |class |void |static |import java\.)",
            "cpp": r"(#include |std::|cout|cin|namespace |template<)",
            "go": r"(func |package |import \(|type |struct {|interface)",
            "rust": r"(fn |let |mut |impl |pub |mod |use )",
        }

        # Text patterns (natural language)
        text_indicators = {
            "en": r"\bthe\b|\band\b|\bto\b|\bof\b|\bin\b",
            "de": r"\bder\b|\bdie\b|\bund\b|\bden\b|\bdas\b",
            "fr": r"\ble\b|\bla\b|\bles\b|\bdes\b|\bet\b",
            "es": r"\bel\b|\bla\b|\bque\b|\bde\b|\bly\b",
        }

        coverage = {}

        # Analyze code coverage
        for lang, pattern in code_patterns.items():
            matches = sum(1 for t in texts if re.search(pattern, t))
            if matches > 0:
                # Sample from this language
                sample = [t for t in texts if re.search(pattern, t)][:min(100, len(texts))]
                covered, total = self._compute_coverage(sample)
                coverage[lang] = covered / max(1, total)

        # Analyze natural language coverage
        for lang, pattern in text_indicators.items():
            matches = sum(1 for t in texts if re.search(pattern, t, re.IGNORECASE))
            if matches > 0:
                sample = [t for t in texts if re.search(pattern, t, re.IGNORECASE)][:min(100, len(texts))]
                covered, total = self._compute_coverage(sample)
                coverage[f"nl_{lang}"] = covered / max(1, total)

        # Default coverage
        if not coverage:
            covered, total = self._compute_coverage(texts[:1000])
            coverage["default"] = covered / max(1, total)

        return coverage

    def _compute_coverage(self, texts: List[str]) -> Tuple[int, int]:
        """Compute vocabulary coverage for texts."""
        covered = 0
        total = 0

        for text in texts:
            tokens = re.findall(self.tokenizer.pat, text)
            for token in tokens:
                # Check if token is in vocabulary
                if token in self.tokenizer.vocab:
                    covered += 1
                total += 1

        return covered, total

    def get_token_frequency(
        self,
        texts: List[str],
    ) -> Dict[str, int]:
        """
        Get token frequency distribution.

        Args:
            texts: Sample texts

        Returns:
            Dictionary of token -> frequency
        """
        freq = Counter()

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for token_id in tokens:
                token = self.tokenizer.inverse_vocab.get(token_id)
                if token:
                    freq[token] += 1

        return dict(freq)

    def get_coverage_curve(
        self,
        texts: List[str],
    ) -> List[Tuple[int, float]]:
        """
        Get coverage curve as vocabulary grows.

        Args:
            texts: Sample texts

        Returns:
            List of (vocab_size, coverage) points
        """
        # Get frequencies
        freq = self.get_token_frequency(texts)
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])

        # Compute cumulative coverage
        points = []
        vocab_set: Set[str] = set()
        total_tokens = sum(freq.values())

        for token, _ in sorted_tokens:
            vocab_set.add(token)
            # Estimate coverage
            covered = sum(freq[t] for t in vocab_set if t in freq)
            coverage = covered / max(1, total_tokens)
            points.append((len(vocab_set), coverage))

            if len(points) >= 100:
                break

        return points

    def _get_token_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Get distribution of token types."""
        counts = {
            "special": 0,
            "byte": 0,
            "merged": 0,
            "rare": 0,
        }

        special_ids = {
            self.tokenizer.vocab.get(t)
            for t in self.tokenizer.special_tokens.values()
        }
        additional_ids = {
            self.tokenizer.vocab.get(t)
            for t in self.tokenizer.additional_special_tokens
        }

        special_ids.update(additional_ids)

        for text in texts[:min(1000, len(texts))]:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for token_id in tokens:
                if token_id in special_ids:
                    counts["special"] += 1
                elif token_id < 256:
                    counts["byte"] += 1
                else:
                    counts["merged"] += 1

        total = sum(counts.values())
        if total > 0:
            return {k: v / total for k, v in counts.items()}
        return counts


def analyze_vocabulary(
    tokenizer: BPETokenizer,
    texts: List[str],
) -> VocabAnalysis:
    """
    Convenience function to analyze vocabulary.

    Args:
        tokenizer: Tokenizer to analyze
        texts: Sample texts

    Returns:
        VocabAnalysis results
    """
    analyzer = VocabularyAnalyzer(tokenizer)
    return analyzer.analyze(texts)