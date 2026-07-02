"""
Tokenizer Evaluator for Expera AI.

Evaluates tokenizer quality:
- Compression metrics
- Coverage metrics
- Code vs text comparison
- Cross-tokenizer comparison
"""

import regex as re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .bpe_tokenizer import BPETokenizer


@dataclass
class EvaluationResult:
    """Tokenizer evaluation results."""
    avg_tokens_per_word: float
    avg_chars_per_token: float
    compression_ratio: float
    vocabulary_utilization: float
    out_of_vocab_rate: float
    code_metrics: Dict[str, float] = field(default_factory=dict)
    text_metrics: Dict[str, float] = field(default_factory=dict)


class TokenizerEvaluator:
    """
    Evaluate tokenizer quality and performance.

    Provides metrics for:
    - Compression efficiency
    - Vocabulary utilization
    - Out-of-vocabulary rates
    - Code vs text comparison
    """

    def __init__(self, tokenizer: BPETokenizer):
        """
        Initialize evaluator.

        Args:
            tokenizer: BPETokenizer to evaluate
        """
        self.tokenizer = tokenizer

    def evaluate(
        self,
        texts: List[str],
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation.

        Args:
            texts: Sample texts for evaluation

        Returns:
            EvaluationResult with metrics
        """
        # Basic metrics
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(re.findall(r'\w+', t)) for t in texts)

        # Token counts
        total_tokens = 0
        unk_count = 0
        unk_token_id = self.tokenizer.vocab.get(self.tokenizer.special_tokens["unk_token"])

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            unk_count += sum(1 for t in tokens if t == unk_token_id)

        # Compute metrics
        avg_tokens_per_word = total_tokens / max(1, total_words)
        avg_chars_per_token = total_chars / max(1, total_tokens)
        compression_ratio = total_chars / max(1, total_tokens)

        # Vocabulary utilization
        unique_tokens = set()
        for text in texts[:min(1000, len(texts))]:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            unique_tokens.update(tokens)
        vocab_util = len(unique_tokens) / max(1, len(self.tokenizer.vocab))

        # OOV rate
        oov_rate = unk_count / max(1, total_tokens)

        # Separate code and text metrics
        code_texts = self._extract_code(texts)
        plain_texts = [t for t in texts if t not in code_texts]

        code_metrics = self._evaluate_set(code_texts) if code_texts else {}
        text_metrics = self._evaluate_set(plain_texts) if plain_texts else {}

        return EvaluationResult(
            avg_tokens_per_word=avg_tokens_per_word,
            avg_chars_per_token=avg_chars_per_token,
            compression_ratio=compression_ratio,
            vocabulary_utilization=vocab_util,
            out_of_vocab_rate=oov_rate,
            code_metrics=code_metrics,
            text_metrics=text_metrics,
        )

    def _extract_code(self, texts: List[str]) -> List[str]:
        """Extract code-like texts."""
        code_indicators = [
            r'\bdef\b', r'\bclass\b', r'\bfunction\b',
            r'\bimport\b', r'\breturn\b', r'\bif\s*\(',
            r'\{', r'\}', r'\)', r';',
        ]
        return [t for t in texts if any(re.search(p, t) for p in code_indicators)]

    def _evaluate_set(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate metrics for a set of texts."""
        if not texts:
            return {}

        total_chars = sum(len(t) for t in texts)
        total_tokens = 0

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)

        return {
            "avg_chars_per_token": total_chars / max(1, total_tokens),
            "sample_count": len(texts),
        }

    def evaluate_code(
        self,
        code_samples: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate tokenizer on code samples.

        Args:
            code_samples: Code texts

        Returns:
            Dictionary of code-specific metrics
        """
        return self._evaluate_set(code_samples)

    def evaluate_text(
        self,
        text_samples: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate tokenizer on natural language samples.

        Args:
            text_samples: Natural language texts

        Returns:
            Dictionary of text-specific metrics
        """
        return self._evaluate_set(text_samples)

    def compare(
        self,
        other_tokenizer: BPETokenizer,
        texts: List[str],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compare two tokenizers on the same texts.

        Args:
            other_tokenizer: Another tokenizer to compare
            texts: Sample texts

        Returns:
            Dictionary of (our_metric, other_metric) tuples
        """
        # Our metrics
        our_total = 0
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            our_total += len(tokens)

        # Other metrics
        other_total = 0
        for text in texts:
            tokens = other_tokenizer.encode(text, add_special_tokens=False)
            other_total += len(tokens)

        return {
            "token_count": (our_total, other_total),
            "avg_tokens_per_doc": (
                our_total / max(1, len(texts)),
                other_total / max(1, len(texts)),
            ),
        }


def evaluate_tokenizer(
    tokenizer: BPETokenizer,
    texts: List[str],
) -> EvaluationResult:
    """
    Convenience function to evaluate tokenizer.

    Args:
        tokenizer: Tokenizer to evaluate
        texts: Sample texts

    Returns:
        EvaluationResult
    """
    evaluator = TokenizerEvaluator(tokenizer)
    return evaluator.evaluate(texts)