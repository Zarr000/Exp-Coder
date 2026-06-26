"""
Multi-dimensional Quality Filtering for Expera AI.

Scores documents across multiple quality dimensions:
1. Length distribution (too short/long)
2. Special character ratio (gibberish detection)
3. Repetition ratio (duplicate n-gram density)
4. Code-to-comment ratio (code quality)
5. Perplexity against reference (optional)
6. Composite quality score

Supports configurable scoring pipelines and thresholds.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any


@dataclass
class QualityScore:
    """Quality score for a single document."""
    overall: float
    dimensions: Dict[str, float]
    passed: bool = True


class QualityFilter:
    """
    Multi-dimensional quality filter with configurable pipeline.
    
    Quality dimensions are scored from 0.0 (worst) to 1.0 (best).
    The composite score is a weighted combination.
    Threshold can be absolute or percentile-based.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.3,
        min_doc_length: int = 50,
        max_doc_length: int = 1000000,
        max_repetition_ratio: float = 0.3,
        max_special_char_ratio: float = 0.4,
        min_code_comment_ratio: float = 0.01,
    ):
        self.weights = weights or {
            "length_score": 0.15,
            "special_char_score": 0.25,
            "repetition_score": 0.30,
            "code_comment_score": 0.10,
            "line_length_score": 0.20,
        }
        self.threshold = threshold
        
        # Configurable thresholds
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.max_repetition_ratio = max_repetition_ratio
        self.max_special_char_ratio = max_special_char_ratio
        self.min_code_comment_ratio = min_code_comment_ratio
    
    def score_length(self, text: str) -> float:
        """Score based on document length."""
        length = len(text)
        if length < self.min_doc_length:
            return length / self.min_doc_length
        if length > self.max_doc_length:
            return self.max_doc_length / length
        return 1.0
    
    def score_special_chars(self, text: str) -> float:
        """
        Score based on special character ratio.
        
        High ratio of non-alphanumeric chars suggests gibberish.
        """
        if not text:
            return 0.0
        
        # Count alphanumeric vs total characters
        alnum = sum(1 for c in text if c.isalnum() or c.isspace())
        ratio = 1.0 - (alnum / len(text))
        
        if ratio > self.max_special_char_ratio:
            return max(0.0, 1.0 - ratio)
        return 1.0
    
    def score_repetition(self, text: str) -> float:
        """
        Score based on n-gram repetition density.
        
        High repetition suggests boilerplate or spam.
        """
        if len(text) < 100:
            return 1.0
        
        # Check line repetition
        lines = text.split('\n')
        if len(lines) > 10:
            unique_lines = len(set(lines))
            line_rep_ratio = 1.0 - (unique_lines / len(lines))
        else:
            line_rep_ratio = 0.0
        
        # Check character 5-gram repetition
        n = 5
        if len(text) > n:
            ngrams = set()
            total = 0
            repeated = 0
            for i in range(len(text) - n):
                ng = text[i:i+n]
                if ng in ngrams:
                    repeated += 1
                ngrams.add(ng)
                total += 1
            ngram_rep_ratio = repeated / max(1, total)
        else:
            ngram_rep_ratio = 0.0
        
        # Combined score
        rep_ratio = max(line_rep_ratio, ngram_rep_ratio)
        if rep_ratio > self.max_repetition_ratio:
            return max(0.0, 1.0 - rep_ratio)
        return 1.0
    
    def score_line_lengths(self, text: str) -> float:
        """
        Score based on line length distribution.
        
        Penalizes documents with extremely long or short lines.
        """
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return 0.0
        
        long_lines = sum(1 for l in lines if len(l) > 500)
        short_lines = sum(1 for l in lines if len(l) < 10)
        
        long_ratio = long_lines / len(lines)
        short_ratio = short_lines / len(lines)
        
        # Penalize extreme values
        score = 1.0
        if long_ratio > 0.3:
            score -= (long_ratio - 0.3) * 2
        if short_ratio > 0.5:
            score -= (short_ratio - 0.5)
        
        return max(0.0, score)
    
    def score_code_comment_ratio(self, text: str) -> float:
        """
        Score code based on comment ratio.
        
        Code with no comments may be low quality.
        Code that is mostly comments may be documentation.
        """
        # Simple comment detection
        comment_lines = 0
        total_code_lines = 0
        
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            total_code_lines += 1
            if stripped.startswith('#') or stripped.startswith('//') or \
               stripped.startswith('/*') or stripped.startswith('*') or \
               stripped.startswith('--') or stripped.startswith('%'):
                comment_lines += 1
        
        if total_code_lines == 0:
            return 0.5  # Neutral for non-code
        
        ratio = comment_lines / total_code_lines
        
        # Ideal: some comments but not too many
        if ratio < self.min_code_comment_ratio:
            return ratio / self.min_code_comment_ratio * 0.5  # Penalize no comments
        elif ratio > 0.8:
            return max(0.0, 1.0 - ratio)  # Penalize too many comments
        return 1.0
    
    def compute(self, text: str) -> QualityScore:
        """
        Compute quality scores for a document.
        
        Args:
            text: Document text
            
        Returns:
            QualityScore with per-dimension and overall scores
        """
        dimensions = {
            "length_score": self.score_length(text),
            "special_char_score": self.score_special_chars(text),
            "repetition_score": self.score_repetition(text),
            "code_comment_score": self.score_code_comment_ratio(text),
            "line_length_score": self.score_line_lengths(text),
        }
        
        # Weighted composite score
        overall = sum(
            dimensions[dim] * self.weights.get(dim, 0.0)
            for dim in dimensions
        )
        total_weight = sum(self.weights.get(dim, 0.0) for dim in dimensions)
        if total_weight > 0:
            overall /= total_weight
        
        return QualityScore(
            overall=overall,
            dimensions=dimensions,
            passed=overall >= self.threshold,
        )
    
    def filter_batch(
        self,
        texts: List[str],
        batch_size: int = 1000,
    ) -> List[bool]:
        """
        Filter multiple documents.
        
        Args:
            texts: List of document texts
            batch_size: Processing batch size
            
        Returns:
            List of boolean pass/fail results
        """
        results = []
        for text in texts:
            score = self.compute(text)
            results.append(score.passed)
        return results