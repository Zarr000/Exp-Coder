"""
Dataset statistics and reporting for Expera AI.

Tracks:
- Per-shard and global statistics
- Language distributions
- Quality score distributions
- Token counts and compression ratios
- Live monitoring during training
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class DatasetStats:
    """Comprehensive dataset statistics."""
    name: str = ""
    total_documents: int = 0
    total_bytes: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    quality_distribution: Dict[str, float] = field(default_factory=dict)
    avg_doc_length: float = 0.0
    avg_tokens_per_seq: float = 0.0
    compression_ratio: float = 0.0
    duplicates_removed: int = 0
    documents_filtered: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_documents": self.total_documents,
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "languages": dict(self.languages),
            "quality_distribution": dict(self.quality_distribution),
            "avg_doc_length": self.avg_doc_length,
            "avg_tokens_per_seq": self.avg_tokens_per_seq,
            "compression_ratio": self.compression_ratio,
            "duplicates_removed": self.duplicates_removed,
            "documents_filtered": self.documents_filtered,
            "processing_time": self.processing_time,
        }
    
    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name}",
            f"  Documents: {self.total_documents:,}",
            f"  Tokens: {self.total_tokens:,}",
            f"  Sequences: {self.total_sequences:,}",
            f"  Avg doc length: {self.avg_doc_length:.1f} chars",
            f"  Compression: {self.compression_ratio:.2f} chars/token",
            f"  Duplicates removed: {self.duplicates_removed:,}",
            f"  Filtered: {self.documents_filtered:,}",
            f"  Processing time: {self.processing_time:.1f}s",
        ]
        if self.languages:
            lines.append("  Languages:")
            for lang, count in sorted(self.languages.items(), key=lambda x: -x[1]):
                pct = count / max(1, self.total_documents) * 100
                lines.append(f"    {lang}: {count:,} ({pct:.1f}%)")
        return '\n'.join(lines)


class StatsCollector:
    """
    Collects and aggregates statistics during dataset processing.
    
    Supports:
    - Incremental updates (streaming)
    - Per-shard aggregation
    - Live reporting
    - JSON export
    """
    
    def __init__(self, name: str = "dataset"):
        self.name = name
        self.stats = DatasetStats(name=name)
        self._start_time = time.time()
        self._doc_lengths: List[int] = []
        self._quality_scores: List[float] = []
    
    def record_document(self, text: str, language: str = "unknown",
                        quality_score: float = 0.0) -> None:
        """Record a single document's statistics."""
        self.stats.total_documents += 1
        self.stats.total_bytes += len(text.encode('utf-8'))
        self._doc_lengths.append(len(text))
        self._quality_scores.append(quality_score)
        
        # Language tracking
        self.stats.languages[language] = self.stats.languages.get(language, 0) + 1
    
    def record_tokens(self, num_tokens: int) -> None:
        """Record token count."""
        self.stats.total_tokens += num_tokens
    
    def record_sequence(self, length: int) -> None:
        """Record a packed sequence."""
        self.stats.total_sequences += 1
    
    def record_duplicate(self) -> None:
        """Record a duplicate removal."""
        self.stats.duplicates_removed += 1
    
    def record_filtered(self) -> None:
        """Record a filtered document."""
        self.stats.documents_filtered += 1
    
    def finalize(self) -> DatasetStats:
        """Compute final statistics."""
        self.stats.processing_time = time.time() - self._start_time
        
        if self._doc_lengths:
            self.stats.avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        
        if self.stats.total_tokens > 0:
            self.stats.compression_ratio = self.stats.total_bytes / self.stats.total_tokens
        
        if self.stats.total_sequences > 0:
            self.stats.avg_tokens_per_seq = self.stats.total_tokens / self.stats.total_sequences
        
        # Quality distribution
        if self._quality_scores:
            scores = sorted(self._quality_scores)
            n = len(scores)
            self.stats.quality_distribution = {
                "min": scores[0],
                "p25": scores[n // 4],
                "median": scores[n // 2],
                "p75": scores[3 * n // 4],
                "max": scores[-1],
                "mean": sum(scores) / n,
            }
        
        return self.stats
    
    def report(self) -> str:
        """Generate live progress report."""
        elapsed = time.time() - self._start_time
        docs_per_sec = self.stats.total_documents / max(1, elapsed)
        return (
            f"[{self.name}] {self.stats.total_documents:,} docs | "
            f"{self.stats.total_tokens:,} tokens | "
            f"{docs_per_sec:.0f} docs/s | "
            f"{elapsed:.0f}s elapsed"
        )