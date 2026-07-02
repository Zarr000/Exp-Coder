"""
Tokenizer Benchmark for Expera AI.

Benchmarks tokenizer performance:
- Encoding speed
- Decoding speed
- Batch processing
- Token counting
"""

import time
from dataclasses import dataclass
from typing import List, Callable, Optional
import random

from .bpe_tokenizer import BPETokenizer


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    name: str
    encode_time_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    avg_tokens_per_text: float


class TokenizerBenchmark:
    """
    Benchmark tokenizer performance.

    Measures:
    - Encode speed (tokens/second)
    - Decode speed
    - Batch throughput
    - Memory usage (if psutil available)
    """

    def __init__(self, tokenizer: BPETokenizer):
        """
        Initialize benchmark.

        Args:
            tokenizer: Tokenizer to benchmark
        """
        self.tokenizer = tokenizer

    def benchmark_encode(
        self,
        texts: List[str],
        warmup: int = 100,
        iterations: int = 1000,
    ) -> BenchmarkResult:
        """
        Benchmark encoding speed.

        Args:
            texts: Sample texts
            warmup: Warmup iterations
            iterations: Benchmark iterations

        Returns:
            BenchmarkResult
        """
        texts = texts * ((iterations // len(texts)) + 1)

        # Warmup
        for text in texts[:warmup]:
            self.tokenizer.encode(text, add_special_tokens=False)

        # Benchmark
        start = time.perf_counter()
        total_tokens = 0

        for text in texts[:iterations]:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)

        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            name="encode",
            encode_time_ms=elapsed * 1000,
            decode_time_ms=0,
            tokens_per_sec=total_tokens / max(0.001, elapsed),
            avg_tokens_per_text=total_tokens / iterations,
        )

    def benchmark_decode(
        self,
        texts: List[str],
        warmup: int = 100,
        iterations: int = 1000,
    ) -> BenchmarkResult:
        """
        Benchmark decoding speed.

        Args:
            texts: Sample texts (will be encoded first)
            warmup: Warmup iterations
            iterations: Benchmark iterations

        Returns:
            BenchmarkResult
        """
        # Pre-encode texts
        encoded = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            encoded.append(tokens)

        encoded = encoded * ((iterations // len(encoded)) + 1)

        # Warmup
        for tokens in encoded[:warmup]:
            self.tokenizer.decode(tokens, skip_special_tokens=True)

        # Benchmark
        start = time.perf_counter()

        for tokens in encoded[:iterations]:
            self.tokenizer.decode(tokens, skip_special_tokens=True)

        elapsed = time.perf_counter() - start

        # Estimate tokens decoded
        total_tokens = sum(len(t) for t in encoded[:iterations])

        return BenchmarkResult(
            name="decode",
            encode_time_ms=0,
            decode_time_ms=elapsed * 1000,
            tokens_per_sec=total_tokens / max(0.001, elapsed),
            avg_tokens_per_text=0,
        )

    def benchmark_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Benchmark batch processing.

        Args:
            texts: Sample texts
            batch_size: Batch size
            iterations: Number of batches

        Returns:
            BenchmarkResult
        """
        # Warmup
        for i in range(min(warmup, len(texts))):
            batch = texts[i * batch_size:(i + 1) * batch_size]
            for text in batch:
                self.tokenizer.encode(text, add_special_tokens=False)

        # Benchmark
        start = time.perf_counter()
        total_tokens = 0

        for _ in range(iterations):
            batch = texts[:batch_size]
            for text in batch:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)

        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            name="batch",
            encode_time_ms=elapsed * 1000,
            decode_time_ms=0,
            tokens_per_sec=total_tokens / max(0.001, elapsed),
            avg_tokens_per_text=total_tokens / (iterations * batch_size),
        )

    def benchmark_all(
        self,
        texts: List[str],
    ) -> List[BenchmarkResult]:
        """
        Run all benchmarks.

        Args:
            texts: Sample texts

        Returns:
            List of BenchmarkResults
        """
        results = []

        # Encode benchmark
        results.append(self.benchmark_encode(texts))

        # Decode benchmark
        results.append(self.benchmark_decode(texts))

        # Batch benchmark
        results.append(self.benchmark_batch(texts))

        return results


def benchmark_tokenizer(
    tokenizer: BPETokenizer,
    texts: List[str],
) -> List[BenchmarkResult]:
    """
    Convenience function to benchmark tokenizer.

    Args:
        tokenizer: Tokenizer to benchmark
        texts: Sample texts

    Returns:
        List of BenchmarkResults
    """
    benchmark = TokenizerBenchmark(tokenizer)
    return benchmark.benchmark_all(texts)