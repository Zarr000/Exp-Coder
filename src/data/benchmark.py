"""
Benchmarking utilities for Expera AI data pipeline.

Provides benchmark scripts for:
- Throughput measurement (documents/second, tokens/second)
- Memory usage tracking
- Tokenization speed
- DataLoader performance
- Profiling hooks for bottleneck identification
"""

import time
import gc
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Iterator
from contextlib import contextmanager

# Optional: psutil for memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    metric: str  # 'docs_per_sec', 'tokens_per_sec', 'mb_per_sec', etc.
    value: float
    unit: str
    duration_sec: float
    samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    rss_mb: float
    vms_mb: float
    timestamp: float


class Profiler:
    """
    Profiler for identifying bottlenecks in the data pipeline.

    Tracks timing and memory for different pipeline stages,
    providing reports to identify optimization opportunities.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_PSUTIL
        self._stages: Dict[str, List[float]] = {}
        self._memory_snapshots: Dict[str, List[MemorySnapshot]] = {}
        self._current_stage: Optional[str] = None
        self._stage_start: Optional[float] = None
        self._process = psutil.Process() if HAS_PSUTIL else None

    def start_stage(self, stage_name: str) -> None:
        """Start timing a pipeline stage."""
        if not self.enabled:
            return
        if self._current_stage is not None:
            self.end_stage()
        self._current_stage = stage_name
        self._stage_start = time.perf_counter()
        if stage_name not in self._stages:
            self._stages[stage_name] = []

    def end_stage(self) -> Optional[float]:
        """End timing the current stage and return elapsed time."""
        if not self.enabled or self._current_stage is None:
            return None
        elapsed = time.perf_counter() - self._stage_start
        self._stages[self._current_stage].append(elapsed)
        self._current_stage = None
        self._stage_start = None
        return elapsed

    def record_memory(self, label: str) -> None:
        """Record memory usage for current stage."""
        if not self.enabled:
            return
        snapshot = MemorySnapshot(
            rss_mb=self._process.memory_info().rss / 1024 / 1024,
            vms_mb=self._process.memory_info().vms / 1024 / 1024,
            timestamp=time.perf_counter(),
        )
        if label not in self._memory_snapshots:
            self._memory_snapshots[label] = []
        self._memory_snapshots[label].append(snapshot)

    def get_stage_report(self) -> Dict[str, Dict[str, float]]:
        """Get timing report for all stages."""
        report = {}
        for stage, times in self._stages.items():
            if not times:
                continue
            report[stage] = {
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
            }
        return report

    def get_memory_report(self) -> Dict[str, Dict[str, float]]:
        """Get memory report for all labels."""
        report = {}
        for label, snapshots in self._memory_snapshots.items():
            if not snapshots:
                continue
            rss = [s.rss_mb for s in snapshots]
            report[label] = {
                'count': len(snapshots),
                'rss_mean': sum(rss) / len(rss),
                'rss_peak': max(rss),
                'rss_min': min(rss),
            }
        return report

    def print_report(self) -> None:
        """Print formatted report to stdout."""
        print("\n=== Timing Report ===")
        print(f"{'Stage':<30} {'Count':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
        print("-" * 86)
        for stage, stats in self.get_stage_report().items():
            print(f"{stage:<30} {stats['count']:>8} {stats['total']:>10.3f} {stats['mean']*1000:>10.3f} {stats['min']*1000:>10.3f} {stats['max']*1000:>10.3f}")

        if self._memory_snapshots:
            print("\n=== Memory Report (MB) ===")
            print(f"{'Label':<30} {'Count':>8} {'Mean':>10} {'Peak':>10} {'Min':>10}")
            print("-" * 66)
            for label, stats in self.get_memory_report().items():
                print(f"{label:<30} {stats['count']:>8} {stats['rss_mean']:>10.1f} {stats['rss_peak']:>10.1f} {stats['rss_min']:>10.1f}")

    def reset(self) -> None:
        """Reset all profiling data."""
        self._stages.clear()
        self._memory_snapshots.clear()
        self._current_stage = None
        self._stage_start = None


@contextmanager
def profile_stage(profiler: Profiler, stage_name: str):
    """Context manager for profiling a stage."""
    profiler.start_stage(stage_name)
    try:
        yield profiler
    finally:
        profiler.end_stage()


def benchmark_throughput(
    generator_fn: Callable[[], Iterator[Any]],
    num_samples: int = 1000,
    warmup: int = 100,
) -> BenchmarkResult:
    """
    Benchmark document/inference throughput.

    Args:
        generator_fn: Function that yields documents/results
        num_samples: Number of samples to measure
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with throughput metrics
    """
    # Warmup
    gen = generator_fn()
    for i, _ in enumerate(gen):
        if i >= warmup:
            break

    # Measure
    start_time = time.perf_counter()

    gen = generator_fn()
    count = 0
    for _ in gen:
        count += 1
        if count >= num_samples:
            break

    elapsed = time.perf_counter() - start_time

    return BenchmarkResult(
        name="throughput",
        metric="docs_per_sec",
        value=count / max(elapsed, 0.0001),
        unit="docs/s",
        duration_sec=elapsed,
        samples=count,
    )


def benchmark_tokenization(
    tokenizer: Any,
    texts: List[str],
    max_texts: int = 10000,
) -> BenchmarkResult:
    """
    Benchmark tokenization speed.

    Args:
        tokenizer: Tokenizer with encode method
        texts: List of texts to tokenize
        max_texts: Maximum texts to process

    Returns:
        BenchmarkResult with tokenization metrics
    """
    texts = texts[:max_texts]

    # Warmup
    for text in texts[:min(10, len(texts))]:
        tokenizer.encode(text)

    # Measure
    start_time = time.perf_counter()

    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)

    elapsed = time.perf_counter() - start_time

    return BenchmarkResult(
        name="tokenization",
        metric="tokens_per_sec",
        value=total_tokens / max(elapsed, 0.0001),
        unit="tokens/s",
        duration_sec=elapsed,
        samples=len(texts),
        metadata={"total_tokens": total_tokens},
    )


def benchmark_dataloader(
    dataloader_loader: Callable[[], Iterator[Any]],
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> BenchmarkResult:
    """
    Benchmark DataLoader performance.

    Args:
        dataloader_loader: Function that returns DataLoader
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup batches

    Returns:
        BenchmarkResult with dataloader metrics
    """
    loader = dataloader_loader()

    # Warmup
    for i, _ in enumerate(loader):
        if i >= warmup_batches:
            break

    # Measure
    start_time = time.perf_counter()

    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        # Count tokens
        if hasattr(batch, 'input_ids'):
            tokens = batch.input_ids
        elif isinstance(batch, dict) and 'input_ids' in batch:
            tokens = batch['input_ids']
        else:
            tokens = None

        if tokens is not None:
            if hasattr(tokens, 'numel'):
                total_tokens += tokens.numel()
            else:
                total_tokens += sum(len(t) for t in tokens)

    elapsed = time.perf_counter() - start_time

    return BenchmarkResult(
        name="dataloader",
        metric="tokens_per_sec",
        value=total_tokens / max(elapsed, 0.0001),
        unit="tokens/s",
        duration_sec=elapsed,
        samples=num_batches,
        metadata={"total_tokens": total_tokens},
    )


def benchmark_memory(
    test_fn: Callable[[], Any],
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """
    Benchmark memory usage of a function.

    Args:
        test_fn: Function to benchmark
        num_runs: Number of runs for averaging

    Returns:
        List of BenchmarkResults for each run
    """
    if not HAS_PSUTIL:
        return []

    process = psutil.Process()
    results = []

    for run in range(num_runs):
        gc.collect()
        gc.disable()

        # Measure before
        mem_before = process.memory_info().rss / 1024 / 1024

        # Run function
        test_fn()

        # Measure after
        mem_after = process.memory_info().rss / 1024 / 1024

        gc.enable()

        results.append(BenchmarkResult(
            name=f"memory_run_{run}",
            metric="memory_delta",
            value=mem_after - mem_before,
            unit="MB",
            duration_sec=0,
            samples=1,
        ))

    return results


class PipelineProfiler:
    """
    End-to-end pipeline profiler.

    Profiles all stages of the data pipeline to identify
    bottlenecks and optimization opportunities.
    """

    def __init__(self):
        self.profiler = Profiler()

    def profile_iteration(
        self,
        texts: List[str],
        tokenizer: Any,
        preprocessor: Optional[Any] = None,
        quality_filter: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Profile a single iteration through the pipeline.

        Args:
            texts: Input texts
            tokenizer: Tokenizer
            preprocessor: Optional text preprocessor
            quality_filter: Optional quality filter

        Returns:
            Profiling results
        """
        results = {
            "texts_processed": 0,
            "texts_filtered": 0,
            "total_tokens": 0,
            "stages": {},
        }

        for text in texts:
            # Stage 1: Preprocessing
            if preprocessor:
                self.profiler.start_stage("preprocess")
                cleaned = preprocessor(text)
                self.profiler.end_stage()
                if cleaned is None:
                    results["texts_filtered"] += 1
                    continue
            else:
                cleaned = text

            # Stage 2: Quality filtering
            if quality_filter:
                self.profiler.start_stage("quality_filter")
                score = quality_filter.compute(cleaned)
                self.profiler.end_stage()
                if not score.passed:
                    results["texts_filtered"] += 1
                    continue

            # Stage 3: Tokenization
            self.profiler.start_stage("tokenize")
            tokens = tokenizer.encode(cleaned)
            self.profiler.end_stage()

            results["total_tokens"] += len(tokens)
            results["texts_processed"] += 1

        results["stages"] = self.profiler.get_stage_report()
        return results

    def get_report(self) -> str:
        """Get formatted pipeline report."""
        self.profiler.print_report()
        return ""


def run_full_benchmark(
    tokenizer: Any,
    sample_texts: List[str],
    data_loader_fn: Optional[Callable[[], Iterator[Any]]] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Run complete benchmark suite.

    Args:
        tokenizer: Tokenizer to benchmark
        sample_texts: Sample texts for testing
        data_loader_fn: Optional DataLoader factory

    Returns:
        Dictionary of benchmark results
    """
    results = {}

    # Tokenization benchmark
    print("Running tokenization benchmark...")
    results["tokenization"] = benchmark_tokenization(tokenizer, sample_texts)
    print(f"  Tokens/sec: {results['tokenization'].value:,.0f}")

    # DataLoader benchmark
    if data_loader_fn:
        print("Running dataloader benchmark...")
        results["dataloader"] = benchmark_dataloader(data_loader_fn)
        print(f"  Tokens/sec: {results['dataloader'].value:,.0f}")

    # Memory benchmark
    if HAS_PSUTIL:
        print("Running memory benchmark...")
        results["memory"] = benchmark_memory(lambda: tokenizer.encode(sample_texts[0]))[0]
        print(f"  Memory delta: {results['memory'].value:.1f} MB")
    else:
        print("Skipping memory benchmark (psutil not available)")

    return results