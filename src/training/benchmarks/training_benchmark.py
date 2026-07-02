"""
Training benchmarking for Expera AI.

Provides:
- Throughput benchmarking
- Memory benchmarking
- Scaling benchmarks
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

# Try to import pynvml for GPU metrics
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass
class BenchmarkResult:
    """Result of training benchmark."""
    name: str
    batch_size: int = 0
    seq_len: int = 0
    num_steps: int = 0

    # Timing
    total_time: float = 0.0
    avg_step_time: float = 0.0

    # Throughput
    tokens_per_sec: float = 0.0
    batches_per_sec: float = 0.0

    # Memory
    gpu_memory_gb: float = 0.0
    peak_memory_gb: float = 0.0

    # GPU util
    avg_gpu_util: float = 0.0

    # Additional metrics
    extra: Dict[str, float] = field(default_factory=dict)


class TrainingBenchmark:
    """
    Benchmark training performance.

    Measures:
    - Throughput (tokens/sec)
    - Memory usage
    - GPU utilization
    - Scaling efficiency
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model
        self.device = torch.device(device)
        self.use_amp = use_amp

        # Initialize pynvml
        if HAS_PYNVML:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def benchmark_throughput(
        self,
        batch_size: int = 8,
        seq_len: int = 1024,
        num_steps: int = 100,
        warmup_steps: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark throughput.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_steps: Number of steps to benchmark
            warmup_steps: Number of warmup steps

        Returns:
            BenchmarkResult
        """
        self.model.train()

        # Create dummy input
        input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(self.device)
        labels = input_ids.clone()

        # Dummy optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(warmup_steps):
            output = self.model(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        step_times = []

        for _ in range(num_steps):
            start = time.time()

            output = self.model(input_ids)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            step_times.append(time.time() - start)

        # Compute metrics
        total_time = sum(step_times)
        avg_step_time = total_time / num_steps

        tokens_per_sec = (batch_size * seq_len) / avg_step_time
        batches_per_sec = 1 / avg_step_time

        # Memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_memory = 0.0
            peak_memory = 0.0

        # GPU util
        avg_gpu_util = 0.0
        if self.handle:
            try:
                avg_gpu_util = sum(
                    pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    for _ in range(num_steps)
                ) / num_steps
            except Exception:
                pass

        return BenchmarkResult(
            name="throughput",
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            total_time=total_time,
            avg_step_time=avg_step_time,
            tokens_per_sec=tokens_per_sec,
            batches_per_sec=batches_per_sec,
            gpu_memory_gb=gpu_memory,
            peak_memory_gb=peak_memory,
            avg_gpu_util=avg_gpu_util,
        )

    def benchmark_scaling(
        self,
        batch_sizes: List[int],
        seq_len: int = 1024,
        num_steps: int = 50,
    ) -> List[BenchmarkResult]:
        """
        Benchmark scaling with different batch sizes.

        Args:
            batch_sizes: List of batch sizes
            seq_len: Sequence length
            num_steps: Number of steps

        Returns:
            List of benchmark results
        """
        results = []

        for bs in batch_sizes:
            result = self.benchmark_throughput(
                batch_size=bs,
                seq_len=seq_len,
                num_steps=num_steps,
            )
            results.append(result)

        return results

    def benchmark_memory(
        self,
        batch_size: int = 8,
        seq_len: int = 1024,
    ) -> Dict[str, float]:
        """
        Benchmark memory usage.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Memory metrics
        """
        if not torch.cuda.is_available():
            return {}

        # Reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.model.train()

        # Forward
        input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(self.device)

        output = self.model(input_ids)

        forward_memory = torch.cuda.memory_allocated() / 1e9

        # Backward
        loss = output.sum()
        loss.backward()

        backward_memory = torch.cuda.memory_allocated() / 1e9

        # Optimizer step
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        optimizer.step()

        optimizer_memory = torch.cuda.memory_allocated() / 1e9

        # Peak
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        return {
            "forward_memory_gb": forward_memory,
            "backward_memory_gb": backward_memory,
            "optimizer_memory_gb": optimizer_memory,
            "peak_memory_gb": peak_memory,
        }

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not HAS_PYNVML or not self.handle:
            return {}

        try:
            name = pynvml.nvmlDeviceGetName(self.handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            return {
                "name": name,
                "total_memory_gb": memory.total / 1e9,
                "free_memory_gb": memory.free / 1e9,
                "used_memory_gb": memory.used / 1e9,
            }
        except Exception:
            return {}

    def close(self) -> None:
        """Cleanup."""
        if HAS_PYNVML:
            pynvml.nvmlExit()


def run_training_benchmark(
    model: nn.Module,
    batch_size: int = 8,
    seq_len: int = 1024,
    num_steps: int = 100,
) -> BenchmarkResult:
    """
    Run training benchmark.

    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of steps

    Returns:
        BenchmarkResult
    """
    benchmark = TrainingBenchmark(model)
    result = benchmark.benchmark_throughput(
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
    )
    benchmark.close()
    return result