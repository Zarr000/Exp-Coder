"""
DataLoader factory for Expera AI.

Creates PyTorch DataLoaders with:
- Multi-worker support
- Prefetching
- Custom collation
- Profiling hooks
- Benchmarking support
"""

import time
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    batch_size: int = 1  # 1 for pre-batched IterableDataset
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    timeout: int = 0


def collate_batch(batch):
    """Collate function for pre-batched data."""
    if len(batch) == 0:
        return {}
    if isinstance(batch[0], dict):
        return batch[0]
    return batch[0]


def get_dataloader(
    dataset: IterableDataset,
    config: Optional[DataLoaderConfig] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: PyTorch IterableDataset
        config: DataLoader configuration
        collate_fn: Custom collate function
        
    Returns:
        Configured DataLoader
    """
    if config is None:
        config = DataLoaderConfig()
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
        timeout=config.timeout,
        collate_fn=collate_fn or collate_batch,
    )


class DataLoaderBenchmark:
    """Benchmark data loading performance."""
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.metrics: Dict[str, List[float]] = {
            "batch_times": [],
            "tokens_per_sec": [],
            "memory_mb": [],
        }
    
    def run(self, num_batches: int = 100) -> Dict[str, float]:
        """Run benchmark and return summary metrics."""
        import psutil
        process = psutil.Process()
        
        start_time = time.time()
        total_tokens = 0
        
        for i, batch in enumerate(self.dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Count tokens
            if hasattr(batch, 'input_ids'):
                tokens = sum(len(ids) for ids in batch.input_ids)
            elif isinstance(batch, dict) and 'input_ids' in batch:
                tokens = batch['input_ids'].numel()
            else:
                tokens = 0
            total_tokens += tokens
            
            batch_time = time.time() - batch_start
            self.metrics["batch_times"].append(batch_time)
            self.metrics["tokens_per_sec"].append(tokens / max(0.001, batch_time))
            self.metrics["memory_mb"].append(process.memory_info().rss / 1024 / 1024)
        
        elapsed = time.time() - start_time
        
        return {
            "total_batches": len(self.metrics["batch_times"]),
            "total_tokens": total_tokens,
            "total_time": elapsed,
            "avg_batch_time": sum(self.metrics["batch_times"]) / max(1, len(self.metrics["batch_times"])),
            "avg_tokens_per_sec": total_tokens / max(0.001, elapsed),
            "peak_memory_mb": max(self.metrics["memory_mb"]) if self.metrics["memory_mb"] else 0,
        }