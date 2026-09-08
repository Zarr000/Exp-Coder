"""
GPU Monitor for Expera AI.

Monitors GPU resources:
- Memory usage
- Compute utilization
- Temperature
- Power usage

Usage:
    monitor = GPUMonitor()
    stats = monitor.get_stats()
    print(f"GPU: {stats['memory_used']} / {stats['memory_total']} MB")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU statistics snapshot."""

    name: str
    device_id: int
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    compute_capability: tuple[int, int]
    utilization: float  # 0-100
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    timestamp: float = field(default_factory=time.time)


@dataclass
class GPUMonitorConfig:
    """GPU monitor configuration."""

    polling_interval: float = 1.0  # seconds
    history_size: int = 60  # samples to keep
    alert_threshold: float = 90.0  # utilization % to alert
    memory_threshold: float = 90.0  # memory % to alert


class GPUMonitor:
    """
    Monitors GPU resources.

    Features:
    - Real-time stats
    - History tracking
    - Alerts
    """

    def __init__(self, config: Optional[GPUMonitorConfig] = None):
        """Initialize GPU monitor."""
        self.config = config or GPUMonitorConfig()
        self._history: list[GPUStats] = []
        self._pynvml_initialized = False
        self._nvml_handle = None

    def _init_nvml(self) -> bool:
        """Initialize NVML."""
        if self._pynvml_initialized:
            return True

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml
            self._pynvml_initialized = True
            logger.info("NVML initialized")
            return True
        except ImportError:
            logger.debug("pynvml not available, using torch fallback")
            return False
        except Exception as e:
            logger.warning(f"NVML init failed: {e}")
            return False

    def get_stats(self, device_id: int = 0) -> Optional[GPUStats]:
        """Get current GPU stats."""
        # Try NVML first
        if self._init_nvml() and self._nvml_handle:
            return self._get_stats_nvml(device_id)

        # Fallback to torch
        return self._get_stats_torch(device_id)

    def _get_stats_nvml(self, device_id: int) -> Optional[GPUStats]:
        """Get stats via NVML."""
        nvml = self._nvml_handle

        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(device_id)

            # Memory
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = mem_info.total // (1024 * 1024)
            memory_used = mem_info.used // (1024 * 1024)
            memory_free = mem_info.free // (1024 * 1024)

            # Name
            name = nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Compute capability
            cap = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            utilization = 0.0

            # Temperature
            try:
                temperature = nvml.nvmlDeviceGetTemperature(
                    handle, nvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temperature = None

            # Power
            try:
                power = nvml.nvmlDeviceGetPowerUsage(handle)
                power = power / 1000.0  # mW to W
            except Exception:
                power = None

            return GPUStats(
                name=name,
                device_id=device_id,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_free=memory_free,
                compute_capability=cap,
                utilization=utilization,
                temperature=temperature,
                power_usage=power,
            )

        except Exception as e:
            logger.warning(f"NVML stats failed: {e}")
            return None

    def _get_stats_torch(self, device_id: int) -> Optional[GPUStats]:
        """Get stats via PyTorch."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            # Memory
            memory_total = torch.cuda.get_device_properties(device_id).total_memory // (1024 * 1024)
            memory_allocated = torch.cuda.memory_allocated(device_id) // (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved(device_id) // (1024 * 1024)
            memory_used = memory_allocated
            memory_free = memory_total - memory_used

            # Name
            name = torch.cuda.get_device_name(device_id)

            # Compute capability
            cap = torch.cuda.get_device_capability(device_id)

            return GPUStats(
                name=name,
                device_id=device_id,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_free=memory_free,
                compute_capability=cap,
                utilization=0.0,
            )

        except Exception as e:
            logger.warning(f"Torch stats failed: {e}")
            return None

    def get_all_stats(self) -> list[GPUStats]:
        """Get stats for all GPUs."""
        stats = []

        # Try torch to count devices
        try:
            import torch
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = 1

        for i in range(device_count):
            s = self.get_stats(i)
            if s:
                stats.append(s)

        return stats

    def can_fit(self, model_size_mb: int, device_id: int = 0, safety_margin: float = 0.9) -> bool:
        """Check if model fits in GPU memory."""
        stats = self.get_stats(device_id)
        if not stats:
            return False

        available = stats.memory_free * safety_margin
        return model_size_mb <= available

    def get_utilization(self, device_id: int = 0) -> float:
        """Get GPU utilization percentage."""
        stats = self.get_stats(device_id)
        if not stats:
            return 0.0

        if stats.memory_total == 0:
            return 0.0

        return (stats.memory_used / stats.memory_total) * 100

    def should_offload(self, device_id: int = 0, threshold: float = 90.0) -> bool:
        """Check if should offload to CPU or remote."""
        utilization = self.get_utilization(device_id)
        return utilization >= threshold

    def record_stats(self, stats: GPUStats) -> None:
        """Record stats to history."""
        self._history.append(stats)

        # Trim history
        if len(self._history) > self.config.history_size:
            self._history = self._history[-self.config.history_size:]

    def get_history(self, count: int = 60) -> list[GPUStats]:
        """Get stats history."""
        return self._history[-count:]

    def get_average_utilization(self, count: int = 60) -> float:
        """Get average utilization over history."""
        history = self.get_history(count)
        if not history:
            return 0.0

        return sum(s.utilization for s in history) / len(history)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._pynvml_initialized and self._nvml_handle:
            try:
                self._nvml_handle.nvmlShutdown()
            except Exception:
                pass

        self._pynvml_initialized = False
        self._nvml_handle = None


def get_gpu_monitor() -> GPUMonitor:
    """Get default GPU monitor."""
    return GPUMonitor()


# Export
__all__ = [
    "GPUMonitor",
    "GPUMonitorConfig",
    "GPUStats",
    "get_gpu_monitor",
]