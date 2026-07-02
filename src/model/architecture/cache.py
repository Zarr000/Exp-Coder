"""
KV Cache Manager for Expera AI.

Efficient key-value cache management for autoregressive generation.
Features:
- Pre-allocation
- Paged memory
- Sliding window eviction
- Dynamic batching support
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn


@dataclass
class CacheConfig:
    """KV Cache configuration."""
    max_seq_len: int = 32768
    max_batch_size: int = 1
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class KVCache:
    """
    Single KV cache entry.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Allocate cache
        self.k_cache = torch.zeros(
            num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )

        self._seq_len = 0

    def _get_position(self) -> int:
        """Get current position."""
        return self._seq_len

    def _increment_position(self) -> None:
        """Increment position."""
        self._seq_len += 1

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append new key-value pairs.

        Args:
            k: (num_heads, 1, head_dim)
            v: (num_heads, 1, head_dim)
        """
        seq_idx = self._seq_len
        self.k_cache[:, seq_idx] = k.squeeze(1)
        self.v_cache[:, seq_idx] = v.squeeze(1)
        self._seq_len += 1

    def update_at(self, k: torch.Tensor, v: torch.Tensor, position: int) -> None:
        """Update at specific position."""
        self.k_cache[:, position] = k.squeeze(1)
        self.v_cache[:, position] = v.squeeze(1)

    def get(self, start: int = 0, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV pairs."""
        if end is None:
            end = self._seq_len
        return (
            self.k_cache[:, start:end].unsqueeze(0),
            self.v_cache[:, start:end].unsqueeze(0),
        )

    @property
    def seq_len(self) -> int:
        return self._seq_len


class KVCacheManager:
    """
    Manages multiple KV caches for all layers.

    Features:
    - Pre-allocated memory
    - Sliding window eviction
    - Paged memory support
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        sliding_window: Optional[int] = None,
    ):
        self.config = config or CacheConfig()
        self.sliding_window = sliding_window

        # Create caches for each layer
        self.caches: Dict[int, KVCache] = {}
        self._initialized = False
        self._positions: Dict[int, int] = {}

    def _get_position(self, layer_idx: int) -> int:
        """Get current position for layer."""
        return self._positions.get(layer_idx, 0)

    def _increment_position(self, layer_idx: int) -> int:
        """Increment position for layer and return new position."""
        self._positions[layer_idx] = self._get_position(layer_idx) + 1
        return self._positions[layer_idx]

    def _initialize(self) -> None:
        """Initialize all KV caches."""
        if self._initialized:
            return

        for layer_idx in range(self.config.num_layers):
            self.caches[layer_idx] = KVCache(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                dtype=self.config.dtype,
                device=self.config.device,
            )

        self._initialized = True

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Update cache for a layer.

        Args:
            layer_idx: Layer index
            k: Key tensor (batch, num_heads, 1, head_dim)
            v: Value tensor (batch, num_heads, 1, head_dim)
        """
        if not self._initialized:
            self._initialize()

        # Check if we need to evict (sliding window)
        cache = self.caches.get(layer_idx)
        if cache is None:
            cache = KVCache(
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            self.caches[layer_idx] = cache

        # Apply sliding window eviction
        if self.sliding_window and cache.seq_len >= self.sliding_window:
            self._evict(layer_idx)

        cache.append(k, v)

    def _evict(self, layer_idx: int) -> None:
        """Evict old entries for sliding window."""
        cache = self.caches.get(layer_idx)
        if cache is None:
            return

        # Shift cache (simplified - real implementation would use ring buffer)
        if cache.seq_len >= self.sliding_window:
            evict_len = cache.seq_len - self.sliding_window + 1
            cache.k_cache[:, :-evict_len] = cache.k_cache[:, evict_len:]
            cache.v_cache[:, :-evict_len] = cache.v_cache[:, evict_len:]
            cache._seq_len -= evict_len

    def get(
        self,
        layer_idx: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache for a layer."""
        if not self._initialized:
            self._initialize()

        cache = self.caches.get(layer_idx)
        if cache is None:
            # Return empty tensors
            return (
                torch.zeros(
                    1, self.config.num_heads, 0, self.config.head_dim,
                    device=self.config.device,
                ),
                torch.zeros(
                    1, self.config.num_heads, 0, self.config.head_dim,
                    device=self.config.device,
                ),
            )

        return cache.get(start, end)

    def get_all(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get all caches."""
        if not self._initialized:
            self._initialize()

        return {idx: cache.get() for idx, cache in self.caches.items()}

    def reset(self) -> None:
        """Reset all caches."""
        if not self._initialized:
            return

        for cache in self.caches.values():
            cache._seq_len = 0
            cache.k_cache.zero_()
            cache.v_cache.zero_()

    def to(self, device: str) -> "KVCacheManager":
        """Move to device."""
        self.config.device = device
        for cache in self.caches.values():
            cache.k_cache = cache.k_cache.to(device)
            cache.v_cache = cache.v_cache.to(device)
        return self


class PagedKVCacheManager(KVCacheManager):
    """
    Paged KV cache manager.
    Uses block-based memory allocation.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        block_size: int = 16,
    ):
        super().__init__(config)
        self.block_size = block_size
        self.page_table: Dict[int, Dict[int, int]] = {}  # layer -> block_idx -> page
        self._positions: Dict[int, int] = {}  # layer -> position

    def _initialize(self) -> None:
        """Initialize paged caches."""
        num_blocks = math.ceil(self.config.max_seq_len / self.block_size)

        for layer_idx in range(self.config.num_layers):
            # Pre-allocate pages
            pages = [
                torch.zeros(
                    self.config.num_heads,
                    self.block_size,
                    self.config.head_dim,
                    dtype=self.config.dtype,
                    device=self.config.device,
                )
                for _ in range(num_blocks)
            ]
            self.caches[layer_idx] = pages

        self._initialized = True

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Update with paged allocation."""
        if not self._initialized:
            self._initialize()

        # Get current position
        pos = self._get_position(layer_idx)
        block_idx = pos // self.block_size
        offset = pos % self.block_size

        # Get or create block
        if layer_idx not in self.caches:
            self.caches[layer_idx] = []

        while len(self.caches[layer_idx]) <= block_idx:
            self.caches[layer_idx].append(torch.zeros(
                self.config.num_heads,
                self.block_size,
                self.config.head_dim,
                dtype=self.config.dtype,
                device=self.config.device,
            ))

        # Store in page
        self.caches[layer_idx][block_idx][:, offset] = k.squeeze(1)
        # Track position
        self._increment_position(layer_idx)


class StreamingKVCacheManager(KVCacheManager):
    """
    Streaming KV cache with automatic eviction.
    For very long contexts without explicit max_seq_len.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        max_memory_gb: float = 20.0,
    ):
        super().__init__(config)
        self.max_memory_gb = max_memory_gb

    def _compute_max_seq_from_memory(self) -> int:
        """Compute max sequence from memory budget."""
        # Estimate memory per token
        bytes_per_token = (
            self.config.num_heads *
            self.config.head_dim *
            2 *  # K and V
            self.config.num_layers *
            2  # bfloat16 = 2 bytes
        )

        max_tokens = int(self.max_memory_gb * 1e9 / bytes_per_token)
        return min(max_tokens, self.config.max_seq_len)


__all__ = [
    "CacheConfig",
    "KVCache",
    "KVCacheManager",
    "PagedKVCacheManager",
    "StreamingKVCacheManager",
]