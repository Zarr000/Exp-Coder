"""
KV Cache Management for Expera AI.

Provides efficient KV cache handling for inference:
- Cache allocation
- Cache reuse/promotion
- Cache eviction
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import logging

import torch
import torch.nn as nn
from torch import Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KVCache:
    """
    KV (Key-Value) cache for attention.

    Stores:
        - keys: [batch, num_heads, seq_len, head_dim]
        - values: [batch, num_heads, seq_len, head_dim]
    """
    keys: Tensor
    values: Tensor
    seq_len: int = 0
    max_seq_len: int = 0

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get cache shape."""
        return self.keys.shape


class KVCacheManager:
    """
    Manages KV caches for efficient inference.

    Features:
        - Pre-allocate caches
        - Cache promotion (move partial to full)
        - Cache reuse
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize cache manager.

        Args:
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            device: Device
            dtype: Data type
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype

        # Pre-allocate caches
        self.caches: Dict[int, KVCache] = {}
        self._allocate_caches()

        logger.info(f"KVCacheManager: {num_layers} layers, {max_batch_size}x{max_seq_len}")

    def _allocate_caches(self):
        """Pre-allocate KV caches."""
        for layer_idx in range(self.num_layers):
            keys = torch.zeros(
                (self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            values = torch.zeros(
                (self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.caches[layer_idx] = KVCache(
                keys=keys,
                values=values,
                seq_len=0,
                max_seq_len=self.max_seq_len,
            )

    def get_cache(self, layer_idx: int) -> KVCache:
        """Get cache for layer."""
        return self.caches.get(layer_idx)

    def update(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
    ):
        """Update cache with new keys/values."""
        cache = self.caches[layer_idx]
        seq_len = keys.shape[2]  # seq dim

        # Update cache
        cache.keys[:, :, :seq_len, :] = keys
        cache.values[:, :, :seq_len, :] = values
        cache.seq_len = seq_len

    def clear(self, layer_idx: Optional[int] = None):
        """Clear cache(s)."""
        if layer_idx is not None:
            cache = self.caches[layer_idx]
            cache.keys.zero_()
            cache.values.zero_()
            cache.seq_len = 0
        else:
            for cache in self.caches.values():
                cache.keys.zero_()
                cache.values.zero_()
                cache.seq_len = 0

    def get_prefix_cache(
        self,
        layer_idx: int,
        seq_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """Get prefix cache for given seq_len."""
        cache = self.caches[layer_idx]
        return (
            cache.keys[:, :, :seq_len, :],
            cache.values[:, :, :seq_len, :],
        )


class CacheConfig:
    """Cache configuration."""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_type: str = "static",  # static, sliding, ring
        sliding_window: int = 4096,
        cache_layer_indices: Optional[List[int]] = None,
    ):
        self.enable_cache = enable_cache
        self.cache_type = cache_type
        self.sliding_window = sliding_window
        self.cache_layer_indices = cache_layer_indices


__all__ = [
    "KVCache",
    "KVCacheManager",
    "CacheConfig",
]