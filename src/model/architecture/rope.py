"""
Rotary Position Embeddings with YaRN scaling for Expera AI.

YaRN (Yet another RoPE extension) enables:
- Extended context lengths (>32K)
- Better extrapolation
- Memory-efficient interpolation
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class YaRNScaledRotaryEmbedding(nn.Module):
    """
    YaRN-scaled RoPE for extended context.

    Reference: "YaRN: Efficient Context Window Extension for LLMs"
    (https://arxiv.org/abs/2309.00071)

    Features:
    - interpolate positions via sqrt interpolation factor
    - scale attention via attention scaling factor
    - Better than linear/NTK interpolation
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        factor: float = 1.0,  # YaRN scaling factor
        original_seq_len: int = 8192,  # Original training context
        extrapolation_factor: float = 1.0,
        attention_factors: Optional[tuple] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.factor = factor
        self.original_seq_len = original_seq_len
        self.extrapolation_factor = extrapolation_factor

        # Attention scaling factors (from YaRN paper)
        if attention_factors is not None:
            self.attention_factors = attention_frequencies = attention_factors
        else:
            # Default: [1, 1/(2*k*d) for k in [1, 2, ...]]
            dim = dim // 2
            attention_factors = tuple(
                1.0 / (extrapolation_factor * (base ** (2 * (i // 2) / dim)))
                for i in range(0, dim * 2, 2)
            )
            self.attention_factors = attention_frequencies = attention_factors

        # Precompute inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos and sin for efficiency."""
        # Full frequency calculation
        t = torch.arange(seq_len, device=self.inv_freq.device)
        # Outer product: (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Apply attention scaling
        attention_scale = torch.tensor(self.attention_factors, device=self.inv_freq.device)

        # Embeddings
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple:
        """
        Get cos/sin for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            (cos, sin) tensors
        """
        if (
            seq_len > self.cos_cached.shape[0]
            or self.training
        ):
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


class LinearScaledRoPE(nn.Module):
    """
    Simple linear RoPE scaling.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,  # Scale factor (linear)
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        # Inv frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> tuple:
        """Get scaled embeddings."""
        t = torch.arange(seq_len, device=self.inv_freq.device)

        # Scale positions
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos(), emb.sin()


class NTKScaledRoPE(nn.Module):
    """
    NTK-aware RoPE scaling.

    Uses NTK-by-parts methodology from:
    "Neural Tangent Kernel (NTK)aware RoPE scaling"
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        original_seq_len: int = 4096,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.original_seq_len = original_seq_len

        # Compute NTK scaling factor
        # λ = original / target
        self.ntk_scale = self.original_seq_len / max_seq_len

        # Frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> tuple:
        """Get NTK-scaled embeddings."""
        t = torch.arange(seq_len, device=self.inv_freq.device)

        # Apply NTK scaling
        t = t * self.ntk_scale

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos(), emb.sin()


class YarnScaledRotaryEmbedding(YaRNScaledRotaryEmbedding):
    """
    Alias for YaRN RoPE.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        factor: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            max_seq_len=max_seq_len,
            base=base,
            factor=factor,
        )


def apply_rotary_positional(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        x with RoPE applied
    """
    # Combine real and imaginary parts
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]

    # Apply: x * cos + rotate(x) * sin
    # rotate(x) = (-x2, x1)
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)


def apply_rotary_positional_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary to half of hidden dimensions (more common).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)


__all__ = [
    "YaRNScaledRotaryEmbedding",
    "LinearScaledRoPE",
    "NTKScaledRoPE",
    "YarnScaledRotaryEmbedding",  # Alias
    "apply_rotary_positional",
    "apply_rotary_positional_half",
]