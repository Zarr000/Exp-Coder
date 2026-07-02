"""
RMSNorm implementation for Expera AI.

RMSNorm (Root Mean Square Layer Normalization) is:
- Faster than LayerNorm (no mean calculation)
- More stable for LLMs
- Used in Llama, Mistral, etc.
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm only uses root mean square of inputs:
        RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)

    Advantages:
    - No bias term (simpler)
    - Faster (no mean computation)
    - Often better generalization
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Initialize RMSNorm.

        Args:
            normalized_shape: Feature dimension
            eps: Epsilon for numerical stability
            weight: Optional weight parameter (default: ones)
            bias: Optional bias parameter (default: None - RMSNorm typically has no bias)
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Weight (gamma)
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(normalized_shape))

        # Bias (beta) - optional, default no bias
        if bias is not None:
            self.bias = bias
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor (..., normalized_shape)

        Returns:
            Normalized tensor
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Using rsqrt for efficiency: 1/sqrt(x)
        norm = torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )

        # Normalize and apply weight
        output = x * norm * self.weight

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'


class FusedRMSNorm(nn.Module):
    """
    Fused kernel version of RMSNorm for better performance.

    Uses torch.nn.functional.normalize for potential GPU optimization.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused RMSNorm."""
        # Compute RMS
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()

        # Normalize and scale
        return x / rms * self.weight


class T5RMSNorm(nn.Module):
    """
    T5-style RMSNorm with bias.

    Based on "T5: Text-to-Text Transfer Transformer" but using RMS.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        bias: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply T5-style RMSNorm."""
        # Compute normalization factor
        norm = torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )

        # Normalize
        output = x * norm
        # Scale
        output = output * self.weight

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Functional RMSNorm for use in places where module overhead is undesired.

    Args:
        x: Input tensor
        weight: Normalization weight
        eps: Epsilon for stability

    Returns:
        Normalized tensor
    """
    norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return x * norm * weight


# Alias for backward compatibility
LayerNorm = nn.LayerNorm

__all__ = [
    "RMSNorm",
    "FusedRMSNorm",
    "T5RMSNorm",
    "rms_norm",
    "LayerNorm",  # Keep for compatibility
]