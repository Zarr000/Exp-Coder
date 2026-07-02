"""
Grouped Feed-Forward Network (Grouped FFN) for Expera AI.

Grouped FFN processes tokens in groups for more efficient computation
compared to dense FFN, with benefits for long sequences.

Reference: "Grouped-Query Attention (GQA)" inspired architecture
for FFN layers, with shared computation within groups.

Features:
- Grouped token processing
- Shared hidden computation within groups
- Configurable group sizes
- Memory efficiency
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GroupedFFN(nn.Module):
    """
    Grouped Feed-Forward Network.

    Processes tokens in groups with shared hidden computation,
    reducing memory and compute overhead for long sequences.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_groups: int = 4,
        activation: str = "silu",
        bias: bool = False,
        dropout: float = 0.0,
        chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 2.666)  # ~2/3 of 4*dim
        self.num_groups = num_groups
        self.chunk_size = chunk_size or (dim // num_groups)

        self.activation = activation
        self.use_bias = bias

        # Input projection
        self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=bias)

        # Output projection
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward through grouped FFN.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Get dimensions
        batch_size, seq_len, dim = x.shape

        # Compute gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Apply activation (typically SiLU/Swish)
        if self.activation == "silu":
            activated = F.silu(gate)
        elif self.activation == "gelu":
            activated = F.gelu(gate)
        elif self.activation == "relu":
            activated = F.relu(gate)
        else:
            activated = F.silu(gate)

        # Element-wise multiply with up projection
        intermediate = activated * up

        # Apply dropout
        if self.dropout is not None:
            intermediate = self.dropout(intermediate)

        # Project down
        output = self.down_proj(intermediate)

        return output

    def forward_chunked(self, x: Tensor) -> Tensor:
        """
        Forward with chunked processing for memory efficiency.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        chunk_size = self.chunk_size
        batch_size, seq_len, dim = x.shape

        # Reshape into chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        outputs = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)

            chunk = x[:, start:end, :]
            chunk_out = self.forward(chunk)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=1)


class MultiQueryFFN(nn.Module):
    """
    Multi-Query FFN variant.

    Uses fewer projection matrices than having separate FFNs per query,
    inspired by Multi-Query Attention.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_groups: int = 4,
        shared_intermediate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 2.666)
        self.num_groups = num_groups
        self.shared_intermediate = shared_intermediate

        # Single shared intermediate projection (shared across groups)
        self.shared_gate = nn.Linear(dim, self.hidden_dim, bias=False)
        self.shared_up = nn.Linear(dim, self.hidden_dim, bias=False)

        # Output projections per group
        if not shared_intermediate:
            self.group_down = nn.ModuleList([
                nn.Linear(self.hidden_dim, dim, bias=False)
                for _ in range(num_groups)
            ])
        else:
            self.group_down = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        group_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward through Multi-Query FFN.

        Args:
            x: Input tensor (batch, seq_len, dim)
            group_indices: Optional group assignment (batch, seq_len)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Shared intermediate computation
        gate = self.shared_gate(x)
        up = self.shared_up(x)
        intermediate = F.silu(gate) * up

        # Group-specific output projection
        if self.shared_intermediate:
            output = self.group_down(intermediate)
        else:
            # Apply per-group projection
            if group_indices is None:
                # Default to group 0
                group_indices = torch.zeros(
                    x.shape[0], x.shape[1],
                    dtype=torch.long, device=x.device
                )

            batch_size, seq_len, _ = x.shape
            output = torch.zeros_like(x)

            for group_id in range(self.num_groups):
                mask = group_indices == group_id
                if mask.any():
                    output[mask] = self.group_down[group_id](intermediate[mask])

        return output


class ParallellFFN(nn.Module):
    """
    Parallel FFN (used in some Transformer variants).

    Processes FFN in parallel with attention.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 2.666)

        # Parallel projections
        self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward in parallel."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FusedFFN(nn.Module):
    """
    Fused FFN that combines multiple activations.

    Uses a single large linear layer for efficiency.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 2.666)

        # Gate and up projections combined
        self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with fused activations."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ExpertPoolFFN(nn.Module):
    """
    Expert Pool FFN with dynamic expert selection.

    Maintains a pool of experts and selects dynamically.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or int(dim * 2.666)
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim, bias=False),
            )
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with expert selection."""
        batch_size, seq_len, _ = x.shape

        # Get routing weights
        router_out = self.router(x)  # (batch, seq_len, num_experts)
        top_k_weights, top_k_indices = torch.topk(router_out, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(x)

        # Apply top-k experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]
            weight = top_k_weights[..., i].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    output[mask] += expert_out * weight[mask]

        return output


class GroupedFFNConfig:
    """Configuration for Grouped FFN."""

    def __init__(
        self,
        dim: int = 4096,
        hidden_dim: Optional[int] = None,
        num_groups: int = 4,
        use_chunked: bool = False,
        chunk_size: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size
        self.activation = activation
        self.dropout = dropout


def create_grouped_ffn(
    dim: int,
    hidden_dim: Optional[int] = None,
    num_groups: int = 4,
    ffn_type: str = "grouped",
    **kwargs,
) -> nn.Module:
    """
    Create a grouped FFN.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension
        num_groups: Number of groups
        ffn_type: Type of FFN (grouped, multiquery, parallel, fused, expert_pool)
        **kwargs: Additional arguments

    Returns:
        GroupedFFN module
    """
    if ffn_type == "grouped":
        return GroupedFFN(dim, hidden_dim, num_groups, **kwargs)
    elif ffn_type == "multiquery":
        return MultiQueryFFN(dim, hidden_dim, num_groups)
    elif ffn_type == "parallel":
        return ParallellFFN(dim, hidden_dim)
    elif ffn_type == "fused":
        return FusedFFN(dim, hidden_dim)
    elif ffn_type == "expert_pool":
        return ExpertPoolFFN(dim, hidden_dim)
    else:
        return GroupedFFN(dim, hidden_dim, num_groups)


__all__ = [
    "GroupedFFN",
    "MultiQueryFFN",
    "ParallellFFN",
    "FusedFFN",
    "ExpertPoolFFN",
    "GroupedFFNConfig",
    "create_grouped_ffn",
]