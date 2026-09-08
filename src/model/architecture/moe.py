"""
Mixture of Experts (MoE) for Expera AI.

MoE enables sparse activation of expert networks, allowing
model scaling without proportional compute cost.

Reference: "Outrageously Large Neural Networks"
(https://arxiv.org/abs/1704.01279)

Features:
- Top-k gating
- Load balancing loss
- Expert capacity planning
- Router z-loss
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts."""
    num_experts: int = 8
    top_k: int = 2  # Number of experts to activate per token
    dropout: float = 0.0
    noisy_gating: bool = True
    noise_std: float = 0.1
    capacity_factor: float = 1.25  # Expert capacity multiplier
    eval_capacity_factor: float = 2.0  # Higher capacity during eval
    drop_tokens: bool = True  # Drop tokens when capacity exceeded


class Expert(nn.Module):
    """
    Single expert network.

    Each expert is a small feedforward network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Gate проекция
        self.wg = nn.Linear(input_dim, hidden_dim, bias=False)
        # Up проекция
        self.wup = nn.Linear(input_dim, hidden_dim, bias=False)
        # Down проекция
        self.wdown = nn.Linear(hidden_dim, output_dim, bias=False)

        # Activation function
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward through expert."""
        return self.wdown(self.act(self.wup(x)))


class GatingMechanism(nn.Module):
    """
    MoE gating mechanism with top-k selection.

    Implements noisy top-k gating from the Switch Transformer paper.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noisy_gating: bool = True,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.noise_std = noise_std

        # Gate weights
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute gating scores.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            (gates, expert_indices, expert_weights)
        """
        # Compute gate logits
        logits = self.gate(x)  # (batch, seq_len, num_experts)

        if self.noisy_gating and self.training:
            # Add noise for load balancing
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Get top-k experts
        gates, expert_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over top-k
        gates = F.softmax(gates, dim=-1)

        # Create mask for selected experts
        expert_mask = torch.zeros_like(logits).scatter_(-1, expert_indices, 1.0)

        return gates, expert_indices, expert_mask


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    Sparse MoE with expert capacity and load balancing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        config: Optional[MoEConfig] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        self.config = config or MoEConfig()
        self.top_k = self.config.top_k

        # Experts
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        # Gating
        self.gate = GatingMechanism(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            noisy_gating=self.config.noisy_gating,
            noise_std=self.config.noise_std,
        )

        # Layer norm
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Forward through MoE.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            (output, metadata)
        """
        batch_size, seq_len, hidden_dim = x.shape
        input_dim = x.shape[-1]

        # Normalize input
        x_norm = self.norm(x)

        # Get gating
        gates, expert_indices, expert_mask = self.gate(x_norm)

        # Initialize output
        output = torch.zeros_like(x)

        # Track expert usage
        expert_usage = torch.zeros(self.num_experts, device=x.device)

        # Apply experts
        capacity = int(seq_len * self.config.capacity_factor)
        flat_x = x_norm.view(-1, x_norm.shape[-1])
        flat_output = output.view(-1, x.shape[-1])

        for k in range(self.top_k):
            # Get indices for k-th expert
            expert_idx = expert_indices[..., k]  # (batch, seq_len)
            expert_gate = gates[..., k].unsqueeze(-1)  # (batch, seq_len, 1)
            flat_gate = expert_gate.view(-1, 1)

            # Apply each expert
            for expert_id in range(self.num_experts):
                # Find tokens for this expert (flattened across batch and seq)
                expert_tokens = (expert_idx == expert_id).reshape(-1)  # (batch*seq,)
                tokens_for_expert = flat_x[expert_tokens]

                if expert_tokens.any() and tokens_for_expert.shape[0] > capacity:
                    # Deterministically keep the first `capacity` routed tokens
                    # (independent of random routing) so the layer never
                    # crashes on shape mismatch.
                    kept_positions = expert_tokens.nonzero(as_tuple=False)[:capacity, 0]
                    tokens_for_expert = flat_x[kept_positions]
                    expert_tokens = torch.zeros_like(expert_tokens)
                    expert_tokens[kept_positions] = True

                if not expert_tokens.any():
                    continue

                expert_output = self.experts[expert_id](tokens_for_expert)
                flat_output[expert_tokens] += expert_output * flat_gate[expert_tokens]
                expert_usage[expert_id] += expert_tokens.sum()

        # Compute metadata
        metadata = {
            "expert_usage": expert_usage,
            "expert_indices": expert_indices,
            "gates": gates,
            "capacity": capacity,
            "utilization": expert_usage / (batch_size * seq_len),
        }

        return output, metadata

    def get_load_balancing_loss(
        self,
        expert_indices: Tensor,
        gate_weights: Tensor,
    ) -> Tensor:
        """
        Compute load balancing loss.

        Encourages equal usage of all experts.

        Args:
            expert_indices: (batch, seq_len, top_k)
            gate_weights: (batch, seq_len, top_k)

        Returns:
            Load balancing loss
        """
        # Get batch size
        batch_size = expert_indices.shape[0]

        # Calculate fraction of tokens per expert
        num_experts = self.num_experts
        expert_counts = torch.zeros(num_experts, device=expert_indices.device)

        for k in range(self.top_k):
            expert_counts += (expert_indices[..., k] == torch.arange(
                num_experts, device=expert_indices.device
            ).unsqueeze(0).unsqueeze(0)).sum(dim=(0, 1))

        # Fraction of tokens
        expert_fraction = expert_counts / (expert_counts.sum() + 1e-10)

        # Calculate auxiliary loss
        load_balance_loss = num_experts * (expert_fraction ** 2).sum()

        return load_balance_loss


class SparseMoEBlock(nn.Module):
    """
    Complete MoE transformer block.

    Combines MoE with residual connection and layer norm.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # MoE layer
        self.moe = MoELayer(
            input_dim=dim,
            output_dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Layer norms
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, dict]:
        """Forward through MoE block."""
        # Pre-norm
        residual = x
        x = self.pre_norm(x)

        # MoE
        moe_output, metadata = self.moe(x)

        # Dropout
        if self.dropout is not None:
            moe_output = self.dropout(moe_output)

        # Residual
        x = residual + moe_output

        # Post-norm
        x = self.post_norm(x)

        return x, metadata


class GroupedExperts(nn.Module):
    """
    Grouped experts for efficient MoE.

    Groups experts and processes with shared computation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_groups: int = 4,
        experts_per_group: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group

        # Each group shares hidden computation
        self.group_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * experts_per_group),
                nn.SiLU(),
            )
            for _ in range(num_groups)
        ])

        # Expert output projection
        self.output_proj = nn.Linear(input_dim * experts_per_group, output_dim)

    def forward(self, x: Tensor, group_indices: Tensor) -> Tensor:
        """
        Forward with grouped experts.

        Args:
            x: Input (batch, seq_len, input_dim)
            group_indices: Group assignment (batch, seq_len)

        Returns:
            Output (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        output = torch.zeros_like(x)

        for group_id in range(self.num_groups):
            # Get tokens for this group
            group_mask = group_indices == group_id
            group_tokens = x[group_mask]

            if group_tokens.shape[0] == 0:
                continue

            # Apply group transform
            group_output = self.group_transform[group_id](group_tokens)  # (num_tokens, hidden * experts_per_group)

            # Split into expert outputs
            expert_outputs = group_output.view(
                group_tokens.shape[0], self.experts_per_group, -1
            )  # (num_tokens, experts_per_group, hidden)

            # Aggregate (mean pooling)
            aggregated = expert_outputs.mean(dim=1)  # (num_tokens, hidden)

            # Project to output
            projected = self.output_proj(aggregated)

            # Fill output
            output[group_mask] = projected

        return output


__all__ = [
    "MoEConfig",
    "Expert",
    "GatingMechanism",
    "MoELayer",
    "SparseMoEBlock",
    "GroupedExperts",
]