"""
Feed-Forward Network for Expera AI

Implements:
1. Standard FFN with GELU activation
2. SwiGLU variant (used in modern LLMs like LLaMA)
3. Configurable activation functions

Mathematical Formulation (Standard):
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

Mathematical Formulation (SwiGLU):
    FFN(x) = (Swish(xW_1) ⊙ (xW_3))W_2
    
    where Swish(x) = x * sigmoid(βx)
    and SwiGLU uses 2/3 of the intermediate size

Architecture decisions:
- Pre-LayerNorm before FFN
- GELU activation for standard variant
- Optional SwiGLU for better performance
- Proper initialization for training stability
"""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.
    
    Smooth approximation of ReLU with non-zero gradients for negative values.
    
    Exact formula:
        GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
    
    Tanh approximation (used in BERT/GPT):
        GELU(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
    
    Advantages over ReLU:
    - Smooth gradient everywhere
    - Non-zero gradient for negative values
    - Better performance in deep networks
    """
    
    def __init__(self, approximate: Literal['tanh', 'none'] = 'tanh'):
        """
        Initialize GELU activation.
        
        Args:
            approximate: Approximation method ('tanh' or 'none')
        """
        super().__init__()
        self.approximate = approximate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        return F.gelu(x, approximate=self.approximate)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (used in LLaMA, PaLM, etc.).
    
    SwiGLU(x, y) = Swish(x) * y
    
    where Swish(x) = x * sigmoid(βx)
    
    In standard FFN:
        SwiGLU_FFN(x) = (Swish(xW_gate) ⊙ (xW_up))W_down
    
    Note: SwiGLU requires 3 weight matrices instead of 2,
    but typically uses 2/3 of the intermediate size to compensate.
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Initialize SwiGLU activation.
        
        Args:
            beta: Beta parameter for Swish activation (default: 1.0)
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor (from up projection)
            gate: Gate tensor (from gate projection)
            
        Returns:
            Activated values
        """
        # Swish activation
        swish = gate * torch.sigmoid(self.beta * gate)
        # Element-wise multiplication
        return swish * x


class FeedForward(nn.Module):
    """
    Feed-Forward Network with configurable activation.
    
    Supports two architectures:
    1. Standard FFN: GELU(xW_1)W_2
    2. SwiGLU FFN: (Swish(xW_gate) * (xW_up))W_down
    
    Standard FFN dimensions:
        hidden_size -> 4 * hidden_size -> hidden_size
    
    SwiGLU FFN dimensions:
        hidden_size -> (8/3) * hidden_size -> hidden_size
        (uses 2 weight matrices for intermediate, so dimension is 2/3 of standard)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        """
        Initialize feed-forward network.
        
        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension of FFN
            activation: Activation function ('gelu', 'relu', 'swiglu', 'silu')
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_name = activation
        
        if activation in ["gelu", "relu", "silu"]:
            # Standard FFN: 2 linear layers
            self.gate_proj = None  # Not used for standard FFN
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            
        elif activation == "swiglu":
            # SwiGLU FFN: 3 linear layers
            # Intermediate size is typically hidden_size * 8/3 ≈ 2.67 * hidden_size
            # to compensate for the extra projection
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Activation function
        if activation == "gelu":
            self.activation = GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()  # SiLU = Swish with beta=1
        elif activation == "swiglu":
            self.activation = SwiGLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize FFN weights."""
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        
        if self.gate_proj is not None:
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        if self.activation_name == "swiglu":
            # SwiGLU: (Swish(xW_gate) * (xW_up))W_down
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = self.activation(gate, up)
        else:
            # Standard: Activation(xW_up)W_down
            hidden = self.up_proj(x)
            hidden = self.activation(hidden)
            
        # Down projection
        hidden = self.down_proj(hidden)
        hidden = self.dropout(hidden)
        
        return hidden
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"activation={self.activation_name}"
        )