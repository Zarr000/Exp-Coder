"""
Transformer Block for Expera AI

Implements a single transformer block with:
1. Pre-LayerNorm architecture (more stable training)
2. Multi-Head Attention with RoPE and GQA
3. Feed-Forward Network with GELU/SwiGLU
4. Residual connections
5. Optional memory bank integration

Architecture:
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

Pre-LayerNorm vs Post-LayerNorm:
- Pre-LayerNorm: Apply LN before each sublayer
  Benefits: More stable gradients, easier to train
  Used in: GPT-2, LLaMA, most modern transformers
  
- Post-LayerNorm: Apply LN after each sublayer
  Benefits: Better performance at small scale
  Used in: Original Transformer paper
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-layer normalization.
    
    Contains:
    - Layer norm
    - Multi-head self-attention
    - Residual connection
    - Layer norm
    - Feed-forward network
    - Residual connection
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
    ):
        """
        Initialize transformer block.
        
        Args:
            hidden_size: Dimension of hidden states
            num_heads: Number of attention heads
            intermediate_size: FFN hidden dimension
            num_kv_heads: Number of key-value heads (GQA)
            head_dim: Dimension per head
            activation: FFN activation function
            dropout: Dropout probability
            attention_dropout: Attention dropout
            use_rope: Whether to use RoPE
            rope_theta: RoPE theta parameter
            max_position_embeddings: Maximum sequence length
            layer_norm_eps: Layer norm epsilon
        """
        super().__init__()
        
        # Layer norms (pre-norm)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Self-attention
        from .attention import MultiHeadAttention
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        
        # Feed-forward network
        from .feedforward import FeedForward
        self.mlp = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout=dropout,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of transformer block.
        
        Args:
            hidden_states: Input (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached K,V states
            use_cache: Whether to return cache
            
        Returns:
            Tuple of (output, key_value_cache)
        """
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # Feed-forward with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.self_attn.hidden_size}, "
            f"num_heads={self.self_attn.num_heads}, "
            f"intermediate_size={self.mlp.intermediate_size}"
        )