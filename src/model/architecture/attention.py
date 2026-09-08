"""
Multi-Head Attention mechanism for Expera AI

Implements:
1. Scaled Dot-Product Attention with causal masking
2. Multi-Head Attention with RoPE integration
3. Grouped Query Attention (GQA) support
4. Efficient key-value caching for autoregressive generation

Key innovations:
- Rotary Position Embeddings (RoPE) integration
- Grouped Query Attention for efficient inference
- Efficient key-value caching for autoregressive generation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryPositionalEmbedding


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with causal masking.
    
    Mathematical Formulation:
        Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    where:
    - Q, K, V are query, key, value matrices
    - d_k is the key dimension (head_dim)
    - Causal mask prevents attending to future positions
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: (batch, num_heads, seq_len, head_dim)
            key: (batch, num_heads, seq_len, head_dim)
            value: (batch, num_heads, seq_len, head_dim)
            mask: Attention mask or None
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        _, _, _, head_dim = query.shape
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(head_dim)
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask
            
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        
        if need_weights:
            return context, attention_weights
        return context, None


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE and GQA support.
    
    Supports:
    1. Standard Multi-Head Attention (MHA)
    2. Grouped Query Attention (GQA) - fewer KV heads than Q heads
    3. Rotary Position Embeddings (RoPE)
    4. Efficient KV caching for autoregressive generation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        # Head dimension
        if head_dim is None:
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim
            
        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Rotary Position Embeddings
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=max_position_embeddings,
                base=rope_theta,
            )
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize attention weights with normal distribution."""
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
        
    def _create_causal_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal attention mask (upper triangular).
        
        Args:
            seq_len: Sequence length
            dtype: Data type
            device: Device
            
        Returns:
            Causal mask (1, 1, seq_len, seq_len)
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _repeat_kv(
        self,
        x: torch.Tensor,
        n_rep: int,
    ) -> torch.Tensor:
        """
        Repeat key/value heads for Grouped Query Attention.
        
        Args:
            x: (batch, num_kv_heads, seq_len, head_dim)
            n_rep: Number of repetitions
            
        Returns:
            (batch, num_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return x
            
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            past_key_value: Cached K,V states for generation
            position_ids: Position IDs for RoPE
            use_cache: Whether to return K,V cache
            
        Returns:
            Tuple of (output tensor, key/value cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rope:
            q, k = self.rope(q, k, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            
        key_value_cache = (k, v) if use_cache else None
        
        # Repeat KV heads for GQA
        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)
        
        # Create causal mask if needed
        if attention_mask is None:
            current_seq_len = k.size(2)
            attention_mask = self._create_causal_mask(
                seq_len=current_seq_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if past_key_value is not None:
                attention_mask = attention_mask[:, :, -seq_len:, :]
                
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape to (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        output = self.dropout(output)
        
        return output, key_value_cache
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"use_rope={self.use_rope}"
        )