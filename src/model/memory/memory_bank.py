"""
Memory Bank for Expera AI

An external memory system that allows the model to store and retrieve
information beyond the fixed context window. This enables:
1. Long-term information retention across long documents
2. Project-specific knowledge storage
3. Efficient context management for code understanding

Architecture:
- Slots: Fixed-size memory bank N × d
- Write: Update memories with new information
- Read: Retrieve relevant memories via attention
- Update: Decay old memories, reinforce important ones
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    External memory bank with read/write/update operations.
    
    Memory is structured as a fixed-size matrix M ∈ ℝ^(N × d)
    where N is the number of memory slots and d is the hidden dimension.
    
    Operations:
    - Read: Attend over memory slots using current hidden state as query
    - Write: Update memory slots with new information
    - Update: Apply decay and importance-based updates
    """
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.1,
        decay_rate: float = 0.99,
    ):
        """
        Initialize memory bank.
        
        Args:
            hidden_size: Dimension of hidden states
            memory_size: Number of memory slots
            num_heads: Number of attention heads for memory retrieval
            dropout: Dropout probability
            decay_rate: Memory decay rate per step
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.decay_rate = decay_rate
        
        # Memory projection layers
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Write projection
        self.write_proj = nn.Linear(hidden_size, memory_size, bias=False)
        
        # Gating mechanism
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize memory bank weights."""
        for module in [self.query_proj, self.key_proj, self.value_proj,
                       self.output_proj, self.write_proj, self.gate_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def read(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Read from memory using attention.
        
        Args:
            query: Query tensor (batch, seq_len, hidden_size)
            memory: Memory bank (batch, memory_size, hidden_size)
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights, updated_query)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project query, key, value
        q = self.query_proj(query)
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.memory_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.memory_size, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        memory_output = torch.matmul(attn_weights, v)
        memory_output = memory_output.transpose(1, 2).contiguous()
        memory_output = memory_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        memory_output = self.output_proj(memory_output)
        
        # Gated integration with query
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([query, memory_output], dim=-1))
        )
        output = gate * memory_output + (1 - gate) * query
        
        if need_weights:
            return output, attn_weights, memory
        return output, None, memory
    
    def write(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Write new information to memory.
        
        Uses an importance-based update mechanism where the write
        strength is determined by the hidden state content.
        
        Args:
            hidden_states: New information (batch, seq_len, hidden_size)
            memory: Current memory bank (batch, memory_size, hidden_size)
            
        Returns:
            Updated memory bank (batch, memory_size, hidden_size)
        """
        # Compute write weights (importance scores)
        write_weights = self.write_proj(hidden_states.mean(dim=1))
        write_weights = F.softmax(write_weights, dim=-1)
        
        # Compute content to write (use mean pooling)
        content = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        # Update memory: M_new = decay * M_old + write_weights * content
        memory = memory * self.decay_rate
        memory = memory + write_weights.unsqueeze(-1) * content.unsqueeze(1)
        
        return memory
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        return_memory: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: read from memory, optionally update.
        
        Args:
            hidden_states: Current hidden states (batch, seq_len, hidden_size)
            memory: Current memory bank (batch, memory_size, hidden_size)
            update_memory: Whether to update memory with new information
            return_memory: Whether to return updated memory
            
        Returns:
            Tuple of (output, updated_memory)
        """
        # Initialize memory if not provided
        if memory is None:
            memory = torch.zeros(
                hidden_states.size(0), self.memory_size, self.hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            
        # Read from memory
        output, _, memory = self.read(hidden_states, memory)
        
        # Write to memory
        if update_memory:
            memory = self.write(hidden_states, memory)
            
        if return_memory:
            return output, memory
        return output, None