"""Token and Position Embeddings for Expera AI."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) - applies rotation to Q/K vectors."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.freqs.device)
        angles = torch.outer(positions, self.freqs)
        angles = angles.unsqueeze(-1).repeat(1, 1, 2).flatten(1)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                positions: torch.Tensor = None) -> tuple:
        seq_len = q.size(2)
        head_dim = q.size(-1)
        if positions is None:
            positions = torch.arange(seq_len, device=q.device)
        
        # Get cos/sin for this sequence length and head dimension
        cos = self.cos_cache[:seq_len, :head_dim].to(q.dtype)
        sin = self.sin_cache[:seq_len, :head_dim].to(q.dtype)
        
        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class TokenEmbedding(nn.Module):
    """Convert token IDs to dense vectors."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 padding_idx: int = None, std: float = 0.02):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(input_ids)


class EmbeddingModule(nn.Module):
    """Combined embedding with RoPE support."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int = 2048,
                 padding_idx: int = None, dropout: float = 0.1,
                 std: float = 0.02, rope_theta: float = 10000.0):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size, padding_idx, std)
        self.rope = RotaryPositionalEmbedding(hidden_size, max_position_embeddings, rope_theta)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, input_ids: torch.LongTensor,
                positions: torch.Tensor = None) -> torch.Tensor:
        embeddings = self.token_embedding(input_ids)
        embeddings = self.dropout(embeddings)
        embeddings = self.ln(embeddings)
        return embeddings