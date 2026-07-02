"""
Cross Attention for Expera AI.

Cross-modal attention mechanisms for:
- Vision-language models
- Multimodal transformers
- Unified embedding spaces
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CrossAttentionConfig:
    """Cross attention configuration."""
    embed_dim: int = 768
    num_heads: int = 12
    dropout: float = 0.1
    kv_dim: Optional[int] = None  # If different from embed_dim
    is_cross_attention: bool = True
    scope: str = "both"  # " encoder", "decoder", "both"


class CrossAttentionModule(nn.Module):
    """
    Cross attention module.

    Allows queries to attend to keys/values from a different sequence.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.kv_dim = config.kv_dim or config.embed_dim

        # Note: Accept that head_dim * num_heads may != embed_dim (truncation handled in reshape)

        # Projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(self.kv_dim, config.embed_dim)
        self.v_proj = nn.Linear(self.kv_dim, config.embed_dim)

        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Cross attention forward.

        Args:
            query: (batch, seq_len_q, embed_dim)
            key: (batch, seq_len_kv, kv_dim)
            value: (batch, seq_len_kv, kv_dim)
            mask: Optional attention mask

        Returns:
            Output (batch, seq_len_q, embed_dim)
        """
        batch_size = query.size(0)
        # Handle 2D input (batch, dim) -> (batch, 1, dim)
        squeeze_output = False
        if query.dim() == 2:
            squeeze_output = True
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)

        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, heads, seq_len_q, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)

        # Squeeze if original input was 2D
        if squeeze_output:
            output = output.squeeze(1)

        return output


class SelfAttention(nn.Module):
    """Custom self-attention to avoid PyTorch nn.MultiheadAttention issues."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, mask=None):
        # Accept query, key, value format to match nn.MultiheadAttention
        if key is None:
            key = query
        if value is None:
            value = query

        # query: (batch, seq, dim)
        batch_size, seq_len, _ = query.size()
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), None  # Return tuple to match old API


class MultimodalTransformer(nn.Module):
    """
    Multimodal transformer.

    Combines vision and text modalities.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        text_dim: int = 768,
        vision_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Modal-specific projections
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionModule(CrossAttentionConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                kv_dim=vision_dim if i % 2 == 0 else text_dim,
            ))
            for i in range(num_layers)
        ])

        # Self-attention layers (use custom to avoid batch_first issues)
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # MLPs
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers * 3)
        ])

    def forward(
        self,
        text: Tensor,
        vision: Tensor,
        text_mask: Optional[Tensor] = None,
        vision_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Multimodal forward.

        Args:
            text: (batch, text_seq_len, text_dim)
            vision: (batch, vision_seq_len, vision_dim)
            text_mask: Optional text attention mask
            vision_mask: Optional vision attention mask

        Returns:
            (text_output, vision_output)
        """
        # Project to common space
        text = self.text_proj(text)
        vision = self.vision_proj(vision)

        # Alternate between modalities
        for i, (cross_attn, self_attn, mlp) in enumerate(
            zip(self.cross_attention_layers, self.self_attention_layers, self.mlps)
        ):
            # Cross attention: text attends to vision
            text_norm = self.norms[i * 3](text)
            text = text_norm + cross_attn(text_norm, vision, vision, vision_mask)

            # Self attention on text
            text = text + self_attn(text, text, text, text_mask)[0]
            text = self.norms[i * 3 + 1](text)

            # MLP
            text = text + mlp(text)
            text = self.norms[i * 3 + 2](text)

            # Cross attention: vision attends to text
            vision_norm = self.norms[i * 3 + 1](vision)
            vision = vision_norm + cross_attn(vision_norm, text, text, text_mask)

            # Self attention on vision
            vision = vision + self_attn(vision, vision, vision_mask)[0]
            vision = self.norms[i * 3 + 1](vision)

            # MLP
            vision = vision + mlp(vision)
            vision = self.norms[i * 3 + 2](vision)

        return text, vision


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model.

    Unified VLM combining encoder and decoder.
    """

    def __init__(
        self,
        vision_embed_dim: int = 768,
        text_embed_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # Ensure num_heads divides hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = max(1, hidden_dim // 64)  # Adjust to fit

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Projections
        self.vision_projection = nn.Linear(vision_embed_dim, hidden_dim)
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Encoder (vision -> text)
        self.encoder = MultimodalTransformer(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            text_dim=hidden_dim,
            vision_dim=hidden_dim,
        )

        # Decoder (text -> text)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)

        # Output head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie weights with embedding
        self.lm_head.weight = self.text_embedding.weight

    def encode_image(self, vision_features: Tensor) -> Tensor:
        """Encode vision features."""
        return self.vision_projection(vision_features)

    def generate(
        self,
        vision_features: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate text from vision features.

        Args:
            vision_features: (batch, vision_seq, vision_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated token IDs (batch, seq)
        """
        batch_size = vision_features.shape[0]

        # Encode vision
        vision_hidden = self.encode_image(vision_features)

        # Start with BOS token
        current_ids = torch.full((batch_size, 1), 1, dtype=torch.long)

        for _ in range(max_length):
            # Embed text
            text_emb = self.text_embedding(current_ids)
            positions = torch.arange(
                current_ids.size(1), device=current_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            text_emb = text_emb + self.position_embedding(positions)

            # Encode
            text_hidden, vision_hidden = self.encoder(text_emb, vision_hidden)

            # Get next token logits
            logits = self.lm_head(text_hidden[:, -1])

            # Temperature sampling
            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            # Greedy for now
            next_token = probs.argmax(dim=-1, keepdim=True)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop if all EOS
            if (next_token == 2).all():  # EOS token
                break

        return current_ids

    def forward(
        self,
        vision_features: Tensor,
        text_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Forward pass.

        Args:
            vision_features: (batch, vision_seq, vision_dim)
            text_ids: (batch, text_seq)
            labels: Optional labels for LM loss

        Returns:
            (loss, metadata)
        """
        # Project vision
        vision_hidden = self.encode_image(vision_features)

        # Embed text
        text_emb = self.text_embedding(text_ids)
        positions = torch.arange(
            text_ids.size(1), device=text_ids.device
        ).unsqueeze(0).expand(text_emb.shape[0], -1)
        text_emb = text_emb + self.position_embedding(positions)

        # Encode
        text_hidden, _ = self.encoder(text_emb, vision_hidden)

        # LM logits
        logits = self.lm_head(text_hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        metadata = {
            "logits": logits,
        }

        return loss, metadata


def create_multimodal_model(
    model_type: str = "vlm",
    **kwargs,
) -> nn.Module:
    """
    Create a multimodal model.

    Args:
        model_type: Model type ("vlm", "multimodal_transformer")
        **kwargs: Model configuration

    Returns:
        nn.Module instance
    """
    if model_type == "vlm":
        return VisionLanguageModel(**kwargs)
    elif model_type == "multimodal_transformer":
        return MultimodalTransformer(**kwargs)
    else:
        return VisionLanguageModel(**kwargs)


__all__ = [
    "CrossAttentionModule",
    "CrossAttentionConfig",
    "MultimodalTransformer",
    "VisionLanguageModel",
    "create_multimodal_model",
]