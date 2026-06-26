"""
Tests for Expera AI model architecture.
Tests model initialization, forward pass, and basic functionality.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from src.model.architecture import (
    ExperaModel,
    MultiHeadAttention,
    FeedForward,
    RotaryPositionalEmbedding,
    TokenEmbedding,
    EmbeddingModule,
    TransformerBlock,
)


class TestExperaModel:
    """Test suite for Expera AI model."""
    
    @pytest.fixture
    def model_config(self):
        """Default model configuration for testing."""
        return {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "activation": "gelu",
            "dropout": 0.0,  # No dropout for testing
            "use_rope": True,
        }
    
    @pytest.fixture
    def model(self, model_config):
        """Create model instance."""
        return ExperaModel(**model_config)
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        params = model.get_num_params()
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["total"] == params["trainable"]
        
    def test_forward_pass(self, model):
        """Test forward pass with random input."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        outputs = model(input_ids, use_cache=False, return_dict=True)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)
        
    def test_forward_with_cache(self, model):
        """Test forward pass with KV cache."""
        batch_size = 2
        
        # First forward pass
        input_ids_1 = torch.randint(0, 100, (batch_size, 8))
        outputs_1 = model(input_ids_1, use_cache=True, return_dict=True)
        
        assert outputs_1["past_key_values"] is not None
        assert len(outputs_1["past_key_values"]) == 2  # num_layers
        
        # Second forward pass with cache
        input_ids_2 = torch.randint(0, 100, (batch_size, 4))
        outputs_2 = model(
            input_ids_2,
            use_cache=True,
            past_key_values=outputs_1["past_key_values"],
            return_dict=True,
        )
        
        assert outputs_2["past_key_values"] is not None
        
    def test_generate(self, model):
        """Test text generation."""
        batch_size = 1
        input_ids = torch.randint(0, 100, (batch_size, 4))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
        )
        
        assert generated.shape[1] == 14  # 4 input + 10 generated
        
    def test_generate_with_sampling(self, model):
        """Test text generation with sampling."""
        batch_size = 1
        input_ids = torch.randint(0, 100, (batch_size, 4))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
        
        assert generated.shape[1] == 14


class TestComponents:
    """Test suite for individual model components."""
    
    @pytest.fixture
    def hidden_size(self):
        return 64
    
    @pytest.fixture
    def num_heads(self):
        return 4
    
    @pytest.fixture
    def batch_seq(self):
        return 2, 8
    
    def test_rotary_embedding(self, hidden_size, batch_seq):
        """Test RoPE module."""
        batch, seq = batch_seq
        rope = RotaryPositionalEmbedding(dim=hidden_size, max_seq_len=128)
        
        q = torch.randn(batch, 4, seq, hidden_size // 4)
        k = torch.randn(batch, 4, seq, hidden_size // 4)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
    def test_multi_head_attention(self, hidden_size, num_heads, batch_seq):
        """Test multi-head attention."""
        batch, seq = batch_seq
        attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
        )
        
        x = torch.randn(batch, seq, hidden_size)
        output, cache = attn(x)
        
        assert output.shape == (batch, seq, hidden_size)
        assert cache is None  # No cache by default
        
    def test_multi_head_attention_with_cache(self, hidden_size, num_heads):
        """Test attention with KV cache."""
        batch = 2
        attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
        )
        
        # First pass
        x1 = torch.randn(batch, 4, hidden_size)
        output1, cache = attn(x1, use_cache=True)
        
        assert cache is not None
        assert len(cache) == 2  # K and V
        
        # Second pass
        x2 = torch.randn(batch, 2, hidden_size)
        output2, cache = attn(x2, past_key_value=cache, use_cache=True)
        
        assert output2.shape == (batch, 2, hidden_size)
        
    def test_feed_forward(self, hidden_size, batch_seq):
        """Test feed-forward network."""
        batch, seq = batch_seq
        ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            activation="gelu",
            dropout=0.0,
        )
        
        x = torch.randn(batch, seq, hidden_size)
        output = ffn(x)
        
        assert output.shape == (batch, seq, hidden_size)
        
    def test_feed_forward_swiglu(self, hidden_size, batch_seq):
        """Test SwiGLU feed-forward network."""
        batch, seq = batch_seq
        ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=int(8 * hidden_size / 3),
            activation="swiglu",
            dropout=0.0,
        )
        
        x = torch.randn(batch, seq, hidden_size)
        output = ffn(x)
        
        assert output.shape == (batch, seq, hidden_size)
        
    def test_transformer_block(self, hidden_size, num_heads, batch_seq):
        """Test transformer block."""
        batch, seq = batch_seq
        block = TransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=hidden_size * 4,
            dropout=0.0,
        )
        
        x = torch.randn(batch, seq, hidden_size)
        output, cache = block(x, use_cache=True)
        
        assert output.shape == (batch, seq, hidden_size)
        assert cache is not None
        
    def test_embedding_module(self, batch_seq):
        """Test embedding module."""
        batch, seq = batch_seq
        embed = EmbeddingModule(
            vocab_size=1000,
            hidden_size=64,
            max_position_embeddings=128,
        )
        
        input_ids = torch.randint(0, 100, (batch, seq))
        positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        
        embeddings = embed(input_ids, positions)
        
        assert embeddings.shape == (batch, seq, 64)


class TestModelScale:
    """Test model scaling with different sizes."""
    
    def test_tiny_model(self):
        """Test tiny model variant."""
        model = ExperaModel(
            vocab_size=1000,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            max_position_embeddings=64,
        )
        
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        outputs = model(input_ids, return_dict=True)
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)
        
    def test_gqa_model(self):
        """Test model with Grouped Query Attention."""
        model = ExperaModel(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,  # GQA: 2 KV heads
            intermediate_size=256,
        )
        
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        outputs = model(input_ids, return_dict=True)
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])