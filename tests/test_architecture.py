"""
Tests for Expera AI Phase 4 Architecture modules.

Tests:
- RMSNorm
- YaRN RoPE scaling
- KV Cache Manager
- FlashAttention with Sliding Window
- Speculative Decoding
- MoE infrastructure
- Grouped FFN
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

from src.model.architecture.norm import RMSNorm
from src.model.architecture.rope import (
    YaRNScaledRotaryEmbedding,
    LinearScaledRoPE,
    NTKScaledRoPE,
    apply_rotary_positional,
    apply_rotary_positional_half,
)
from src.model.architecture.cache import (
    CacheConfig,
    KVCache,
    KVCacheManager,
    PagedKVCacheManager,
    StreamingKVCacheManager,
)
from src.model.architecture.flash_attention import (
    FlashAttention,
    FlashAttentionVarlen,
    PagedAttention,
    HAS_FLASH_ATTN,
)
from src.model.architecture.speculative import (
    SpeculativeDecoder,
    SpeculativeConfig,
    NGramSpeculator,
    TreeSpeculator,
)
from src.model.architecture.moe import (
    MoEConfig,
    Expert,
    GatingMechanism,
    MoELayer,
    SparseMoEBlock,
)
from src.model.architecture.grouped_ffn import (
    GroupedFFN,
    MultiQueryFFN,
    ParallellFFN,
    FusedFFN,
    ExpertPoolFFN,
    create_grouped_ffn,
)


class TestRMSNorm:
    """Test suite for RMSNorm."""

    @pytest.fixture
    def rmsnorm(self):
        return RMSNorm(normalized_shape=128)

    def test_initialization(self, rmsnorm):
        """Test RMSNorm initializes with correct parameters."""
        assert rmsnorm.weight.shape == (128,)
        assert rmsnorm.eps == 1e-6

    def test_forward(self, rmsnorm):
        """Test RMSNorm forward pass."""
        x = torch.randn(2, 4, 128)
        output = rmsnorm(x)
        assert output.shape == x.shape

    def test_output_range(self, rmsnorm):
        """Test RMSNorm produces normalized output."""
        x = torch.randn(2, 4, 128)
        output = rmsnorm(x)
        # Check RMS is approximately 1 for each token
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestYaRNRoPE:
    """Test suite for YaRN RoPE scaling."""

    @pytest.fixture
    def yarn_embed(self):
        return YaRNScaledRotaryEmbedding(dim=64, max_seq_len=512)

    def test_initialization(self, yarn_embed):
        """Test YaRN initializes correctly."""
        assert yarn_embed.dim == 64
        assert yarn_embed.max_seq_len == 512

    def test_forward(self, yarn_embed):
        """Test YaRN forward produces cos/sin."""
        cos, sin = yarn_embed(seq_len=32)
        assert cos.shape[0] == 32
        assert sin.shape[0] == 32

    def test_ntk_rope(self):
        """Test NTK RoPE scaling."""
        ntk = NTKScaledRoPE(dim=64, max_seq_len=512, original_seq_len=256)
        cos, sin = ntk(seq_len=32)
        assert cos.shape[0] == 32


class TestKVCache:
    """Test suite for KV Cache Manager."""

    @pytest.fixture
    def cache_config(self):
        return CacheConfig(
            max_seq_len=512,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            device="cpu",
        )

    @pytest.fixture
    def cache_manager(self, cache_config):
        manager = KVCacheManager(config=cache_config, sliding_window=128)
        manager._initialize()
        return manager

    def test_initialization(self, cache_manager):
        """Test KVCacheManager initializes."""
        assert cache_manager._initialized  # Fixture calls _initialize()

    def test_update(self, cache_manager):
        """Test updating cache."""
        k = torch.randn(4, 1, 32)  # (num_heads, 1, head_dim)
        v = torch.randn(4, 1, 32)
        cache_manager.update(layer_idx=0, k=k, v=v)
        assert cache_manager._initialized

    def test_get(self, cache_manager):
        """Test getting from cache."""
        k, v = cache_manager.get(layer_idx=0)
        assert k.shape[0] == 1  # batch
        assert k.shape[2] == 0  # empty initially

    def test_sliding_window(self, cache_manager):
        """Test sliding window eviction."""
        k = torch.randn(4, 1, 32)
        for _ in range(150):
            cache_manager.update(layer_idx=0, k=k, v=k)
        # Should have at most sliding_window tokens
        k_out, _ = cache_manager.get(layer_idx=0)
        assert k_out.shape[2] <= 128


class TestPagedKVCache:
    """Test suite for Paged KV Cache."""

    @pytest.fixture
    def paged_cache(self):
        config = CacheConfig(max_seq_len=512, num_layers=4, num_heads=4, head_dim=32, device="cpu")
        return PagedKVCacheManager(config=config, block_size=16)

    def test_paged_update(self, paged_cache):
        """Test paged cache update."""
        k = torch.randn(4, 1, 32)  # (num_heads, 1, head_dim)
        paged_cache.update(layer_idx=0, k=k, v=k)
        assert paged_cache._initialized


class TestFlashAttention:
    """Test suite for FlashAttention."""

    @pytest.fixture
    def flash_attn(self):
        return FlashAttention(embed_dim=128, num_heads=4)

    def test_initialization(self, flash_attn):
        """Test FlashAttention initializes."""
        assert flash_attn.embed_dim == 128
        assert flash_attn.num_heads == 4
        assert flash_attn.head_dim == 32

    def test_forward(self, flash_attn):
        """Test FlashAttention forward."""
        x = torch.randn(2, 8, 128)
        output, weights = flash_attn(x, x, x, is_causal=True)
        assert output.shape == x.shape

    def test_causal_mask(self, flash_attn):
        """Test causal masking."""
        x = torch.randn(2, 4, 128)
        output, _ = flash_attn(x, x, x, is_causal=True)
        assert output.shape == x.shape

    def test_sliding_window(self):
        """Test sliding window attention."""
        flash_attn = FlashAttention(
            embed_dim=128,
            num_heads=4,
            window_size=(3, 0),  # Left=3, Right=0 (causal)
        )
        x = torch.randn(2, 8, 128)
        output, _ = flash_attn(x, x, x, is_causal=True)
        assert output.shape == x.shape


class TestSpeculativeDecoding:
    """Test suite for Speculative Decoding."""

    @pytest.fixture
    def mock_draft_model(self):
        model = nn.Linear(128, 32000)
        return model

    @pytest.fixture
    def mock_target_model(self):
        model = nn.Linear(128, 32000)
        return model

    def test_speculative_config(self):
        """Test SpeculativeConfig."""
        config = SpeculativeConfig(draft_max_tokens=4)
        assert config.draft_max_tokens == 4

    def test_ngram_speculator(self):
        """Test N-gram speculator."""
        speculator = NGramSpeculator(vocab_size=32000, ngram_size=4)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        tokens, probs = speculator(input_ids, max_tokens=2)
        assert tokens.shape[1] == 2

    def test_tree_speculator(self, mock_draft_model):
        """Test Tree speculator."""
        # Wrap in simple module that outputs logits
        # Linear(128, 32000) expects embeddings (batch, seq_len, 128), not token IDs
        class DraftWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                # x is (batch, seq_len) token IDs, convert to embeddings
                embeddings = torch.randn(x.shape[0], x.shape[1], 128, device=x.device)
                return self.model(embeddings)
        speculator = TreeSpeculator(
            draft_model=DraftWrapper(mock_draft_model),
            vocab_size=32000,
            branch_factor=3,
        )
        input_ids = torch.tensor([[1, 2, 3]])  # (batch, seq_len) token IDs
        tokens, probs = speculator(input_ids)
        assert tokens.shape[1] == 3


class TestMoE:
    """Test suite for MoE."""

    @pytest.fixture
    def moe_layer(self):
        return MoELayer(
            input_dim=128,
            output_dim=128,
            hidden_dim=256,
            num_experts=4,
            top_k=2,
        )

    def test_expert_creation(self):
        """Test Expert creation."""
        expert = Expert(128, 128, 256)
        x = torch.randn(2, 4, 128)
        out = expert(x)
        assert out.shape == x.shape

    def test_moe_layer(self, moe_layer):
        """Test MoE layer forward."""
        x = torch.randn(2, 4, 128)
        out, metadata = moe_layer(x)
        assert out.shape == x.shape
        assert "expert_usage" in metadata

    def test_load_balancing(self, moe_layer):
        """Test load balancing loss computation."""
        x = torch.randn(2, 4, 128)
        _, metadata = moe_layer(x)
        # Should have expert usage tracked
        assert metadata["expert_usage"].shape[0] == 4


class TestGroupedFFN:
    """Test suite for Grouped FFN."""

    @pytest.fixture
    def grouped_ffn(self):
        return GroupedFFN(dim=128, hidden_dim=256, num_groups=4)

    def test_grouped_ffn_forward(self, grouped_ffn):
        """Test GroupedFFN forward."""
        x = torch.randn(2, 4, 128)
        out = grouped_ffn(x)
        assert out.shape == x.shape

    def test_multiquery_ffn(self):
        """Test MultiQueryFFN."""
        ffnn = MultiQueryFFN(dim=128, hidden_dim=256, num_groups=4)
        x = torch.randn(2, 4, 128)
        out = ffnn(x)
        assert out.shape == x.shape

    def test_parallel_ffn(self):
        """Test ParallelFFN."""
        ffnn = ParallellFFN(dim=128, hidden_dim=256)
        x = torch.randn(2, 4, 128)
        out = ffnn(x)
        assert out.shape == x.shape

    def test_fused_ffn(self):
        """Test FusedFFN."""
        ffnn = FusedFFN(dim=128, hidden_dim=256)
        x = torch.randn(2, 4, 128)
        out = ffnn(x)
        assert out.shape == x.shape

    def test_expert_pool(self):
        """Test ExpertPoolFFN."""
        ffnn = ExpertPoolFFN(dim=128, hidden_dim=256, num_experts=4, top_k=2)
        x = torch.randn(2, 4, 128)
        out = ffnn(x)
        assert out.shape == x.shape


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_rmsnorm_with_attention(self):
        """Test RMSNorm with attention."""
        norm = RMSNorm(normalized_shape=128)
        attn = FlashAttention(embed_dim=128, num_heads=4)

        x = torch.randn(2, 4, 128)
        x_norm = norm(x)
        out, _ = attn(x_norm, x_norm, x_norm)
        assert out.shape == x.shape

    def test_yarn_with_model(self):
        """Test YaRN with RoPE application."""
        embed = YaRNScaledRotaryEmbedding(dim=64, max_seq_len=256)
        x = torch.randn(2, 4, 64)

        cos, sin = embed(seq_len=4)
        x_rope = apply_rotary_positional(x, cos, sin)
        assert x_rope.shape == x.shape


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])