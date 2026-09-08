# Phase 4: Model Architecture Upgrade

## Overview

Optimize Expera AI for:
- Coding intelligence
- Long-context understanding
- Future multimodal integration
- Efficient inference on consumer GPUs

## Goals

1. Improve code understanding and generation
2. Support 128K+ context length
3. Optimize for consumer GPU inference (24GB VRAM)
4. Maintain backward compatibility
5. Provide configurable architecture switches

## Core Implementations

### 1. RMSNorm

Faster, more stable than LayerNorm.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        # RMS normalization
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight + (self.bias if self.bias is not None else 0)
```

### 2. SwiGLU

Swish-Gated Linear Unit - better than ReLU/GELU for LLMs.

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        # SwiGLU: silu(x @ W) * (x @ V)
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
```

### 3. FlashAttention v2

Memory-efficient attention.

```python
class FlashAttention(nn.Module):
    def forward(self, q, k, v, mask=None, window_size=None):
        # FlashAttention-2 style with optional sliding window
        # O(n) memory instead of O(n^2)
        return flash_attn_func(q, k, v, mask=mask,
                              window_size=window_size)
```

### 4. KV Cache Manager

Efficient key-value cache management.

```python
class KVCacheManager:
    def __init__(self, max_seq_len: int = 32768):
        self.kv_cache = {}  # layer_idx -> (k_cache, v_cache)
        self.max_seq_len = max_seq_len

    def update(self, layer_idx, k, v):
        # Update cache with eviction for max_seq_len
        pass

    def get(self, layer_idx):
        return self.kv_cache.get(layer_idx, (None, None))
```

### 5. YaRN RoPE scaling

Extended context with RoPE interpolation.

```python
class YaRNScaledRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=32768, base=10000, factor=1.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.factor = factor  # YaRN interpolation factor

    def forward(self, seq_len):
        # Interpolated RoPE with YaRN scaling
        pass
```

### 6. Sliding Window Attention

Efficient long-context with local attention.

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size: int = 4096):
        self.window_size = window_size

    def forward(self, q, k, v, mask=None):
        # Local attention within window_size
        pass
```

### 7. Speculative Decoding

Faster autoregressive generation.

```python
class SpeculativeDecoder:
    def __init__(self, model, draft_model, n_tokens=4):
        self.model = model
        self.draft_model = draft_model
        self.n_tokens = n_tokens

    def decode(self, prompt, max_tokens):
        # Draft tokens with small model, verify with main model
        pass
```

## Optional Implementations

### 8. Mixture of Experts

```python
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        self.experts = nn.ModuleList([FeedForward(dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = top_k
```

### 9. Grouped FFN

```python
class GroupedFFN(nn.Module):
    def forward(self, x):
        # Group queries, apply FFN, scatter back
        pass
```

## Architecture Configuration

```python
@dataclass
class ModelConfig:
    # Normalization
    use_rms_norm: bool = True

    # Activation
    activation: str = "swiglu"  # "swiglu", "gelu", "relu"

    # Attention
    use_flash_attention: bool = True
    flash_attention_version: int = 2

    # Context
    max_context_length: int = 32768
    use_rope_scaling: bool = True
    rope_scaling_type: str = "yarn"  # "yarn", "linear"

    # Sliding window
    use_sliding_window: bool = False
    sliding_window_size: int = 4096

    # KV Cache
    kv_cache_max_seq_len: int = 32768

    # Optional
    use_moe: bool = False
    use_grouped_ffn: bool = False
```

## Backward Compatibility

All new features disabled by default (`ModelConfig` defaults). Existing models load without changes.

## Module Structure

```
src/model/architecture/
├── __init__.py
├── attention.py          # FlashAttention, SlidingWindowAttention
├── feedforward.py       # SwiGLU, GroupedFFN
├── norm.py             # RMSNorm
├── rope.py            # YaRNScaledRotaryEmbedding
├── cache.py           # KVCacheManager
├── speculative.py     # SpeculativeDecoder
├── moe.py             # MoELayer (optional)
├── transformer_block.py  # Updated
└── expera_model.py      # Updated with config switches
```

## Benchmarks

| Feature | Memory | Throughput | Context |
|---------|--------|------------|---------|
| RMSNorm | baseline | baseline | baseline |
| SwiGLU | baseline | +10-15% | baseline |
| FlashAttention v2 | -50% | +10-20% | baseline |
| YaRN + 32K | 0 | 0 | 32K |
| Sliding Window | -30% | +15% | 128K |
| Speculative | +5% | +2-3x | baseline |

## Tests Required

1. Norm correctness
2. Attention equivalence
3. Memory usage
4. Throughput
5. Long context extrapolation
6. Backward compatibility
7. Speculative decoding accuracy

## Exit Criteria

1. All core implementations complete
2. Optional implementations complete (if requested)
3. All tests passing
4. Benchmarks complete
5. Documentation complete
6. Backward compatibility verified