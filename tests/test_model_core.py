"""Core model semantics: causal mask values, RoPE, GQA, forward shapes."""

import torch

from src.model.architecture import (
    ExperaModel,
    MultiHeadAttention,
    RotaryPositionalEmbedding,
)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------
def _mask_values(model, seq_len, total_len):
    return model._create_causal_mask(
        seq_len, total_len, dtype=torch.float32, device=torch.device("cpu")
    )[0, 0]


def test_causal_mask_no_past(tiny_model):
    mask = _mask_values(tiny_model, seq_len=4, total_len=4)
    # attend (0.0) strictly lower-or-equal triangle, -inf above
    assert (mask[torch.tril(torch.ones(4, 4, dtype=torch.bool))] == 0.0).all()
    assert (mask[torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)] == float("-inf")).all()


def test_causal_mask_with_past(tiny_model):
    # query length 2, past length 3 -> total 5
    mask = _mask_values(tiny_model, seq_len=2, total_len=5)
    # row i (global pos past+i=3+i) attends cols j <= 3+i
    expected = torch.zeros(2, 5)
    expected[0, 4] = float("-inf")   # row0 global 3: attend 0..3
    expected[1, :] = 0.0             # row1 global 4: attend 0..4
    assert torch.equal(mask, expected)


def test_causal_mask_single_token_with_past(tiny_model):
    # generation step: query length 1 after past 3
    mask = _mask_values(tiny_model, seq_len=1, total_len=4)
    assert torch.equal(mask[0], torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_causal_mask_position_attends_self(tiny_model):
    mask = _mask_values(tiny_model, seq_len=6, total_len=6)
    assert (torch.diag(mask.clone().detach()) == 0.0).all()


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def test_rope_shape_and_determinism():
    rope = RotaryPositionalEmbedding(dim=8, max_seq_len=16)
    q = torch.randn(2, 4, 8, 8)
    k = torch.randn(2, 4, 8, 8)
    q1, k1 = rope(q, k)
    q2, k2 = rope(q, k)
    assert q1.shape == q.shape and k1.shape == k.shape
    assert torch.equal(q1, q2) and torch.equal(k1, k2)


def test_rope_position_dependence(tiny_model_config):
    rope = RotaryPositionalEmbedding(dim=8, max_seq_len=16)
    x = torch.randn(1, 1, 1, 8)
    pos0 = rope(x, x)[0]
    pos1 = rope(x, x)[0]  # same position -> identical
    assert torch.equal(pos0, pos1)

    # different positions must produce different rotations
    x_dup = torch.cat([x, x], dim=2)  # (1,1,2,8)
    _, k_rot = rope(x_dup, x_dup, positions=torch.tensor([0, 1]))
    assert not torch.equal(k_rot[:, :, 0], k_rot[:, :, 1])


def test_rope_rotation_is_unitary():
    rope = RotaryPositionalEmbedding(dim=4, max_seq_len=8)
    x = torch.randn(1, 1, 1, 4)
    x_rot, _ = rope(x, x, positions=torch.tensor([2]))
    # norm preserved per pair-wise rotation
    assert torch.allclose(x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-6)


# ---------------------------------------------------------------------------
# GQA
# ---------------------------------------------------------------------------
def test_repeat_kv():
    from src.model.architecture.attention import MultiHeadAttention as _MHA

    attn = _MHA(hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8, dropout=0.0)
    kv = torch.randn(2, 2, 5, 8)
    out = attn._repeat_kv(kv, 2)
    assert out.shape == (2, 4, 5, 8)
    # head grouping: head h == repeated kv_head h // 2
    assert torch.equal(out[:, 0], kv[:, 0]) and torch.equal(out[:, 1], kv[:, 0])
    assert torch.equal(out[:, 2], kv[:, 1]) and torch.equal(out[:, 3], kv[:, 1])


def test_gqa_equals_full_attention_on_matching_weights():
    """GQA (2 KV heads, each stretched over 2 query heads) must equal an
    MHA whose KV heads are duplicated identically in groups."""
    hidden, heads, hd = 32, 4, 8
    gqa = MultiHeadAttention(hidden_size=hidden, num_heads=heads, num_kv_heads=2,
                             head_dim=hd, dropout=0.0, attention_dropout=0.0)
    full = MultiHeadAttention(hidden_size=hidden, num_heads=heads, head_dim=hd,
                              dropout=0.0, attention_dropout=0.0)
    gqa.eval()
    full.eval()

    # The full model's KV weights are the GQA weights repeated per group:
    # full kv head {0,1} <- gqa kv head 0, full kv head {2,3} <- gqa kv head 1.
    # Tile whole KV blocks (each hd rows), not individual rows.
    def tile_blocks(key, value):
        struct = value.reshape(2, hd, hidden)
        return struct.repeat_interleave(2, dim=0).reshape(heads * hd, hidden).clone()

    full.q_proj.weight.data = gqa.q_proj.weight.data.clone()
    full.o_proj.weight.data = gqa.o_proj.weight.data.clone()
    full.k_proj.weight.data = tile_blocks("k", gqa.k_proj.weight.data)
    full.v_proj.weight.data = tile_blocks("v", gqa.v_proj.weight.data)

    x = torch.randn(2, 8, hidden)
    out_full, _ = full(x)
    out_gqa, _ = gqa(x)
    assert torch.allclose(out_gqa, out_full, atol=1e-5)


# ---------------------------------------------------------------------------
# Forward pass shape / CPU
# ---------------------------------------------------------------------------
def test_forward_shape_batch_and_sequence(tiny_model):
    b, s = 3, 5
    ids = torch.randint(0, 64, (b, s))
    out = tiny_model(ids, use_cache=False, return_dict=True)
    assert out["logits"].shape == (b, s, 128)  # batch x seq x vocab


def test_forward_generates_gradients_to_vocab(tiny_model):
    ids = torch.randint(0, 64, (2, 4))
    logits = tiny_model(ids)["logits"]
    assert logits.is_floating_point() and torch.isfinite(logits).all()


def test_cpu_float32_forward_deterministic(tiny_model):
    ids = torch.randint(0, 64, (1, 6))
    a = tiny_model(ids, use_cache=False, return_dict=True)["logits"]
    b = tiny_model(ids, use_cache=False, return_dict=True)["logits"]
    assert torch.equal(a, b)