"""Tokenizer round-trip / serialization / consistency tests."""

from src.config import ModelConfig, check_vocab_consistency
from src.tokenizer import BPETokenizer

SAMPLES = [
    "hello world",
    "def fibonacci(n):\n    return n",
    "the quick brown fox jumps over the lazy dog",
    "import torch; x = torch.randn(3, 3)",
    "SELECT * FROM users WHERE age > 18;",
]


def _train(texts, vocab_size=1000):
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, verbose=False)
    return tokenizer


def test_encode_decode_roundtrip():
    tokenizer = _train(SAMPLES)
    for text in SAMPLES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        # BPE must never fall back to <unk> for known-ascii text
        assert tokenizer.unk_id() not in ids, (text, ids)
        assert tokenizer.decode(ids) == text


def test_no_unknown_for_spaces():
    tokenizer = _train(SAMPLES)
    ids = tokenizer.encode("space expected", add_special_tokens=False)
    assert tokenizer.unk_id() not in ids
    assert tokenizer.decode(ids) == "space expected"


def test_special_tokens_are_pinned():
    tokenizer = _train(SAMPLES)
    ids = tokenizer.encode("hello world", add_special_tokens=True)
    assert ids[0] == tokenizer.bos_id()
    assert ids[-1] == tokenizer.eos_id()
    assert tokenizer.bos_id() != tokenizer.eos_id() != tokenizer.pad_id() != tokenizer.unk_id()


def test_compat_tokenizer_api():
    tokenizer = _train(SAMPLES)
    for method in ("eos_id", "bos_id", "pad_id", "unk_id"):
        assert tokenizer.__getattribute__(method)() >= 0
    assert tokenizer.id_to_piece(tokenizer.token_to_id("h")) == "h"
    assert tokenizer.decode_ids([tokenizer.token_to_id("h")]) == "h"
    assert tokenizer.eos_token_id == tokenizer.eos_id()
    assert tokenizer.pad_token_id == tokenizer.pad_id()


def test_save_load_roundtrip(tmp_path):
    tokenizer = _train(SAMPLES)
    out = tmp_path / "tok"
    tokenizer.save(str(out))

    loaded = BPETokenizer.load(str(out))
    assert loaded.vocab == tokenizer.vocab
    assert loaded.merge_ranks == tokenizer.merge_ranks

    for text in SAMPLES:
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == tokenizer.decode(tokenizer.encode(text))


def test_canonical_vocab_matches_model_config():
    # Canonical 120M uses vocab 50304; a tokenizer with the same target is
    # consistent, and the actual trained vocab never exceeds the model size.
    canonical = ModelConfig()  # defaults == canonical Exp-Coder 120M
    assert canonical.vocab_size == 50304

    untrained = BPETokenizer(vocab_size=canonical.vocab_size)
    check_vocab_consistency(untrained, canonical)

    trained_small = BPETokenizer(vocab_size=canonical.vocab_size)
    trained_small.train(SAMPLES, verbose=False)
    check_vocab_consistency(trained_small, canonical)


def test_vocab_mismatch_fails_loudly():
    tokenizer = BPETokenizer(vocab_size=32000)
    canonical = ModelConfig()  # 50304
    try:
        check_vocab_consistency(tokenizer, canonical)
    except ValueError as exc:
        assert "32000" in str(exc) and "50304" in str(exc)
    else:
        raise AssertionError("expected ValueError for vocab mismatch")