"""Causal LM dataset determinism and label alignment."""

import torch
from torch.utils.data import DataLoader

from src.data import CausalLMDataset, load_text_corpus

TEXTS = [
    "def add(a, b):\n    return a + b",
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "while True:\n    x += 1\n    print(x)",
]


def test_block_determinism(tiny_tokenizer):
    ds1 = CausalLMDataset(TEXTS, tiny_tokenizer, sequence_length=16)
    ds2 = CausalLMDataset(TEXTS, tiny_tokenizer, sequence_length=16)
    assert len(ds1) == len(ds2)
    for i in range(len(ds1)):
        assert torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])
        assert torch.equal(ds1[i]["labels"], ds2[i]["labels"])


def test_labels_equal_inputs(tiny_tokenizer):
    ds = CausalLMDataset(TEXTS, tiny_tokenizer, sequence_length=16)
    for i in range(len(ds)):
        item = ds[i]
        assert torch.equal(item["input_ids"], item["labels"])
        assert item["input_ids"].shape == torch.Size([16])


def test_blocks_are_contiguous_token_slices(tiny_tokenizer):
    ds = CausalLMDataset(TEXTS, tiny_tokenizer, sequence_length=8)
    tokens = ds.tokens
    # block i must equal tokens[8i : 8i+8] exactly (no shuffling inside)
    for i in range(min(len(ds), 3)):
        assert ds[i]["input_ids"].tolist() == tokens[i * 8: i * 8 + 8]


def test_dataloader_batches(tiny_tokenizer):
    ds = CausalLMDataset(TEXTS, tiny_tokenizer, sequence_length=8)
    loader = DataLoader(ds, batch_size=2)
    for batch in loader:
        assert batch["input_ids"].shape[1] == 8
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["input_ids"].shape[0] in (1, 2)


def test_load_text_corpus_file_and_dir(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello from file", encoding="utf-8")
    assert load_text_corpus(f) == ["hello from file"]

    sub = tmp_path / "corpus"
    sub.mkdir()
    (sub / "one.md").write_text("doc one", encoding="utf-8")
    (sub / "two.jsonl").write_text(
        '{"text": "a json line"}\nplain line\n', encoding="utf-8"
    )
    docs = load_text_corpus(sub)
    assert "doc one" in docs
    assert "a json line" in docs
    assert "plain line" in docs