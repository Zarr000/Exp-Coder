"""Shared pytest fixtures for Exp-Coder core tests."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_model_config():
    """Deterministic tiny transformer (GQA, no dropout) for fast unit tests."""
    return {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "activation": "gelu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "use_rope": True,
    }


@pytest.fixture
def tiny_model(tiny_model_config):
    from src.model.architecture import ExperaModel

    torch.manual_seed(0)
    return ExperaModel(**tiny_model_config)


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """Trained small BPE tokenizer (matches tiny_model vocab)."""
    from src.tokenizer import BPETokenizer

    texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "hello world this is the quick brown fox",
        "the quick brown fox jumps over the lazy dog",
        "def hello_world():\n    print('hello world')",
        "import torch; x = torch.randn(3, 3)",
        "class Stack:\n    def push(self, item):\n        self.items.append(item)",
    ] * 20
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(texts, verbose=False)
    path = tmp_path / "tok"
    tokenizer.save(str(path))
    return BPETokenizer.load(str(path))