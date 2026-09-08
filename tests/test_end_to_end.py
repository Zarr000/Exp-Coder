"""Tiny end-to-end core pipeline smoke test.

small corpus -> BPE tokenizer -> dataset -> DataLoader -> tiny ExperaModel
-> loss -> backward -> optimizer -> checkpoint -> reload -> inference

Everything runs on CPU and must stay fast.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import check_vocab_consistency
from src.data import CausalLMDataset
from src.model.architecture import ExperaModel
from src.tokenizer import BPETokenizer
from src.training import LanguageModelingLoss

CORPUS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]",
    "numbers = [1, 2, 3, 4, 5]\ntotal = sum(numbers)\nprint(total)",
    "class Node:\n    def __init__(self, value):\n        self.value = value",
] * 15


def test_end_to_end_cpu(tmp_path):
    torch.manual_seed(7)

    # tokenizer (min byte-level vocab = 256 bytes + special tokens)
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(CORPUS, verbose=False)
    tok_dir = tmp_path / "tokenizer"
    tokenizer.save(str(tok_dir))
    tokenizer = BPETokenizer.load(str(tok_dir))

    from src.config import ModelConfig

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=96,
        max_position_embeddings=256,
        activation="gelu",
        dropout=0.1,
        attention_dropout=0.1,
    )
    check_vocab_consistency(tokenizer, model_cfg)

    # dataset + dataloader
    ds = CausalLMDataset(CORPUS, tokenizer, sequence_length=64)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = ExperaModel(**model_cfg.to_model_kwargs())
    crit = LanguageModelingLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # train a few steps
    losses = []
    for batch in loader:
        ids = batch["input_ids"]
        logits = model(ids, use_cache=False, return_dict=True)["logits"]
        loss = crit(logits, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if len(losses) >= 8:
            break
    assert all(torch.isfinite(torch.tensor(l)).item() for l in losses)

    # checkpoint
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "format": "exp-coder-v1",
        "step": len(losses),
        "model_state_dict": model.state_dict(),
    }, ckpt_path)

    # reload + inference
    model2 = ExperaModel(**model_cfg.to_model_kwargs())
    model2.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
    model2.eval()

    from src.inference import GenerationConfig, Generator
    from src.inference.generator import DecodingStrategy

    gen = Generator(model2, tokenizer, GenerationConfig(
        strategy=DecodingStrategy.GREEDY, max_new_tokens=10, use_cache=True))
    text = gen.generate("def fibonacci")
    assert isinstance(text, str)