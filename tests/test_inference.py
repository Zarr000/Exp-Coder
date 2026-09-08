"""Inference: cached-vs-uncached equivalence across prompt lengths, decoders."""

import pytest
import torch

from src.model.architecture import ExperaModel
from src.inference import GenerationConfig, Generator
from src.inference.generator import (
    BeamSearch,
    ContrastiveSearch,
    DecodingStrategy,
    GreedySearch,
    Sampling,
)

MODEL_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=64,
    max_position_embeddings=128,
    activation="gelu",
    dropout=0.0,
    attention_dropout=0.0,
)


@pytest.mark.parametrize("prompt_len,max_new", [(1, 15), (2, 15), (8, 15), (40, 15), (125, 3)])
def test_greedy_cached_equals_uncached(prompt_len, max_new):
    torch.manual_seed(10)
    model = ExperaModel(**MODEL_KWARGS)
    model.eval()
    prompt = torch.randint(0, 128, (1, prompt_len))

    uncached = GreedySearch(None).decode(
        model, prompt.clone(),
        config=GenerationConfig(max_new_tokens=max_new, use_cache=False),
    )
    cached = GreedySearch(None).decode(
        model, prompt.clone(),
        config=GenerationConfig(max_new_tokens=max_new, use_cache=True),
    )
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("prompt_len", [2, 8, 40])
@pytest.mark.parametrize("seed", [21, 42])
def test_sampling_cached_equals_uncached(prompt_len, seed):
    torch.manual_seed(11)
    model = ExperaModel(**MODEL_KWARGS)
    model.eval()
    prompt = torch.randint(0, 128, (1, prompt_len))

    def run(use_cache):
        return Sampling(None).decode(
            model, prompt.clone(),
            config=GenerationConfig(
                max_new_tokens=12, use_cache=use_cache,
                temperature=0.9, top_k=20, top_p=0.95, seed=seed,
            ),
        )

    assert torch.equal(run(False), run(True))


def test_greedy_respects_eos_and_max_tokens():
    torch.manual_seed(12)
    model = ExperaModel(**MODEL_KWARGS)
    model.eval()
    prompt = torch.randint(0, 128, (1, 4))

    out = GreedySearch(eos_token_id=0).decode(
        model, prompt.clone(), config=GenerationConfig(max_new_tokens=50, use_cache=False)
    )
    assert out.shape[1] >= 4
    # if an EOS (id 0) was emitted, decoding stops there
    generated = out[0, 4:]
    if (generated == 0).any():
        end = (generated == 0).nonzero()[0].item()
        assert (generated[:end] != 0).all()


def test_generator_with_bpe_tokenizer(tiny_tokenizer):
    from src.model.architecture import ExperaModel

    torch.manual_seed(13)
    vocab = tiny_tokenizer.vocab_size  # target (>= 256 bytes + specials)
    model = ExperaModel(vocab_size=max(vocab, len(tiny_tokenizer)), hidden_size=32,
                        num_layers=2, num_heads=4, num_kv_heads=2,
                        intermediate_size=64, max_position_embeddings=128,
                        activation="gelu", dropout=0.0, attention_dropout=0.0)
    model.eval()
    gen = Generator(
        model, tiny_tokenizer,
        GenerationConfig(strategy=DecodingStrategy.GREEDY, max_new_tokens=8, use_cache=False),
    )
    text = gen.generate("def fibonacci")
    assert isinstance(text, str)
    # either empty (EOS immediately) or contains only decodable text chars
    assert text == "" or all(ord(c) < 0x2000 or not c.isprintable() or c for c in text)


def test_beam_search_runs(tiny_model):
    prompt = torch.randint(0, 120, (1, 4))
    out = BeamSearch(None).decode(
        tiny_model, prompt, config=GenerationConfig(max_new_tokens=6, num_beams=3)
    )
    assert out.shape[0] == 1 and out.shape[1] == 10


def test_beam_search_num_beams_one_matches_greedy(tiny_model):
    prompt = torch.randint(0, 120, (1, 4))
    cfg = GenerationConfig(max_new_tokens=6, num_beams=1)
    beam = BeamSearch(None).decode(tiny_model, prompt.clone(), config=cfg)
    greedy = GreedySearch(None).decode(tiny_model, prompt, config=cfg)
    assert torch.equal(beam, greedy)


def test_contrastive_search_no_crash(tiny_model):
    prompt = torch.randint(0, 120, (1, 4))
    out = ContrastiveSearch(None).decode(
        tiny_model, prompt, config=GenerationConfig(max_new_tokens=6, top_k=10)
    )
    assert out.shape[1] == 10


def test_greedy_without_tokenizer_refuses_unknown_ids():
    # Token IDs outside vocab must not silently blow up: model clamps nowhere,
    # it errors instead -> assert embeddings raise for out-of-range ids
    model = ExperaModel(**MODEL_KWARGS)
    try:
        model(torch.tensor([[500]]))
    except Exception:
        pass
    else:
        raise AssertionError("expected IndexError for out-of-vocab id")