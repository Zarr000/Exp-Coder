"""Canonical Exp-Coder 120M parameter count and vocab consistency."""

import pytest
import torch

from src.config import ModelConfig, load_model_config
from src.model.architecture import ExperaModel

CONFIG_PATH = "configs/exp_coder_120m.yaml"


@pytest.fixture(scope="module")
def canonical_cfg():
    return load_model_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def canonical_model(canonical_cfg):
    torch.manual_seed(0)
    return ExperaModel(**canonical_cfg.to_model_kwargs())


def test_canonical_config_values(canonical_cfg):
    assert (canonical_cfg.vocab_size, canonical_cfg.hidden_size,
            canonical_cfg.num_layers, canonical_cfg.num_heads) == (50304, 768, 12, 12)
    assert canonical_cfg.max_position_embeddings == 2048
    assert canonical_cfg.validate() is None


def test_canonical_parameter_count(canonical_model):
    total = sum(p.numel() for p in canonical_model.parameters())
    assert abs(total - 123_600_000) < 1_500_000, f"measured {total}"


def test_estimate_matches_measured(canonical_cfg, canonical_model):
    measured = sum(p.numel() for p in canonical_model.parameters())
    assert abs(canonical_cfg.estimate_num_parameters() - measured) < measured * 0.01


def test_canonical_model_builds_and_forwards(canonical_model):
    with torch.no_grad():
        logits = canonical_model(torch.tensor([[5, 9, 3]]), return_dict=True)["logits"]
    assert logits.shape == (1, 3, 50304)
    assert torch.isfinite(logits).all()


def test_canonical_vocab_is_50304():
    assert ModelConfig().vocab_size == 50304
    assert load_model_config(CONFIG_PATH).vocab_size == 50304