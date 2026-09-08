"""Exp-Coder public facade (exp_coder.* aliases) and config loader."""

import pytest


def test_exp_coder_alias_maps_to_src():
    from exp_coder.model import ExperaModel
    import src.model.architecture as arch

    assert ExperaModel is arch.ExperaModel


def test_exp_coder_submodules_alias():
    import exp_coder.training
    import exp_coder.tokenizer
    import exp_coder.inference
    import exp_coder.data
    import exp_coder.utils
    import exp_coder.config
    assert exp_coder.training.Trainer is not None
    assert exp_coder.tokenizer.BPETokenizer is not None


def test_canonical_config_loader():
    from exp_coder.config import load_model_config

    cfg = load_model_config("configs/exp_coder_120m.yaml")
    assert cfg.vocab_size == 50304


def test_training_config_loader():
    from exp_coder.config import load_training_config

    cfg = load_training_config("configs/training.yaml")
    assert cfg["seed"] == 42
    assert "batch_size" in cfg and "max_steps" in cfg


def test_config_loader_rejects_missing_file():
    from exp_coder.config import load_model_config

    with pytest.raises(FileNotFoundError):
        load_model_config("configs/does_not_exist.yaml")


def test_model_config_validate_gqa_dims():
    from exp_coder.config import ModelConfig

    ModelConfig(vocab_size=64, hidden_size=64, num_heads=4, num_kv_heads=2).validate()

    with pytest.raises(ValueError):
        ModelConfig(vocab_size=64, hidden_size=50, num_heads=4).validate()