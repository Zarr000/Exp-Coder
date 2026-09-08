"""Exp-Coder canonical configuration and loading.

A single source of truth for the Exp-Coder model architecture and training runs.

The canonical development model is *Exp-Coder 120M*
(``configs/exp_coder_120m.yaml``, vocab 50304 / hidden 768 / 12 layers,
approximately 123.6M parameters, tied embeddings). Smaller configs under
``configs/`` are explicitly labelled debug/tiny and are only for fast
iteration and tests.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class ModelConfig:
    """Exp-Coder transformer architecture.

    Fields mirror the arguments accepted by
    ``src.model.architecture.ExperaModel`` (decoder-only, pre-norm, RoPE,
    optional GQA, tied embeddings).
    """

    vocab_size: int = 50304
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # None -> full attention (== num_heads)
    head_dim: Optional[int] = None  # None -> hidden_size // num_heads
    intermediate_size: Optional[int] = None  # None -> derived (4x or 8/3x)
    max_position_embeddings: int = 2048
    activation: str = "gelu"
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_rope: bool = True
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    padding_idx: Optional[int] = None
    std: float = 0.02

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments accepted by ``ExperaModel``."""
        return asdict(self)

    def estimate_num_parameters(self) -> int:
        """Analytical parameter estimate (tied embeddings, no LM head).

        Matches the measured count for the canonical 120M config
        (50304 / 768 / 12 / gelu): ~123.6M.
        """
        v, h, l = self.vocab_size, self.hidden_size, self.num_layers
        kv = self.num_kv_heads if self.num_kv_heads else self.num_heads

        embedding = v * h
        attention = h * h + 2 * kv * (h // self.num_heads) * h + h * h
        ffn_dim = self.intermediate_size
        if ffn_dim is None:
            ffn_dim = int(8 * h / 3) if self.activation == "swiglu" else 4 * h
        projections = 3 if self.activation in ("swiglu",) else 2
        ffn = ffn_dim * h * projections
        layer = attention + ffn + 2 * h
        return embedding + l * layer

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load a canonical model config from YAML."""
        data = _load_yaml(path)
        block = data.get("model", data)
        known = {f for f in cls.__dataclass_fields__ if f != "name"}
        params = {k: v for k, v in block.items() if k in known}
        return cls(**params)

    def validate(self) -> None:
        """Structural sanity checks; raises on impossible dimensions."""
        if self.hidden_size % self.num_heads != 0 or (
            self.num_kv_heads and self.num_heads % self.num_kv_heads != 0
        ):
            raise ValueError(
                f"num_heads={self.num_heads} not compatible with "
                f"hidden_size={self.hidden_size} / num_kv_heads={self.num_kv_heads}"
            )


def _load_yaml(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: Union[str, Path]) -> ModelConfig:
    """Load a canonical model configuration file."""
    return ModelConfig.from_yaml(path)


def load_training_config(path: Union[str, Path]) -> dict:
    """Load a YAML training-run configuration (flat-ish dict)."""
    data = _load_yaml(path)
    train = data.get("training", data)
    if not isinstance(train, dict):
        raise ValueError(f"Invalid training config: {path}")
    return {k: v for k, v in train.items()}


def check_vocab_consistency(tokenizer: Any, model_config: ModelConfig) -> None:
    """Assert the tokenizer vocabulary is compatible with the model.

    Raises a clear ``ValueError`` (never silently resizes anything) when the
    tokenizer target or actual vocabulary would exceed the model embedding.
    """
    target = getattr(tokenizer, "vocab_size", None)
    actual = len(tokenizer) if hasattr(tokenizer, "__len__") else None

    if target is not None and target != model_config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size %s does not match model vocabulary "
            "size %s." % (target, model_config.vocab_size)
        )
    if actual is not None and actual > model_config.vocab_size:
        raise ValueError(
            "Tokenizer actual vocabulary size %s exceeds model vocabulary "
            "size %s. Embeddings must be at least as large as the tokenizer "
            "vocabulary." % (actual, model_config.vocab_size)
        )