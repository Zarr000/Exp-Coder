"""
Exp-Coder text generation.

Decoding strategies:
- Greedy search (causal, KV-cache aware)
- Sampling (temperature, top-k, top-p, KV-cache aware)
- Beam search (correct, simple; experimental)
- Contrastive search (experimental)

``Generator`` is the high-level facade; ``InferencePipeline`` builds on it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecodingStrategy(Enum):
    """Decoding strategy types."""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM = "beam"
    CONTRASTIVE = "contrastive"


@dataclass
class GenerationConfig:
    """Generation configuration."""
    # Decoding
    strategy: DecodingStrategy = DecodingStrategy.SAMPLING
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    max_length: int = 4096

    # Sampling
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    # Beam search
    num_beams: int = 1
    early_stopping: bool = False
    length_penalty: float = 1.0

    # Contrastive search
    penalty_alpha: float = 0.6

    # Output
    echo: bool = False
    stop_strings: Optional[List[str]] = None
    seed: Optional[int] = None

    # Misc
    use_cache: bool = True


class BaseDecoder(ABC):
    """Base class for token-by-token decoders."""

    @abstractmethod
    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: "GenerationConfig" = None,
    ) -> Tensor:
        """Decode and return full sequence (prompt + generated tokens)."""
        raise NotImplementedError


def _prefill_or_step(
    model: nn.Module,
    input_ids: Tensor,
    past_key_values: Optional[Any],
    use_cache: bool,
) -> Any:
    """Run the model for one generation step.

    First call on the full prompt returns a cache; subsequent calls feed only
    the last token plus ``past_key_values`` (incremental KV cache).
    """
    if use_cache and past_key_values is not None:
        outputs = model(
            input_ids[:, -1:],
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
    else:
        outputs = model(input_ids, use_cache=use_cache, return_dict=True)
    return outputs


def _apply_repetition_penalty(logits: Tensor, input_ids: Tensor, penalty: float) -> Tensor:
    """Penalize tokens already present in the sequence."""
    score = torch.gather(logits, -1, input_ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits.scatter_(-1, input_ids, score)
    return logits


def _apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep only the top-k logits; mask everything else to -inf."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set whose prob sum >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    mask = mask.scatter(-1, sorted_indices, mask)
    return logits.masked_fill(mask, float("-inf"))


class GreedySearch(BaseDecoder):
    """Greedy (argmax) decoding with optional incremental KV cache."""

    def __init__(self, eos_token_id: Optional[int] = None):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: "GenerationConfig" = None,
    ) -> Tensor:
        config = config or GenerationConfig()
        model.eval()
        max_new = config.max_new_tokens
        use_cache = config.use_cache
        past_key_values = None
        generated = 0

        with torch.no_grad():
            while generated < max_new:
                outputs = _prefill_or_step(model, input_ids, past_key_values, use_cache)
                past_key_values = outputs["past_key_values"] if use_cache else None
                next_logits = outputs["logits"][:, -1, :]

                if config.repeat_penalty != 1.0:
                    next_logits = _apply_repetition_penalty(
                        next_logits, input_ids, config.repeat_penalty
                    )

                next_token = next_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated += 1

                if (self.eos_token_id is not None
                        and (next_token == self.eos_token_id).all()):
                    break

        return input_ids


class Sampling(BaseDecoder):
    """Temperature / top-k / top-p sampling with incremental KV cache."""

    def __init__(self, eos_token_id: Optional[int] = None):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: "GenerationConfig" = None,
    ) -> Tensor:
        config = config or GenerationConfig()
        model.eval()
        max_new = config.max_new_tokens
        use_cache = config.use_cache

        if config.seed is not None:
            torch.manual_seed(config.seed)

        past_key_values = None
        generated = 0

        with torch.no_grad():
            while generated < max_new:
                outputs = _prefill_or_step(model, input_ids, past_key_values, use_cache)
                past_key_values = outputs["past_key_values"] if use_cache else None
                next_logits = outputs["logits"][:, -1, :]

                if config.repeat_penalty != 1.0:
                    next_logits = _apply_repetition_penalty(
                        next_logits, input_ids, config.repeat_penalty
                    )
                if config.temperature != 1.0:
                    next_logits = next_logits / config.temperature
                next_logits = _apply_top_k(next_logits, config.top_k)
                if config.top_p < 1.0:
                    next_logits = _apply_top_p(next_logits, config.top_p)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated += 1

                if (self.eos_token_id is not None
                        and (next_token == self.eos_token_id).all()):
                    break

        return input_ids


class BeamSearch(BaseDecoder):
    """Simple, arithmetic-correct beam search (experimental).

    Runs without the KV cache and does not mask finished hypotheses, so it is
    a reference implementation, not a tuned one.
    """

    def __init__(self, eos_token_id: Optional[int] = None):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: "GenerationConfig" = None,
    ) -> Tensor:
        config = config or GenerationConfig()
        model.eval()
        batch_size = input_ids.shape[0]
        num_beams = max(1, config.num_beams)
        if num_beams == 1:
            return GreedySearch(self.eos_token_id).decode(
                model, input_ids, attention_mask, config
            )

        beams = input_ids.repeat_interleave(num_beams, dim=0)
        log_probs_acc = torch.zeros(batch_size * num_beams, device=input_ids.device)

        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                logits = model(beams, return_dict=True)["logits"][:, -1, :]
                log_prob = F.log_softmax(logits, dim=-1)
                log_prob = log_prob + log_probs_acc.unsqueeze(-1)
                log_prob = log_prob.view(batch_size, num_beams * logits.size(-1))

                top_scores, top_pos = torch.topk(log_prob, num_beams, dim=-1)
                beam_prev = top_pos // logits.size(-1)
                tokens = top_pos % logits.size(-1)

                beam_prev_global = (
                    torch.arange(batch_size, device=beams.device).unsqueeze(-1)
                    * num_beams
                    + beam_prev
                ).reshape(-1)
                beams = torch.cat(
                    [beams[beam_prev_global], tokens.reshape(-1, 1)], dim=-1
                )
                log_probs_acc = top_scores.reshape(-1)

                if (self.eos_token_id is not None
                        and (beams[:, -1] == self.eos_token_id).all()):
                    break

        best = log_probs_acc.view(batch_size, num_beams).argmax(dim=-1)
        return beams.view(batch_size, num_beams, -1)[
            torch.arange(batch_size, device=beams.device), best
        ]


class ContrastiveSearch(BaseDecoder):
    """Contrastive search (experimental, minimally repaired)."""

    def __init__(self, eos_token_id: Optional[int] = None):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: "GenerationConfig" = None,
    ) -> Tensor:
        config = config or GenerationConfig()
        model.eval()
        max_new = config.max_new_tokens
        penalty_alpha = config.penalty_alpha
        top_k = config.top_k
        generated = 0

        with torch.no_grad():
            while generated < max_new:
                outputs = model(input_ids, return_dict=True)
                next_logits = outputs["logits"][:, -1, :]

                if config.repeat_penalty != 1.0:
                    next_logits = _apply_repetition_penalty(
                        next_logits, input_ids, config.repeat_penalty
                    )

                scores = next_logits.clone()
                for idx in range(input_ids.shape[0]):
                    top_k_vals, top_k_idx = torch.topk(scores[idx], top_k)
                    for tok in input_ids[idx]:
                        if tok in top_k_idx:
                            scores[idx, tok] *= penalty_alpha
                        else:
                            scores[idx, tok] = float("-inf")

                probs = F.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated += 1

                if (self.eos_token_id is not None
                        and (next_token == self.eos_token_id).all()):
                    break

        return input_ids


def is_valid_generation_token(token_id: int, tokenizer: Any) -> bool:
    """
    Check whether a token is a reasonable text-generation output.

    Filters special/control tokens and empty decodes.
    """
    if tokenizer is None:
        return True

    eos_id = tokenizer.eos_id()
    bos_id = tokenizer.bos_id()
    pad_id = tokenizer.pad_id()
    unk_id = tokenizer.unk_id()

    if token_id == eos_id:
        return False
    if pad_id >= 0 and token_id == pad_id:
        return False

    piece = tokenizer.id_to_piece(token_id)

    if token_id == unk_id:
        return False

    if piece.startswith("<") and piece.endswith(">"):
        return False

    try:
        decoded = tokenizer.decode_ids([token_id])
        if not decoded or not decoded.strip():
            return False
    except Exception:
        return False

    return True


class Generator:
    """Text generator facade over the decoding strategies."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        config: "GenerationConfig" = None,
        debug: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.debug = debug

        eos_token_id = tokenizer.eos_id() if tokenizer else None

        strategy = self.config.strategy
        if strategy == DecodingStrategy.GREEDY:
            self.decoder = GreedySearch(eos_token_id)
        elif strategy == DecodingStrategy.BEAM:
            self.decoder = BeamSearch(eos_token_id)
        elif strategy == DecodingStrategy.CONTRASTIVE:
            self.decoder = ContrastiveSearch(eos_token_id)
        else:
            self.decoder = Sampling(eos_token_id)

    def generate(
        self,
        prompt: Union[str, List[str], Tensor],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text from a prompt (or list of prompts)."""
        if isinstance(prompt, (str, Tensor)):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        input_ids = self._tokenize(prompts)

        config = self._copy_config(**kwargs)
        if max_new_tokens is not None:
            config.max_new_tokens = max_new_tokens
        if temperature is not None:
            config.temperature = temperature
        if top_p is not None:
            config.top_p = top_p
        if top_k is not None:
            config.top_k = top_k

        output_ids = self.decoder.decode(self.model, input_ids, config=config)

        if self.debug:
            start = (input_ids.shape[1], output_ids.shape[1])
            logger.info(
                "DEBUG generate: prompt=%d tokens, output=%d tokens",
                start[0], start[1],
            )

        return self._decode(output_ids, input_ids)

    def _tokenize(self, prompts: List[Any]) -> Tensor:
        if self.tokenizer is None:
            return torch.randint(0, 1000, (len(prompts), 10))

        all_ids = []
        for prompt in prompts:
            if isinstance(prompt, Tensor):
                all_ids.append(prompt.reshape(-1).tolist())
            else:
                all_ids.append(self.tokenizer.encode(str(prompt)))

        max_len = max(len(ids) for ids in all_ids)
        pad_id = self.tokenizer.pad_id() if hasattr(self.tokenizer, "pad_id") else 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
        return torch.tensor(padded, dtype=torch.long)

    def _decode(self, output_ids: Tensor, input_ids: Tensor) -> Union[str, List[str]]:
        if self.tokenizer is None:
            return "Generated text (no tokenizer)"

        output_ids = output_ids.detach().cpu()
        input_len = input_ids.shape[1]
        texts = []
        for ids in output_ids:
            gen_ids = ids[input_len:].tolist()
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))

        if len(texts) == 1:
            return self._filter_response(texts[0])
        return [self._filter_response(t) for t in texts]

    @staticmethod
    def _filter_response(text: str) -> str:
        """Clean control characters; preserve everything else."""
        if not text:
            return ""
        import re
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
        return cleaned

    def _copy_config(self, **kwargs) -> "GenerationConfig":
        config = GenerationConfig(
            strategy=self.config.strategy,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
            num_beams=self.config.num_beams,
            use_cache=self.config.use_cache,
        )
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


__all__ = [
    "Generator",
    "GenerationConfig",
    "DecodingStrategy",
    "GreedySearch",
    "Sampling",
    "BeamSearch",
    "ContrastiveSearch",
    "BaseDecoder",
]