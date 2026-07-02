"""
Text Generation for Expera AI.

Supports various decoding strategies:
- Greedy search
- Sampling (temperature, top-k, top-p)
- Beam search
- Contrastive search
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
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

    # Contrastive
    penalty_alpha: float = 0.6
    top_k: int = 4

    # Output
    echo: bool = False
    stop_strings: Optional[List[str]] = None
    seed: Optional[int] = None

    # Misc
    use_cache: bool = True


class BaseDecoder(ABC):
    """Base decoder class."""

    @abstractmethod
    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: GenerationConfig = None,
    ) -> Tensor:
        """Decode next tokens."""
        pass


class GreedySearch(BaseDecoder):
    """Greedy search decoder."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: GenerationConfig = None,
    ) -> Tensor:
        """Greedy decode."""
        config = config or GenerationConfig()
        model.eval()

        max_new_tokens = config.max_new_tokens
        max_length = config.max_length
        use_cache = config.use_cache

        # Setup
        device = input_ids.device
        batch_size = input_ids.shape[0]
        input_len = input_ids.shape[1]
        generated_tokens = 0

        # Initialize
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Generation loop
        with torch.no_grad():
            while generated_tokens < max_new_tokens and generated_tokens < max_length:
                # Forward
                if use_cache and cur_len > input_ids.shape[1]:
                    # Use KV cache for incremental decoding
                    outputs = model(
                        input_ids[:, -1:],
                        attention_mask=attention_mask[:, -1:],
                        use_cache=True,
                    )
                    logits = outputs["logits"]
                    next_token_logits = logits[:, -1, :]
                else:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                    )
                    logits = outputs["logits"]
                    next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                if config.repeat_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, config.repeat_penalty
                    )

                # Greedy selection
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Append
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
                generated_tokens += 1

                # Check for EOS
                if (next_token == self.eos_token_id).all():
                    break

        return input_ids

    def _apply_repetition_penalty(
        self, logits: Tensor, input_ids: Tensor, penalty: float
    ) -> Tensor:
        """Apply repetition penalty."""
        score = torch.gather(logits, -1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(-1, input_ids, score)
        return logits


class Sampling(BaseDecoder):
    """Sampling decoder with temperature, top-k, and top-p."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: GenerationConfig = None,
    ) -> Tensor:
        """Sample decode."""
        config = config or GenerationConfig()
        model.eval()

        max_new_tokens = config.max_new_tokens
        max_length = config.max_length
        use_cache = config.use_cache

        # RNG
        if config.seed is not None:
            torch.manual_seed(config.seed)

        device = input_ids.device
        batch_size = input_ids.shape[0]
        input_len = input_ids.shape[1]
        generated_tokens = 0

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Generation loop
        with torch.no_grad():
            while generated_tokens < max_new_tokens and generated_tokens < max_length:
                # Forward
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                )
                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                if config.repeat_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, config.repeat_penalty
                    )

                # Apply temperature
                if config.temperature != 1.0:
                    next_token_logits = next_token_logits / config.temperature

                # Apply top-k filtering
                if config.top_k > 0:
                    next_token_logits = self._apply_top_k(next_token_logits, config.top_k)

                # Apply top-p (nucleus) filtering
                if config.top_p < 1.0:
                    next_token_logits = self._apply_top_p(next_token_logits, config.top_p)

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
                generated_tokens += 1

                # Check for EOS
                if (next_token == self.eos_token_id).all():
                    break

        return input_ids

    def _apply_repetition_penalty(
        self, logits: Tensor, input_ids: Tensor, penalty: float
    ) -> Tensor:
        """Apply repetition penalty."""
        score = torch.gather(logits, -1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(-1, input_ids, score)
        return logits

    def _apply_top_k(self, logits: Tensor, k: int) -> Tensor:
        """Apply top-k filtering."""
        # Get top k indices - this has same shape as logits
        top_k_indices = torch.topk(logits, k, dim=-1).indices
        # Create a mask of -inf
        mask = torch.full_like(logits, float("-inf"))
        # Scatter 0.0 at top-k positions (keeps them)
        mask.scatter_(-1, top_k_indices, 0.0)
        return logits + mask

    def _apply_top_p(self, logits: Tensor, p: float) -> Tensor:
        """Apply nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask tokens above p
        mask = cumulative_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        # Scatter back
        mask = mask.scatter(-1, sorted_indices, mask)
        return logits.masked_fill(mask, float("-inf"))


class BeamSearch(BaseDecoder):
    """Beam search decoder."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: GenerationConfig = None,
    ) -> Tensor:
        """Beam search decode."""
        config = config or GenerationConfig()
        model.eval()

        num_beams = config.num_beams
        max_new_tokens = config.max_new_tokens
        length_penalty = config.length_penalty

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Expand input for beams
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        # Beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = float("-inf")
        beam_scores = beam_scores.view(-1)

        # Generation
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                if config.repeat_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, config.repeat_penalty
                    )

                # Compute log probs
                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Add beam scores
                log_probs = log_probs + beam_scores.unsqueeze(-1)

                # Flatten for top-k
                log_probs = log_probs.view(batch_size, -1)

                # Select top beams
                top_scores, next_tokens = torch.topk(
                    log_probs, num_beams, dim=-1, largest=True, sorted=True
                )

                # Update beam scores
                beam_scores = top_scores.view(-1)

                # Update input IDs
                next_tokens = next_tokens.view(-1, 1)
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)

                # Check for EOS
                if (next_tokens == self.eos_token_id).any():
                    break

        # Select best beam
        best_beam = beam_scores.view(batch_size, num_beams).argmax(dim=-1)
        return input_ids.view(batch_size, num_beams, -1)[
            torch.arange(batch_size, device=device), best_beam
        ]

    def _apply_repetition_penalty(
        self, logits: Tensor, input_ids: Tensor, penalty: float
    ) -> Tensor:
        """Apply repetition penalty."""
        score = torch.gather(logits, -1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(-1, input_ids, score)
        return logits


class ContrastiveSearch(BaseDecoder):
    """Contrastive search decoder."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        config: GenerationConfig = None,
    ) -> Tensor:
        """Contrastive decode."""
        config = config or GenerationConfig()
        model.eval()

        max_new_tokens = config.max_new_tokens
        penalty_alpha = config.penalty_alpha
        top_k = config.top_k

        device = input_ids.device
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Generation loop
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                if config.repeat_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, config.repeat_penalty
                    )
                    # Mask previously seen tokens
                    for tok in input_ids[0]:
                        next_token_logits[0, tok] = float("-inf")

                # Top-k selection with penalty
                scores = next_token_logits.clone()
                for idx in range(batch_size):
                    top_k_vals, top_k_idx = torch.topk(scores[idx], top_k)
                    for i, tok in enumerate(input_ids[idx]):
                        if tok in top_k_idx:
                            scores[idx, tok] *= penalty_alpha
                        else:
                            scores[idx, tok] = float("-inf")

                # Sample from top-k
                probs = F.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
                generated_tokens += 1

                # Check for EOS
                if (next_token == self.eos_token_id).all():
                    break

        return input_ids

    def _apply_repetition_penalty(
        self, logits: Tensor, input_ids: Tensor, penalty: float
    ) -> Tensor:
        """Apply repetition penalty."""
        score = torch.gather(logits, -1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(-1, input_ids, score)
        return logits


def is_valid_generation_token(token_id: int, tokenizer: Any) -> bool:
    """
    Check if a token is valid for generation output.

    Filters:
    - Empty strings after decode
    - Control tokens
    - Invalid special IDs

    Returns:
        True if token should be kept in output
    """
    if tokenizer is None:
        return True

    # Check special IDs
    eos_id = tokenizer.eos_id()
    bos_id = tokenizer.bos_id()
    pad_id = tokenizer.pad_id()
    unk_id = tokenizer.unk_id()

    # Skip EOS immediately
    if token_id == eos_id:
        return False

    # Skip padding
    if pad_id >= 0 and token_id == pad_id:
        return False

    # Check piece decode
    piece = tokenizer.id_to_piece(token_id)

    # Skip unknown token
    if token_id == unk_id and unk_id != 0:  # ID 0 might be real
        return False

    # Skip control tokens (start with < and end with >)
    if piece.startswith("<") and piece.endswith(">"):
        # Allow known good tokens
        if piece in ["<unk>", "<s>", "</s>"]:
            return True
        return False

    # Decode and check if empty
    try:
        decoded = tokenizer.decode_ids([token_id])
        if not decoded or not decoded.strip():
            return False
    except:
        return False

    return True


class Generator:
    """Text generator with multiple decoding strategies."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        config: GenerationConfig = None,
        debug: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.debug = debug

        # Select decoder
        eos_token_id = tokenizer.eos_id() if tokenizer else 2

        if self.config.strategy == DecodingStrategy.GREEDY:
            self.decoder = GreedySearch(eos_token_id)
        elif self.config.strategy == DecodingStrategy.SAMPLING:
            self.decoder = Sampling(eos_token_id)
        elif self.config.strategy == DecodingStrategy.BEAM:
            self.decoder = BeamSearch(eos_token_id)
        elif self.config.strategy == DecodingStrategy.CONTRASTIVE:
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
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt(s)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            **kwargs: Additional config overrides

        Returns:
            Generated text(s)
        """
        # Tokenize
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        input_ids = self._tokenize(prompts)

        # Override config
        config = self._copy_config(**kwargs)
        if max_new_tokens is not None:
            config.max_new_tokens = max_new_tokens
        if temperature is not None:
            config.temperature = temperature
        if top_p is not None:
            config.top_p = top_p
        if top_k is not None:
            config.top_k = top_k

        # Generate
        output_ids = self.decoder.decode(
            self.model, input_ids, config=config
        )

        # Debug logging
        if self.debug:
            import time
            start_time = time.time()
            logger.debug(f"DEBUG generate:")
            logger.debug(f"  prompt: {repr(prompt)[:100]}")
            logger.debug(f"  input_ids: {input_ids.tolist()}")
            logger.debug(f"  generated_ids: {output_ids.tolist()}")
            logger.debug(f"  first predicted token: {output_ids[0, input_ids.shape[1]].item()}")
            logger.debug(f"  generation time: {time.time() - start_time:.4f}s")

        # Decode
        return self._decode(output_ids, input_ids)

    def _tokenize(self, prompts: List[str]) -> Tensor:
        """Tokenize prompts."""
        if self.tokenizer:
            # SentencePiece encode returns a list of ids
            # Handle both single string and list of strings
            if len(prompts) == 1:
                ids = self.tokenizer.encode(prompts[0])
                input_ids = torch.tensor([ids], dtype=torch.long)
            else:
                # Batch processing - encode each prompt
                all_ids = []
                for prompt in prompts:
                    ids = self.tokenizer.encode(prompt)
                    all_ids.append(ids)
                input_ids = torch.tensor(all_ids, dtype=torch.long)
        else:
            # Dummy tokenization
            input_ids = torch.randint(0, 1000, (len(prompts), 10))

        return input_ids

    def _decode(self, output_ids: Tensor, input_ids: Tensor) -> Union[str, List[str]]:
        """Decode output IDs."""
        if self.tokenizer:
            # Convert to numpy and handle different tensor shapes
            output_ids = output_ids.cpu()
            input_len = input_ids.shape[1]
            texts = []

            # Handle both 2D and 1D tensors
            if output_ids.dim() == 2:
                for ids in output_ids:
                    # Remove input tokens (only decode newly generated)
                    gen_ids = ids[input_len:].tolist()
                    text = self.tokenizer.decode(gen_ids)
                    texts.append(text)
            else:
                gen_ids = output_ids[input_len:].tolist()
                text = self.tokenizer.decode(gen_ids)
                texts.append(text)

            # Check for empty or special-token-only responses
            result = texts[0] if len(texts) == 1 else texts
            return self._filter_response(result)
        else:
            return "Generated text (no tokenizer)"

    def _filter_response(self, text: str) -> str:
        """Filter and clean response text."""
        if not text:
            return ""

        # Check for empty after strip
        stripped = text.strip()
        if not stripped:
            return ""

        # For Phase 12: Allow repetitive output from undertrained models
        # The model IS generating text - don't filter it out completely
        # Only filter completely empty or all-whitespace
        if len(stripped) == 0:
            return ""

        # Filter out control characters but keep basic punctuation
        import re
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        result = cleaned.strip()
        # Only return empty if literally nothing left
        return result if result else ""

    def _copy_config(self, **kwargs) -> GenerationConfig:
        """Copy config with overrides."""
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