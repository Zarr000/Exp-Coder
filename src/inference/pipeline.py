"""
End-to-End Inference Pipeline for Expera AI.

Provides high-level inference API:
- Model loading and initialization
- Text generation
- Batch processing
- Streaming support
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Callable, AsyncIterator
import logging

import torch
import torch.nn as nn
from torch import Tensor

from .generator import Generator, GenerationConfig, DecodingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextGenerationOutput:
    """Text generation output."""
    text: Union[str, List[str]]
    tokens: Optional[Tensor] = None
    prompt: Optional[Union[str, List[str]]] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[Union[str, List[str]]] = None

    def __str__(self) -> str:
        return self.text


class InferencePipeline:
    """
    End-to-end inference pipeline.

    Usage:
        pipeline = InferencePipeline(model_path="checkpoints/final/")
        output = pipeline.generate("Hello, how are you?")
        print(output.text)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize pipeline.

        Args:
            model: Model instance (or load from model_path)
            tokenizer: Tokenizer instance
            model_path: Path to checkpoint
            device: Device to use ("cuda", "cpu", or None for auto)
            config: Generation config
            dtype: Model dtype (torch.float16, torch.bfloat16, etc.)
        """
        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        if model is None and model_path:
            model = self._load_model(model_path, dtype)
        elif model is None:
            logger.warning("No model provided, using dummy")
            from src.model.architecture import ExperaModel
            model = ExperaModel()

        self.model = model.to(self.device)
        self.model.eval()

        # Load tokenizer if model_path provided and no tokenizer passed
        if tokenizer is None and model_path:
            tokenizer = self._load_tokenizer(model_path)

        self.tokenizer = tokenizer

        # Generation config
        self.config = config or GenerationConfig()

        # Generator
        self.generator = Generator(self.model, tokenizer, self.config)

        logger.info(f"Pipeline initialized on {self.device}")

    def _load_model(self, model_path: str, dtype: Optional[torch.dtype] = None) -> nn.Module:
        """Load model from checkpoint."""
        import json
        from pathlib import Path
        from src.model.architecture import ExperaModel

        model_path = Path(model_path)

        # Load config.json
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        # Determine checkpoint file path
        if model_path.is_file() and model_path.suffix in ['.pt', '.pth', '.bin']:
            ckpt_path = model_path
        else:
            # Try common checkpoint filenames
            for name in ['model.pt', 'pytorch_model.bin', 'model.bin', 'consolidated.00.pth']:
                ckpt_path = model_path / name
                if ckpt_path.exists():
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Create model with config values
        model = ExperaModel(
            vocab_size=config.get("vocab_size", 50304),
            hidden_size=config.get("hidden_size", 768),
            num_layers=config.get("num_hidden_layers", 12),
            num_heads=config.get("num_attention_heads", 8),
            num_kv_heads=config.get("num_key_value_heads", 4),
            max_position_embeddings=config.get("max_position_embeddings", 2048),
        )

        model.load_state_dict(state_dict, strict=False)

        if dtype:
            model = model.to(dtype)

        logger.info(f"Loaded model from {ckpt_path}")
        return model

    def _load_tokenizer(self, model_path: str) -> Any:
        """Load a tokenizer for the checkpoint directory.

        Prefers the canonical Exp-Coder byte-level BPE format
        (``vocab.json`` / ``merges.txt`` / ``config.json``), falling back to
        a legacy SentencePiece ``tokenizer.model``.
        """
        from pathlib import Path
        from src.tokenizer import BPETokenizer

        model_path = Path(model_path)

        for candidate in (model_path, model_path / "tokenizer"):
            if (candidate / "config.json").exists() and (candidate / "vocab.json").exists():
                try:
                    return BPETokenizer.load(str(candidate))
                except Exception as exc:
                    logger.warning(
                        "BPETokenizer load failed for %s (%s); trying SentencePiece",
                        candidate, exc,
                    )

        import sentencepiece as spm
        for subdir in ("", "tokenizer"):
            tokenizer_path = model_path / subdir / "tokenizer.model"
            if tokenizer_path.exists():
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.load(str(tokenizer_path))
                logger.info("Loaded SentencePiece tokenizer from %s", tokenizer_path)
                return tokenizer

        logger.warning(
            "No tokenizer found in %s (expected BPETokenizer vocab.json or "
            "SentencePiece tokenizer.model); using fallback", model_path,
        )
        return None

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: bool = False,
        **kwargs
    ) -> TextGenerationOutput:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (>0 enables sampling)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            do_sample: Enable sampling
            **kwargs: Additional generation config

        Returns:
            TextGenerationOutput with generated text
        """
        # Config overrides
        gen_config = self.config
        if do_sample or temperature is not None and temperature > 0:
            gen_config.strategy = DecodingStrategy.SAMPLING
        elif do_sample:
            gen_config.strategy = DecodingStrategy.GREEDY

        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None and temperature > 0:
            gen_config.temperature = temperature
        if top_p is not None:
            gen_config.top_p = top_p
        if top_k is not None:
            gen_config.top_k = top_k
        if repetition_penalty is not None:
            gen_config.repeat_penalty = repetition_penalty

        # Generate
        text = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )

        # Handle empty/bad responses - provide fallback
        import random
        _is_bad = False

        if not text or (isinstance(text, str) and len(text.strip()) == 0):
            _is_bad = True
        elif isinstance(text, str):
            # Check for repetitive/unknown chars - common with undertrained models
            stripped = text.strip()
            if stripped:
                # Check ratio of unique characters
                unique_ratio = len(set(stripped)) / max(len(stripped), 1)
                # Check for repetitive content
                is_repetitive = len(set(stripped)) <= 2 or stripped.count(stripped[0]) > len(stripped) * 0.7
                # Check for unknown char (question mark indicates unknown)
                has_unknown = '⁄' in stripped or '?' in stripped or '▯' in stripped
                if is_repetitive or has_unknown or unique_ratio < 0.3:
                    _is_bad = True

        if _is_bad:
            # For repetitive output, keep it as-is (model is working but collapsed to single token)
            # Just clean high Unicode artifacts but preserve the content
            import re
            text_filtered = text
            for ch in ['\u2047', '\u2581', '\u2582', '\u2583']:  # ⁏ ░ ▒ ▓
                text_filtered = text_filtered.replace(ch, '')
            text_filtered = text_filtered.replace('\u200b', '')  # ZWSP
            text_filtered = re.sub(r'[\ufffe-\uffff]', '', text_filtered)
            text_filtered = text_filtered.strip()
            # Always preserve output if it has any content
            if text_filtered:
                text = text_filtered
                logger.info(f"Cleaned output: {repr(text)[:50]}")
            else:
                text = text

        # Usage info
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return TextGenerationOutput(
            text=text,
            prompt=prompt,
            usage=usage,
            finish_reason="stop" if isinstance(text, str) else ["stop"] * len(text),
        )

    def stream(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generated text.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            callback: Callback for each token/batch
            **kwargs: Generation config

        Yields:
            Generated text chunks
        """
        raise NotImplementedError("Streaming not implemented yet")
        # yield text_chunk

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        batch_size: int = 8,
        **kwargs
    ) -> List[TextGenerationOutput]:
        """
        Batch generate.

        Args:
            prompts: List of prompts
            max_new_tokens: Maximum tokens per prompt
            batch_size: Batch size
            **kwargs: Generation config

        Returns:
            List of outputs
        """
        outputs = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            output = self.generate(batch, max_new_tokens=max_new_tokens, **kwargs)
            outputs.append(output)

        return outputs

    def get_num_params(self) -> Dict[str, int]:
        """Get model parameter count."""
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "total": num_params,
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }


__all__ = [
    "InferencePipeline",
    "TextGenerationOutput",
]