"""
Local Inference Engine.

Runs inference locally on available GPU/CPU.

Usage:
    python -m src.runtime.local_inference --model checkpoints/expera-350m --prompt "Hello"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class LocalConfig:
    """Local inference configuration."""

    model_path: Path = Path("checkpoints/expera-350m")
    device: str = "cuda"  # cuda, cpu, mps
    dtype: str = "bf16"  # bf16, fp16, fp32

    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1

    use_cache: bool = True
    stream: bool = True


@dataclass
class InferenceResult:
    """Result from inference."""

    text: str
    tokens: int
    latency_ms: float
    finish_reason: str  # stop, length, timeout


class LocalInferenceEngine:
    """Local inference engine using transformers."""

    def __init__(self, config: LocalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    async def load(self) -> None:
        """Load model and tokenizer."""
        if self._loaded:
            return

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        dtype = torch.float16
        if self.config.dtype == "bf16":
            dtype = torch.bfloat16

        logger.info(f"Loading model from {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        self._loaded = True
        logger.info(f"Model loaded on {device}")

    async def unload(self) -> None:
        """Unload model to free memory."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> InferenceResult:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop: Stop sequences

        Returns:
            InferenceResult with generated text
        """
        await self.load()

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k

        start_time = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=self.config.use_cache,
            )

        generated = outputs[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in text:
                    text = text.split(stop_seq)[0]

        latency_ms = (time.perf_counter() - start_time) * 1000
        finish_reason = "stop" if len(generated) < max_new_tokens else "length"

        return InferenceResult(
            text=text,
            tokens=len(generated),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        callback=None,
    ) -> InferenceResult:
        """
        Generate with streaming.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            callback: Called for each new token

        Returns:
            InferenceResult with full text
        """
        await self.load()

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        start_time = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        generated_ids = input_ids.clone()
        all_text = prompt

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k
                if self.config.top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, self.config.top_k)[0][:, -1:]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus)
                if self.config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                new_text = self.tokenizer.decode(
                    next_token[0], skip_special_tokens=True
                )
                all_text += new_text

                if callback:
                    await callback(new_text)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            text=all_text[len(prompt) :],
            tokens=generated_ids.shape[1] - input_ids.shape[1],
            latency_ms=latency_ms,
            finish_reason="stop",
        )

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> InferenceResult:
        """Generate response in chat format."""
        prompt = self._format_chat(messages)
        return await self.generate(prompt, **kwargs)

    def _format_chat(self, messages: list[dict]) -> str:
        """Format messages as prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)


async def main():
    parser = argparse.ArgumentParser(description="Local inference engine")
    parser.add_argument("--model", default="checkpoints/expera-350m")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    args = parser.parse_args()

    config = LocalConfig(
        model_path=Path(args.model),
        device=args.device,
    )

    engine = LocalInferenceEngine(config)

    if args.stream:
        collected = []
        async def on_token(token: str):
            collected.append(token)
            print(token, end="", flush=True)

        result = await engine.stream_generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            callback=on_token,
        )
        print()
    else:
        result = await engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result.text)

    logger.info(
        f"Generated {result.tokens} tokens in {result.latency_ms:.0f}ms "
        f"({result.tokens / result.latency_ms * 1000:.1f} tok/s)"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())