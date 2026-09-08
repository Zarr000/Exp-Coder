"""
Hybrid Router.

Routes requests to local or remote inference based on:
- Request size
- Latency requirements
- Availability

Usage:
    python -m src.runtime.hybrid_router --prompt "Hello"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.runtime.local_inference import LocalConfig, LocalInferenceEngine
from src.runtime.remote_inference import RemoteConfig, RemoteInferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Hybrid inference configuration."""

    # Local settings
    local_model_path: Path = Path("checkpoints/expera-350m")
    local_device: str = "cuda"
    local_dtype: str = "bf16"

    # Remote settings
    remote_url: str = "http://localhost:8000"
    remote_api_key: str = ""
    remote_model: str = "expera-350m"

    # Routing preferences
    prefer_local: bool = True
    local_max_tokens: int = 512
    remote_max_tokens: int = 2048
    local_latency_threshold_ms: float = 5000

    # Fallback settings
    fallback_to_remote: bool = True
    fallback_to_cpu: bool = True


@dataclass
class RouteDecision:
    """Routing decision."""

    target: str  # local, remote
    reason: str


class HybridRouter:
    """Routes inference between local and remote engines."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.local_engine = LocalInferenceEngine(
            LocalConfig(
                model_path=config.local_model_path,
                device=config.local_device,
                dtype=config.local_dtype,
            )
        )
        self.remote_engine = None
        self._remote_available = None
        self._last_check = 0.0

    async def _check_remote(self) -> bool:
        """Check if remote is available."""
        import time

        now = time.perf_counter()
        if now - self._last_check < 10:  # Cache for 10s
            return self._remote_available

        self._last_check = now
        if not self.remote_engine:
            self.remote_engine = RemoteInferenceEngine(
                RemoteConfig(
                    api_url=self.config.remote_url,
                    api_key=self.config.remote_api_key,
                    model=self.config.remote_model,
                )
            )
            await self.remote_engine.__aenter__()

        try:
            self._remote_available = await self.remote_engine.health_check()
        except Exception:
            self._remote_available = False

        return self._remote_available

    def _decide_route(self, prompt: str, max_tokens: int) -> RouteDecision:
        """Decide which engine to use."""
        # Check token limits
        if max_tokens > self.config.local_max_tokens and self.config.prefer_local:
            return RouteDecision(
                target="remote",
                reason=f"exceeds local limit ({max_tokens} > {self.config.local_max_tokens})",
            )

        # Check prompt length
        if len(prompt) > 4000 and self.config.prefer_local:
            return RouteDecision(
                target="remote",
                reason=f"long prompt ({len(prompt)} chars)",
            )

        # Use preference
        if self.config.prefer_local:
            return RouteDecision(target="local", reason="preferred")

        return RouteDecision(target="remote", reason="default")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """
        Generate text using hybrid inference.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Dict with text, tokens, latency_ms, source, finish_reason
        """
        max_tokens = max_tokens or self.config.local_max_tokens
        decision = self._decide_route(prompt, max_tokens)

        logger.info(f"Routing to {decision.target}: {decision.reason}")

        try:
            if decision.target == "local":
                result = await self.local_engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return {
                    "text": result.text,
                    "tokens": result.tokens,
                    "latency_ms": result.latency_ms,
                    "source": "local",
                    "finish_reason": result.finish_reason,
                }
            else:
                result = await self.remote_engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return {
                    "text": result["text"],
                    "tokens": result["tokens"],
                    "latency_ms": result["latency_ms"],
                    "source": "remote",
                    "finish_reason": result["finish_reason"],
                }

        except Exception as e:
            logger.error(f"{decision.target} failed: {e}")

            # Fallback
            if decision.target == "local" and self.config.fallback_to_remote:
                logger.info("Falling back to remote")
                result = await self.remote_engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return {
                    "text": result["text"],
                    "tokens": result["tokens"],
                    "latency_ms": result["latency_ms"],
                    "source": "remote",
                    "finish_reason": result["finish_reason"],
                }

            raise

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        callback=None,
    ) -> dict:
        """Generate with streaming."""
        max_tokens = max_tokens or self.config.local_max_tokens
        decision = self._decide_route(prompt, max_tokens)

        if decision.target == "local":
            result = await self.local_engine.stream_generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                callback=callback,
            )
            return {
                "text": result.text,
                "tokens": result.tokens,
                "latency_ms": result.latency_ms,
                "source": "local",
                "finish_reason": result.finish_reason,
            }
        else:
            result = await self.remote_engine.stream_generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                callback=callback,
            )
            return {
                "text": result["text"],
                "tokens": result["tokens"],
                "latency_ms": result["latency_ms"],
                "source": "remote",
                "finish_reason": result["finish_reason"],
            }

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """Chat using hybrid inference."""
        # Convert messages to prompt
        prompt = self._format_chat(messages)
        max_tokens = kwargs.get("max_tokens", self.config.local_max_tokens)

        result = await self.generate(prompt, max_tokens=max_tokens, **kwargs)
        return result

    def _format_chat(self, messages: list[dict]) -> str:
        """Format chat messages as prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    async def close(self) -> None:
        """Close engines."""
        if self.remote_engine:
            await self.remote_engine.__aexit__()


async def main():
    parser = argparse.ArgumentParser(description="Hybrid router")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prefer-local", action="store_true", default=True)
    parser.add_argument("--prefer-remote", action="store_false", dest="prefer_local")
    args = parser.parse_args()

    config = HybridConfig(prefer_local=args.prefer_local)

    router = HybridRouter(config)

    try:
        result = await router.generate(
            args.prompt,
            max_tokens=args.max_tokens,
        )
        print(result["text"])
        logger.info(f"Source: {result['source']}, Latency: {result['latency_ms']:.0f}ms")
    finally:
        await router.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())