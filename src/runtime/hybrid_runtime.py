"""
Hybrid Runtime for Expera AI.

Combines local and remote runtimes:
- Local for fast, private inference
- Remote for heavy loads / unavailable local resources
- Automatic load balancing
- Failover handling

Usage:
    runtime = HybridRuntime()
    async for chunk in runtime.generate(prompt):
        print(chunk, end="")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from .local_runtime import LocalRuntime, GPUInfo
from .remote_runtime import RemoteRuntime, RemoteConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSelector:
    """Configurable runtime selection logic."""

    prefer_local: bool = True
    local_threshold: int = 2048  # Max tokens for local
    remote_threshold: int = 8192  # Max tokens for remote
    min_gpu_memory_mb: int = 512  # Minimum GPU memory

    def choose_runtime(
        self,
        gpu_info: Optional[GPUInfo],
        prompt_length: int,
    ) -> str:
        """Choose runtime: 'local' or 'remote'."""
        # Check local capability
        if self.prefer_local and gpu_info and gpu_info.is_available:
            if gpu_info.memory_free >= self.min_gpu_memory_mb:
                if prompt_length <= self.local_threshold:
                    return "local"

        # Fall back to remote
        return "remote"

    def get_max_tokens(self, runtime: str) -> int:
        """Get max tokens for runtime."""
        if runtime == "local":
            return self.local_threshold
        return self.remote_threshold


@dataclass
class HybridConfig:
    """Hybrid runtime configuration."""

    local_enabled: bool = True
    remote_enabled: bool = True
    remote_endpoint: Optional[str] = None
    remote_api_key: Optional[str] = None
    fallback_to_remote: bool = True
    selector: RuntimeSelector = field(default_factory=RuntimeSelector)

    # Routing
    routes: dict[str, str] = field(default_factory=dict)  # task_type -> runtime


class HybridRuntime:
    """
    Hybrid local + remote runtime.

    Features:
    - Automatic runtime selection
    - Load balancing
    - Failover
    - Metrics
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        """Initialize hybrid runtime."""
        self.config = config or HybridConfig()
        self.local: Optional[LocalRuntime] = None
        self.remote: Optional[RemoteRuntime] = None
        self._stats = {
            "local_requests": 0,
            "remote_requests": 0,
            "fallbacks": 0,
        }

        # Initialize runtimes
        if self.config.local_enabled:
            self.local = LocalRuntime()
        if self.config.remote_enabled:
            remote_config = RemoteConfig(
                endpoint=self.config.remote_endpoint or "http://localhost:8000",
                api_key=self.config.remote_api_key,
            )
            self.remote = RemoteRuntime(remote_config)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
        runtime: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text with automatic runtime selection."""
        # Get GPU info
        gpu_info = None
        if self.local:
            gpu_info = self.local.detect_gpu()

        # Choose runtime if not specified
        if runtime is None:
            runtime = self.config.selector.choose_runtime(
                gpu_info,
                len(prompt.split()),
            )

        # Try primary runtime
        try:
            if runtime == "local" and self.local:
                logger.info("Using local runtime")
                self._stats["local_requests"] += 1

                async for chunk in self.local.generate(
                    prompt,
                    max_tokens=min(max_tokens, self.config.selector.local_threshold),
                    temperature=temperature,
                    stream=stream,
                ):
                    yield chunk
                return

            elif runtime == "remote" and self.remote:
                logger.info("Using remote runtime")
                self._stats["remote_requests"] += 1

                async for chunk in self.remote.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                ):
                    yield chunk
                return

        except Exception as e:
            logger.warning(f"Runtime {runtime} failed: {e}")

            # Fallback if enabled
            if self.config.fallback_to_remote:
                fallback = "remote" if runtime == "local" else "local"
                if fallback == "remote" and self.remote:
                    logger.info(f"Falling back to remote")
                    self._stats["fallbacks"] += 1
                    self._stats["remote_requests"] += 1

                    async for chunk in self.remote.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                    ):
                        yield chunk
                    return
                elif fallback == "local" and self.local:
                    logger.info(f"Falling back to local")
                    self._stats["fallbacks"] += 1
                    self._stats["local_requests"] += 1

                    async for chunk in self.local.generate(
                        prompt,
                        max_tokens=self.config.selector.local_threshold,
                        temperature=temperature,
                        stream=stream,
                    ):
                        yield chunk
                    return

        # All runtimes failed
        yield f"[Error: All runtimes failed]"

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Chat with automatic runtime selection."""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        async for chunk in self.generate(prompt, **kwargs):
            yield chunk

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI messages to prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)

    async def health_check(self) -> dict[str, bool]:
        """Check health of all runtimes."""
        results = {}

        if self.local:
            results["local"] = await self.local.health_check()

        if self.remote:
            results["remote"] = await self.remote.health_check()

        return results

    def get_stats(self) -> dict[str, int]:
        """Get runtime statistics."""
        return self._stats.copy()

    async def close(self) -> None:
        """Close all runtimes."""
        if self.local:
            await self.local.close()
        if self.remote:
            await self.remote.close()

    async def __aenter__(self) -> "HybridRuntime":
        """Async context manager."""
        return self

    async def __aexit__(self, *args) -> None:
        """Exit context."""
        await self.close()


def get_hybrid_runtime(
    remote_endpoint: Optional[str] = None,
    remote_api_key: Optional[str] = None,
) -> HybridRuntime:
    """Get default hybrid runtime."""
    config = HybridConfig(
        remote_endpoint=remote_endpoint,
        remote_api_key=remote_api_key,
    )
    return HybridRuntime(config)


# Export
__all__ = [
    "HybridRuntime",
    "HybridConfig",
    "RuntimeSelector",
    "get_hybrid_runtime",
]