"""
Remote Runtime for Expera AI.

Provides remote VPS/cloud inference:
- Connect to remote endpoints
- Stream responses
- Handle authentication
- Automatic retry

Usage:
    runtime = RemoteRuntime(endpoint="https://api.experaa.com")
    async for chunk in runtime.generate(prompt):
        print(chunk, end="")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class RemoteConfig:
    """Remote runtime configuration."""
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None
    model: str = "expera-coder-120m"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    stream: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class RemoteResponse:
    """Response from remote endpoint."""
    text: str
    finish_reason: str
    usage: dict
    model: str
    latency_ms: float


class RemoteRuntime:
    """
    Remote inference runtime.

    Supports:
    - REST API
    - Streaming
    - Authentication
    - Automatic retry
    """

    def __init__(self, config: Optional[RemoteConfig] = None):
        """Initialize remote runtime."""
        self.config = config or RemoteConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._connected = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
        return self.session

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate text from remote endpoint."""
        session = await self._get_session()

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                async with session.post(
                    f"{self.config.endpoint}/v1/generate",
                    json=payload,
                ) as response:
                    if response.status == 200:
                        if stream:
                            async for line in response.content:
                                if line:
                                    try:
                                        data = json.loads(line)
                                        if "text" in data:
                                            yield data["text"]
                                    except json.JSONDecodeError:
                                        continue
                        else:
                            data = await response.json()
                            yield data.get("text", "")
                        return

                    elif response.status == 429:
                        # Rate limited
                        wait = self.config.retry_delay * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)

                    else:
                        error = await response.text()
                        logger.error(f"API error {response.status}: {error}")
                        last_error = error

            except aiohttp.ClientError as e:
                logger.warning(f"Connection error (attempt {attempt+1}): {e}")
                last_error = str(e)
                await asyncio.sleep(self.config.retry_delay)

            except asyncio.TimeoutError:
                logger.warning(f"Timeout (attempt {attempt+1})")
                last_error = "Timeout"
                await asyncio.sleep(self.config.retry_delay)

        # All retries failed
        yield f"[Error: Failed after {self.config.max_retries} attempts - {last_error}]"

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Chat with remote endpoint using OpenAI-compatible format."""
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

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if remote endpoint is healthy."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.endpoint}/health") as response:
                return response.status == 200
        except Exception:
            return False

    async def __aenter__(self) -> "RemoteRuntime":
        """Async context manager."""
        await self._get_session()
        return self

    async def __aexit__(self, *args) -> None:
        """Exit context."""
        await self.close()


def get_runtime(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RemoteRuntime:
    """Get default remote runtime."""
    config = RemoteConfig(
        endpoint=endpoint or "http://localhost:8000",
        api_key=api_key,
    )
    return RemoteRuntime(config)


# Export
__all__ = [
    "RemoteRuntime",
    "RemoteConfig",
    "RemoteResponse",
    "get_runtime",
]