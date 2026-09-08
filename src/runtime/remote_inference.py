"""
Remote Inference Engine.

Runs inference on remote VPS/server API.

Usage:
    python -m src.runtime.remote_inference --url http://localhost:8000 --prompt "Hello"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class RemoteConfig:
    """Remote inference configuration."""

    api_url: str = "http://localhost:8000"
    api_key: str = ""

    model: str = "expera-350m"

    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1

    stream: bool = True
    timeout: int = 120


class RemoteInferenceEngine:
    """Remote inference engine via HTTP API."""

    def __init__(self, config: RemoteConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {},
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        )
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if remote API is available."""
        try:
            async with self._session.get(
                f"{self.config.api_url}/health"
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> dict:
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
            Dict with text, tokens, latency_ms, finish_reason
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or [],
            "stream": False,
        }

        start_time = time.perf_counter()

        async with self._session.post(
            f"{self.config.api_url}/v1/completions", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"API error: {resp.status} - {text}")

            result = await resp.json()

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text": result.get("text", ""),
            "tokens": result.get("tokens", 0),
            "latency_ms": latency_ms,
            "finish_reason": result.get("finish_reason", "stop"),
        }

    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        callback=None,
    ) -> dict:
        """
        Generate with streaming.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            callback: Called for each new token chunk

        Returns:
            Dict with full generated text
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start_time = time.perf_counter()
        all_text = ""

        async with self._session.post(
            f"{self.config.api_url}/v1/completions",
            json=payload,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"API error: {resp.status} - {text}")

            async for line in resp.content:
                if not line:
                    continue

                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue

                if line == "data: [DONE]":
                    break

                try:
                    data = line[5:].strip()
                    chunk = json.loads(data)
                    token = chunk.get("text", "")
                    all_text += token

                    if callback:
                        await callback(token)
                except json.JSONDecodeError:
                    continue

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text": all_text,
            "tokens": len(all_text.split()),
            "latency_ms": latency_ms,
            "finish_reason": "stop",
        }

    async def chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """Generate response in chat format (OpenAI-compatible)."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": kwargs.get("stream", False),
        }

        start_time = time.perf_counter()

        async with self._session.post(
            f"{self.config.api_url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"API error: {resp.status} - {text}")

            result = await resp.json()

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "tokens": result.get("usage", {}).get("completion_tokens", 0),
            "latency_ms": latency_ms,
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason", "stop"),
        }

    async def embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings for texts."""
        payload = {
            "model": f"{self.config.model}-embedding",
            "input": texts,
        }

        async with self._session.post(
            f"{self.config.api_url}/v1/embeddings", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"API error: {resp.status} - {text}")

            result = await resp.json()
            return [item["embedding"] for item in result.get("data", [])]


import json  # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description="Remote inference engine")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="expera-350m")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    args = parser.parse_args()

    config = RemoteConfig(api_url=args.url, model=args.model)

    async with RemoteInferenceEngine(config) as engine:
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
            print(result["text"])

        logger.info(
            f"Generated {result['tokens']} tokens in {result['latency_ms']:.0f}ms"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())