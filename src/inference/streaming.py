"""
Streaming Generation for Expera AI.

Provides streaming text generation with callbacks.
"""

from typing import Optional, Callable, AsyncIterator, Union, List, Awaitable, Any
from dataclasses import dataclass
import asyncio
import logging

import torch
import torch.nn as nn
from torch import Tensor

from .generator import Generator, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Streaming text chunk."""
    text: str
    is_final: bool = False
    token_id: Optional[int] = None


# Type alias for callbacks
StreamCallback = Callable[[StreamChunk], None]
AsyncStreamCallback = Callable[[StreamChunk], Awaitable[None]]


class StreamingGenerator:
    """
    Streaming text generator.

    Usage:
        async def on_chunk(chunk):
            print(chunk.text, end="", flush=True)

        generator = StreamingGenerator(model, tokenizer)
        async for chunk in generator.stream("Hello", callback=on_chunk):
            pass
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """Initialize streaming generator."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.generator = Generator(model, tokenizer, self.config)

    def stream(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        callback: Optional[StreamCallback] = None,
        chunk_size: int = 1,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream generated text.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            callback: Callback for each chunk
            chunk_size: Number of tokens per chunk
            **kwargs: Generation config

        Yields:
            StreamChunk objects
        """
        # Sync wrapper for simplicity
        # In production, use async generator
        import queue
        import threading

        q = queue.Queue()
        done = threading.Event()

        def generate_in_thread():
            try:
                output = self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )

                # Parse output into chunks
                if isinstance(output, str):
                    # Yield in chunks
                    for i in range(0, len(output), chunk_size):
                        chunk_text = output[i:i + chunk_size]
                        q.put(StreamChunk(text=chunk_text, is_final=False))

                q.put(None)  # Signal done

            except Exception as e:
                q.put(e)

        # Start generation thread
        thread = threading.Thread(target=generate_in_thread)
        thread.start()

        # Yield chunks
        while True:
            item = q.get()
            if item is None:
                break

            if isinstance(item, Exception):
                raise item

            if callback:
                callback(item)

            if item.is_final:
                break

            yield item

        thread.join()

    def stream_sync(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Synchronous streaming (blocking).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens
            on_token: Callback for each token
            **kwargs: Generation config

        Returns:
            Full generated text
        """
        # Simple streaming generation
        config = self.config
        if max_new_tokens:
            config.max_new_tokens = max_new_tokens

        # Generator
        full_text = ""

        # For now, just generate and return
        # Real streaming needs model changes
        output = self.generator.generate(prompt, **kwargs)

        if on_token:
            # Callback for each character (approximate)
            for char in str(output):
                on_token(char)
                full_text += char
        else:
            full_text = str(output)

        return full_text


__all__ = [
    "StreamingGenerator",
    "StreamCallback",
    "StreamChunk",
]