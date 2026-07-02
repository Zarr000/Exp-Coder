"""
Memory Summarizer.

Summarizes conversation history and memory content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import tiktoken


@dataclass
class Summary:
    """Memory summary."""

    content: str
    token_count: int
    conversation_id: str


class MemorySummarizer:
    """
    Summarizes memory content for efficient retrieval.

    Uses tiktoken for accurate token counting.
    """

    def __init__(self, model: str = "cl100k_base", max_tokens: int = 2048) -> None:
        """Initialize summarizer."""
        self.encoder = tiktoken.get_encoding(model)
        self.max_tokens = max_tokens
        self._cache: dict[str, Summary] = {}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))

    def truncate_for_tokens(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to fit within token limit."""
        max_tokens = max_tokens or self.max_tokens
        tokens = self.encoder.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated = tokens[:max_tokens]
        return self.encoder.decode(truncated)

    def create_conversation_id(self, messages: list[dict[str, Any]]) -> str:
        """Create deterministic conversation ID."""
        content = "".join(m.get("content", "") for m in messages)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def summarize_messages(self, messages: list[dict[str, Any]], max_tokens: Optional[int] = None) -> Summary:
        """Summarize a conversation history."""
        max_tokens = max_tokens or self.max_tokens

        combined = "\n\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        )

        conversation_id = self.create_conversation_id(messages)

        if conversation_id in self._cache:
            cached = self._cache[conversation_id]
            if cached.token_count <= max_tokens:
                return cached

        truncated = self.truncate_for_tokens(combined, max_tokens)
        token_count = self.count_tokens(truncated)

        summary = Summary(
            content=truncated,
            token_count=token_count,
            conversation_id=conversation_id
        )

        self._cache[conversation_id] = summary
        return summary

    def extract_key_points(self, text: str, num_points: int = 5) -> list[str]:
        """Extract key points from text."""
        sentences = text.replace("\n", " ").split(". ")

        key_points = []
        for sentence in sentences:
            if len(sentence) > 20:
                key_points.append(sentence.strip() + ".")
                if len(key_points) >= num_points:
                    break

        return key_points

    def get_cache_size(self) -> int:
        """Get number of cached summaries."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear summary cache."""
        self._cache.clear()