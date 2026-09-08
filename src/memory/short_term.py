"""
Short Term Memory for Expera AI.

Manages current conversation context:
- Recent messages
- Temporary state
- Context window

Usage:
    memory = ShortTermMemory(max_length=10)
    memory.add(user="Hello", assistant="Hi there!")
    context = memory.get_context()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in memory."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShortTermConfig:
    """Short term memory configuration."""

    max_length: int = 10  # Max messages
    max_tokens: int = 4096  # Max tokens
    include_system: bool = True


class ShortTermMemory:
    """
    Short term memory.

    Features:
    - Message history
    - Token limit
    - Context retrieval
    """

    def __init__(self, config: Optional[ShortTermConfig] = None):
        """Initialize short term memory."""
        self.config = config or ShortTermConfig()
        self._messages: list[Message] = []

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a message."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        self._messages.append(message)

        # Trim if needed
        self._trim()

    def _trim(self) -> None:
        """Trim message history."""
        if len(self._messages) > self.config.max_length:
            # Keep system message if present
            keep = self.config.max_length
            if self._messages and self._messages[0].role == "system":
                keep += 1

            self._messages = self._messages[-keep:]

    def get_messages(self) -> list[Message]:
        """Get all messages."""
        return self._messages.copy()

    def get_context(self) -> list[dict]:
        """Get context for model."""
        messages = []

        for msg in self._messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return messages

    def get_last(self, n: int = 1) -> list[Message]:
        """Get last n messages."""
        return self._messages[-n:]

    def get_last_user(self) -> Optional[str]:
        """Get last user message."""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant(self) -> Optional[str]:
        """Get last assistant message."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def count_tokens(self) -> int:
        """Estimate token count."""
        total = 0
        for msg in self._messages:
            # Rough estimate: 1 token ≈ 4 chars
            total += len(msg.content) // 4
        return total

    def clear(self) -> None:
        """Clear memory."""
        # Keep system message if configured
        if self.config.include_system and self._messages:
            if self._messages[0].role == "system":
                self._messages = [self._messages[0]]
                return

        self._messages.clear()

    def is_full(self) -> bool:
        """Check if memory is full."""
        if len(self._messages) >= self.config.max_length:
            return True

        if self.count_tokens() >= self.config.max_tokens:
            return True

        return False

    def __len__(self) -> int:
        """Get message count."""
        return len(self._messages)


# Export
__all__ = [
    "ShortTermMemory",
    "ShortTermConfig",
    "Message",
]