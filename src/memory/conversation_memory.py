"""
Conversation Memory for Expera AI.

Full conversation memory system:
- Combines short-term and long-term
- Summary generation
- Context management

Usage:
    memory = ConversationMemory()
    await memory.add_message("user", "Hello")
    summary = await memory.get_summary()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .short_term import ShortTermMemory, ShortTermConfig
from .long_term import LongTermMemory, LongTermConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Current conversation state."""

    conversation_id: str
    started_at: float = field(default_factory=time.time)
    message_count: int = 0
    summary: Optional[str] = None


class ConversationMemory:
    """
    Full conversation memory.

    Combines:
    - Short term (recent messages)
    - Long term (persistent)
    - Summaries
    """

    def __init__(
        self,
        conversation_id: str = "default",
        short_term_config: Optional[ShortTermConfig] = None,
        long_term_config: Optional[LongTermConfig] = None,
    ):
        """Initialize conversation memory."""
        self.conversation_id = conversation_id

        # Initialize components
        self.short_term = ShortTermMemory(short_term_config)
        self.long_term = LongTermMemory(long_term_config)

        # State
        self.state = ConversationState(conversation_id=conversation_id)

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a message to conversation."""
        self.short_term.add(role, content, metadata)
        self.state.message_count += 1

    async def get_context(self) -> list[dict]:
        """Get conversation context."""
        return self.short_term.get_context()

    async def get_summary(self, force: bool = False) -> Optional[str]:
        """Get conversation summary."""
        # Return existing if not forced
        if not force and self.state.summary:
            return self.state.summary

        # Generate summary from recent messages
        messages = self.short_term.get_messages()
        if not messages:
            return None

        # Simple summary: first and last few messages
        if len(messages) <= 4:
            self.state.summary = f"Conversation with {len(messages)} messages"
        else:
            self.state.summary = (
                f"Conversation started with {messages[0].content[:50]}... "
                f"and {len(messages) - 2} more messages, "
                f"ending with {messages[-1].content[:50]}..."
            )

        return self.state.summary

    async def store_fact(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Store a fact in long term."""
        self.long_term.store(key, value, category="conversation")

    async def retrieve_fact(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a fact."""
        return self.long_term.retrieve(key, default)

    async def clear(self) -> None:
        """Clear conversation."""
        self.short_term.clear()
        self.state = ConversationState(
            conversation_id=self.conversation_id,
        )

    def is_empty(self) -> bool:
        """Check if conversation is empty."""
        return len(self.short_term) == 0

    def message_count(self) -> int:
        """Get message count."""
        return self.state.message_count


# Export
__all__ = [
    "ConversationMemory",
    "ConversationState",
]