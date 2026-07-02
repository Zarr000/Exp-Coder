"""
Memory Manager.

Manages all memory systems: short-term, long-term, episodic, semantic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .short_term import ShortTermMemory
from .long_term import LongTermMemory


@dataclass
class MemoryConfig:
    """Memory manager configuration."""

    short_term_max_items: int = 100
    long_term_enabled: bool = True
    episodic_enabled: bool = True
    semantic_enabled: bool = True
    auto_save_interval: int = 300


@dataclass
class MemoryItem:
    """Single memory item."""

    content: str
    memory_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """
    Unified memory manager.

    Coordinates short-term, long-term, episodic, and semantic memory.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        """Initialize memory manager."""
        self.config = config or MemoryConfig()
        self.short_term = ShortTermMemory(max_items=self.config.short_term_max_items)
        self.long_term = LongTermMemory() if self.config.long_term_enabled else None
        self._lock = asyncio.Lock()

    async def add(self, content: str, memory_type: str = "short_term", metadata: Optional[dict[str, Any]] = None) -> None:
        """Add memory item."""
        async with self._lock:
            item = MemoryItem(
                content=content,
                memory_type=memory_type,
                metadata=metadata or {}
            )

            if memory_type == "short_term":
                await self.short_term.add(item)
            elif memory_type == "long_term" and self.long_term:
                await self.long_term.add(item)

    async def get_recent(self, limit: int = 10, memory_type: Optional[str] = None) -> list[MemoryItem]:
        """Get recent memories."""
        async with self._lock:
            if memory_type == "short_term" or memory_type is None:
                return await self.short_term.get_recent(limit)
            elif memory_type == "long_term" and self.long_term:
                return await self.long_term.get_recent(limit)
            return []

    async def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Search memories."""
        results: list[MemoryItem] = []

        if self.long_term:
            results.extend(await self.long_term.search(query, limit))

        results.extend(await self.short_term.search(query, limit))

        return results[:limit]

    async def clear(self, memory_type: Optional[str] = None) -> None:
        """Clear memories."""
        async with self._lock:
            if memory_type == "short_term" or memory_type is None:
                await self.short_term.clear()
            if memory_type == "long_term" and self.long_term:
                await self.long_term.clear()

    async def save(self) -> None:
        """Persist memories to storage."""
        if self.long_term:
            await self.long_term.save()

    async def load(self) -> None:
        """Load memories from storage."""
        if self.long_term:
            await self.long_term.load()