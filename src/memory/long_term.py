"""
Long Term Memory for Expera AI.

Manages persistent memory:
- User preferences
- Past conversations
- Learned facts

Usage:
    memory = LongTermMemory()
    memory.store("user_prefers_python", True)
    prefs = memory.retrieve("user_prefers_python")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A long term memory entry."""

    key: str
    value: Any
    category: str = "general"
    timestamp: float = 0.0
    access_count: int = 0


@dataclass
class LongTermConfig:
    """Long term memory configuration."""

    storage_path: str = "data/memory"
    max_entries: int = 10000
    auto_save: bool = True


class LongTermMemory:
    """
    Long term memory.

    Features:
    - Persistent storage
    - Categories
    - Search
    - Auto-save
    """

    def __init__(self, config: Optional[LongTermConfig] = None):
        """Initialize long term memory."""
        self.config = config or LongTermConfig()
        self._entries: dict[str, MemoryEntry] = {}
        self._storage_path = Path(self.config.storage_path)
        self._dirty = False

    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
    ) -> None:
        """Store a memory."""
        import time

        entry = MemoryEntry(
            key=key,
            value=value,
            category=category,
            timestamp=time.time(),
        )

        self._entries[key] = entry
        self._dirty = True

        # Auto-save
        if self.config.auto_save:
            self.save()

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a memory."""
        entry = self._entries.get(key)
        if entry is None:
            return default

        entry.access_count += 1
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete a memory."""
        if key in self._entries:
            del self._entries[key]
            self._dirty = True
            return True
        return False

    def get_category(self, category: str) -> list[MemoryEntry]:
        """Get all entries in a category."""
        return [
            e for e in self._entries.values()
            if e.category == category
        ]

    def search(self, query: str) -> list[MemoryEntry]:
        """Search memories."""
        query_lower = query.lower()
        results = []

        for entry in self._entries.values():
            key_lower = entry.key.lower()
            value_str = str(entry.value).lower()

            if query_lower in key_lower or query_lower in value_str:
                results.append(entry)

        return results

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get recent memories."""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_entries[:n]

    def get_frequent(self, n: int = 10) -> list[MemoryEntry]:
        """Get most accessed memories."""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.access_count,
            reverse=True,
        )
        return sorted_entries[:n]

    def save(self, path: Optional[str] = None) -> bool:
        """Save to file."""
        try:
            save_path = Path(path) if path else self._storage_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for key, entry in self._entries.items():
                data[key] = {
                    "value": entry.value,
                    "category": entry.category,
                    "timestamp": entry.timestamp,
                    "access_count": entry.access_count,
                }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            logger.info(f"Saved {len(data)} memories")
            return True

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """Load from file."""
        import time

        try:
            load_path = Path(path) if path else self._storage_path
            if not load_path.exists():
                return False

            with open(load_path) as f:
                data = json.load(f)

            for key, entry_data in data.items():
                self._entries[key] = MemoryEntry(
                    key=key,
                    value=entry_data["value"],
                    category=entry_data.get("category", "general"),
                    timestamp=entry_data.get("timestamp", time.time()),
                    access_count=entry_data.get("access_count", 0),
                )

            logger.info(f"Loaded {len(self._entries)} memories")
            return True

        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()
        self._dirty = True

    def __len__(self) -> int:
        """Get entry count."""
        return len(self._entries)


# Export
__all__ = [
    "LongTermMemory",
    "LongTermConfig",
    "MemoryEntry",
]