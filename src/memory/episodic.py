"""
Episodic Memory for Expera AI.

Stores conversation episodes and interactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Episode:
    """Conversation episode."""

    id: str
    start_time: datetime
    end_time: datetime
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """
    Episodic memory storage.

    Stores and retrieves conversation episodes.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        """Initialize episodic memory."""
        self.storage_path = storage_path
        self._episodes: dict[str, Episode] = {}
        self._load()

    def _load(self) -> None:
        """Load episodes from storage."""
        if self.storage_path:
            path = self.storage_path / "episodes.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._episodes = {
                    k: Episode(
                        id=e["id"],
                        start_time=datetime.fromisoformat(e["start_time"]),
                        end_time=datetime.fromisoformat(e["end_time"]),
                        messages=e.get("messages", []),
                        summary=e.get("summary"),
                        metadata=e.get("metadata", {}),
                    )
                    for k, e in data.items()
                }

    def _save(self) -> None:
        """Save episodes to storage."""
        if self.storage_path:
            path = self.storage_path / "episodes.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                k: {
                    "id": e.id,
                    "start_time": e.start_time.isoformat(),
                    "end_time": e.end_time.isoformat(),
                    "messages": e.messages,
                    "summary": e.summary,
                    "metadata": e.metadata,
                }
                for k, e in self._episodes.items()
            }
            path.write_text(json.dumps(data, indent=2))

    def start_episode(self, metadata: Optional[dict[str, Any]] = None) -> Episode:
        """Start new episode."""
        episode = Episode(
            id=datetime.now().isoformat(),
            start_time=datetime.now(),
            end_time=datetime.now(),
            metadata=metadata or {},
        )
        self._episodes[episode.id] = episode
        return episode

    def add_message(self, episode_id: str, message: dict[str, Any]) -> bool:
        """Add message to episode."""
        if episode_id in self._episodes:
            self._episodes[episode_id].messages.append(message)
            self._episodes[episode_id].end_time = datetime.now()
            return True
        return False

    def end_episode(self, episode_id: str, summary: Optional[str] = None) -> bool:
        """End episode."""
        if episode_id in self._episodes:
            self._episodes[episode_id].summary = summary
            self._episodes[episode_id].end_time = datetime.now()
            self._save()
            return True
        return False

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID."""
        return self._episodes.get(episode_id)

    def get_recent(self, limit: int = 10) -> list[Episode]:
        """Get recent episodes."""
        episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.start_time,
            reverse=True,
        )
        return episodes[:limit]

    def search(self, query: str) -> list[Episode]:
        """Search episodes."""
        results = []
        for episode in self._episodes.values():
            if query.lower() in episode.summary.lower():
                results.append(episode)
            else:
                for msg in episode.messages:
                    if query.lower() in str(msg).lower():
                        results.append(episode)
                        break
        return results

    def summarize(self, episode_id: str) -> Optional[str]:
        """Get episode summary."""
        episode = self._episodes.get(episode_id)
        if episode:
            if episode.summary:
                return episode.summary
            return f"Episode with {len(episode.messages)} messages"
        return None

    def delete_episode(self, episode_id: str) -> bool:
        """Delete episode."""
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            self._save()
            return True
        return False

    def count(self) -> int:
        """Count episodes."""
        return len(self._episodes)