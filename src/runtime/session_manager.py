"""
Session Manager for Expera AI.

Manages user sessions:
- Session creation/deletion
- Session state
- Concurrent sessions
- Session limits
- Cleanup

Usage:
    manager = SessionManager()
    session_id = await manager.create_session(user_id="user123")
    session = manager.get_session(session_id)
    await manager.delete_session(session_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """User session."""

    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class SessionConfig:
    """Session manager configuration."""

    max_sessions: int = 100
    session_timeout: float = 3600  # seconds
    cleanup_interval: float = 60  # seconds
    max_history: int = 100


class SessionManager:
    """
    Manages user sessions.

    Features:
    - Session lifecycle
    - Concurrent session limits
    - Automatic cleanup
    - Session data storage
    """

    def __init__(self, config: Optional[SessionConfig] = None):
        """Initialize session manager."""
        self.config = config or SessionConfig()
        self._sessions: dict[str, Session] = {}
        self._user_sessions: dict[str, set[str]] = {}  # user_id -> session_ids
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def create_session(
        self,
        user_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Create a new session."""
        async with self._lock:
            # Check max sessions
            if len(self._sessions) >= self.config.max_sessions:
                # Try cleanup first
                await self._cleanup_stale()

                if len(self._sessions) >= self.config.max_sessions:
                    logger.warning("Max sessions reached")
                    return None

            # Check user max sessions
            user_sessions = self._user_sessions.get(user_id, set())
            if len(user_sessions) >= 10:  # Max 10 per user
                logger.warning(f"Max sessions for user {user_id}")
                return None

            # Create session
            session_id = str(uuid.uuid4())
            session = Session(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
            )

            self._sessions[session_id] = session

            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)

            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def update_session(
        self,
        session_id: str,
        data: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update session data."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.last_activity = time.time()

            if data:
                session.data.update(data)

            if metadata:
                session.metadata.update(metadata)

            return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False

            # Remove from user sessions
            user_id = session.user_id
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]

            logger.info(f"Deleted session {session_id}")
            return True

    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user."""
        async with self._lock:
            session_ids = self._user_sessions.get(user_id, set()).copy()
            count = 0

            for session_id in session_ids:
                if await self.delete_session(session_id):
                    count += 1

            return count

    async def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all sessions for a user."""
        async with self._lock:
            session_ids = self._user_sessions.get(user_id, set())
            return [
                self._sessions[sid]
                for sid in session_ids
                if sid in self._sessions
            ]

    async def touch_session(self, session_id: str) -> bool:
        """Update last activity time."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_activity = time.time()
                return True
            return False

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

    async def _cleanup_stale(self) -> int:
        """Clean up stale sessions."""
        now = time.time()
        stale = []

        for session_id, session in self._sessions.items():
            if not session.is_active:
                stale.append(session_id)
                continue

            if now - session.last_activity > self.config.session_timeout:
                stale.append(session_id)

        for session_id in stale:
            await self.delete_session(session_id)

        if stale:
            logger.info(f"Cleaned up {len(stale)} stale sessions")

        return len(stale)

    async def get_stats(self) -> dict:
        """Get session manager statistics."""
        async with self._lock:
            active = sum(1 for s in self._sessions.values() if s.is_active)
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": active,
                "total_users": len(self._user_sessions),
            }

    async def cleanup(self) -> None:
        """Clean up all sessions."""
        async with self._lock:
            session_ids = list(self._sessions.keys())

        for session_id in session_ids:
            await self.delete_session(session_id)

        await self.stop()


# Export
__all__ = [
    "SessionManager",
    "Session",
    "SessionConfig",
]