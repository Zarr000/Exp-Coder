"""
Provider Manager for Expera AI.

Manages inference providers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProviderType(Enum):
    """Provider types."""

    LOCAL_GPU = "local_gpu"
    LOCAL_CPU = "local_cpu"
    VPS = "vps"
    CLOUD = "cloud"


@dataclass
class ProviderConfig:
    """Provider configuration."""

    name: str
    provider_type: ProviderType
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: str = "default"
    max_tokens: int = 2048
    temperature: float = 0.7
    priority: int = 0


@dataclass
class ProviderStatus:
    """Provider status."""

    name: str
    available: bool
    latency: float = 0.0
    error_count: int = 0
    request_count: int = 0


class ProviderManager:
    """
    Manages inference providers.

    Features:
    - Provider registration
    - Health monitoring
    - Automatic failover
    - Load balancing
    """

    def __init__(self) -> None:
        """Initialize provider manager."""
        self._providers: dict[str, ProviderConfig] = {}
        self._status: dict[str, ProviderStatus] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register(self, config: ProviderConfig) -> None:
        """Register provider."""
        self._providers[config.name] = config
        self._status[config.name] = ProviderStatus(
            name=config.name,
            available=True,
        )
        self._locks[config.name] = asyncio.Lock()

    def unregister(self, name: str) -> bool:
        """Unregister provider."""
        if name in self._providers:
            del self._providers[name]
            del self._status[name]
            if name in self._locks:
                del self._locks[name]
            return True
        return False

    def get(self, name: str) -> Optional[ProviderConfig]:
        """Get provider config."""
        return self._providers.get(name)

    def get_type(self, provider_type: ProviderType) -> list[ProviderConfig]:
        """Get providers by type."""
        return [
            p for p in self._providers.values()
            if p.provider_type == provider_type
        ]

    def get_available(self) -> list[ProviderConfig]:
        """Get available providers."""
        return [
            p for p in self._providers.values()
            if self._status[p.name].available
        ]

    def update_status(self, name: str, status: ProviderStatus) -> None:
        """Update provider status."""
        self._status[name] = status

    def get_status(self, name: str) -> Optional[ProviderStatus]:
        """Get provider status."""
        return self._status.get(name)

    def set_available(self, name: str, available: bool) -> None:
        """Set provider availability."""
        if name in self._status:
            self._status[name].available = available
            self._status[name].error_count = 0

    def increment_error(self, name: str) -> None:
        """Increment error count."""
        if name in self._status:
            self._status[name].error_count += 1
            if self._status[name].error_count >= 3:
                self._status[name].available = False

    def get_best(self) -> Optional[ProviderConfig]:
        """Get best available provider."""
        available = self.get_available()

        if not available:
            return None

        available.sort(key=lambda p: p.priority, reverse=True)
        return available[0]

    def list_providers(self) -> list[str]:
        """List provider names."""
        return list(self._providers.keys())

    def count(self) -> int:
        """Count providers."""
        return len(self._providers)