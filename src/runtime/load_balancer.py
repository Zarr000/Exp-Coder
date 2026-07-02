"""
Load Balancer for Expera AI.

Distributes requests across providers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoadStats:
    """Load statistics."""

    provider: str
    requests: int = 0
    errors: int = 0
    total_latency: float = 0.0
    last_request: float = 0.0


class LoadBalancer:
    """
    Load balancer.

    Strategies:
    - Round-robin
    - Least connections
    - Least latency
    - Weighted
    """

    def __init__(self, strategy: str = "weighted") -> None:
        """Initialize load balancer."""
        self.strategy = strategy
        self._stats: dict[str, LoadStats] = {}
        self._current: int = 0

    def select(self, providers: list[str]) -> Optional[str]:
        """Select provider based on strategy."""
        if not providers:
            return None

        if self.strategy == "round_robin":
            return self._round_robin(providers)

        elif self.strategy == "least_connections":
            return self._least_connections(providers)

        elif self.strategy == "least_latency":
            return self._least_latency(providers)

        elif self.strategy == "weighted":
            return self._weighted(providers)

        return providers[0]

    def _round_robin(self, providers: list[str]) -> str:
        """Round-robin selection."""
        provider = providers[self._current % len(providers)]
        self._current += 1
        return provider

    def _least_connections(self, providers: list[str]) -> str:
        """Select provider with least active connections."""
        min_requests = float("inf")
        selected = providers[0]

        for provider in providers:
            stats = self._stats.get(provider, LoadStats(provider=provider))
            if stats.requests < min_requests:
                min_requests = stats.requests
                selected = provider

        return selected

    def _least_latency(self, providers: list[str]) -> str:
        """Select provider with lowest latency."""
        min_latency = float("inf")
        selected = providers[0]

        for provider in providers:
            stats = self._stats.get(provider, LoadStats(provider=provider))
            if stats.total_latency > 0:
                avg = stats.total_latency / stats.requests
            else:
                avg = float("inf")

            if avg < min_latency:
                min_latency = avg
                selected = provider

        return selected

    def _weighted(self, providers: list[str]) -> str:
        """Weighted selection (random)."""
        import random
        return random.choice(providers)

    def record_request(self, provider: str) -> None:
        """Record request."""
        if provider not in self._stats:
            self._stats[provider] = LoadStats(provider=provider)

        self._stats[provider].requests += 1
        self._stats[provider].last_request = time.time()

    def record_response(self, provider: str, latency: float, error: bool = False) -> None:
        """Record response."""
        if provider not in self._stats:
            self._stats[provider] = LoadStats(provider=provider)

        self._stats[provider].total_latency += latency

        if error:
            self._stats[provider].errors += 1

    def get_stats(self, provider: str) -> Optional[LoadStats]:
        """Get stats for provider."""
        return self._stats.get(provider)

    def get_all_stats(self) -> dict[str, LoadStats]:
        """Get all stats."""
        return self._stats

    def reset_stats(self, provider: Optional[str] = None) -> None:
        """Reset stats."""
        if provider:
            if provider in self._stats:
                self._stats[provider] = LoadStats(provider=provider)
        else:
            self._stats.clear()