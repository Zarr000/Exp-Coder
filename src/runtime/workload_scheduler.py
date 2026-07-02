"""
Workload Scheduler.

Schedules and balances inference workloads across:
- Multiple local GPUs
- Remote VPS instances

Usage:
    python -m src.runtime.workload_scheduler --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.runtime.hybrid_router import HybridConfig, HybridRouter
from src.runtime.local_inference import LocalConfig, LocalInferenceEngine
from src.runtime.remote_inference import RemoteConfig, RemoteInferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    """A worker instance."""

    id: str
    type: str  # local, remote
    endpoint: str
    available: bool = True
    current_load: int = 0
    max_concurrent: int = 1
    avg_latency_ms: float = 0
    total_requests: int = 0
    total_tokens: int = 0


@dataclass
class SchedulingConfig:
    """Scheduling configuration."""

    # Workers
    local_workers: list[dict] = field(default_factory=list)
    remote_workers: list[dict] = field(default_factory=list)

    # Scheduling strategy
    strategy: str = "least_load"  # least_load, round_robin, fastest, hash

    # Limits
    max_queue_size: int = 100
    request_timeout: int = 120

    # Health check
    health_check_interval: int = 30
    max_failure_count: int = 3


@dataclass
class Request:
    """A queued request."""

    id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    priority: int = 0
    created_at: float = field(default_factory=time.perf_counter)
    worker_id: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class WorkloadScheduler:
    """Schedules inference requests across workers."""

    def __init__(self, config: SchedulingConfig):
        self.config = config
        self.workers: dict[str, Worker] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._stats = {"requests": 0, "tokens": 0, "errors": 0}

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True

        # Register workers
        for worker_cfg in self.config.local_workers:
            worker = Worker(
                id=f"local_{worker_cfg['id']}",
                type="local",
                endpoint=worker_cfg.get("endpoint", "local"),
                max_concurrent=worker_cfg.get("max_concurrent", 1),
            )
            self.workers[worker.id] = worker

        for worker_cfg in self.config.remote_workers:
            worker = Worker(
                id=f"remote_{worker_cfg['id']}",
                type="remote",
                endpoint=worker_cfg["url"],
                max_concurrent=worker_cfg.get("max_concurrent", 10),
            )
            self.workers[worker.id] = worker

        logger.info(f"Registered {len(self.workers)} workers")

        # Start workers
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        self._scheduler_task.cancel()
        self._health_task.cancel()

    def _select_worker(self, request: Request) -> Optional[Worker]:
        """Select best worker for request."""
        if not self.workers:
            return None

        available = [w for w in self.workers.values() if w.available]
        if not available:
            return None

        if self.config.strategy == "least_load":
            return min(available, key=lambda w: w.current_load / w.max_concurrent)

        if self.config.strategy == "fastest":
            return min(available, key=lambda w: w.avg_latency_ms or float("inf"))

        if self.config.strategy == "round_robin":
            # Simple round-robin based on total requests
            return min(available, key=lambda w: w.total_requests)

        if self.config.strategy == "hash":
            # Hash-based routing for consistency
            hash_val = int(hashlib.md5(request.prompt.encode()).hexdigest(), 16)
            return available[hash_val % len(available)]

        return available[0]

    async def _execute_request(
        self,
        request: Request,
        worker: Worker,
    ) -> dict:
        """Execute request on worker."""
        worker.current_load += 1

        try:
            if worker.type == "local":
                engine = LocalInferenceEngine(
                    LocalConfig(model_path=Path(worker.endpoint))
                )
                result = await engine.generate(
                    request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                return {
                    "text": result.text,
                    "tokens": result.tokens,
                    "latency_ms": result.latency_ms,
                    "worker_id": worker.id,
                }
            else:
                engine = RemoteInferenceEngine(
                    RemoteConfig(api_url=worker.endpoint)
                )
                async with engine:
                    result = await engine.generate(
                        request.prompt,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                    return {
                        "text": result["text"],
                        "tokens": result["tokens"],
                        "latency_ms": result["latency_ms"],
                        "worker_id": worker.id,
                    }
        finally:
            worker.current_load -= 1

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Get request from queue
                request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            # Select worker
            worker = self._select_worker(request)
            if not worker:
                # No available worker, requeue
                await asyncio.sleep(0.1)
                if self.queue.qsize() < self.config.max_queue_size:
                    self.queue.put_nowait(request)
                continue

            # Execute
            request.worker_id = worker.id
            worker.available = False

            try:
                start_time = time.perf_counter()
                result = await asyncio.wait_for(
                    self._execute_request(request, worker),
                    timeout=self.config.request_timeout,
                )
                request.result = result

                # Update stats
                worker.total_requests += 1
                worker.total_tokens += result["tokens"]
                worker.avg_latency_ms = (
                    (worker.avg_latency_ms * (worker.total_requests - 1) + result["latency_ms"])
                    / worker.total_requests
                )

                self._stats["requests"] += 1
                self._stats["tokens"] += result["tokens"]

            except asyncio.TimeoutError:
                request.error = f"Timeout after {self.config.request_timeout}s"
                self._stats["errors"] += 1

            except Exception as e:
                request.error = str(e)
                self._stats["errors"] += 1
                logger.error(f"Worker {worker.id} error: {e}")

            worker.available = True

    async def _health_loop(self) -> None:
        """Health check loop."""
        while self._running:
            await asyncio.sleep(self.config.health_check_interval)

            for worker in self.workers.values():
                try:
                    if worker.type == "remote":
                        engine = RemoteInferenceEngine(
                            RemoteConfig(api_url=worker.endpoint)
                        )
                        async with engine:
                            worker.available = await engine.health_check()
                except Exception:
                    worker.available = False

    async def submit(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Submit a request, returns request ID."""
        import uuid

        request = Request(
            id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        await self.queue.put(request)
        return request.id

    async def get_result(self, request_id: str) -> Optional[dict]:
        """Get result for request ID."""
        # This is simplified - in practice, track completed requests
        return None

    def get_stats(self) -> dict:
        """Get scheduler stats."""
        return {
            **self._stats,
            "queue_size": self.queue.qsize(),
            "workers": {
                w.id: {
                    "available": w.available,
                    "load": w.current_load,
                    "avg_latency": w.avg_latency_ms,
                }
                for w in self.workers.values()
            },
        }


async def main():
    parser = argparse.ArgumentParser(description="Workload scheduler")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--strategy", default="least_load")
    args = parser.parse_args()

    config = SchedulingConfig(strategy=args.strategy)

    scheduler = WorkloadScheduler(config)
    await scheduler.start()

    try:
        # Demo: submit some requests
        for i in range(5):
            await scheduler.submit(f"Test prompt {i}")
            print(f"Submitted request {i}")

        await asyncio.sleep(2)
        print(scheduler.get_stats())

    finally:
        await scheduler.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())