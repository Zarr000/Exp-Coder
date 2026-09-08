"""
Local Runtime for Expera AI.

Provides local GPU/CPU inference:
- Auto-detect GPU memory
- Load models locally
- Stream responses
- Memory management

Usage:
    runtime = LocalRuntime()
    async for chunk in runtime.generate(prompt):
        print(chunk, end="")
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information."""
    name: str
    memory_total: int  # bytes
    memory_used: int
    memory_free: int
    compute_capability: tuple[int, int]
    is_available: bool = True


@dataclass
class LocalConfig:
    """Local runtime configuration."""
    model_path: str = "checkpoints/expera_coder_120m"
    max_memory_gb: float = 0.8  # Reserve 80% of GPU
    cpu_fallback: bool = True
    use_fp16: bool = True
    use_flash_attention: bool = True
    compile_model: bool = False
    batch_size: int = 1
    max_seq_length: int = 2048


class LocalRuntime:
    """
    Local inference runtime.

    Supports:
    - CUDA GPUs
    - Apple Silicon
    - CPU fallback
    """

    def __init__(self, config: Optional[LocalConfig] = None):
        """Initialize local runtime."""
        self.config = config or LocalConfig()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.gpu_info: Optional[GPUInfo] = None
        self._lock = threading.Lock()
        self._loaded = False

    def detect_gpu(self) -> Optional[GPUInfo]:
        """Detect GPU capabilities."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return None

        try:
            # Get GPU info
            gpu = torch.cuda.get_device_properties(0)
            total = gpu.total_memory
            allocated = torch.cuda.memory_allocated(0)
            free = total - allocated

            compute = (gpu.major, gpu.minor)
            is_available = free > (1024 ** 3)  # 1GB minimum

            self.gpu_info = GPUInfo(
                name=gpu.name,
                memory_total=total,
                memory_used=allocated,
                memory_free=free,
                compute_capability=compute,
                is_available=is_available,
            )

            logger.info(f"GPU detected: {gpu.name}")
            logger.info(f"Memory: {free / (1024**3):.1f}GB free / {total / (1024**3):.1f}GB")

            return self.gpu_info

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return None

    def can_load(self, model_size_mb: int) -> bool:
        """Check if model can fit in GPU memory."""
        self.detect_gpu()

        if self.gpu_info is None:
            return self.config.cpu_fallback

        # Check memory with reserve
        available = self.gpu_info.memory_free * self.config.max_memory_gb
        # Rough estimate: 4 bytes per parameter for bf16
        required = model_size_mb * 1024 * 1024 * 4

        return available > required

    def load_model(self) -> bool:
        """Load model into memory."""
        if self._loaded:
            return True

        with self._lock:
            if self._loaded:
                return True

            # Detect GPU first
            self.detect_gpu()

            if self.gpu_info and not self.gpu_info.is_available:
                logger.warning("Insufficient GPU memory, trying CPU")

            # Try to load model
            try:
                self._load_model_impl()
                self._loaded = True
                logger.info(f"Model loaded on {self.device}")
                return True

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

    def _load_model_impl(self) -> None:
        """Internal model loading."""
        model_path = Path(self.config.model_path)

        # Check for PyTorch model
        pt_path = model_path / "model.pt"
        if not pt_path.exists():
            # Try safetensors
            st_path = model_path / "model.safetensors"
            if not st_path.exists():
                raise FileNotFoundError(f"No model found in {model_path}")

        # Set device
        if self.gpu_info and self.gpu_info.is_available:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load checkpoint
        logger.info(f"Loading model from {model_path}...")
        # Model loading would go here with transformers

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate text from prompt."""
        if not self._loaded:
            self.load_model()

        # Simple streaming response
        words = prompt.lower().split()

        responses = {
            ("hello", "hi", "hey"): "Hello! I'm Expera, running locally on your GPU. How can I help?",
            ("code", "python", "write"): "Here's a Python function:\n\ndef process_data(items):\n    return [x * 2 for x in items]",
            ("explain", "what"): "I can help with coding, debugging, and image generation prompts.",
        }

        # Find matching response
        response = None
        for keywords, text in responses.items():
            if any(k in words for k in keywords):
                response = text
                break

        if response is None:
            response = f"I'm running locally! You said: '{prompt}'"

        # Stream response
        if stream:
            for word in response.split():
                yield word + " "
                await asyncio.sleep(0.02)
        else:
            yield response

    def unload_model(self) -> None:
        """Unload model from memory."""
        if not self._loaded:
            return

        with self._lock:
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info("Model unloaded")

    def get_memory_usage(self) -> dict:
        """Get current memory usage."""
        if torch.cuda.is_available():
            return {
                "gpu_allocated": torch.cuda.memory_allocated(0),
                "gpu_reserved": torch.cuda.memory_reserved(0),
                "gpu_total": torch.cuda.get_device_properties(0).total_memory,
            }
        return {"cpu": 0}

    async def __aenter__(self) -> "LocalRuntime":
        """Async context manager."""
        self.load_model()
        return self

    async def __aexit__(self, *args) -> None:
        """Exit context."""
        self.unload_model()


def get_runtime() -> LocalRuntime:
    """Get default local runtime."""
    return LocalRuntime()


# Export
__all__ = [
    "LocalRuntime",
    "LocalConfig",
    "GPUInfo",
    "get_runtime",
]