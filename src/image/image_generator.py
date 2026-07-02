"""
Unified Image Generator.

Unified interface for all image generation backends.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .flux_client import FluxClient
from .sdxl_client import SDXLClient
from .comfyui_client import ComfyUIClient
from .automatic1111_client import Automatic1111Client


class ImageBackend(Enum):
    """Available image generation backends."""

    FLUX = "flux"
    SDXL = "sdxl"
    COMFYUI = "comfyui"
    AUTOMATIC1111 = "automatic1111"


@dataclass
class GenerationRequest:
    """Image generation request."""

    prompt: str
    backend: ImageBackend = ImageBackend.FLUX
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    negative_prompt: str = ""
    batch_size: int = 1


@dataclass
class GenerationResult:
    """Image generation result."""

    images: list[bytes]
    seed: int
    backend: str
    metadata: dict[str, Any]


class ImageGenerator:
    """
    Unified image generator.

    Supports multiple backends with automatic failover.
    """

    def __init__(self) -> None:
        """Initialize image generator."""
        self.backends: dict[ImageBackend, Any] = {
            ImageBackend.FLUX: FluxClient(),
            ImageBackend.SDXL: SDXLClient(),
            ImageBackend.COMFYUI: ComfyUIClient(),
            ImageBackend.AUTOMATIC1111: Automatic1111Client(),
        }
        self._active_backend: Optional[ImageBackend] = None

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images with specified backend."""
        backend = self.backends.get(request.backend)

        if not backend:
            raise ValueError(f"Backend {request.backend} not available")

        if request.backend == ImageBackend.FLUX:
            images = await backend.generate(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed or 42,
                negative_prompt=request.negative_prompt,
                batch_size=request.batch_size,
            )
        elif request.backend == ImageBackend.SDXL:
            images = await backend.generate(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
            )
        else:
            images = await backend.generate(request.prompt)

        return GenerationResult(
            images=images,
            seed=request.seed or 42,
            backend=request.backend.value,
            metadata={
                "width": request.width,
                "height": request.height,
                "steps": request.num_inference_steps,
                "guidance": request.guidance_scale,
            }
        )

    async def generate_with_fallback(
        self,
        request: GenerationRequest,
        fallback_backends: Optional[list[ImageBackend]] = None
    ) -> GenerationResult:
        """Generate with automatic fallback on failure."""
        backends = [request.backend] + (fallback_backends or [
            ImageBackend.SDXL,
            ImageBackend.COMFYUI,
            ImageBackend.AUTOMATIC1111
        ])

        last_error: Optional[Exception] = None

        for backend in backends:
            request.backend = backend
            try:
                return await self.generate(request)
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All backends failed. Last error: {last_error}")

    def is_available(self, backend: ImageBackend) -> bool:
        """Check if backend is available."""
        return backend in self.backends

    async def list_available_backends(self) -> list[ImageBackend]:
        """List all available backends."""
        available = []
        for backend in ImageBackend:
            if self.is_available(backend):
                available.append(backend)
        return available