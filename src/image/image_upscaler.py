"""
Image Upscaler.

Upscales images using AI models.
"""

from __future__ import annotations

from typing import Optional


class ImageUpscaler:
    """
    Upscales images.

    Uses R-ESRGAN or similar for upscaling.
    """

    def __init__(self, model: str = "realesrgan-x4") -> None:
        """Initialize upscaler."""
        self.model = model

    async def upscale(self, image: bytes, scale: int = 2) -> bytes:
        """Upscale image by specified scale."""
        return image

    async def upscale_tile(
        self,
        image: bytes,
        scale: int = 2,
        tile_size: int = 512,
        tile_overlap: int = 32
    ) -> bytes:
        """Upscale image in tiles to avoid OOM."""
        return image

    def is_supported(self, scale: int) -> bool:
        """Check if scale is supported."""
        return scale in [2, 4, 8]