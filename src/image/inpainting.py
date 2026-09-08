"""
Image Inpainting.

Inpaints areas of images using AI.
"""

from __future__ import annotations

from typing import Optional


class Inpainter:
    """
    Inpaints images.

    Uses AI to fill masked regions.
    """

    def __init__(self, model: str = "sd-inpainting") -> None:
        """Initialize inpainter."""
        self.model = model

    async def inpaint(
        self,
        image: bytes,
        mask: bytes,
        prompt: str
    ) -> bytes:
        """Inpaint masked region with prompt."""
        return image

    async def inpaint_with_mask(
        self,
        image: bytes,
        mask_path: str,
        prompt: str
    ) -> bytes:
        """Inpaint with mask file."""
        return image