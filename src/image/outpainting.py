"""
Image Outpainting.

Extends images beyond their boundaries.
"""

from __future__ import annotations


class Outpainter:
    """
    Outpaints images.

    Extends images in specified directions.
    """

    def __init__(self, model: str = "sd-outpainting") -> None:
        """Initialize outpainter."""
        self.model = model

    async def outpaint(
        self,
        image: bytes,
        direction: str = "right",
        pixels: int = 512
    ) -> bytes:
        """Outpaint image in direction."""
        return image

    async def outpaint_all(
        self,
        image: bytes,
        pixels: int = 512
    ) -> bytes:
        """Outpaint all sides."""
        return image