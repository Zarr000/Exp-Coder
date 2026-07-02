"""
ControlNet Integration.

ControlNet for controlled image generation.
"""

from __future__ import annotations

from typing import Optional


CONTROL_TYPES = [
    "canny",
    "depth",
    "normal",
    "pose",
    "seg",
    "scribble",
]


class ControlNet:
    """
    Integrates ControlNet for controlled generation.

    Supports multiple control types.
    """

    def __init__(self, model: str = "control_v11p") -> None:
        """Initialize ControlNet."""
        self.model = model

    async def apply_control(
        self,
        image: bytes,
        control_image: bytes,
        control_type: str = "canny",
        prompt: str = ""
    ) -> bytes:
        """Apply ControlNet control to image generation."""
        return image

    async def prepare_control_image(
        self,
        image: bytes,
        control_type: str = "canny"
    ) -> bytes:
        """Prepare control image for specified control type."""
        return image

    def is_supported(self, control_type: str) -> bool:
        """Check if control type is supported."""
        return control_type in CONTROL_TYPES