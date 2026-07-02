"""
Image Upscaler for Expera AI.

Upscales images:
- AI upscaling (RealESRGAN)
- Traditional upscaling
- Batch processing

Usage:
    upscaler = ImageUpscaler()
    upscaled = await upscaler.upscale(image, model="realesrgan-x4")
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class UpscaleConfig:
    """Upscaler configuration."""

    default_model: str = "realesrgan-x4"
    max_size: int = 4096
    tile_size: int = 256
    tile_pad: int = 10


# Available models
UPSCALE_MODELS = {
    "realesrgan-x4": {
        "scale": 4,
        "description": "RealESRGAN x4 - General purpose",
    },
    "realesrgan-x2": {
        "scale": 2,
        "description": "RealESRGAN x2 - Faster",
    },
    "realesrgan-x8": {
        "scale": 8,
        "description": "RealESRGAN x8 - High magnification",
    },
    "RealESRGAN_x4plus": {
        "scale": 4,
        "description": "RealESRGAN x4+ - Improved quality",
    },
    "RealESRGAN_x4plus_anime": {
        "scale": 4,
        "description": "RealESRGAN x4+ Anime - Anime optimized",
    },
}


class ImageUpscaler:
    """
    Upscales images.

    Features:
    - Multiple models
    - Tile-based processing
    - Batch upscaling
    """

    def __init__(self, endpoint: Optional[str] = None, config: Optional[UpscaleConfig] = None):
        """Initialize upscaler."""
        self.endpoint = endpoint or "http://localhost:7860"
        self.config = config or UpscaleConfig()
        self.models = UPSCALE_MODELS.copy()
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def upscale(
        self,
        image: bytes,
        model: Optional[str] = None,
        scale: Optional[int] = None,
    ) -> bytes:
        """Upscale image."""
        model = model or self.config.default_model

        if self.endpoint:
            return await self._upscale_remote(image, model)

        return await self._upscale_traditional(image, scale=4)

    async def _upscale_remote(
        self,
        image: bytes,
        model: str,
    ) -> bytes:
        """Upscale via remote API."""
        session = await self._get_session()

        b64_image = base64.b64encode(image).decode("utf-8")

        payload = {
            "image": b64_image,
            "model": model,
        }

        async with session.post(
            f"{self.endpoint}/sdapi/v1/extra-single-image",
            json=payload,
        ) as response:
            if response.status == 200:
                result = await response.json()
                return base64.b64decode(result["image"])
            else:
                raise ValueError(f"API error: {response.status}")

    async def _upscale_traditional(
        self,
        image: bytes,
        scale: int = 4,
    ) -> bytes:
        """Traditional upscaling using PIL."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image))

            # Get new size
            new_width = img.width * scale
            new_height = img.height * scale

            # Use LANCZOS for high-quality upscaling
            img = img.resize(
                (new_width, new_height),
                Image.LANCZOS,
            )

            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        except ImportError:
            logger.error("PIL not available")
            return image
        except Exception as e:
            logger.error(f"Upscale failed: {e}")
            return image

    async def upscale_tile(
        self,
        image: bytes,
        model: str = "realesrgan-x4",
    ) -> bytes:
        """Upscale with tiling for large images."""
        try:
            from PIL import Image
            import math

            img = Image.open(BytesIO(image))
            scale = self.models.get(model, {}).get("scale", 4)

            # Get tile size
            tile_size = self.config.tile_size
            pad = self.config.tile_pad

            # Calculate tiles
            tiles_x = math.ceil(img.width / tile_size)
            tiles_y = math.ceil(img.height / tile_size)

            # Process each tile
            result = Image.new(img.mode, (img.width * scale, img.height * scale))

            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Get tile bounds
                    left = x * tile_size
                    top = y * tile_size
                    right = min(left + tile_size + pad, img.width)
                    bottom = min(top + tile_size + pad, img.height)

                    # Extract tile
                    tile = img.crop((left, top, right, bottom))

                    # Upscale tile
                    tile = await self.upscale_tile_single(tile, model)

                    # Calculate output position
                    out_left = left * scale
                    out_top = top * scale
                    out_right = right * scale
                    out_bottom = bottom * scale

                    # Paste result
                    result.paste(tile, (out_left, out_top))

            buf = BytesIO()
            result.save(buf, format="PNG")
            return buf.getvalue()

        except ImportError:
            return await self._upscale_traditional(image, scale)

    async def upscale_tile_single(
        self,
        image: Image,
        model: str,
    ) -> Image:
        """Upscale single tile (helper)."""
        # Save temp, upscale, load back
        buf = BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()

        if self.endpoint:
            data = await self._upscale_remote(data, model)

        return Image.open(BytesIO(data))

    async def batch_upscale(
        self,
        images: list[bytes],
        model: Optional[str] = None,
    ) -> list[bytes]:
        """Upscale multiple images."""
        results = []

        for image in images:
            result = await self.upscale(image, model)
            results.append(result)

        return results

    def get_models(self) -> dict[str, dict]:
        """Get available upscale models."""
        return self.models.copy()

    def get_info(self, image: bytes) -> dict:
        """Get image info."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image))
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
            }
        except Exception:
            return {}

    async def close(self) -> None:
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


# Export
__all__ = [
    "ImageUpscaler",
    "UpscaleConfig",
    "UPSCALE_MODELS",
]