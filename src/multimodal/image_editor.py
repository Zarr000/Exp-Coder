"""
Image Editor for Expera AI.

Edits images:
- Crop, resize, rotate
- Filters and effects
- Color adjustment
- Inpainting

Usage:
    editor = ImageEditor()
    edited = await editor.apply_mask(image, mask, prompt)
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
class EditRequest:
    """Image edit request."""

    image: bytes
    mask: Optional[bytes] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    strength: float = 0.75
    steps: int = 20
    cfg_scale: float = 7.5


@dataclass
class EditConfig:
    """Image editor configuration."""

    max_size: int = 2048
    default_format: str = "png"
    quality: int = 95


class ImageEditor:
    """
    Edits images.

    Features:
    - Inpainting/outpainting
    - Image editing
    - Crop/resize
    - Filters
    """

    def __init__(self, endpoint: Optional[str] = None, config: Optional[EditConfig] = None):
        """Initialize image editor."""
        self.endpoint = endpoint or "http://localhost:7860"
        self.config = config or EditConfig()
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def inpaint(
        self,
        image: bytes,
        mask: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> bytes:
        """Inpaint image with mask."""
        if not self.endpoint:
            logger.warning("No endpoint configured")
            return image

        session = await self._get_session()

        b64_image = base64.b64encode(image).decode("utf-8")
        b64_mask = base64.b64encode(mask).decode("utf-8")

        payload = {
            "init_images": [b64_image],
            "mask_image": b64_mask,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": strength,
            "inpaint_full_res": False,
            "inpaint_mask_full_res": True,
        }

        async with session.post(
            f"{self.endpoint}/sdapi/v1/img2img",
            json=payload,
        ) as response:
            if response.status == 200:
                result = await response.json()
                return base64.b64decode(result["images"][0])
            else:
                raise ValueError(f"API error: {response.status}")

    async def outpaint(
        self,
        image: bytes,
        prompt: str,
        direction: str = "right",
        pixels: int = 512,
    ) -> bytes:
        """Outpaint (extend) image."""
        # Direction offsets
        directions = {
            "left": (-pixels, 0),
            "right": (pixels, 0),
            "up": (0, -pixels),
            "down": (0, pixels),
        }

        offset = directions.get(direction, (pixels, 0))

        if not self.endpoint:
            return image

        session = await self._get_session()

        b64_image = base64.b64encode(image).decode("utf-8")

        payload = {
            "init_images": [b64_image],
            "prompt": prompt,
            "denoising_strength": 0.75,
            "image_seed": -1,
        }

        async with session.post(
            f"{self.endpoint}/sdapi/v1/exten",
            json=payload,
        ) as response:
            if response.status == 200:
                result = await response.json()
                return base64.b64decode(result["image"])
            else:
                raise ValueError(f"API error: {response.status}")

    def resize(self, image: bytes, width: int, height: int) -> bytes:
        """Resize image."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image))
            img = img.resize((width, height), Image.LANCZOS)

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            logger.error("PIL not available")
            return image
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            return image

    def crop(self, image: bytes, x: int, y: int, width: int, height: int) -> bytes:
        """Crop image."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image))
            img = img.crop((x, y, x + width, y + height))

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            logger.error("PIL not available")
            return image
        except Exception as e:
            logger.error(f"Crop failed: {e}")
            return image

    def rotate(self, image: bytes, angle: float) -> bytes:
        """Rotate image."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image))
            img = img.rotate(angle, expand=True)

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            logger.error("PIL not available")
            return image
        except Exception as e:
            logger.error(f"Rotate failed: {e}")
            return image

    def adjust_brightness(self, image: bytes, factor: float = 1.0) -> bytes:
        """Adjust brightness."""
        try:
            from PIL import Image, ImageEnhance

            img = Image.open(BytesIO(image))
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            return image
        except Exception as e:
            logger.error(f"Brightness adjust failed: {e}")
            return image

    def adjust_contrast(self, image: bytes, factor: float = 1.0) -> bytes:
        """Adjust contrast."""
        try:
            from PIL import Image, ImageEnhance

            img = Image.open(BytesIO(image))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            return image
        except Exception as e:
            logger.error(f"Contrast adjust failed: {e}")
            return image

    def adjust_saturation(self, image: bytes, factor: float = 1.0) -> bytes:
        """Adjust saturation."""
        try:
            from PIL import Image, ImageEnhance

            img = Image.open(BytesIO(image))
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            return image
        except Exception as e:
            logger.error(f"Saturation adjust failed: {e}")
            return image

    def grayscale(self, image: bytes) -> bytes:
        """Convert to grayscale."""
        return self.adjust_saturation(image, 0)

    def blur(self, image: bytes, radius: int = 5) -> bytes:
        """Apply blur."""
        try:
            from PIL import Image, ImageFilter

            img = Image.open(BytesIO(image))
            img = img.filter(ImageFilter.GaussianBlur(radius))

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            return image
        except Exception as e:
            logger.error(f"Blur failed: {e}")
            return image

    def sharpen(self, image: bytes) -> bytes:
        """Apply sharpen."""
        try:
            from PIL import Image, ImageFilter

            img = Image.open(BytesIO(image))
            img = img.filter(ImageFilter.SHARPEN)

            buf = BytesIO()
            img.save(buf, format=self.config.default_format.upper())
            return buf.getvalue()

        except ImportError:
            return image
        except Exception as e:
            logger.error(f"Sharpen failed: {e}")
            return image

    def get_info(self, image: bytes) -> dict:
        """Get image info."""
        try:
            from PIL import Image

            img = Image.open(BytesIO(image))
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
            }
        except Exception:
            return {}

    def save(self, image: bytes, path: str) -> bool:
        """Save image to file."""
        try:
            Path(path).write_bytes(image)
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    async def close(self) -> None:
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


# Export
__all__ = [
    "ImageEditor",
    "EditRequest",
    "EditConfig",
]