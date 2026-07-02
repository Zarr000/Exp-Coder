"""
Image Generator for Expera AI.

Generates images:
- Text-to-image
- Image-to-image
- Inpainting/Outpainting
- ControlNet support

Usage:
    generator = ImageGenerator()
    image = await generator.generate("a cat sitting", model="sdxl")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Image generation request."""

    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    seed: Optional[int] = None
    model: Optional[str] = None
    style: Optional[str] = None


@dataclass
class GenerationResult:
    """Image generation result."""

    images: list[bytes]
    seed: int
    info: dict


class ImageGenerator:
    """
    Generates images from text.

    Supports:
    - Stable Diffusion
    - SDXL
    - ControlNet
    - Remote API
    """

    def __init__(self, endpoint: Optional[str] = None):
        """Initialize image generator."""
        self.endpoint = endpoint or "http://localhost:7860"
        self.session: Optional[aiohttp.ClientSession] = None
        self._default_model = "sd-v1-5"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
        model: Optional[str] = None,
    ) -> GenerationResult:
        """Generate image from prompt."""
        # Use remote API if endpoint configured
        if self.endpoint:
            return await self._generate_remote(
                prompt, negative_prompt, width, height, steps, cfg_scale, seed, model
            )

        # Placeholder for local generation
        logger.warning("No endpoint configured, returning placeholder")
        return GenerationResult(
            images=[self._create_placeholder(width, height)],
            seed=seed or 0,
            info={"model": model or self._default_model},
        )

    async def _generate_remote(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: Optional[int],
        model: Optional[str],
    ) -> GenerationResult:
        """Generate using remote API."""
        session = await self._get_session()

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed or -1,
        }

        model = model or self._default_model
        if model != self._default_model:
            payload["model_id"] = model

        try:
            async with session.post(
                f"{self.endpoint}/sdapi/v1/txt2img",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    images = []

                    for b64_img in result.get("images", []):
                        img_data = base64.b64decode(b64_img)
                        images.append(img_data)

                    return GenerationResult(
                        images=images,
                        seed=result.get("seed", seed or 0),
                        info=result.get("info", {}),
                    )
                else:
                    error = await response.text()
                    logger.error(f"API error: {error}")
                    raise ValueError(f"API error: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            raise

    def _create_placeholder(self, width: int, height: int) -> bytes:
        """Create placeholder image."""
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (width, height), color="#1a1a2e")
            draw = ImageDraw.Draw(img)

            # Draw a simple gradient-like pattern
            for i in range(0, width, 50):
                color = tuple(min(255, 26 + i // 2) for _ in range(3))
                draw.rectangle([i, 0, i + 50, height], fill=color)

            # Add text
            draw.text((width // 2 - 100, height // 2), "Expera Image", fill="white")
            draw.text((width // 2 - 120, height // 2 + 30), "(Placeholder)", fill="gray")

            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        except ImportError:
            # Return minimal PNG if PIL not available
            return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"

    async def generate_variants(
        self,
        base_image: bytes,
        prompt: str,
        strength: float = 0.75,
        count: int = 4,
    ) -> list[GenerationResult]:
        """Generate variants of an image."""
        results = []

        for i in range(count):
            result = await self.img2img(
                base_image,
                prompt,
                strength=strength,
                seed=None,
            )
            results.append(result)
            await asyncio.sleep(0.1)

        return results

    async def img2img(
        self,
        init_image: bytes,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Generate image from image."""
        if self.endpoint:
            return await self._img2img_remote(
                init_image, prompt, negative_prompt, strength, steps, cfg_scale, seed
            )

        return GenerationResult(
            images=[self._create_placeholder(1024, 1024)],
            seed=seed or 0,
            info={},
        )

    async def _img2img_remote(
        self,
        init_image: bytes,
        prompt: str,
        negative_prompt: Optional[str],
        strength: float,
        steps: int,
        cfg_scale: float,
        seed: Optional[int],
    ) -> GenerationResult:
        """Image-to-image via remote API."""
        session = await self._get_session()

        # Encode image as base64
        b64_image = base64.b64encode(init_image).decode("utf-8")

        payload = {
            "init_images": [b64_image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed or -1,
        }

        try:
            async with session.post(
                f"{self.endpoint}/sdapi/v1/img2img",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    images = []

                    for b64_img in result.get("images", []):
                        img_data = base64.b64decode(b64_img)
                        images.append(img_data)

                    return GenerationResult(
                        images=images,
                        seed=result.get("seed", seed or 0),
                        info=result.get("info", {}),
                    )
                else:
                    error = await response.text()
                    raise ValueError(f"API error: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            raise

    async def upscale(
        self,
        image: bytes,
        model: str = "realesrgan-x4",
    ) -> bytes:
        """Upscale image."""
        if self.endpoint:
            return await self._upscale_remote(image, model)

        return image

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

    def save_image(
        self,
        image: bytes,
        path: str,
    ) -> bool:
        """Save image to file."""
        try:
            Path(path).write_bytes(image)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


# Export
__all__ = [
    "ImageGenerator",
    "GenerationRequest",
    "GenerationResult",
]