"""
Automatic1111 Client.

Interface to Automatic1111's Stable Diffusion WebUI API.

Usage:
    python -m src.image.automatic1111_client --prompt "a cat" --output output.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class Automatic1111Config:
    """Automatic1111 client configuration."""

    api_url: str = "http://localhost:7860"

    width: int = 512
    height: int = 512
    steps: int = 20
    guidance: float = 7.5
    seed: int = -1

    batch_size: int = 1
    enable_hr: bool = False
    denoising_strength: float = 0.75


class Automatic1111Client:
    """Client for Automatic1111 API."""

    def __init__(self, config: Automatic1111Config):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list[bytes]:
        """
        Generate images using txt2img.

        Args:
            prompt: Text description
            negative_prompt: Things to avoid
            width: Image width
            height: Image height
            steps: Sampling steps
            guidance: Guidance scale
            seed: Random seed

        Returns:
            List of generated images as bytes
        """
        width = width or self.config.width
        height = height or self.config.height
        steps = steps or self.config.steps
        guidance = guidance or self.config.guidance

        if seed is None:
            seed = self.config.seed if self.config.seed != -1 else -1

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": guidance,
            "seed": seed,
            "batch_size": self.config.batch_size,
            "enable_hr": self.config.enable_hr,
            "denoising_strength": self.config.denoising_strength,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/txt2img", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"txt2img error: {resp.status} - {text}")

            result = await resp.json()
            images = []

            for b64_image in result.get("images", []):
                image_data = base64.b64decode(b64_image)
                images.append(image_data)

            return images

    async def img2img(
        self,
        prompt: str,
        image_data: bytes,
        negative_prompt: str = "",
        strength: float = 0.75,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list[bytes]:
        """
        Generate using img2img (image-to-image).

        Args:
            prompt: Text description
            image_data: Input image as bytes
            negative_prompt: Things to avoid
            strength: Denoising strength (0-1)
            width: Image width
            height: Image height
            steps: Sampling steps
            guidance: Guidance scale
            seed: Random seed

        Returns:
            List of generated images as bytes
        """
        width = width or self.config.width
        height = height or self.config.height
        steps = steps or self.config.steps
        guidance = guidance or self.config.guidance

        if seed is None:
            seed = self.config.seed if self.config.seed != -1 else -1

        image_b64 = base64.b64encode(image_data).decode()

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_images": [image_b64],
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": guidance,
            "seed": seed,
            "denoising_strength": strength,
            "include_init_images": True,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/img2img", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"img2img error: {resp.status} - {text}")

            result = await resp.json()
            images = []

            for b64_image in result.get("images", []):
                image_data = base64.b64decode(b64_image)
                images.append(image_data)

            return images

    async def inpaint(
        self,
        prompt: str,
        image_data: bytes,
        mask_data: bytes,
        negative_prompt: str = "",
        inpaint_full: bool = False,
        inpaint_masked: bool = False,
        mask_blur: int = 0,
    ) -> list[bytes]:
        """
        Inpaint/mask parts of an image.

        Args:
            prompt: Text description
            image_data: Input image as bytes
            mask_data: Mask image as bytes (white = paint, black = keep)
            negative_prompt: Things to avoid
            inpaint_full: Inpaint full image
            inpaint_masked: Inpaint only masked region

        Returns:
            List of generated images as bytes
        """
        image_b64 = base64.b64encode(image_data).decode()
        mask_b64 = base64.b64encode(mask_data).decode()

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_images": [image_b64],
            "mask": mask_b64,
            "width": self.config.width,
            "height": self.config.height,
            "steps": self.config.steps,
            "cfg_scale": self.config.guidance,
            "seed": self.config.seed,
            "inpaint_full": inpaint_full,
            "inpaint_masked": inpaint_masked,
            "mask_blur": mask_blur,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/img2img", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"inpaint error: {resp.status} - {text}")

            result = await resp.json()
            return [base64.b64decode(img) for img in result.get("images", [])]

    async def extra_single_image(
        self,
        image_data: bytes,
        upscaling_resize: float = 2.0,
        upscaler_1: str = "R-ESRGAN 4x+",
        exfgraup: bool = False,
    ) -> bytes:
        """Upscale a single image."""
        image_b64 = base64.b64encode(image_data).decode()

        payload = {
            "image": image_b64,
            "upscaling_resize": upscaling_resize,
            "upscaler_1": upscaler_1,
            "exfgraup": exfgraup,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/extra-single-image", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Extra error: {resp.status} - {text}")

            result = await resp.json()
            return base64.b64decode(result["image"])

    async def interrogate(
        self,
        image_data: bytes,
    ) -> str:
        """Interrogate an image for prompt generation."""
        image_b64 = base64.b64encode(image_data).decode()

        payload = {"image": image_b64}

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/interrogate", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Interrogate error: {resp.status} - {text}")

            result = await resp.json()
            return result["caption"]

    async def get_options(self) -> dict:
        """Get current API options."""
        async with self._session.get(
            f"{self.config.api_url}/sdapi/v1/options"
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Options error: {resp.status}")
            return await resp.json()

    async def set_options(self, options: dict) -> dict:
        """Set API options."""
        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/options", json=options
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Options error: {resp.status}")
            return await resp.json()