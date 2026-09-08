"""
Image Agent.

Specialized agent for image generation and editing.

Usage:
    python -m src.agents.image_agent --task "generate a sunset" --output image.png
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.image import (
    FluxClient,
    SDXLClient,
    ComfyUIClient,
    Automatic1111Client,
    style_presets,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    """Image generation configuration."""

    model: str = "flux"  # flux, sdxl, comfyui, automatic1111
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    seed: int = -1

    style: Optional[str] = None
    quality: str = "high"  # low, medium, high, ultra


@dataclass
class GenerationRequest:
    """Image generation request."""

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    seed: int = -1
    style: Optional[str] = None
    output_path: Optional[Path] = None


class ImageAgent:
    """Agent for image generation tasks."""

    def __init__(
        self,
        config: Optional[ImageConfig] = None,
        api_url: Optional[str] = None,
    ):
        self.config = config or ImageConfig()
        self.api_url = api_url
        self._clients = {}

    def _get_client(self, model: Optional[str] = None):
        """Get appropriate client."""
        model = model or self.config.model

        if model in self._clients:
            return self._clients[model]

        if model == "flux":
            from src.image import FluxConfig
            config = FluxConfig(api_url=self.api_url or "http://localhost:7860")
            client = FluxClient(config)

        elif model == "sdxl":
            from src.image import SDXLConfig
            config = SDXLConfig(api_url=self.api_url or "http://localhost:7860")
            client = SDXLClient(config)

        elif model == "comfyui":
            from src.image import ComfyUIConfig
            config = ComfyUIConfig(api_url=self.api_url or "http://localhost:8188")
            client = ComfyUIClient(config)

        else:  # automatic1111
            from src.image import Automatic1111Config
            config = Automatic1111Config(
                api_url=self.api_url or "http://localhost:7860"
            )
            client = Automatic1111Client(config)

        self._clients[model] = client
        return client

    async def generate(
        self,
        request: GenerationRequest,
    ) -> list[bytes]:
        """Generate image from prompt."""
        if request.style:
            request.prompt = await self._enhance_prompt(
                request.prompt, request.style
            )

        client = self._get_client()

        async with client:
            images = await client.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance=request.guidance,
                seed=request.seed,
            )

        return images

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt for better results."""
        return await self._enhance_prompt(prompt, self.config.style or "photorealistic")

    async def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Internal prompt enhancement."""
        from src.image import get_style_prompt

        return await get_style_prompt(prompt, style)

    async def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        **kwargs,
    ) -> list[bytes]:
        """Generate multiple variations."""
        client = self._get_client()

        images = []
        async with client:
            for i in range(num_variations):
                seed = kwargs.get("seed", -1)
                if seed == -1:
                    seed = i * 1000 + 42

                imgs = await client.generate(prompt=prompt, seed=seed, **kwargs)
                images.extend(imgs)

        return images

    async def regenerate(
        self,
        request: GenerationRequest,
    ) -> list[bytes]:
        """Regenerate with different seed."""
        request.seed = -1  # Force random
        return await self.generate(request)

    async def inpaint(
        self,
        prompt: str,
        image_path: Path,
        mask_path: Path,
    ) -> list[bytes]:
        """Inpaint part of an image."""
        client = self._get_client()

        with open(image_path, "rb") as f:
            image_data = f.read()
        with open(mask_path, "rb") as f:
            mask_data = f.read()

        if hasattr(client, "inpaint"):
            async with client:
                return await client.inpaint(
                    prompt=prompt,
                    image_data=image_data,
                    mask_data=mask_data,
                )

        return []

    async def upscale(
        self,
        image_path: Path,
        scale: int = 2,
    ) -> bytes:
        """Upscale an image."""
        client = self._get_client("automatic1111")

        with open(image_path, "rb") as f:
            image_data = f.read()

        async with client:
            return await client.extra_single_image(
                image_data,
                upscaling_resize=float(scale),
            )

    async def image_to_image(
        self,
        prompt: str,
        image_path: Path,
        strength: float = 0.7,
    ) -> list[bytes]:
        """Transform image based on prompt."""
        client = self._get_client()

        with open(image_path, "rb") as f:
            image_data = f.read()

        if hasattr(client, "img2img"):
            async with client:
                return await client.img2img(
                    prompt=prompt,
                    image_data=image_data,
                    strength=strength,
                )

        return []


async def main():
    parser = argparse.ArgumentParser(description="Image agent")
    parser.add_argument("--task", required=True, help="Image task")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--model", default="flux")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--style", default="photorealistic")
    parser.add_argument("--url", help="API URL")
    args = parser.parse_args()

    config = ImageConfig(
        model=args.model,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        style=args.style,
    )

    agent = ImageAgent(config, api_url=args.url)

    request = GenerationRequest(
        prompt=args.task,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
    )

    if args.output:
        request.output_path = Path(args.output)

    images = await agent.generate(request)

    if args.output and images:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(images[0])

        logger.info(f"Saved: {output_path}")
    elif images:
        logger.info(f"Generated {len(images)} images")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())