"""
SDXL Image Generation Client.

Interface to Stable Diffusion XL models.

Usage:
    python -m src.image.sdxl_client --prompt "a cat sitting on a bench" --output output.png
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
class SDXLConfig:
    """SDXL client configuration."""

    api_url: str = "http://localhost:7860"

    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 7.5
    seed: int = -1

    batch_size: int = 1

    # SDXL-specific
    refiner_steps: int = 10
    has_refiner: bool = True


class SDXLClient:
    """Client for SDXL image generation API."""

    def __init__(self, config: SDXLConfig):
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
        use_refiner: bool = True,
    ) -> list[bytes]:
        """
        Generate images using SDXL.

        Args:
            prompt: Text description
            negative_prompt: Things to avoid
            width: Image width (1024 for SDXL)
            height: Image height (1024 for SDXL)
            steps: Number of sampling steps
            guidance: Guidance scale
            seed: Random seed
            use_refiner: Use refiner model

        Returns:
            List of generated images as bytes
        """
        width = width or self.config.width
        height = height or self.config.height
        steps = steps or self.config.steps
        guidance = guidance or self.config.guidance
        seed = seed if self.config.seed != -1 else None

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "batch_size": self.config.batch_size,
            "enable_refiner": use_refiner and self.config.has_refiner,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/txt2img", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"SDXL API error: {resp.status} - {text}")

            result = await resp.json()
            images = []

            for b64_image in result.get("images", []):
                image_data = base64.b64decode(b64_image)
                images.append(image_data)

            return images

    async def generate_with_style(
        self,
        prompt: str,
        style: str = "photorealistic",
    ) -> list[bytes]:
        """Generate with a specific style preset."""
        style_prompts = {
            "photorealistic": "photo quality, realistic, detailed, 4k",
            "anime": "anime style, cel shaded, vibrant colors, manga",
            "illustration": "illustration, artistic, colorful, detailed",
            "oil_painting": "oil painting style, brushstrokes, classical",
            "watercolor": "watercolor painting, soft colors, flowing",
            "3d": "3d render, blender, octane, detailed, unreal engine",
        }

        full_prompt = f"{prompt}, {style_prompts.get(style, '')}"
        return await self.generate(full_prompt)

    async def ip_adapter(
        self,
        prompt: str,
        image_path: str,
        strength: float = 0.7,
    ) -> list[bytes]:
        """Generate with IP-Adapter reference image."""
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "prompt": prompt,
            "image": image_b64,
            "ip_adapter_strength": strength,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/ip_adapter", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"IP-Adapter error: {resp.status} - {text}")

            result = await resp.json()
            return [base64.b64decode(img) for img in result.get("images", [])]


style_presets = {
    "photorealistic": "cinematic photo, detailed, realistic, sharp, 8k",
    "anime": "anime, manga style, vibrant, cel shaded",
    "digital_art": "digital art, intricate details, vibrant colors",
    "oil_painting": "oil painting, visible brushstrokes, classical art",
    "watercolor": "watercolor painting, soft, flowing, delicate",
    "concept_art": "concept art, detailed, sci-fi, fantasy",
    "3d_render": "3d render, blender, octane, unreal engine, detailed",
    "minimalist": "minimalist, clean, simple, modern design",
}


async def get_style_prompt(prompt: str, style: str) -> str:
    """Get prompt with style enhancement."""
    style_prompt = style_presets.get(style, "")
    if style_prompt:
        return f"{prompt}, {style_prompt}"
    return prompt


async def main():
    parser = argparse.ArgumentParser(description="SDXL image generation client")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--style", default="photorealistic", help="Style preset")
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()

    config = SDXLConfig(
        api_url=args.url,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    async with SDXLClient(config) as client:
        if args.style != "none":
            prompt = await get_style_prompt(args.prompt, args.style)
        else:
            prompt = args.prompt

        images = await client.generate(prompt, negative_prompt=args.negative)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            for i, image_data in enumerate(images):
                if len(images) > 1:
                    output_file = output_path.with_stem(f"{output_path.stem}_{i}")
                else:
                    output_file = output_path

                with open(output_file, "wb") as f:
                    f.write(image_data)
                logger.info(f"Saved: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())