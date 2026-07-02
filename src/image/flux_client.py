"""
Flux Image Generation Client.

Interface to Flux text-to-image models.

Usage:
    python -m src.image.flux_client --prompt "a sunset over mountains" --output output.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class FluxConfig:
    """Flux client configuration."""

    api_url: str = "http://localhost:7860"
    api_key: str = ""

    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    seed: int = -1

    batch_size: int = 1
    prompt_prefix: str = "photorealistic, high quality, detailed: "
    prompt_suffix: str = ""


class FluxClient:
    """Client for Flux image generation API."""

    def __init__(self, config: FluxConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    def _build_prompt(self, prompt: str) -> str:
        """Build full prompt with prefix/suffix."""
        parts = [self.config.prompt_prefix, prompt.strip(), self.config.prompt_suffix]
        return "".join(p for p in parts if p)

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
        Generate images from text prompt.

        Args:
            prompt: Text description
            negative_prompt: Things to avoid
            width: Image width
            height: Image height
            num_steps: Number of sampling steps
            guidance_scale: Guidance scale
            seed: Random seed

        Returns:
            List of generated images as bytes
        """
        width = width or self.config.width
        height = height or self.config.height
        steps = steps or self.config.steps
        guidance = guidance or self.config.guidance
        seed = seed if seed != -1 else None

        full_prompt = self._build_prompt(prompt)

        payload = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "batch_size": self.config.batch_size,
        }

        async with self._session.post(
            f"{self.config.api_url}/sdapi/v1/txt2img", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Flux API error: {resp.status} - {text}")

            result = await resp.json()
            images = []

            for b64_image in result.get("images", []):
                image_data = base64.b64decode(b64_image)
                images.append(image_data)

            return images

    async def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
    ) -> list[bytes]:
        """Generate multiple variations of the same prompt."""
        images = []
        for i in range(num_variations):
            seed = i * 1000 + 42
            imgs = await self.generate(prompt, seed=seed)
            images.extend(imgs)
        return images


async def enhance_prompt(prompt: str, model_client=None) -> str:
    """
    Enhance a simple prompt for better image generation.

    The LLM can improve prompts by adding:
    - Quality descriptors
    - Lighting details
    - Composition hints
    - Style information
    """
    enhancements = {
        "sunset": "golden hour, warm lighting, dramatic clouds, orange and pink sky",
        "portrait": "professional lighting, sharp focus, detailed skin texture",
        "landscape": "wide angle, depth of field, natural lighting, dramatic composition",
        "city": "urban perspective, architectural photography, golden hour",
        "forest": "lush vegetation, dappled light, misty atmosphere",
        "ocean": "wave motion blur, dramatic sky, maritime photography",
    }

    prompt_lower = prompt.lower()
    for keyword, enhancement in enhancements.items():
        if keyword in prompt_lower:
            prompt = f"{prompt}, {enhancement}"

    return prompt


def generate_image_edit_prompt(
    task: str,
    current_image: str = "",
    target_style: str = "",
) -> dict[str, str]:
    """
    Generate prompts for image editing workflow.

    Returns dict with 'inpaint' and 'outpaint' prompts.
    """
    prompts = {
        "inpaint": f"replace {task} with {target_style}" if target_style else task,
        "outpaint": f"extend with {task}, seamless continuation",
    }
    return prompts


async def main():
    parser = argparse.ArgumentParser(description="Flux image generation client")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()

    config = FluxConfig(
        api_url=args.url,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    async with FluxClient(config) as client:
        images = await client.generate(
            args.prompt,
            negative_prompt=args.negative,
        )

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