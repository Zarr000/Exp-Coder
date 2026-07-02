"""
Image Pipeline.

Multi-step image generation with enhancements.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from .image_generator import ImageBackend, ImageGenerator, GenerationRequest, GenerationResult
from .prompt_enhancer import PromptEnhancer
from .image_upscaler import ImageUpscaler


@dataclass
class PipelineConfig:
    """Image pipeline configuration."""

    enhance_prompt: bool = True
    upscale: bool = False
    upscale_factor: int = 2
    enhance_face: bool = False
    remove_background: bool = False
    num_variants: int = 1


@dataclass
class PipelineResult:
    """Image pipeline result."""

    images: list[bytes]
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""


class ImagePipeline:
    """
    Multi-step image generation pipeline.

    Combines prompt enhancement, generation, and post-processing.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize image pipeline."""
        self.config = config or PipelineConfig()
        self.generator = ImageGenerator()
        self.enhancer = PromptEnhancer()
        self.upscaler = ImageUpscaler() if self.config.upscale else None

    async def generate(
        self,
        prompt: str,
        backend: ImageBackend = ImageBackend.FLUX,
        seed: Optional[int] = None
    ) -> PipelineResult:
        """Generate image with full pipeline."""
        final_prompt = prompt

        if self.config.enhance_prompt:
            enhanced = await self.enhancer.enhance(prompt)
            final_prompt = enhanced
            await asyncio.sleep(0)

        image_result = await self.generator.generate(
            GenerationRequest(
                prompt=final_prompt,
                backend=backend,
                seed=seed,
                batch_size=self.config.num_variants,
            )
        )

        images = image_result.images

        if self.config.upscale and self.upscaler and images:
            upscaled = await self.upscaler.upscale(
                images[0],
                scale=self.config.upscale_factor
            )
            images = [upscaled]

        metadata = {
            "original_prompt": prompt,
            "enhanced_prompt": final_prompt,
            "backend": backend.value,
            "config": {
                "enhance_prompt": self.config.enhance_prompt,
                "upscale": self.config.upscale,
                "upscale_factor": self.config.upscale_factor,
            }
        }

        return PipelineResult(
            images=images,
            metadata=metadata,
            prompt=final_prompt
        )

    async def generate_variants(
        self,
        prompt: str,
        num_variants: int = 4,
        backend: ImageBackend = ImageBackend.FLUX
    ) -> list[PipelineResult]:
        """Generate multiple variants of the same prompt."""
        tasks = []
        for i in range(num_variants):
            seed = 42 + i
            task = self.generate(prompt, backend, seed)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def inpaint(
        self,
        image: bytes,
        mask: bytes,
        prompt: str,
        backend: ImageBackend = ImageBackend.COMFYUI
    ) -> bytes:
        """Inpaint image with mask."""
        from .inpainting import Inpainter
        inpainter = Inpainter()
        return await inpainter.inpaint(image, mask, prompt)

    async def outpaint(
        self,
        image: bytes,
        direction: str = "right",
        pixels: int = 512
    ) -> bytes:
        """Outpaint image in specified direction."""
        from .outpainting import Outpainter
        outpainter = Outpainter()
        return await outpainter.outpaint(image, direction, pixels)

    async def edit_with_controlnet(
        self,
        image: bytes,
        control_image: bytes,
        control_type: str = "canny",
        prompt: str = ""
    ) -> bytes:
        """Edit image with ControlNet."""
        from .controlnet import ControlNet
        controlnet = ControlNet()
        return await controlnet.apply_control(image, control_image, control_type, prompt)