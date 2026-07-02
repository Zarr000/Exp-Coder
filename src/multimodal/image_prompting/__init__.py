"""
Image Prompting for Expera AI.

Image prompt generation for:
- Stable Diffusion
- Flux
- Midjourney
- DALL-E
- Future text-to-image models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

import torch
import re


class PromptBackend(Enum):
    """Image prompting backends."""
    STABLE_DIFFUSION = "stable_diffusion"
    FLUX = "flux"
    MIDJOURNEY = "midjourney"
    DALLE = "dalle"


@dataclass
class ImagePromptConfig:
    """Image prompt configuration."""
    backend: PromptBackend = PromptBackend.STABLE_DIFFUSION
    max_length: int = 77  # CLIP token limit
    quality: str = "high"
    style: Optional[str] = None
    aspect_ratio: str = "1:1"  # 1:1, 16:9, 9:16, 4:3, 3:4
    seed: Optional[int] = None


@dataclass
class GeneratedPrompt:
    """Generated image prompt."""
    prompt: str
    negative_prompt: str
    parameters: Dict[str, Any]
    tags: List[str] = field(default_factory=list)


class ImagePromptGenerator(ABC):
    """Base class for image prompt generation."""

    @abstractmethod
    def generate(
        self,
        description: str,
        **kwargs,
    ) -> GeneratedPrompt:
        """
        Generate image prompt from description.

        Args:
            description: Text description
            **kwargs: Additional parameters

        Returns:
            GeneratedPrompt
        """
        pass


class StableDiffusionPromptGenerator(ImagePromptGenerator):
    """
    Stable Diffusion prompt generator.

    Generates prompts optimized for Stable Diffusion.
    """

    def __init__(self, config: ImagePromptConfig):
        self.config = config

        # Quality modifiers
        self.quality_modifiers = {
            "low": "",
            "medium": ", detailed, clear",
            "high": ", highly detailed, intricate, 4k, masterpiece",
            "ultra": ", photorealistic, 8k, ultra detailed, cinematic lighting",
        }

        # Style presets
        self.style_presets = {
            "anime": ", anime style, manga",
            "photorealistic": ", photorealistic, realistic, 8k",
            "oil_painting": ", oil painting, painterly",
            "watercolor": ", watercolor, painting",
            "digital_art": ", digital art, concept art",
            "3d_render": ", 3d render, cgi, blender",
            "pixel_art": ", pixel art, 8-bit",
            "illustration": ", illustration, vector art",
        }

        # Aspect ratio mappings
        self.aspect_ratios = {
            "1:1": (512, 512),
            "16:9": (768, 432),
            "9:16": (432, 768),
            "4:3": (512, 384),
            "3:4": (384, 512),
            "21:9": (832, 352),
        }

    def generate(
        self,
        description: str,
        quality: Optional[str] = None,
        style: Optional[str] = None,
        **kwargs,
    ) -> GeneratedPrompt:
        """Generate Stable Diffusion prompt."""
        quality = quality or self.config.quality
        style = style or self.config.style

        # Build positive prompt
        parts = [description]

        # Add quality modifier
        if quality in self.quality_modifiers:
            parts.append(self.quality_modifiers[quality])

        # Add style
        if style and style in self.style_presets:
            parts.append(self.style_presets[style])

        prompt = "".join(parts)

        # Truncate to max length
        if len(prompt) > self.config.max_length * 4:  # Approximate token chars
            prompt = prompt[: self.config.max_length * 4]

        # Negative prompt
        negative_prompt = (
            "low quality, blurry, distorted, deformed, "
            "ugly, bad anatomy, watermark, text"
        )

        # Get resolution
        width, height = self.aspect_ratios.get(
            self.config.aspect_ratio, (512, 512)
        )

        parameters = {
            "width": width,
            "height": height,
            "seed": self.config.seed,
            "steps": 25 if quality == "high" else 20,
            "guidance_scale": 7.5,
        }

        # Extract tags
        tags = self._extract_tags(prompt)

        return GeneratedPrompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            parameters=parameters,
            tags=tags,
        )

    def _extract_tags(self, prompt: str) -> List[str]:
        """Extract tags from prompt."""
        # Simple extraction
        tags = []
        keywords = [
            "photorealistic", "anime", "painting", "3d", "digital",
            "detailed", "intricate", "cinematic", "portrait", "landscape",
        ]
        for kw in keywords:
            if kw.lower() in prompt.lower():
                tags.append(kw)
        return tags


class FluxPromptGenerator(ImagePromptGenerator):
    """
    Flux prompt generator.

    Generates prompts for Flux models.
    """

    def __init__(self, config: ImagePromptConfig):
        self.config = config

        # Flux quality modifiers
        self.quality_modifiers = {
            "low": "",
            "medium": ", sharp focus",
            "high": ", high detail, sharp focus, detailed",
            "ultra": ", highest detail, film grain, photorealistic",
        }

    def generate(
        self,
        description: str,
        **kwargs,
    ) -> GeneratedPrompt:
        """Generate Flux prompt."""
        quality = kwargs.get("quality", self.config.quality)

        parts = [description]
        if quality in self.quality_modifiers:
            parts.append(self.quality_modifiers[quality])

        prompt = "".join(parts)

        negative_prompt = "blurry, low quality, distorted"

        parameters = {
            "width": 512,
            "height": 512,
            "seed": self.config.seed,
            "steps": 20,
            "guidance": 3.5,
        }

        return GeneratedPrompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            parameters=parameters,
            tags=description.split()[:5],
        )


class MidjourneyPromptGenerator(ImagePromptGenerator):
    """
    Midjourney prompt generator.

    Generates prompts in Midjourney style.
    """

    def __init__(self, config: ImagePromptConfig):
        self.config = config

        # Midjourney parameters
        self.versions = ["v5", "v6", "niji"]
        self.params = {
            "--ar": "aspect ratio",
            "--v": "version",
            "--q": "quality",
            "--s": "stylize",
            "--iw": "image weight",
            "--no": "negative",
        }

    def generate(
        self,
        description: str,
        version: str = "v6",
        aspect_ratio: str = "1:1",
        **kwargs,
    ) -> GeneratedPrompt:
        """Generate Midjourney prompt."""
        # Build prompt
        parts = [description]

        # Add parameters
        params_parts = []
        params_parts.append(f"--ar {aspect_ratio}")
        params_parts.append(f"--v {version}")

        if "quality" in kwargs:
            params_parts.append(f"--q {kwargs['quality']}")
        if "stylize" in kwargs:
            params_parts.append(f"--s {kwargs['stylize']}")

        prompt = " ".join(parts)
        params_str = " ".join(params_parts)

        full_prompt = f"{prompt} {params_str}"

        negative_prompt = kwargs.get("negative", "")

        parameters = {
            "aspect_ratio": aspect_ratio,
            "version": version,
            "raw_params": params_parts,
        }

        return GeneratedPrompt(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            parameters=parameters,
            tags=[aspect_ratio, version],
        )


class DALLEPromptGenerator(ImagePromptGenerator):
    """
    DALL-E prompt generator.

    Generates prompts for DALL-E.
    """

    def __init__(self, config: ImagePromptConfig):
        self.config = config

        self.sizes = {
            "256": "256x256",
            "512": "512x512",
            "1024": "1024x1024",
        }

    def generate(
        self,
        description: str,
        size: str = "1024",
        **kwargs,
    ) -> GeneratedPrompt:
        """Generate DALL-E prompt."""
        prompt = description

        # DALL-E doesn't use negative prompts
        negative_prompt = ""

        parameters = {
            "size": self.sizes.get(size, "1024x1024"),
        }

        return GeneratedPrompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            parameters=parameters,
            tags=[],
        )


class PromptOptimizer:
    """
    Prompt optimization utilities.

    Enhances prompts for better results.
    """

    @staticmethod
    def enhance(prompt: str) -> str:
        """Enhance a basic prompt."""
        enhancements = [
            "detailed",
            "high quality",
            "professional photography",
        ]

        # Check if already has quality terms
        prompt_lower = prompt.lower()
        for enh in enhancements:
            if enh not in prompt_lower:
                prompt = f"{prompt}, {enh}"

        return prompt

    @staticmethod
    def remove_redundant(text: str) -> str:
        """Remove redundant words."""
        # Simple deduplication
        words = text.split(", ")
        unique = []
        seen = set()

        for word in words:
            word_lower = word.lower().strip()
            if word_lower not in seen:
                unique.append(word)
                seen.add(word_lower)

        return ", ".join(unique)

    @staticmethod
    def fix_spacing(text: str) -> str:
        """Fix spacing issues."""
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing comma spaces
        text = re.sub(r",\s+", ",", text)
        text = re.sub(r"\s+,", ",", text)
        return text.strip()


def create_prompt_generator(
    backend: PromptBackend = PromptBackend.STABLE_DIFFUSION,
    **kwargs,
) -> ImagePromptGenerator:
    """
    Create an image prompt generator.

    Args:
        backend: Prompt backend
        **kwargs: Additional arguments

    Returns:
        ImagePromptGenerator instance
    """
    config = ImagePromptConfig(backend=backend, **kwargs)

    if backend == PromptBackend.STABLE_DIFFUSION:
        return StableDiffusionPromptGenerator(config)
    elif backend == PromptBackend.FLUX:
        return FluxPromptGenerator(config)
    elif backend == PromptBackend.MIDJOURNEY:
        return MidjourneyPromptGenerator(config)
    elif backend == PromptBackend.DALLE:
        return DALLEPromptGenerator(config)
    else:
        return StableDiffusionPromptGenerator(config)


__all__ = [
    "ImagePromptGenerator",
    "ImagePromptConfig",
    "GeneratedPrompt",
    "PromptBackend",
    "StableDiffusionPromptGenerator",
    "FluxPromptGenerator",
    "MidjourneyPromptGenerator",
    "DALLEPromptGenerator",
    "PromptOptimizer",
    "create_prompt_generator",
]