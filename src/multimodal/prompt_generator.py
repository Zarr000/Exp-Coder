"""
Prompt Generator for Expera AI.

Generates image prompts:
- Text-to-image prompts
- Style presets
- Quality modifiers
- Subject descriptions

Usage:
    generator = PromptGenerator()
    prompt = generator.generate("a cat", style="anime")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Prompt generation configuration."""

    default_style: str = "realistic"
    default_quality: str = "high"
    default_aspect: str = "square"
    include_quality_modifiers: bool = True
    max_length: int = 200


# Style presets
STYLE_PRESETS = {
    "realistic": "photorealistic, detailed, 8k, professional photography",
    "anime": "anime style, manga art, vibrant colors, cel shaded",
    "digital-art": "digital painting, illustration, concept art, artstation",
    "oil-painting": "oil painting, painterly, brushstrokes, traditional art",
    "watercolor": "watercolor painting, soft colors, flowing, delicate",
    "3d-render": "3d render, cgi, octane render, unreal engine",
    "pixel-art": "pixel art, 8-bit, retro game, chiptune",
    "sketch": "sketch, pencil drawing, hand drawn, study",
    "comic": "comic book style, panels, bold lines, vibrant",
    "abstract": "abstract art, abstract composition, patterns",
    "fantasy": "fantasy art, magical, ethereal, detailed",
    "sci-fi": "sci-fi, futuristic, high tech, detailed",
    "portrait": "portrait, head and shoulders, detailed face, professional lighting",
    "landscape": "landscape, scenic, nature, beautiful lighting",
    "architecture": "architecture, building, architectural photography, detailed",
}

# Quality modifiers
QUALITY_MODIFIERS = {
    "low": "good quality",
    "medium": "high quality, detailed",
    "high": "very detailed, 4k, high resolution",
    "ultra": "ultra detailed, 8k, photorealistic, masterpiece",
    "best": "best quality, masterpiece, extremely detailed, 8k, photorealistic",
}

# Aspect ratios
ASPECT_RATIOS = {
    "square": "1:1",
    "portrait": "3:4",
    "landscape": "4:3",
    "wide": "16:9",
    "ultrawide": "21:9",
    "phone": "9:16",
}

# Negative prompt defaults
NEGATIVE_PROMPT = (
    "low quality, blurry, deformed, disfigured, bad anatomy, "
    "bad hands, missing fingers, extra fingers, "
    "text, watermark, signature"
)


class PromptGenerator:
    """
    Generates image prompts.

    Features:
    - Style templates
    - Quality modifiers
    - Aspect ratio
    - Negative prompts
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        """Initialize prompt generator."""
        self.config = config or PromptConfig()
        self.styles = STYLE_PRESETS.copy()
        self.quality = QUALITY_MODIFIERS.copy()

    def generate(
        self,
        subject: str,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        aspect: Optional[str] = None,
        add_keywords: Optional[list[str]] = None,
    ) -> str:
        """Generate a prompt."""
        parts = [subject]

        # Add style
        style = style or self.config.default_style
        if style in self.styles:
            parts.append(self.styles[style])

        # Add quality
        quality = quality or self.config.default_quality
        if self.config.include_quality_modifiers and quality in self.quality:
            parts.append(self.quality[quality])

        # Add custom keywords
        if add_keywords:
            parts.extend(add_keywords)

        # Combine
        prompt = ", ".join(parts)

        # Trim to max length
        if len(prompt) > self.config.max_length:
            prompt = prompt[:self.config.max_length].rsplit(", ", 1)[0]

        return prompt

    def generate_with_composition(
        self,
        subject: str,
        composition: str = "centered",
        lighting: str = "natural",
        environment: str = "studio",
    ) -> str:
        """Generate with composition details."""
        parts = [subject]

        # Composition
        composition_map = {
            "centered": "centered subject",
            "full-body": "full body shot",
            "close-up": "close-up view",
            "wide": "wide shot",
            "over-shoulder": "over the shoulder view",
            "first-person": "first person view",
            "bird-view": "bird's eye view",
            "worm-view": "worm's eye view",
        }
        if composition in composition_map:
            parts.append(composition_map[composition])

        # Lighting
        lighting_map = {
            "natural": "natural lighting",
            "studio": "studio lighting",
            "golden-hour": "golden hour lighting",
            "blue-hour": "blue hour lighting",
            "cinematic": "cinematic lighting",
            "rim-light": "rim lighting",
            "soft-light": "soft lighting",
            "dramatic": "dramatic lighting",
        }
        if lighting in lighting_map:
            parts.append(lighting_map[lighting])

        # Environment
        env_map = {
            "studio": "in studio",
            "outdoor": "outdoor location",
            "indoor": "indoor location",
            "nature": "in nature",
            "urban": "urban environment",
            " void": "in void",
            "neutral": "neutral background",
        }
        if environment in env_map:
            parts.append(env_map[environment])

        return ", ".join(parts)

    def generate_negative(self, additions: Optional[list[str]] = None) -> str:
        """Generate negative prompt."""
        parts = [NEGATIVE_PROMPT]

        if additions:
            parts.extend(additions)

        return ", ".join(parts)

    def get_styles(self) -> list[str]:
        """Get available styles."""
        return list(self.styles.keys())

    def get_qualities(self) -> list[str]:
        """Get available qualities."""
        return list(self.quality.keys())

    def get_aspects(self) -> list[str]:
        """Get available aspect ratios."""
        return list(ASPECT_RATIOS.keys())


# Export
__all__ = [
    "PromptGenerator",
    "PromptConfig",
    "NEGATIVE_PROMPT",
]