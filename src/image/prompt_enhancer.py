"""
Prompt Enhancer.

Enhances image generation prompts with quality modifiers.
"""

from __future__ import annotations

from typing import Optional


QUALITY_MODIFIERS = [
    "high quality",
    "detailed",
    "professional",
    "masterpiece",
    "8k",
    "highly detailed",
]

STYLE_PRESETS = [
    "cinematic",
    "photorealistic",
    "anime",
    "digital art",
    "oil painting",
    "watercolor",
]

LIGHTING_KEYWORDS = [
    "soft lighting",
    "natural lighting",
    "studio lighting",
    "golden hour",
    "dramatic lighting",
    "rim lighting",
]


class PromptEnhancer:
    """
    Enhances image prompts for better results.

    Adds quality modifiers, style presets, and lighting keywords.
    """

    def __init__(
        self,
        add_quality: bool = True,
        add_style: bool = True,
        add_lighting: bool = True
    ) -> None:
        """Initialize prompt enhancer."""
        self.add_quality = add_quality
        self.add_style = add_style
        self.add_lighting = add_lighting

    def enhance(
        self,
        prompt: str,
        style: Optional[str] = None,
        quality: str = "high"
    ) -> str:
        """Enhance prompt with modifiers."""
        enhanced = prompt

        if self.add_quality:
            if quality == "high":
                enhanced += ", " + ", ".join(QUALITY_MODIFIERS[:3])
            elif quality == "medium":
                enhanced += ", " + QUALITY_MODIFIERS[0]

        if self.add_style:
            style = style or "photorealistic"
            enhanced += ", " + style + " style"

        if self.add_lighting:
            enhanced += ", " + LIGHTING_KEYWORDS[0]

        return enhanced

    def extract_style(self, prompt: str) -> Optional[str]:
        """Extract style preset from prompt."""
        for style in STYLE_PRESETS:
            if style.lower() in prompt.lower():
                return style
        return None

    def remove_style(self, prompt: str) -> str:
        """Remove style preset from prompt."""
        result = prompt
        for style in STYLE_PRESETS:
            result = result.replace(style + " style", "")
            result = result.replace(style, "")
        return result.strip(", ").strip()