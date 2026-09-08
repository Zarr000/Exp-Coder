"""
Image Tools for Expera AI.

Image operations:
- Generate
- Edit
- Upscale
- Convert

Usage:
    tools = ImageTools()
    result = await tools.generate("a cat", style="anime")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageOperationResult:
    """Result of image operation."""

    success: bool
    message: str
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None


class ImageTools:
    """
    Image operations.

    Features:
    - Generate images
    - Edit images
    - Upscale
    - Convert formats
    """

    def __init__(
        self,
        generator=None,
        endpoint: Optional[str] = None,
    ):
        """Initialize image tools."""
        self.generator = generator
        self.endpoint = endpoint

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.0,
        save_path: Optional[str] = None,
    ) -> ImageOperationResult:
        """Generate image from prompt."""
        try:
            # Import ImageGenerator if not provided
            if not self.generator:
                from ..image_generator import ImageGenerator

                self.generator = ImageGenerator(self.endpoint)

            result = await self.generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
            )

            images = result.images
            if not images:
                return ImageOperationResult(
                    success=False,
                    message="No images generated",
                )

            image_data = images[0]

            # Save if path provided
            if save_path:
                Path(save_path).write_bytes(image_data)
                return ImageOperationResult(
                    success=True,
                    message=f"Generated and saved to {save_path}",
                    image_path=save_path,
                )

            return ImageOperationResult(
                success=True,
                message="Generated image",
                image_data=image_data,
            )

        except Exception as e:
            logger.error(f"Generate failed: {e}")
            return ImageOperationResult(
                success=False,
                message=str(e),
            )

    async def edit(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
        save_path: Optional[str] = None,
    ) -> ImageOperationResult:
        """Edit image (img2img)."""
        try:
            if not self.generator:
                from ..image_generator import ImageGenerator

                self.generator = ImageGenerator(self.endpoint)

            # Load image
            image_data = Path(image_path).read_bytes()

            result = await self.generator.img2img(
                init_image=image_data,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
            )

            images = result.images
            if not images:
                return ImageOperationResult(
                    success=False,
                    message="No images generated",
                )

            edited_data = images[0]

            if save_path:
                Path(save_path).write_bytes(edited_data)
                return ImageOperationResult(
                    success=True,
                    message=f"Edited and saved to {save_path}",
                    image_path=save_path,
                )

            return ImageOperationResult(
                success=True,
                message="Edited image",
                image_data=edited_data,
            )

        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return ImageOperationResult(
                success=False,
                message=str(e),
            )

    async def upscale(
        self,
        image_path: str,
        model: str = "realesrgan-x4",
        save_path: Optional[str] = None,
    ) -> ImageOperationResult:
        """Upscale image."""
        try:
            from ..image_upscaler import ImageUpscaler

            upscaler = ImageUpscaler(self.endpoint)

            image_data = Path(image_path).read_bytes()
            upscaled_data = await upscaler.upscale(image_data, model)

            if save_path:
                Path(save_path).write_bytes(upscaled_data)
                return ImageOperationResult(
                    success=True,
                    message=f"Upscaled and saved to {save_path}",
                    image_path=save_path,
                )

            return ImageOperationResult(
                success=True,
                message="Upscaled image",
                image_data=upscaled_data,
            )

        except Exception as e:
            logger.error(f"Upscale failed: {e}")
            return ImageOperationResult(
                success=False,
                message=str(e),
            )

    def convert(
        self,
        image_path: str,
        output_format: str = "png",
        save_path: Optional[str] = None,
    ) -> ImageOperationResult:
        """Convert image format."""
        try:
            from PIL import Image

            img = Image.open(image_path)

            if not save_path:
                save_path = str(
                    Path(image_path).with_suffix(f".{output_format}")
                )

            img.save(save_path, format=output_format.upper())

            return ImageOperationResult(
                success=True,
                message=f"Converted to {output_format}",
                image_path=save_path,
            )

        except ImportError:
            return ImageOperationResult(
                success=False,
                message="PIL not available",
            )
        except Exception as e:
            logger.error(f"Convert failed: {e}")
            return ImageOperationResult(
                success=False,
                message=str(e),
            )

    def resize(
        self,
        image_path: str,
        width: int,
        height: int,
        save_path: Optional[str] = None,
    ) -> ImageOperationResult:
        """Resize image."""
        try:
            from PIL import Image

            img = Image.open(image_path)
            img = img.resize((width, height), Image.LANCZOS)

            if not save_path:
                save_path = image_path

            img.save(save_path)

            return ImageOperationResult(
                success=True,
                message=f"Resized to {width}x{height}",
                image_path=save_path,
            )

        except ImportError:
            return ImageOperationResult(
                success=False,
                message="PIL not available",
            )
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            return ImageOperationResult(
                success=False,
                message=str(e),
            )


# Export
__all__ = [
    "ImageTools",
    "ImageOperationResult",
]