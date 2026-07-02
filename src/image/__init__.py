"""
Image System Integration for Expera AI.

Clients for external image generation models:
- Flux
- SDXL
- ComfyUI
- Automatic1111

The LLM acts as:
- Prompt generator
- Editing planner
- Workflow controller
"""

from src.image.flux_client import FluxClient, FluxConfig, enhance_prompt, generate_image_edit_prompt
from src.image.sdxl_client import (
    SDXLClient,
    SDXLConfig,
    get_style_prompt,
    style_presets,
)
from src.image.comfyui_client import (
    ComfyUIClient,
    ComfyUIConfig,
    inpaint_workflow,
    text_to_image_workflow,
    upscaling_workflow,
    image_to_image_workflow,
)
from src.image.automatic1111_client import (
    Automatic1111Client,
    Automatic1111Config,
)

__all__ = [
    # Flux
    "FluxClient",
    "FluxConfig",
    "enhance_prompt",
    "generate_image_edit_prompt",
    # SDXL
    "SDXLClient",
    "SDXLConfig",
    "get_style_prompt",
    "style_presets",
    # ComfyUI
    "ComfyUIClient",
    "ComfyUIConfig",
    "text_to_image_workflow",
    "inpaint_workflow",
    "upscaling_workflow",
    "image_to_image_workflow",
    # Automatic1111
    "Automatic1111Client",
    "Automatic1111Config",
]