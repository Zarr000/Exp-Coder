"""
Multimodal Foundation for Expera AI.

Multimodal capabilities for image understanding, OCR, UI parsing,
image prompting, and cross-modal reasoning.

Modules:
- vision_encoder: Vision encoder abstraction and CLIP interfaces
- image_understanding: Image embedding and understanding pipelines
- ocr: OCR pipeline abstractions
- ui_parser: UI element detection and parsing
- image_prompting: Image prompt generation
- image_editing: Image editing instruction parsing
- cross_attention: Cross-modal attention mechanisms
- embeddings: Unified multimodal embeddings
- benchmarks: Multimodal benchmark suite
"""

from src.multimodal.vision_encoder import (
    VisionEncoder,
    CLIPEncoder,
    ViTEncoder,
    create_vision_encoder,
)
from src.multimodal.image_understanding import (
    ImageUnderstandingPipeline,
    ImageEmbedder,
    create_image_embedder,
)
from src.multimodal.ocr import (
    OCREngine,
    EasyOCREngine,
    TrOCREngine,
    create_ocr_engine,
)
from src.multimodal.ui_parser import (
    UIElementDetector,
    LayoutDetector,
    AccessibilityDetector,
    create_ui_detector,
)
from src.multimodal.image_prompting import (
    ImagePromptGenerator,
    StableDiffusionPromptGenerator,
    FluxPromptGenerator,
    create_prompt_generator,
)
from src.multimodal.image_editing import (
    ImageEditingParser,
    InpaintEditor,
    OutpaintEditor,
    create_editing_parser,
)
from src.multimodal.cross_attention import (
    CrossAttentionModule,
    MultimodalTransformer,
    VisionLanguageModel,
    create_multimodal_model,
)
from src.multimodal.embeddings import (
    MultimodalEmbedder,
    UnifiedEmbeddings,
    create_multimodal_embedder,
)
from src.multimodal.vision import (
    VisionInterface,
    VisionConfig,
    VisionOutput,
    create_vision_interface,
)

__all__ = [
    # Vision encoder
    "VisionEncoder",
    "CLIPEncoder",
    "ViTEncoder",
    "create_vision_encoder",
    # Image understanding
    "ImageUnderstandingPipeline",
    "ImageEmbedder",
    "create_image_embedder",
    # OCR
    "OCREngine",
    "EasyOCREngine",
    "TrOCREngine",
    "create_ocr_engine",
    # UI parsing
    "UIElementDetector",
    "LayoutDetector",
    "AccessibilityDetector",
    "create_ui_detector",
    # Image prompting
    "ImagePromptGenerator",
    "StableDiffusionPromptGenerator",
    "FluxPromptGenerator",
    "create_prompt_generator",
    # Image editing
    "ImageEditingParser",
    "InpaintEditor",
    "OutpaintEditor",
    "create_editing_parser",
    # Cross attention
    "CrossAttentionModule",
    "MultimodalTransformer",
    "VisionLanguageModel",
    "create_multimodal_model",
    # Embeddings
    "MultimodalEmbedder",
    "UnifiedEmbeddings",
    "create_multimodal_embedder",
    # Vision
    "VisionInterface",
    "VisionConfig",
    "VisionOutput",
    "create_vision_interface",
]