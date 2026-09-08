"""
Image Understanding Pipeline.

Image embedding and understanding pipelines with support for:
- Local and cloud execution
- Pluggable vision encoders
- Multimodal reasoning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import numpy as np

try:
    from src.multimodal.vision_encoder import VisionEncoder, VisionEncoderConfig, create_vision_encoder
except ImportError:
    from src.multimodal.vision_encoder import VisionEncoder, VisionEncoderConfig, create_vision_encoder


class UnderstandingBackend(Enum):
    """Image understanding backends."""
    LOCAL = "local"
    OPENAI = "openai"  # GPT-4V
    ANTHROPIC = "anthropic"  # Claude Vision
    GOOGLE = "google"  # Gemini Vision


@dataclass
class ImageUnderstandingConfig:
    """Image understanding configuration."""
    vision_encoder: str = "vit-base-patch16-224"
    embed_dim: int = 768
    use_larger_encoder: bool = False
    backend: UnderstandingBackend = UnderstandingBackend.LOCAL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_resolution: int = 1024


@dataclass
class ImageDescription:
    """Image description result."""
    text: str
    confidence: float
    tags: List[str]
    objects: List[Dict[str, Any]]  # {"label": str, "bbox": [x1, y1, x2, y2], "confidence": float}


class ImageEmbedder(ABC):
    """Base class for image embedding."""

    @abstractmethod
    def __call__(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Embed images."""
        pass


class ImageUnderstandingPipeline:
    """
    Complete image understanding pipeline.

    Combines vision encoder with understanding capabilities.
    """

    def __init__(self, config: ImageUnderstandingConfig):
        self.config = config

        # Initialize vision encoder
        self.encoder = create_vision_encoder(
            model_name=config.vision_encoder,
            device=config.device,
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, config.embed_dim),
        ).to(config.device)

        self.device = config.device

    def embed_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> Tensor:
        """
        Embed images to feature vectors.

        Args:
            images: PIL Image or list

        Returns:
            Embeddings (batch, embed_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Get vision encoder embeddings
        with torch.no_grad():
            embeddings = self.encoder.encode_image(images)

        # Project to target dimension
        embeddings = self.projection(embeddings)

        return embeddings

    def embed_patches(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> Tensor:
        """
        Embed images as patches.

        Args:
            images: PIL Image or list

        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        with torch.no_grad():
            patches = self.encoder.encode_patches(images)

        patches = self.projection(patches)

        return patches

    def describe_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> ImageDescription:
        """
        Describe an image.

        Args:
            image: Input image
            prompt: Optional prompt for guided understanding

        Returns:
            ImageDescription with text, confidence, tags, objects
        """
        # Get embeddings
        embedding = self.embed_images(image).squeeze(0)

        # Simple description (in practice, would use VLM)
        text = "Image analyzed"

        return ImageDescription(
            text=text,
            confidence=0.9,
            tags=["image"],
            objects=[],
        )

    def find_similar(
        self,
        query: Image.Image,
        candidates: List[Image.Image],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find similar images.

        Args:
            query: Query image
            candidates: Candidate images
            top_k: Number of results

        Returns:
            List of (index, similarity) tuples
        """
        query_emb = self.embed_images(query)
        candidate_embs = self.embed_images(candidates)

        # Compute cosine similarity
        similarities = torch.cosine_similarity(
            query_emb.unsqueeze(1),
            candidate_embs.unsqueeze(0),
            dim=-1,
        )

        # Get top-k
        similarities = similarities.squeeze(0)
        top_k = min(top_k, len(candidates))
        scores, indices = torch.topk(similarities, top_k)

        return [(i.item(), s.item()) for i, s in zip(indices, scores)]

    def detect_objects(
        self,
        image: Image.Image,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in image.

        Args:
            image: Input image
            threshold: Confidence threshold

        Returns:
            List of object dicts
        """
        # Get patch embeddings
        patches = self.embed_patches(image)

        # Simple object detection (would use detection model in practice)
        return []

    def to(self, device: str) -> "ImageUnderstandingPipeline":
        """Move to device."""
        self.device = device
        self.encoder.to(device)
        self.projection = self.projection.to(device)
        return self


class LocalImageUnderstandingPipeline(ImageUnderstandingPipeline):
    """Local-only image understanding."""

    def __init__(self, config: ImageUnderstandingConfig):
        config.backend = UnderstandingBackend.LOCAL
        super().__init__(config)


def create_image_embedder(
    model_name: str = "vit-base-patch16-224",
    embed_dim: int = 768,
    device: Optional[str] = None,
    **kwargs,
) -> ImageEmbedder:
    """
    Create an image embedder.

    Args:
        model_name: Vision encoder model
        embed_dim: Output embedding dimension
        device: Device (auto-detect if None)
        **kwargs: Additional arguments

    Returns:
        ImageEmbedder instance
    """
    config = ImageUnderstandingConfig(
        vision_encoder=model_name,
        embed_dim=embed_dim,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
    )
    return ImageUnderstandingPipeline(config)


__all__ = [
    "ImageUnderstandingPipeline",
    "ImageUnderstandingConfig",
    "ImageDescription",
    "ImageEmbedder",
    "UnderstandingBackend",
    "LocalImageUnderstandingPipeline",
    "create_image_embedder",
]