"""
Unified Multimodal Embeddings for Expera AI.

Unified embedding space for:
- Text
- Images
- Audio
- Cross-modal retrieval
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import numpy as np

try:
    from src.multimodal.vision_encoder import create_vision_encoder
except ImportError:
    from src.multimodal.vision_encoder import create_vision_encoder


@dataclass
class MultimodalEmbedderConfig:
    """Multimodal embedder configuration."""
    text_dim: int = 768
    image_dim: int = 768
    audio_dim: int = 768
    output_dim: int = 768
    normalize: bool = True
    temperature: float = 0.07
    use_projection: bool = True


class MultimodalEmbedder(nn.Module):
    """
    Unified multimodal embedder.

    Projects different modalities into a shared embedding space.
    """

    def __init__(self, config: MultimodalEmbedderConfig):
        super().__init__()

        self.config = config

        # Text projection
        if config.use_projection:
            self.text_projection = nn.Sequential(
                nn.Linear(config.text_dim, config.output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.output_dim, config.output_dim),
            )
        else:
            self.text_projection = nn.Identity()

        # Image projection
        if config.use_projection:
            self.image_projection = nn.Sequential(
                nn.Linear(config.image_dim, config.output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.output_dim, config.output_dim),
            )
        else:
            self.image_projection = nn.Identity()

        # Audio projection
        if config.use_projection:
            self.audio_projection = nn.Sequential(
                nn.Linear(config.audio_dim, config.output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.output_dim, config.output_dim),
            )
        else:
            self.audio_projection = nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        for proj in [
            self.text_projection,
            self.image_projection,
            self.audio_projection,
        ]:
            if hasattr(proj, "weight"):
                nn.init.xavier_uniform_(proj.weight)
            elif isinstance(proj, nn.Sequential):
                for module in proj:
                    if hasattr(module, "weight") and isinstance(
                        module, (nn.Linear, nn.Conv2d)
                    ):
                        nn.init.xavier_uniform_(module.weight)

    def embed_text(self, texts: Union[str, List[str]]) -> Tensor:
        """
        Embed text.

        Args:
            texts: Text or list of texts

        Returns:
            Text embeddings (batch, output_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Placeholder - would use text embedding model
        # Using learned embeddings as placeholder
        embeddings = torch.randn(
            len(texts), self.config.text_dim
        )
        embeddings = embeddings.to(next(self.parameters()).device if list(self.parameters()) else "cpu")

        return self.text_projection(embeddings)

    def embed_image(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """
        Embed images.

        Args:
            images: PIL Image or list

        Returns:
            Image embeddings (batch, output_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Would use vision encoder
        # Placeholder embeddings
        embeddings = torch.randn(
            len(images), self.config.image_dim
        )
        embeddings = embeddings.to(next(self.parameters()).device if list(self.parameters()) else "cpu")

        return self.image_projection(embeddings)

    def embed_audio(self, audio: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Embed audio.

        Args:
            audio: Audio tensor/array

        Returns:
            Audio embeddings (batch, output_dim)
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        embeddings = audio.float()
        if audio.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return self.audio_projection(embeddings)

    def forward(
        self,
        text: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            text: Optional text embeddings
            image: Optional image embeddings
            audio: Optional audio embeddings

        Returns:
            Dict of embeddings
        """
        outputs = {}

        if text is not None:
            outputs["text"] = self.text_projection(text)

        if image is not None:
            outputs["image"] = self.image_projection(image)

        if audio is not None:
            outputs["audio"] = self.audio_projection(audio)

        return outputs


class UnifiedEmbeddings:
    """
    Unified embeddings manager.

    Manages embeddings across modalities with cross-modal retrieval.
    """

    def __init__(self, config: MultimodalEmbedderConfig):
        self.config = config
        self.embedder = MultimodalEmbedder(config)

        # Storage for indexed embeddings
        self.text_embeddings: Dict[str, Tensor] = {}
        self.image_embeddings: Dict[str, Tensor] = {}
        self.audio_embeddings: Dict[str, Tensor] = {}

    def add_text(self, key: str, text: str) -> Tensor:
        """Add text embedding."""
        emb = self.embedder.embed_text(text)
        self.text_embeddings[key] = emb
        return emb

    def add_image(self, key: str, image: Image.Image) -> Tensor:
        """Add image embedding."""
        emb = self.embedder.embed_image(image)
        self.image_embeddings[key] = emb
        return emb

    def add_audio(self, key: str, audio: Tensor) -> Tensor:
        """Add audio embedding."""
        emb = self.embedder.embed_audio(audio)
        self.audio_embeddings[key] = emb
        return emb

    def retrieve_text_to_image(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Retrieve images from text query.

        Args:
            query_text: Query text
            top_k: Number of results

        Returns:
            List of (key, score) tuples
        """
        query_emb = self.embedder.embed_text(query_text)

        scores = []
        for key, emb in self.image_embeddings.items():
            score = F.cosine_similarity(query_emb, emb, dim=-1)
            scores.append((key, score.item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve_image_to_text(
        self,
        query_image: Image.Image,
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Retrieve text from image query.

        Args:
            query_image: Query image
            top_k: Number of results

        Returns:
            List of (key, score) tuples
        """
        query_emb = self.embedder.embed_image(query_image)

        scores = []
        for key, emb in self.text_embeddings.items():
            score = F.cosine_similarity(query_emb, emb, dim=-1)
            scores.append((key, score.item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def compute_similarity(
        self,
        emb1: Tensor,
        emb2: Tensor,
    ) -> Tensor:
        """
        Compute similarity between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score
        """
        return F.cosine_similarity(emb1, emb2, dim=-1)

    def normalize(
        self,
        embeddings: Tensor,
    ) -> Tensor:
        """
        Normalize embeddings.

        Args:
            embeddings: Input embeddings

        Returns:
            Normalized embeddings
        """
        return F.normalize(embeddings, p=2, dim=-1)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get embedding index statistics."""
        return {
            "text_count": len(self.text_embeddings),
            "image_count": len(self.image_embeddings),
            "audio_count": len(self.audio_embeddings),
            "dim": self.config.output_dim,
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multimodal learning.

    InfoNCE-style loss for aligning embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings1: Tensor,
        embeddings2: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings1: First modality embeddings (batch, dim)
            embeddings2: Second modality embeddings (batch, dim)
            labels: Optional matching labels

        Returns:
            Loss scalar
        """
        # Normalize
        emb1 = F.normalize(embeddings1, p=2, dim=-1)
        emb2 = F.normalize(embeddings2, p=2, dim=-1)

        # Compute similarity matrix
        similarities = torch.matmul(emb1, emb2.T) / self.temperature

        # Labels are diagonal (matching pairs)
        if labels is None:
            labels = torch.arange(
                embeddings1.size(0),
                device=embeddings1.device,
            )

        # Cross entropy loss
        loss = F.cross_entropy(similarities, labels)

        return loss


def create_multimodal_embedder(
    output_dim: int = 768,
    normalize: bool = True,
    **kwargs,
) -> MultimodalEmbedder:
    """
    Create a multimodal embedder.

    Args:
        output_dim: Output embedding dimension
        normalize: Whether to normalize embeddings
        **kwargs: Additional arguments

    Returns:
        MultimodalEmbedder instance
    """
    config = MultimodalEmbedderConfig(
        output_dim=output_dim,
        normalize=normalize,
        **kwargs,
    )
    return MultimodalEmbedder(config)


__all__ = [
    "MultimodalEmbedder",
    "MultimodalEmbedderConfig",
    "UnifiedEmbeddings",
    "ContrastiveLoss",
    "create_multimodal_embedder",
]