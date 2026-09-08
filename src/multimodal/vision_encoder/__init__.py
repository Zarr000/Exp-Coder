"""
Vision Encoder for Expera AI.

Modular vision encoder abstraction supporting:
- CLIP-compatible encoders
- ViT encoders
- Pluggable backends (local/cloud)

Supports local execution and cloud API backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import numpy as np


class EncoderBackend(Enum):
    """Vision encoder backends."""
    LOCAL = "local"
    CUDA = "cuda"
    TORCH = "torch"
    CLIPE = "clipe"  # OpenCLIP
    OPENAI = "openai"  # OpenAI API
    ANTHROPIC = "anthropic"  # Anthropic API


@dataclass
class VisionEncoderConfig:
    """Vision encoder configuration."""
    model_name: str = "vit-base-patch16-224"
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    backend: EncoderBackend = EncoderBackend.TORCH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    use_pretrained: bool = True


class VisionEncoder(ABC, nn.Module):
    """
    Abstract vision encoder base class.

    Subclasses implement specific encoder architectures.
    """

    def __init__(self, config: VisionEncoderConfig):
        nn.Module.__init__(self)
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

    @abstractmethod
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """
        Encode images to embeddings.

        Args:
            images: PIL Image or list of images

        Returns:
            Image embeddings (batch, embed_dim)
        """
        pass

    @abstractmethod
    def encode_patches(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """
        Encode images as patches with positions.

        Args:
            images: PIL Image or list of images

        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        pass

    def preprocess(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """
        Preprocess images for encoder.

        Args:
            images: PIL Image or list

        Returns:
            Preprocessed tensor (batch, channels, height, width)
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Convert to RGB if needed
        images = [img.convert("RGB") for img in images]

        # Resize
        images = [
            img.resize((self.config.image_size, self.config.image_size))
            for img in images
        ]

        # To tensor
        tensors = [
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            for img in images
        ]

        batch = torch.stack(tensors)
        return batch.to(self.device)

    def to(self, device: str) -> "VisionEncoder":
        """Move to device."""
        self.device = device
        return self


class CLIPEncoder(VisionEncoder):
    """
    CLIP-compatible vision encoder.

    Based on Vision Transformer architecture.
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__(config)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (config.image_size // config.patch_size) ** 2 + 1, config.embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(config.embed_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize linear and layernorm weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Encode images to CLIP embeddings."""
        x = self.preprocess(images)

        # Patch embed
        x = self.patch_embed(x)  # (batch, embed_dim, h//patch, w//patch)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return cls token embedding
        return x[:, 0]

    def encode_patches(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Encode images as patches."""
        x = self.preprocess(images)

        # Patch embed
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add position embedding (without cls)
        x = x + self.pos_embed[:, 1:]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class ViTEncoder(VisionEncoder):
    """
    Vision Transformer (ViT) encoder.

    Standard ViT architecture.
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__(config)

        # Extract patch grid size
        num_patches = (config.image_size // config.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, config.embed_dim, config.patch_size, config.patch_size),
            nn.Flatten(2),
        )

        # CLS and position
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)

        # Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Encode images to ViT embeddings."""
        x = self.preprocess(images)

        # Patch embed: (batch, embed_dim, h//patch, w//patch) -> (batch, num_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x[:, 0]

    def encode_patches(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Encode patches."""
        x = self.preprocess(images)
        # Patch embed: (batch, embed_dim, h//patch, w//patch) -> (batch, num_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Vision Transformer block with attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # MLP
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward through transformer block."""
        # Attention with residual
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)

        # MLP with residual
        x = x + self.mlp(x)
        x = self.norm2(x)

        return x


def create_vision_encoder(
    model_name: str = "vit-base-patch16-224",
    backend: EncoderBackend = EncoderBackend.TORCH,
    device: Optional[str] = None,
    **kwargs,
) -> VisionEncoder:
    """
    Create a vision encoder.

    Args:
        model_name: Model name (e.g., "vit-base-patch16-224", "clip-vit-large-patch14")
        backend: Backend type
        device: Device (auto-detect if None)
        **kwargs: Additional config arguments

    Returns:
        VisionEncoder instance
    """
    # Parse model name
    if "clip" in model_name.lower():
        config = VisionEncoderConfig(
            model_name=model_name,
            backend=backend,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        )
        return CLIPEncoder(config)
    else:
        # Default to ViT
        config = VisionEncoderConfig(
            model_name=model_name,
            backend=backend,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        )
        return ViTEncoder(config)


__all__ = [
    "VisionEncoder",
    "VisionEncoderConfig",
    "EncoderBackend",
    "CLIPEncoder",
    "ViTEncoder",
    "TransformerBlock",
    "create_vision_encoder",
]