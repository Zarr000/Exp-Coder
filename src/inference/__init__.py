"""
Inference System for Expera AI.

Modules:
- generator: Text generation with various decoding strategies
- pipeline: End-to-end inference pipeline
- streaming: Streaming generation support
- cache: KV cache management
"""

from .generator import (
    Generator,
    GenerationConfig,
    DecodingStrategy,
    GreedySearch,
    Sampling,
    BeamSearch,
    ContrastiveSearch,
)
from .pipeline import InferencePipeline, TextGenerationOutput
from .streaming import StreamingGenerator, StreamCallback
from .cache import KVCache, KVCacheManager

__all__ = [
    # Generator
    "Generator",
    "GenerationConfig",
    "DecodingStrategy",
    "GreedySearch",
    "Sampling",
    "BeamSearch",
    "ContrastiveSearch",
    # Pipeline
    "InferencePipeline",
    "TextGenerationOutput",
    # Streaming
    "StreamingGenerator",
    "StreamCallback",
    # Cache
    "KVCache",
    "KVCacheManager",
]