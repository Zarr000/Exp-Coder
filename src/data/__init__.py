"""
Data processing and loading for Expera AI.

Modules:
- dataset_manifest: Dataset provenance and metadata tracking
- shard_manager: Streaming sharded I/O
- deduplicator: MinHash near-dedup + exact dedup
- sequence_packer: Document-boundary-aware packing
- dynamic_batcher: Variable-length batch formation
- dataset_mixer: Temperature-based dataset mixing
- language_detector: Code + natural language detection
- quality_filter: Multi-dimensional quality scoring
- repo_preprocessor: Repository-aware code preprocessing
- preprocessing: Text cleaning and normalization
- dataset_stats: Statistics and reporting
- dataset_validator: Validation utilities
- dataset: Main ExperaDataset orchestrator
- data_loader: Factory for dataloaders
- benchmark: Benchmarking and profiling utilities
"""

from .dataset_manifest import DatasetManifest, DatasetStage, DatasetStatistics, ManifestManager
from .dataset_validator import DatasetValidator, ValidationResult
from .shard_manager import ShardInfo, ShardList, ShardWriter, ShardReader, discover_shards
from .language_detector import LanguageDetector, LanguageResult
from .quality_filter import QualityFilter, QualityScore
from .deduplicator import Deduplicator, DedupResult
from .preprocessing import TextPreprocessor, clean_text
from .sequence_packer import SequencePacker, PackedSequence
from .dynamic_batcher import DynamicBatcher, Batch
from .dataset_mixer import DatasetMixer, MixConfig
from .dataset_stats import StatsCollector, DatasetStats
from .repo_preprocessor import RepoPreprocessor, RepoFile, RepoContext
from .dataset import ExperaDataset
from .causal_lm_dataset import CausalLMDataset, load_text_corpus
from .data_loader import get_dataloader, DataLoaderConfig, DataLoaderBenchmark
from .benchmark import (
    BenchmarkResult,
    Profiler,
    PipelineProfiler,
    benchmark_throughput,
    benchmark_tokenization,
    benchmark_dataloader,
    benchmark_memory,
    run_full_benchmark,
)

__all__ = [
    # Manifest
    "DatasetManifest", "DatasetStage", "DatasetStatistics", "ManifestManager",
    # Validator
    "DatasetValidator", "ValidationResult",
    # Shard Manager
    "ShardInfo", "ShardList", "ShardWriter", "ShardReader", "discover_shards",
    # Language
    "LanguageDetector", "LanguageResult",
    # Quality
    "QualityFilter", "QualityScore",
    # Dedup
    "Deduplicator", "DedupResult",
    # Preprocessing
    "TextPreprocessor", "clean_text",
    # Packing
    "SequencePacker", "PackedSequence",
    # Batching
    "DynamicBatcher", "Batch",
    # Mixing
    "DatasetMixer", "MixConfig",
    # Stats
    "StatsCollector", "DatasetStats",
    # Repo
    "RepoPreprocessor", "RepoFile", "RepoContext",
    # Dataset
    "ExperaDataset",
    # Causal LM dataset (canonical)
    "CausalLMDataset",
    "load_text_corpus",
    # DataLoader
    "get_dataloader", "DataLoaderConfig", "DataLoaderBenchmark",
    # Benchmark
    "BenchmarkResult",
    "Profiler",
    "PipelineProfiler",
    "benchmark_throughput",
    "benchmark_tokenization",
    "benchmark_dataloader",
    "benchmark_memory",
    "run_full_benchmark",
]
