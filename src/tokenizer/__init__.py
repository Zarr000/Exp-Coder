"""
Tokenizer module for Expera AI
Implements Byte-Pair Encoding (BPE) from scratch
"""

from .bpe_tokenizer import BPETokenizer
from .vocab_builder import VocabularyBuilder
from .trainer import TokenizerTrainer, train_tokenizer, train_tokenizer_from_files
from .analyzer import VocabularyAnalyzer, VocabAnalysis, analyze_vocabulary
from .evaluator import TokenizerEvaluator, EvaluationResult, evaluate_tokenizer
from .statistics import StatsCollector, TokenizerStats, collect_stats
from .benchmark import TokenizerBenchmark, BenchmarkResult, benchmark_tokenizer
from .stream_trainer import StreamingTrainer, train_tokenizer_streaming
from .incremental import IncrementalTrainer, expand_tokenizer
from .pruner import VocabularyPruner, prune_vocabulary
from .code_aware import CodeAwareTokenizer, CodeTokenConfig, create_code_tokenizer
from .code_aware import PROGRAMMING_KEYWORDS

__all__ = [
    # Core
    "BPETokenizer",
    "VocabularyBuilder",
    # Training
    "TokenizerTrainer",
    "train_tokenizer",
    "train_tokenizer_from_files",
    # Analysis
    "VocabularyAnalyzer",
    "VocabAnalysis",
    "analyze_vocabulary",
    # Evaluation
    "TokenizerEvaluator",
    "EvaluationResult",
    "evaluate_tokenizer",
    # Statistics
    "StatsCollector",
    "TokenizerStats",
    "collect_stats",
    # Benchmarking
    "TokenizerBenchmark",
    "BenchmarkResult",
    "benchmark_tokenizer",
    # Streaming
    "StreamingTrainer",
    "train_tokenizer_streaming",
    # Incremental
    "IncrementalTrainer",
    "expand_tokenizer",
    # Pruning
    "VocabularyPruner",
    "prune_vocabulary",
    # Code-aware
    "CodeAwareTokenizer",
    "CodeTokenConfig",
    "create_code_tokenizer",
    "PROGRAMMING_KEYWORDS",
]