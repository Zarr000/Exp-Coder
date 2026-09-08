# Phase 2: Tokenizer Training System

## Overview

Build a complete tokenizer training system for Expera AI that supports training Byte-Pair Encoding (BPE) tokenizers from code and text corpora. The system enables vocabulary analysis, evaluation, benchmarking, streaming/incremental training, vocabulary pruning, and code-aware tokenization.

## Goals

1. Produce production-quality tokenizers optimized for code understanding
2. Support massive corpora (100M+ documents) without loading into memory
3. Provide comprehensive analysis and evaluation tools
4. Enable incremental vocabulary growth
5. Support vocabulary pruning and expansion

## Architecture

### Module Structure

```
src/tokenizer/
├── __init__.py
├── bpe_tokenizer.py          # Existing: Core BPE implementation
├── vocab_builder.py         # Existing: Vocabulary building utilities
├── trainer.py             # NEW: Tokenizer trainer
├── analyzer.py           # NEW: Vocabulary analysis
├── evaluator.py         # NEW: Tokenizer evaluation
├── statistics.py        # NEW: Statistics collection
├── benchmark.py        # NEW: Benchmarking utilities
├── stream_trainer.py    # NEW: Streaming trainer
├── incremental.py      # NEW: Incremental training
├── pruner.py          # NEW: Vocabulary pruning
└── code_aware.py     # NEW: Code-aware tokenization
```

### Public APIs

#### trainer.py

```python
class TokenizerTrainer:
    """Train BPE tokenizer on text corpora."""

    def __init__(
        self,
        vocab_size: int = 50304,
        special_tokens: Optional[Dict[str, str]] = None,
        min_frequency: int = 2,
        streaming: bool = False,
    ) -> None

    def train(
        self,
        texts: List[str],
        num_merges: Optional[int] = None,
        verbose: bool = True,
    ) -> BPETokenizer

    def train_from_files(
        self,
        file_paths: List[str],
        num_merges: Optional[int] = None,
        verbose: bool = True,
    ) -> BPETokenizer

    def train_from_iterator(
        self,
        text_iterator: Iterator[str],
        num_merges: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BPETokenizer
```

#### analyzer.py

```python
@dataclass
class VocabAnalysis:
    """Vocabulary analysis results."""
    vocab_size: int
    avg_token_length: float
    max_token_length: int
    min_token_length: int
    coverage: Dict[str, float]  # language -> coverage
    compression_ratio: float
    token_distribution: Dict[str, float]
    special_token_count: int
    byte_token_count: int
    merged_token_count: int


class VocabularyAnalyzer:
    """Analyze tokenizer vocabulary."""

    def __init__(self, tokenizer: BPETokenizer) -> None

    def analyze(self, texts: List[str]) -> VocabAnalysis

    def analyze_coverage(
        self,
        texts: List[str],
    ) -> Dict[str, float]

    def get_token_frequency(
        self,
        texts: List[str],
    ) -> Dict[str, int]

    def get_coverage_curve(
        self,
        texts: List[str],
    ) -> List[Tuple[int, float]]
```

#### evaluator.py

```python
@dataclass
class EvaluationResult:
    """Tokenizer evaluation results."""
    avg_tokens_per_word: float
    avg_chars_per_token: float
    compression_ratio: float
    vocabulary_utilization: float
    out_of_vocab_rate: float
    code_metrics: Dict[str, float]
    text_metrics: Dict[str, float]


class TokenizerEvaluator:
    """Evaluate tokenizer quality."""

    def __init__(self, tokenizer: BPETokenizer) -> None

    def evaluate(
        self,
        texts: List[str],
    ) -> EvaluationResult

    def evaluate_code(
        self,
        code_samples: List[str],
    ) -> Dict[str, float]

    def evaluate_text(
        self,
        text_samples: List[str],
    ) -> Dict[str, float]

    def compare(
        self,
        other_tokenizer: BPETokenizer,
        texts: List[str],
    ) -> Dict[str, Tuple[float, float]]
```

#### statistics.py

```python
@dataclass
class TokenizerStats:
    """Tokenizer statistics."""
    total_documents: int
    total_words: int
    total_tokens_generated: int
    total_bytes: int
    avg_doc_length: float
    avg_word_length: float
    vocab_size: int
    unique_words: int
    unique_bigrams: int
    merge_operations: int


class StatsCollector:
    """Collect tokenizer statistics."""

    def __init__(self) -> None

    def collect(
        self,
        tokenizer: BPETokenizer,
        texts: Iterator[str],
    ) -> TokenizerStats

    def collect_sample(
        self,
        tokenizer: BPETokenizer,
        texts: List[str],
        sample_size: int = 10000,
    ) -> TokenizerStats
```

#### benchmark.py

```python
@dataclass
class BenchmarkResult:
    """Benchmark results."""
    name: str
    encode_time_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    avg_tokens_per_text: float


class TokenizerBenchmark:
    """Benchmark tokenizer performance."""

    def __init__(self, tokenizer: BPETokenizer) -> None

    def benchmark_encode(
        self,
        texts: List[str],
        warmup: int = 100,
    ) -> BenchmarkResult

    def benchmark_decode(
        self,
        token_ids: List[List[int]],
        warmup: int = 100,
    ) -> BenchmarkResult

    def benchmark_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> BenchmarkResult
```

#### stream_trainer.py

```python
class StreamingTrainer:
    """Train tokenizer on streaming data."""

    def __init__(
        self,
        vocab_size: int = 50304,
        min_frequency: int = 2,
        buffer_size: int = 100000,
    ) -> None

    def feed(self, text: str) -> None

    def feed_batch(self, texts: List[str]) -> None

    def train(
        self,
        num_merges: Optional[int] = None,
    ) -> BPETokenizer

    def get_progress(self) -> float
```

#### incremental.py

```python
class IncrementalTrainer:
    """Incrementally expand tokenizer vocabulary."""

    def __init__(
        self,
        base_tokenizer: BPETokenizer,
        target_vocab_size: int,
        min_frequency: int = 2,
    ) -> None

    def add_texts(
        self,
        new_texts: List[str],
    ) -> int  # Number of new merges

    def expand(
        self,
        num_merges: Optional[int] = None,
    ) -> BPETokenizer

    def get_new_tokens(self) -> List[str]
```

#### pruner.py

```python
class VocabularyPruner:
    """Prune tokenizer vocabulary."""

    def __init__(
        self,
        tokenizer: BPETokenizer,
        min_usage: int = 10,
    ) -> None

    def analyze_usage(
        self,
        texts: List[str],
    ) -> Dict[str, int]

    def prune(
        self,
        preserve_tokens: Set[str],
    ) -> BPETokenizer

    def merge_low_freq(
        self,
        threshold: int = 5,
    ) -> Dict[str, str]
```

#### code_aware.py

```python
@dataclass
class CodeTokenConfig:
    """Configuration for code-aware tokenization."""
    split_camel_case: bool = True
    split_snake_case: bool = True
    preserve_keywords: bool = True
    preserve_indentation: bool = True
    max_keyword_length: int = 50


class CodeAwareTokenizer:
    """Tokenizer optimized for code."""

    def __init__(
        self,
        base_tokenizer: BPETokenizer,
        config: Optional[CodeTokenConfig] = None,
    ) -> None

    def tokenize_code(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> List[int]

    def extract_identifiers(
        self,
        code: str,
    ) -> List[str]

    def split_compound(
        self,
        identifier: str,
    ) -> List[str]
```

## Implementation Details

### Training Algorithm

The tokenizer trainer implements GPT-2 style byte-level BPE:

1. **Pre-tokenization**: Split text using regex pattern for word boundaries
2. **Byte encoding**: Convert characters to bytes using GPT-2 byte encoder
3. **BPE merges**: Iteratively merge most frequent adjacent pairs
4. **Vocabulary**: Build vocab from merges (256 base + special + merged)

### Streaming Training

For large corpora (100M+ documents):

1. **Buffer**: Accumulate texts in memory buffer
2. **Frequency counting**: Count word/bigram frequencies incrementally
3. **Periodic merges**: Perform merges when buffer is full
4. **Checkpointing**: Save intermediate results for resumption

### Code-Aware Tokenization

Special handling for code:

1. **CamelCase**: Split `createUser` → `create`, `User`
2. **Snake_case**: Split `my_function` → `my`, `function`
3. **Keywords**: Preserve language keywords
4. **Indentation**: Preserve leading whitespace
5. **Comments**: Preserve block comments

### Memory Management

- **Streaming**: Process texts in chunks, don't load entire corpus
- **Buffer size**: Configurable buffer (default 100K texts)
- **Checkpointing**: Save progress every N texts
- **Disk fallback**: Option to use disk for frequency counting

## Performance Targets

| Metric | Target |
|--------|-------|
| Training speed | 10K texts/sec (streaming) |
| Encode speed | 100K tokens/sec |
| Memory usage | < 10 GB for 1M text buffer |
| Vocabulary analysis | < 30 seconds for 100K texts |

## Tests

All modules require tests covering:

1. Basic functionality
2. Edge cases
3. Error handling
4. Integration with existing BPETokenizer
5. Memory efficiency
6. Output correctness

## Exit Criteria

1. All modules implemented with type hints and docstrings
2. All tests passing
3. Benchmark results meet targets
4. Documentation complete