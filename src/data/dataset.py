"""
ExperaDataset - Main dataset orchestrator for Expera AI.

Integrates all data pipeline components:
- ShardManager for streaming I/O
- LanguageDetector for language awareness
- QualityFilter for quality scoring
- Deduplicator for dedup
- RepoPreprocessor for code repos
- TextPreprocessor for cleaning
- SequencePacker for packing
- DynamicBatcher for batching
- DatasetMixer for mixing
- StatsCollector for statistics

Uses PyTorch IterableDataset for streaming, memory-efficient training.
"""

import torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Optional, Iterator, Any, Tuple
from pathlib import Path

from .shard_manager import ShardReader, ShardList, discover_shards
from .language_detector import LanguageDetector, LanguageResult
from .quality_filter import QualityFilter, QualityScore
from .deduplicator import Deduplicator, DedupResult
from .repo_preprocessor import RepoPreprocessor
from .preprocessing import TextPreprocessor
from .sequence_packer import SequencePacker, PackedSequence
from .dynamic_batcher import DynamicBatcher, Batch
from .dataset_mixer import DatasetMixer, MixConfig
from .dataset_stats import StatsCollector, DatasetStats
from .dataset_manifest import DatasetManifest, DatasetStage, ManifestManager


class ExperaDataset(IterableDataset):
    """
    Streaming dataset for large-scale pretraining.
    
    Integrates all preprocessing, filtering, and packing stages
    into a single IterableDataset that processes data on-the-fly.
    
    Supports:
    - Multiple data sources with mixing
    - Language detection and balancing
    - Quality filtering
    - Deduplication
    - Repository-aware code preprocessing
    - Document-boundary-aware sequence packing
    - Dynamic batching
    - Statistics collection
    """

    def __init__(
        self,
        data_dirs: List[str],
        tokenizer: Any,
        max_length: int = 2048,
        packing_mode: str = 'fixed',
        # Filtering
        enable_quality_filter: bool = True,
        enable_dedup: bool = True,
        enable_language_detection: bool = True,
        enable_repo_preprocessing: bool = False,
        # Mixing
        mix_weights: Optional[Dict[str, float]] = None,
        mix_temperature: float = 1.0,
        # Batching
        max_tokens_per_batch: int = 65536,
        # Shuffling
        shuffle_shards: bool = True,
        seed: Optional[int] = None,
        # Stats
        collect_stats: bool = True,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.packing_mode = packing_mode
        self.seed = seed
        self.collect_stats = collect_stats
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.language_detector = LanguageDetector() if enable_language_detection else None
        self.quality_filter = QualityFilter() if enable_quality_filter else None
        self.deduplicator = Deduplicator() if enable_dedup else None
        self.repo_preprocessor = RepoPreprocessor(enabled=enable_repo_preprocessing)
        self.packer = SequencePacker(
            max_length=max_length,
            packing_mode=packing_mode,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
        )
        self.batcher = DynamicBatcher(
            max_tokens_per_batch=max_tokens_per_batch,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
            seed=seed,
        )
        
        # Initialize mixer
        if mix_weights:
            configs = [MixConfig(name=name, weight=w) for name, w in mix_weights.items()]
            self.mixer = DatasetMixer(configs, global_temperature=mix_temperature, seed=seed)
        else:
            self.mixer = None
        
        # Initialize stats
        self.stats = StatsCollector("expera_dataset") if collect_stats else None
        
        # Discover shards
        self._shard_readers: Dict[str, ShardReader] = {}
        for data_dir in data_dirs:
            shard_list = discover_shards(data_dir)
            if shard_list.total_shards > 0:
                self._shard_readers[data_dir] = ShardReader(
                    shard_list, shuffle_shards=shuffle_shards, seed=seed
                )
    
    def _process_document(self, text: str, source: str = "unknown",
                          filename: Optional[str] = None) -> Optional[List[int]]:
        """Process a single document through the pipeline."""
        # 1. Preprocess
        cleaned = self.preprocessor(text)
        if cleaned is None:
            if self.stats:
                self.stats.record_filtered()
            return None
        
        # 2. Language detection
        language = "unknown"
        if self.language_detector:
            result = self.language_detector.detect(cleaned, filename)
            language = result.language
        
        # 3. Quality filter
        quality_score = 1.0
        if self.quality_filter:
            score = self.quality_filter.compute(cleaned)
            quality_score = score.overall
            if not score.passed:
                if self.stats:
                    self.stats.record_filtered()
                return None
        
        # 4. Deduplication
        if self.deduplicator:
            result = self.deduplicator.check(cleaned)
            if result.is_duplicate:
                if self.stats:
                    self.stats.record_duplicate()
                return None
        
        # 5. Tokenize
        tokens = self.tokenizer.encode(cleaned, add_special_tokens=True)
        
        # 6. Stats
        if self.stats:
            self.stats.record_document(cleaned, language, quality_score)
            self.stats.record_tokens(len(tokens))
        
        return tokens
    
    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Determine which shards this worker handles
        if worker_info is not None:
            # Multi-worker: split shards
            all_readers = list(self._shard_readers.items())
            per_worker = max(1, len(all_readers) // worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(all_readers)
            my_readers = dict(all_readers[start:end])
        else:
            my_readers = self._shard_readers
        
        # Process documents
        token_buffer: List[List[int]] = []
        
        for source, reader in my_readers.items():
            for shard_idx, record in reader.iter_all_records():
                text = record.get("text", "")
                filename = record.get("filename")
                
                tokens = self._process_document(text, source, filename)
                if tokens is None:
                    continue
                
                token_buffer.append(tokens)
                
                # Pack and batch when buffer is full
                if len(token_buffer) >= 1000:
                    for packed in self.packer.pack_multiple(token_buffer):
                        token_buffer = []
                        # Yield batches
                        for batch in self.batcher.batchify([packed.tokens]):
                            yield batch
        
        # Process remaining
        if token_buffer:
            for packed in self.packer.pack_multiple(token_buffer):
                for batch in self.batcher.batchify([packed.tokens]):
                    yield batch
    
    def get_stats(self) -> Optional[DatasetStats]:
        """Get collected statistics."""
        if self.stats:
            return self.stats.finalize()
        return None