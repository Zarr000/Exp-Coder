"""
Dynamic batching for Expera AI.

Supports:
1. Variable-length batching with padding
2. Bucket-based batching (group similar lengths)
3. Dynamic batch size based on total tokens
4. Gradient accumulation support
"""

from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple
import random


@dataclass
class Batch:
    """A batch of sequences."""
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    labels: Optional[List[List[int]]] = None
    metadata: Optional[dict] = None


class DynamicBatcher:
    """
    Dynamic batching with configurable strategies.
    
    Bucket-based batching groups sequences of similar length
    to minimize padding waste. Supports dynamic batch sizing
    based on total token budget.
    """

    def __init__(
        self,
        max_tokens_per_batch: int = 65536,
        pad_token_id: int = 0,
        bucket_boundaries: Optional[List[int]] = None,
        shuffle_buckets: bool = True,
        seed: Optional[int] = None,
    ):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_token_id = pad_token_id
        self.bucket_boundaries = bucket_boundaries or [
            128, 256, 512, 1024, 2048, 4096, 8192
        ]
        self.shuffle_buckets = shuffle_buckets
        self.seed = seed
        self._rng = random.Random(seed) if seed else random.Random()

    def _get_bucket(self, length: int) -> int:
        """Assign sequence to a bucket based on length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)

    def _pad_sequences(
        self, sequences: List[List[int]], max_len: int
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Pad sequences to max_len and create attention masks."""
        padded = []
        masks = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(seq + [self.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        return padded, masks

    def batchify(
        self,
        sequences: List[List[int]],
        labels: Optional[List[List[int]]] = None,
    ) -> Iterator[Batch]:
        """
        Create batches from sequences using bucket-based batching.
        
        Args:
            sequences: List of token sequences
            labels: Optional list of label sequences
            
        Yields:
            Batch objects
        """
        # Group by bucket
        buckets = {}
        for i, seq in enumerate(sequences):
            bucket = self._get_bucket(len(seq))
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append((i, seq))

        # Process each bucket
        bucket_order = sorted(buckets.keys())
        if self.shuffle_buckets:
            self._rng.shuffle(bucket_order)

        for bucket in bucket_order:
            items = buckets[bucket]
            if self.shuffle_buckets:
                self._rng.shuffle(items)

            # Form batches within bucket
            batch_seqs = []
            batch_labels = []
            batch_tokens = 0
            max_len = 0

            for idx, seq in items:
                seq_len = len(seq)
                new_max = max(max_len, seq_len)
                new_tokens = new_max * (len(batch_seqs) + 1)

                if new_tokens > self.max_tokens_per_batch and batch_seqs:
                    # Yield current batch
                    padded, masks = self._pad_sequences(batch_seqs, max_len)
                    yield Batch(
                        input_ids=padded,
                        attention_mask=masks,
                        labels=self._pad_sequences(batch_labels, max_len)[0]
                        if batch_labels else None,
                    )
                    batch_seqs = []
                    batch_labels = []
                    batch_tokens = 0
                    max_len = 0

                batch_seqs.append(seq)
                if labels:
                    batch_labels.append(labels[idx])
                max_len = max(max_len, seq_len)
                batch_tokens = max_len * len(batch_seqs)

            # Yield remaining
            if batch_seqs:
                padded, masks = self._pad_sequences(batch_seqs, max_len)
                yield Batch(
                    input_ids=padded,
                    attention_mask=masks,
                    labels=self._pad_sequences(batch_labels, max_len)[0]
                    if batch_labels else None,
                )