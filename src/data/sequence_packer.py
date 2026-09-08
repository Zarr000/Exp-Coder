"""
Document-boundary-aware sequence packing for Expera AI.

Supports:
1. Fixed-length packing: pack sequences to exact max_length
2. Variable-length packing: pack sequences within [min, max] range
3. Curriculum packing: progressively increase sequence length during training
4. Document boundaries: never pack across document boundaries
"""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple


@dataclass
class PackedSequence:
    """A packed sequence containing one or more documents."""
    tokens: List[int]
    document_boundaries: List[int]  # Positions where documents end
    length: int
    attention_mask: Optional[List[int]] = None


class SequencePacker:
    """
    Packs token sequences into fixed/variable length blocks.
    
    Uses best-fit-decreasing algorithm to minimize wasted tokens
    while respecting document boundaries.
    """

    def __init__(
        self,
        max_length: int = 2048,
        min_length: Optional[int] = None,
        packing_mode: str = 'fixed',
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ):
        self.max_length = max_length
        self.min_length = min_length or max_length
        self.packing_mode = packing_mode  # 'fixed', 'variable', 'curriculum'
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def pack_document(self, tokens: List[int]) -> List[PackedSequence]:
        """
        Pack a single document's tokens into sequences.
        
        If document fits in one sequence, returns single PackedSequence.
        Otherwise, splits across multiple sequences.
        """
        sequences = []
        pos = 0
        while pos < len(tokens):
            end = min(pos + self.max_length, len(tokens))
            seq_tokens = tokens[pos:end]
            
            # Add EOS at end of document
            if end >= len(tokens):
                if len(seq_tokens) < self.max_length:
                    seq_tokens.append(self.eos_token_id)
            
            seq = PackedSequence(
                tokens=seq_tokens,
                document_boundaries=[len(seq_tokens)],
                length=len(seq_tokens),
            )
            sequences.append(seq)
            pos = end
        return sequences

    def pack_multiple(self, token_lists: List[List[int]]) -> Iterator[PackedSequence]:
        """
        Pack multiple documents using best-fit-decreasing.
        
        Documents are sorted by length (largest first) and packed
        into sequences with best-fit to minimize wasted space.
        """
        # Sort documents by length descending
        indexed = sorted(
            [(len(t), t) for t in token_lists],
            key=lambda x: -x[0],
        )

        sequences: List[PackedSequence] = []
        
        for doc_len, doc_tokens in indexed:
            placed = False
            
            # Try to fit into existing sequence
            for seq in sequences:
                space = self.max_length - seq.length - 1  # -1 for EOS
                if doc_len <= space:
                    seq.tokens.extend(doc_tokens)
                    seq.tokens.append(self.eos_token_id)
                    seq.document_boundaries.append(len(seq.tokens))
                    seq.length = len(seq.tokens)
                    placed = True
                    break
            
            if not placed:
                # Need new sequence
                new_seq = PackedSequence(
                    tokens=doc_tokens + [self.eos_token_id],
                    document_boundaries=[len(doc_tokens) + 1],
                    length=len(doc_tokens) + 1,
                )
                sequences.append(new_seq)
        
        # Pad remaining sequences if needed
        for seq in sequences:
            if self.packing_mode == 'fixed' and seq.length < self.max_length:
                padding = [self.pad_token_id] * (self.max_length - seq.length)
                seq.tokens.extend(padding)
                seq.length = self.max_length
                seq.attention_mask = [1] * len(seq.tokens[:seq.length - len(padding)]) + [0] * len(padding)

        yield from sequences

    def curriculum_pack(
        self,
        token_lists: List[List[int]],
        step: int,
        total_steps: int,
        min_ratio: float = 0.25,
    ) -> Iterator[PackedSequence]:
        """
        Curriculum packing: gradually increase sequence length.
        
        Args:
            token_lists: List of token sequences
            step: Current training step
            total_steps: Total training steps
            min_ratio: Minimum sequence length ratio (e.g., 0.25 = 512 for 2048 max)
        """
        progress = step / max(1, total_steps)
        current_max = int(
            self.min_length + (self.max_length - self.min_length) * progress
        )
        
        # Temporarily reduce max_length for this batch
        original_max = self.max_length
        self.max_length = current_max
        results = list(self.pack_multiple(token_lists))
        self.max_length = original_max
        return results