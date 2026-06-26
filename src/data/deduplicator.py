"""
Deduplication for Expera AI.

Supports:
1. Exact deduplication (hash-based) - O(n) memory for content-addressed dedup
2. Near-deduplication (MinHash + LSH) - Detects near-duplicate documents
3. Configurable thresholds and hash sizes
4. Streaming-compatible: processes documents one at a time
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Iterator, Tuple
from collections import defaultdict


@dataclass
class DedupResult:
    """Result of deduplication check."""
    is_duplicate: bool
    duplicate_of: Optional[str] = None  # ID of original document
    similarity: float = 0.0
    method: str = 'exact'  # 'exact' or 'near'


class MinHasher:
    """
    MinHash signature generator for documents.
    
    Uses k independent hash functions to generate a signature
    that approximates Jaccard similarity between documents.
    """
    
    def __init__(self, num_hashes: int = 128, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        self._hash_funcs = self._generate_hash_funcs(num_hashes, seed)
    
    def _generate_hash_funcs(self, n: int, seed: int) -> List[callable]:
        """Generate n independent hash functions."""
        funcs = []
        for i in range(n):
            # Each hash function: (ax + b) mod p
            a = hash(f"a_{seed}_{i}") % 2147483647
            b = hash(f"b_{seed}_{i}") % 2147483647
            p = 2147483647  # Large prime
            funcs.append(lambda x, a=a, b=b, p=p: ((a * hash(x) + b) % p))
        return funcs
    
    def compute_signature(self, tokens: List[str]) -> List[int]:
        """Compute MinHash signature from token list."""
        signature = [float('inf')] * self.num_hashes
        for token in set(tokens):  # Set for shingle uniqueness
            for i, h in enumerate(self._hash_funcs):
                hv = h(token)
                if hv < signature[i]:
                    signature[i] = hv
        return signature
    
    def similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class Deduplicator:
    """
    Document deduplication with exact + near-duplicate detection.
    
    Uses:
    - SHA-256 hash for exact deduplication
    - MinHash + banded LSH for near-duplicate detection
    - Configurable similarity threshold
    """
    
    def __init__(
        self,
        exact_enabled: bool = True,
        near_enabled: bool = True,
        similarity_threshold: float = 0.8,
        num_hashes: int = 128,
        num_bands: int = 16,
        seed: int = 42,
    ):
        self.exact_enabled = exact_enabled
        self.near_enabled = near_enabled
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        
        self.minhasher = MinHasher(num_hashes, seed)
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        
        # Storage for seen documents
        self._exact_hashes: Dict[str, str] = {}  # hash -> doc_id
        self._lsh_buckets: Dict[int, Dict[int, List[Tuple[str, List[int]]]]] = \
            defaultdict(lambda: defaultdict(list))  # band -> bucket -> [(doc_id, sig)]
        self._stats = {"exact_dups": 0, "near_dups": 0, "unique": 0}
    
    def _get_exact_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        # Normalize: strip extra whitespace
        normalized = ' '.join(text.split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _tokenize(self, text: str) -> List[str]:
        """Convert text to character 6-gram tokens."""
        n = 6
        tokens = []
        if len(text) < n:
            return [text]
        for i in range(len(text) - n + 1):
            tokens.append(text[i:i+n])
        return tokens
    
    def _lsh_band_hash(self, signature: List[int], band: int) -> int:
        """Compute bucket hash for a band of the signature."""
        start = band * self.rows_per_band
        end = start + self.rows_per_band
        band_sig = tuple(signature[start:end])
        return hash(band_sig) % 2147483647
    
    def check(self, text: str, doc_id: Optional[str] = None) -> DedupResult:
        """
        Check if a document is a duplicate.
        
        Args:
            text: Document text
            doc_id: Optional document identifier
            
        Returns:
            DedupResult with deduplication status
        """
        if doc_id is None:
            doc_id = hashlib.md5(text.encode()[:100]).hexdigest()
        
        # 1. Exact dedup check
        if self.exact_enabled:
            exact_hash = self._get_exact_hash(text)
            if exact_hash in self._exact_hashes:
                self._stats["exact_dups"] += 1
                return DedupResult(
                    is_duplicate=True,
                    duplicate_of=self._exact_hashes[exact_hash],
                    similarity=1.0,
                    method='exact',
                )
        
        # 2. Near-duplicate check via LSH
        if self.near_enabled and len(text) >= 50:
            tokens = self._tokenize(text)
            signature = self.minhasher.compute_signature(tokens)
            
            candidates: Dict[str, float] = {}
            for band in range(self.num_bands):
                bucket = self._lsh_band_hash(signature, band)
                for existing_id, existing_sig in self._lsh_buckets[band][bucket]:
                    if existing_id not in candidates:
                        sim = self.minhasher.similarity(signature, existing_sig)
                        candidates[existing_id] = sim
            
            # Check candidates against threshold
            best_match = max(candidates, key=candidates.get) if candidates else None
            best_sim = candidates.get(best_match, 0.0) if best_match else 0.0
            
            if best_sim >= self.similarity_threshold:
                self._stats["near_dups"] += 1
                return DedupResult(
                    is_duplicate=True,
                    duplicate_of=best_match,
                    similarity=best_sim,
                    method='near',
                )
            
            # Insert into LSH buckets
            for band in range(self.num_bands):
                bucket = self._lsh_band_hash(signature, band)
                self._lsh_buckets[band][bucket].append((doc_id, signature))
        
        # 3. Not a duplicate - store for future checks
        if self.exact_enabled:
            self._exact_hashes[exact_hash] = doc_id
        
        self._stats["unique"] += 1
        return DedupResult(is_duplicate=False, similarity=0.0)
    
    def check_batch(
        self,
        texts: List[Tuple[str, Optional[str]]],
    ) -> List[DedupResult]:
        """Check multiple documents."""
        return [self.check(text, doc_id) for text, doc_id in texts]
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return dict(self._stats)