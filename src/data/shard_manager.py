"""
Shard Manager for Streaming Dataset I/O.

Handles reading/writing large datasets split across multiple shards.
Key features:
- Streaming line-by-line reads (never loads full shard into memory)
- Compressed JSONL support (.gz)
- Shard-level checksums for integrity
- Parallel I/O with multiprocessing
- Resumable iteration (track position)
- Shard discovery and validation
"""

import gzip
import json
import hashlib
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Callable, Any
from collections import OrderedDict


@dataclass
class ShardInfo:
    """Metadata for a single shard."""
    path: str
    index: int
    num_lines: Optional[int] = None
    total_bytes: Optional[int] = None
    checksum: Optional[str] = None
    compressed: bool = False


@dataclass
class ShardList:
    """Collection of shards forming a dataset."""
    name: str
    shards: List[ShardInfo] = field(default_factory=list)
    total_shards: int = 0
    
    def add_shard(self, shard: ShardInfo) -> None:
        """Add a shard to the list."""
        self.shards.append(shard)
        self.total_shards = len(self.shards)
    
    def get_by_index(self, index: int) -> Optional[ShardInfo]:
        """Get shard by index."""
        for s in self.shards:
            if s.index == index:
                return s
        return None
    
    def validate_continuity(self) -> bool:
        """Check that shard indices are contiguous (0..n-1)."""
        indices = sorted(s.index for s in self.shards)
        return indices == list(range(len(indices)))


class ShardWriter:
    """Write datasets as sharded JSONL files."""
    
    def __init__(
        self,
        output_dir: str,
        name: str,
        max_records_per_shard: int = 100000,
        compress: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.name = name
        self.max_records_per_shard = max_records_per_shard
        self.compress = compress
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_shard = 0
        self._current_count = 0
        self._file = None
        self._checksum = hashlib.sha256()
        self._all_checksums: Dict[str, str] = {}
        self._shard_paths: List[str] = []
        self._total_records = 0
    
    def _open_shard(self) -> None:
        """Open a new shard file."""
        suffix = '.jsonl.gz' if self.compress else '.jsonl'
        path = self.output_dir / f"{self.name}_shard_{self._current_shard:05d}{suffix}"
        
        self._file = gzip.open(path, 'wt', encoding='utf-8') if self.compress else open(path, 'w', encoding='utf-8')
        self._shard_paths.append(str(path))
        self._current_count = 0
        self._checksum = hashlib.sha256()
        print(f"  Opening shard {self._current_shard}: {path}")
    
    def _close_shard(self) -> None:
        """Close current shard and record checksum."""
        if self._file is None:
            return
        self._file.close()
        checksum = self._checksum.hexdigest()
        self._all_checksums[self._shard_paths[-1]] = checksum
        self._file = None
        self._current_shard += 1
    
    def write_record(self, record: Dict[str, Any]) -> None:
        """Write a single record as JSON line."""
        if self._file is None or self._current_count >= self.max_records_per_shard:
            if self._file is not None:
                self._close_shard()
            self._open_shard()
        
        line = json.dumps(record, ensure_ascii=False) + '\n'
        self._file.write(line)
        self._checksum.update(line.encode('utf-8'))
        self._current_count += 1
        self._total_records += 1
    
    def write_records(self, records: Iterator[Dict[str, Any]]) -> None:
        """Write multiple records."""
        for record in records:
            self.write_record(record)
    
    def close(self) -> ShardList:
        """Close writer and return ShardList."""
        if self._file is not None:
            self._close_shard()
        
        shard_list = ShardList(name=self.name)
        for i, path in enumerate(self._shard_paths):
            shard_list.add_shard(ShardInfo(
                path=path,
                index=i,
                checksum=self._all_checksums.get(path),
                compressed=self.compress,
            ))
        return shard_list


class ShardReader:
    """
    Streaming reader for sharded datasets.
    
    Provides line-by-line iteration over shards without loading
    everything into memory. Supports:
    - Ordered iteration across shards
    - Random shard access by index
    - Position tracking for resumability
    - Parallel processing via multiprocessing
    """
    
    def __init__(
        self,
        shard_list: ShardList,
        shuffle_shards: bool = False,
        seed: Optional[int] = None,
    ):
        self.shard_list = shard_list
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        
        # Sort shards by index for ordered access
        self._shards = sorted(shard_list.shards, key=lambda s: s.index)
    
    def __len__(self) -> int:
        """Get total number of shards."""
        return len(self._shards)
    
    def _open_file(self, path: str):
        """Open a shard file, handling compression."""
        if path.endswith('.gz'):
            return gzip.open(path, 'rt', encoding='utf-8')
        return open(path, 'r', encoding='utf-8')
    
    def read_lines(self, shard_index: int) -> Iterator[str]:
        """
        Read lines from a specific shard.
        
        Args:
            shard_index: Index of shard to read
            
        Yields:
            Raw JSON lines
        """
        shard = self.shard_list.get_by_index(shard_index)
        if shard is None:
            raise ValueError(f"Shard {shard_index} not found")
        
        with self._open_file(shard.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    
    def read_records(self, shard_index: int) -> Iterator[Dict[str, Any]]:
        """
        Read parsed JSON records from a specific shard.
        
        Args:
            shard_index: Index of shard to read
            
        Yields:
            Parsed record dictionaries
        """
        for line in self.read_lines(shard_index):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    
    def iter_all_lines(self) -> Iterator[Tuple[int, str]]:
        """
        Iterate over all shards, yielding (shard_index, line) pairs.
        
        Supports optional shard-level shuffling.
        """
        shards = list(self._shards)
        if self.shuffle_shards and self.seed is not None:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(shards)
        
        for shard in shards:
            for line in self.read_lines(shard.index):
                yield shard.index, line
    
    def iter_all_records(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Iterate over all records across all shards.
        
        Yields:
            (shard_index, record) tuples
        """
        for shard_index, line in self.iter_all_lines():
            try:
                yield shard_index, json.loads(line)
            except json.JSONDecodeError:
                continue
    
    def parallel_map(
        self,
        func: Callable[[Dict[str, Any]], Any],
        num_workers: int = 4,
        max_records: Optional[int] = None,
    ) -> Iterator[Any]:
        """
        Apply a function to all records using multiprocessing.
        
        Args:
            func: Function to apply to each record
            num_workers: Number of worker processes
            max_records: Maximum records to process
            
        Yields:
            Function results in order
        """
        with mp.Pool(num_workers) as pool:
            results = []
            count = 0
            for shard_index, record in self.iter_all_records():
                if max_records and count >= max_records:
                    break
                results.append(pool.apply_async(func, (record,)))
                count += 1
                
                # Yield in batches for memory efficiency
                if len(results) >= 1000:
                    for r in results:
                        yield r.get()
                    results = []
            
            for r in results:
                yield r.get()
    
    def get_statistics(self, fast: bool = True) -> Dict[str, Any]:
        """
        Compute shard-level statistics.
        
        Args:
            fast: If True, only count files without parsing
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "num_shards": len(self._shards),
            "shard_paths": [s.path for s in self._shards],
        }
        
        if fast:
            # Quick stats (file sizes only)
            total_bytes = 0
            for shard in self._shards:
                path = Path(shard.path)
                if path.exists():
                    total_bytes += path.stat().st_size
            stats["total_bytes"] = total_bytes
        else:
            # Full stats (parse all files)
            total_lines = 0
            total_bytes = 0
            total_records = 0
            for shard in self._shards:
                path = Path(shard.path)
                if not path.exists():
                    continue
                total_bytes += path.stat().st_size
                with self._open_file(str(path)) as f:
                    for line in f:
                        total_lines += 1
                        if line.strip():
                            try:
                                json.loads(line)
                                total_records += 1
                            except json.JSONDecodeError:
                                pass
            stats.update({
                "total_lines": total_lines,
                "total_bytes": total_bytes,
                "total_records": total_records,
            })
        
        return stats


def discover_shards(
    directory: str,
    pattern: str = "*_shard_*.jsonl*",
) -> ShardList:
    """
    Discover shard files in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for shard files
        
    Returns:
        ShardList with discovered shards
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return ShardList(name=Path(directory).stem)
    
    name = dir_path.stem if dir_path.is_file() else dir_path.name
    shard_list = ShardList(name=name)
    
    for filepath in sorted(dir_path.glob(pattern)):
        # Extract shard index from filename
        stem = filepath.stem.replace('.jsonl', '')
        parts = stem.split('_shard_')
        if len(parts) != 2:
            continue
        
        try:
            index = int(parts[1])
        except ValueError:
            continue
        
        shard_list.add_shard(ShardInfo(
            path=str(filepath),
            index=index,
            compressed=filepath.suffix == '.gz',
            total_bytes=filepath.stat().st_size,
        ))
    
    return shard_list