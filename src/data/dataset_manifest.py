"""
Dataset manifest and provenance tracking for Expera AI.

Tracks:
- Dataset lineage (source → processed → tokenized)
- Per-stage statistics
- Processing parameters
- Versioning
- Reproducibility metadata
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum


class DatasetStage(Enum):
    """Stages in the data processing pipeline."""
    RAW = "raw"
    PROCESSED = "processed"
    TOKENIZED = "tokenized"
    SHARDED = "sharded"


@dataclass
class DatasetStatistics:
    """Statistics for a dataset at a given stage."""
    num_documents: int = 0
    num_tokens: int = 0
    num_bytes: int = 0
    avg_doc_length: float = 0.0
    avg_tokens_per_doc: float = 0.0
    language_distribution: Dict[str, float] = field(default_factory=dict)
    quality_score_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetStatistics":
        return cls(**data)


@dataclass
class StageEntry:
    """Entry for a specific processing stage."""
    stage: DatasetStage
    timestamp: float = field(default_factory=time.time)
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    statistics: Optional[DatasetStatistics] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        if self.statistics:
            d["statistics"] = self.statistics.to_dict()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageEntry":
        data = dict(data)
        data["stage"] = DatasetStage(data["stage"])
        if data.get("statistics"):
            data["statistics"] = DatasetStatistics.from_dict(data["statistics"])
        return cls(**data)


class DatasetManifest:
    """
    Tracks the full provenance of a dataset through all processing stages.
    
    Enables:
    - Reproducibility: exact parameters recorded
    - Debugging: trace issues to specific stages
    - Auditing: know what data was used
    - Incremental processing: skip completed stages
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.created_at = time.time()
        self.stages: Dict[DatasetStage, StageEntry] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_stage(self, entry: StageEntry) -> None:
        """Record a processing stage."""
        self.stages[entry.stage] = entry
    
    def get_stage(self, stage: DatasetStage) -> Optional[StageEntry]:
        """Get a specific stage entry."""
        return self.stages.get(stage)
    
    def has_stage(self, stage: DatasetStage) -> bool:
        """Check if a stage has been completed."""
        return stage in self.stages
    
    def compute_checksum(self, file_paths: List[str]) -> str:
        """Compute a checksum over input files for reproducibility."""
        hasher = hashlib.sha256()
        for fp in sorted(file_paths):
            path = Path(fp)
            if path.exists():
                hasher.update(path.read_bytes())
        return hasher.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "stages": {k.value: v.to_dict() for k, v in self.stages.items()},
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        manifest = cls(name=data["name"], version=data["version"])
        manifest.created_at = data["created_at"]
        manifest.metadata = data.get("metadata", {})
        for stage_str, stage_data in data.get("stages", {}).items():
            stage = DatasetStage(stage_str)
            manifest.stages[stage] = StageEntry.from_dict(stage_data)
        return manifest
    
    def to_json(self, path: str) -> None:
        """Save manifest to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class ManifestManager:
    """
    Manages multiple dataset manifests.
    
    Provides:
    - Registry of all datasets
    - Cross-dataset queries
    - Global statistics
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path
        self.manifests: Dict[str, DatasetManifest] = {}
        if registry_path and Path(registry_path).exists():
            self._load_registry()
    
    def register(self, manifest: DatasetManifest) -> None:
        """Register a dataset manifest."""
        self.manifests[manifest.name] = manifest
        self._save_registry()
    
    def get(self, name: str) -> Optional[DatasetManifest]:
        """Get a manifest by name."""
        return self.manifests.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return list(self.manifests.keys())
    
    def total_documents(self) -> int:
        """Get total documents across all datasets."""
        total = 0
        for m in self.manifests.values():
            for stage in m.stages.values():
                if stage.statistics:
                    total += stage.statistics.num_documents
        return total
    
    def total_tokens(self) -> int:
        """Get total tokens across all datasets."""
        total = 0
        for m in self.manifests.values():
            for stage in m.stages.values():
                if stage.statistics:
                    total += stage.statistics.num_tokens
        return total
    
    def _save_registry(self) -> None:
        if self.registry_path:
            data = {name: m.to_dict() for name, m in self.manifests.items()}
            Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def _load_registry(self) -> None:
        with open(self.registry_path, 'r') as f:
            data = json.load(f)
        for name, mdata in data.items():
            self.manifests[name] = DatasetManifest.from_dict(mdata)