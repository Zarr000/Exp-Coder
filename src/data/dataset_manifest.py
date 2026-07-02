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
    CLEANED = "cleaned"
    PROCESSED = "processed"
    TOKENIZED = "tokenized"
    SHARDED = "sharded"
    READY = "ready"


@dataclass
class DatasetStatistics:
    """Statistics for a dataset at a given stage."""
    total_documents: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    avg_doc_length: float = 0.0
    avg_tokens_per_doc: float = 0.0
    language_distribution: Dict[str, float] = field(default_factory=dict)
    quality_score_distribution: Dict[str, float] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # Backward compatible aliases
    @property
    def num_documents(self) -> int:
        return self.total_documents

    @num_documents.setter
    def num_documents(self, value: int) -> None:
        self.total_documents = value

    @property
    def num_tokens(self) -> int:
        return self.total_tokens

    @num_tokens.setter
    def num_tokens(self, value: int) -> None:
        self.total_tokens = value

    @property
    def num_bytes(self) -> int:
        return self.total_bytes

    @num_bytes.setter
    def num_bytes(self, value: int) -> None:
        self.total_bytes = value
    
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


@dataclass
class ProcessingStep:
    """Record of a single processing step."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    output_hash: str = ""
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingStep":
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

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        stage: Optional[DatasetStage] = None,
        created_at: Optional[float] = None,
        source_paths: Optional[List[str]] = None,
    ):
        self.name = name
        self.version = version
        self.stage = stage or DatasetStage.RAW
        self.created_at = created_at if created_at is not None else time.time()
        self.source_paths = source_paths or []
        self.statistics = DatasetStatistics()
        self.stages: Dict[DatasetStage, StageEntry] = {}
        self.metadata: Dict[str, Any] = {}
        self.processing_steps: List[ProcessingStep] = []
        self._manifest_hash: str = ""

    def add_step(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        input_hash: str = "",
        output_hash: str = "",
        duration: float = 0.0,
    ) -> None:
        """Add a processing step."""
        step = ProcessingStep(
            name=name,
            parameters=parameters or {},
            input_hash=input_hash,
            output_hash=output_hash,
            duration=duration,
        )
        self.processing_steps.append(step)
        self._manifest_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute deterministic hash of manifest content."""
        import json
        content = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "stage": self.stage.value,
                "created_at": self.created_at,
                "source_paths": self.source_paths,
                "processing_steps": [s.to_dict() for s in self.processing_steps],
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @property
    def manifest_hash(self) -> str:
        """Get the manifest hash."""
        return self._manifest_hash
    
    def add_stage(self, entry: StageEntry) -> None:
        """Record a processing stage."""
        self.stages[entry.stage] = entry
    
    def get_stage(self, stage: DatasetStage) -> Optional[StageEntry]:
        """Get a specific stage entry."""
        return self.stages.get(stage)
    
    def has_stage(self, stage: DatasetStage) -> bool:
        """Check if a stage has been completed."""
        return stage in self.stages or stage == self.stage
    
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
            "stage": self.stage.value,
            "created_at": self.created_at,
            "source_paths": self.source_paths,
            "statistics": self.statistics.to_dict(),
            "stages": {k.value: v.to_dict() for k, v in self.stages.items()},
            "processing_steps": [s.to_dict() for s in self.processing_steps],
            "metadata": self.metadata,
            "manifest_hash": self.manifest_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        manifest = cls(
            name=data["name"],
            version=data["version"],
            stage=DatasetStage(data.get("stage", "raw")),
            created_at=data.get("created_at"),
            source_paths=data.get("source_paths", []),
        )
        manifest.statistics = DatasetStatistics.from_dict(data.get("statistics", {}))
        manifest.metadata = data.get("metadata", {})
        manifest._manifest_hash = data.get("manifest_hash", "")
        for stage_str, stage_data in data.get("stages", {}).items():
            stage = DatasetStage(stage_str)
            manifest.stages[stage] = StageEntry.from_dict(stage_data)
        for step_data in data.get("processing_steps", []):
            manifest.processing_steps.append(ProcessingStep.from_dict(step_data))
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

    def create_manifest(
        self,
        name: str,
        version: str,
        stage: DatasetStage,
        source_paths: Optional[List[str]] = None,
    ) -> DatasetManifest:
        """Create a new manifest or update existing with new stage."""
        # Check if existing manifest can be reused
        if name in self.manifests:
            existing = self.manifests[name]
            if existing.version == version:
                # Update stage on existing manifest
                existing.stage = stage
                # Add to stages dict
                entry = StageEntry(stage=stage)
                existing.stages[stage] = entry
                return existing

        # Create new manifest
        manifest = DatasetManifest(
            name=name,
            version=version,
            stage=stage,
            source_paths=source_paths,
        )
        # Also add to stages dict
        manifest.stages[stage] = StageEntry(stage=stage)
        self.manifests[name] = manifest
        return manifest

    def save_manifest(self, manifest: DatasetManifest) -> str:
        """Save manifest to disk and registry."""
        if self.registry_path:
            self._save_registry()
        path = str(Path(self.registry_path).parent / f"{manifest.name}_{manifest.version}.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_json(path)
        return path

    def load_manifest(
        self,
        name: str,
        version: str,
        stage: DatasetStage,
    ) -> Optional[DatasetManifest]:
        """Load a manifest by name, version, and stage."""
        if name in self.manifests:
            m = self.manifests[name]
            if m.version == version and m.stage == stage:
                return m
        # Try to load from disk
        if self.registry_path:
            path = Path(self.registry_path).parent / f"{name}_{version}.json"
            if path.exists():
                return DatasetManifest.from_json(str(path))
        return None

    def get_cached_path(
        self,
        name: str,
        version: str,
        stage: DatasetStage,
    ) -> Optional[str]:
        """Get cached path for a manifest."""
        m = self.load_manifest(name, version, stage)
        if m and hasattr(m, 'cache_path'):
            return m.cache_path
        return None

    def get_all_versions(self, name: str) -> List[str]:
        """Get all versions for a dataset."""
        versions = []
        if name in self.manifests:
            versions.append(self.manifests[name].version)
        # Also scan filesystem
        if self.registry_path:
            parent = Path(self.registry_path).parent
            for p in parent.glob(f"{name}_*.json"):
                v = p.stem.replace(f"{name}_", "")
                if v and v not in versions:
                    versions.append(v)
        return sorted(versions)

    def validate_chain(
        self,
        name: str,
        version: str,
        required_stages: List[DatasetStage],
    ) -> bool:
        """Validate that all required stages exist."""
        m = self.load_manifest(name, version, required_stages[0])
        if not m:
            return False
        for stage in required_stages:
            if not m.has_stage(stage):
                return False
        return True

    def invalidate_cache(
        self,
        name: str,
        version: str,
        stage: DatasetStage,
    ) -> None:
        """Invalidate cached data for a manifest."""
        m = self.load_manifest(name, version, stage)
        if m and self.registry_path:
            path = Path(self.registry_path).parent / f"{name}_{version}.json"
            if path.exists():
                path.unlink()
        if name in self.manifests:
            del self.manifests[name]