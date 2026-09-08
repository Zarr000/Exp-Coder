"""Tests for dataset manifest and metadata tracking."""

import sys
import os
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml

from src.data.dataset_manifest import (
    DatasetManifest,
    DatasetStage,
    DatasetStatistics,
    ManifestManager,
    ProcessingStep,
)


class TestDatasetManifest:
    """Test dataset manifest creation and serialization."""

    def test_create_manifest(self):
        """Test basic manifest creation."""
        manifest = DatasetManifest(
            name="test_dataset",
            version="1.0",
            stage=DatasetStage.RAW,
            created_at=1234567890.0,
        )
        assert manifest.name == "test_dataset"
        assert manifest.version == "1.0"
        assert manifest.stage == DatasetStage.RAW
        assert manifest.manifest_hash == ""

    def test_compute_hash(self):
        """Test deterministic hash computation."""
        manifest = DatasetManifest(
            name="test",
            version="1.0",
            stage=DatasetStage.RAW,
            created_at=1234567890.0,
        )
        h1 = manifest.compute_hash()
        h2 = manifest.compute_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_changes_with_content(self):
        """Test hash changes when manifest content changes."""
        m1 = DatasetManifest(name="test", version="1.0", stage=DatasetStage.RAW, created_at=0)
        m2 = DatasetManifest(name="test", version="2.0", stage=DatasetStage.RAW, created_at=0)
        assert m1.compute_hash() != m2.compute_hash()

    def test_add_step(self):
        """Test recording processing steps."""
        manifest = DatasetManifest(name="test", version="1.0", stage=DatasetStage.RAW, created_at=0)
        manifest.add_step(
            name="clean",
            parameters={"remove_html": True},
            input_hash="abc",
            output_hash="def",
            duration=1.5,
        )
        assert len(manifest.processing_steps) == 1
        assert manifest.processing_steps[0].name == "clean"
        assert manifest.processing_steps[0].parameters == {"remove_html": True}
        assert manifest.manifest_hash != ""


class TestManifestManager:
    """Test manifest manager operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temp directory."""
        manifest_dir = tmp_path / "manifests"
        return ManifestManager(str(manifest_dir))

    def test_create_and_save_load(self, manager):
        """Test round-trip save and load."""
        manifest = manager.create_manifest(
            name="test_dataset",
            version="1.0",
            stage=DatasetStage.RAW,
            source_paths=["/data/file.jsonl"],
        )
        manifest.statistics.total_documents = 1000
        manifest.statistics.languages = {"python": 500, "javascript": 500}

        path = manager.save_manifest(manifest)
        assert Path(path).exists()

        loaded = manager.load_manifest("test_dataset", "1.0", DatasetStage.RAW)
        assert loaded is not None
        assert loaded.name == "test_dataset"
        assert loaded.statistics.total_documents == 1000
        assert loaded.statistics.languages == {"python": 500, "javascript": 500}

    def test_load_nonexistent(self, manager):
        """Test loading non-existent manifest."""
        assert manager.load_manifest("nonexistent", "1.0", DatasetStage.RAW) is None

    def test_get_cached_path(self, manager):
        """Test cached path retrieval."""
        manifest = manager.create_manifest("test", "1.0", DatasetStage.RAW)
        manifest.cache_path = "/data/cached"
        manager.save_manifest(manifest)
        
        assert manager.get_cached_path("test", "1.0", DatasetStage.RAW) == "/data/cached"

    def test_get_all_versions(self, manager):
        """Test version listing."""
        for v in ["1.0", "2.0", "3.0"]:
            m = manager.create_manifest("test", v, DatasetStage.RAW)
            manager.save_manifest(m)

        versions = manager.get_all_versions("test")
        assert versions == ["1.0", "2.0", "3.0"]

    def test_validate_chain(self, manager):
        """Test chain validation."""
        for stage in [DatasetStage.RAW, DatasetStage.CLEANED, DatasetStage.READY]:
            m = manager.create_manifest("test", "1.0", stage)
            manager.save_manifest(m)

        assert manager.validate_chain("test", "1.0", [
            DatasetStage.RAW, DatasetStage.CLEANED, DatasetStage.READY
        ])
        assert not manager.validate_chain("test", "1.0", [
            DatasetStage.RAW, DatasetStage.TOKENIZED
        ])

    def test_invalidate_cache(self, manager):
        """Test cache invalidation."""
        m = manager.create_manifest("test", "1.0", DatasetStage.RAW)
        path = manager.save_manifest(m)
        assert Path(path).exists()

        manager.invalidate_cache("test", "1.0", DatasetStage.RAW)
        assert not Path(path).exists()
        assert manager.load_manifest("test", "1.0", DatasetStage.RAW) is None


class TestDatasetStatistics:
    """Test dataset statistics."""

    def test_statistics_defaults(self):
        """Test default values."""
        stats = DatasetStatistics()
        assert stats.total_documents == 0
        assert stats.total_tokens == 0
        assert stats.languages == {}

    def test_statistics_custom(self):
        """Test custom statistics."""
        stats = DatasetStatistics(
            total_documents=5000,
            total_tokens=1000000,
            languages={"python": 3000, "javascript": 2000},
            quality_scores={"mean_score": 0.85},
        )
        assert stats.total_documents == 5000
        assert stats.languages["python"] == 3000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])