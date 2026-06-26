"""Tests for shard manager (streaming I/O)."""

import sys, os, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.data.shard_manager import (
    ShardInfo, ShardList, ShardWriter, ShardReader, discover_shards
)


class TestShardWriter:
    def test_write_and_read(self, tmp_path):
        writer = ShardWriter(str(tmp_path), "test", max_records_per_shard=10, compress=False)
        for i in range(25):
            writer.write_record({"id": i, "text": f"Document {i}"})
        shard_list = writer.close()
        assert shard_list.total_shards == 3  # 25 records, 10 per shard
        assert len(list(tmp_path.glob("*.jsonl"))) == 3

    def test_write_compressed(self, tmp_path):
        writer = ShardWriter(str(tmp_path), "test", max_records_per_shard=100, compress=True)
        for i in range(50):
            writer.write_record({"id": i})
        shard_list = writer.close()
        for s in shard_list.shards:
            assert s.compressed is True


class TestShardReader:
    @pytest.fixture
    def shard_list(self, tmp_path):
        writer = ShardWriter(str(tmp_path), "test", max_records_per_shard=10, compress=False)
        for i in range(25):
            writer.write_record({"id": i, "text": f"doc_{i}"})
        return writer.close()

    def test_read_records(self, shard_list):
        reader = ShardReader(shard_list)
        records = list(reader.read_records(0))
        assert len(records) == 10
        assert records[0]["id"] == 0

    def test_iter_all_records(self, shard_list):
        reader = ShardReader(shard_list)
        records = list(reader.iter_all_records())
        assert len(records) == 25

    def test_get_statistics_fast(self, shard_list):
        reader = ShardReader(shard_list)
        stats = reader.get_statistics(fast=True)
        assert stats["num_shards"] == 3
        assert stats["total_bytes"] > 0

    def test_get_statistics_slow(self, shard_list):
        reader = ShardReader(shard_list)
        stats = reader.get_statistics(fast=False)
        assert stats["total_records"] == 25


class TestDiscoverShards:
    def test_discover(self, tmp_path):
        writer = ShardWriter(str(tmp_path), "data", max_records_per_shard=100, compress=False)
        for i in range(50):
            writer.write_record({"id": i})
        writer.close()
        discovered = discover_shards(str(tmp_path))
        assert discovered.total_shards == 1

    def test_discover_empty(self, tmp_path):
        discovered = discover_shards(str(tmp_path / "nonexistent"))
        assert discovered.total_shards == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])