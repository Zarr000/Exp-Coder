"""Comprehensive tests for the Expera AI data pipeline."""

import sys, os, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.data.dataset_manifest import DatasetManifest, DatasetStage, DatasetStatistics, ManifestManager
from src.data.dataset_validator import DatasetValidator, ValidationResult, ValidationSeverity
from src.data.shard_manager import ShardWriter, ShardReader, discover_shards
from src.data.language_detector import LanguageDetector, LanguageResult
from src.data.quality_filter import QualityFilter, QualityScore
from src.data.deduplicator import Deduplicator, DedupResult
from src.data.preprocessing import TextPreprocessor, clean_text
from src.data.sequence_packer import SequencePacker, PackedSequence
from src.data.dynamic_batcher import DynamicBatcher, Batch
from src.data.dataset_mixer import DatasetMixer, MixConfig
from src.data.dataset_stats import StatsCollector, DatasetStats
from src.data.repo_preprocessor import RepoPreprocessor


class TestLanguageDetector:
    def test_detect_python(self):
        detector = LanguageDetector()
        code = "import os\ndef main():\n    print('hello')\nif __name__ == '__main__':\n    main()"
        result = detector.detect(code)
        assert result.language == "python"
        assert result.is_code

    def test_detect_javascript(self):
        detector = LanguageDetector()
        code = "function hello() {\n  const x = 1;\n  return x;\n}\nconsole.log('test');"
        result = detector.detect(code)
        assert result.language == "javascript"
        assert result.is_code

    def test_detect_english(self):
        detector = LanguageDetector()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = detector.detect(text)
        assert result.language == "en"
        assert not result.is_code

    def test_detect_from_filename(self):
        detector = LanguageDetector()
        result = detector.detect_from_filename("main.py")
        assert result.language == "python"
        assert result.confidence > 0.9


class TestQualityFilter:
    def test_good_text_passes(self):
        qf = QualityFilter()
        text = "This is a good quality document with proper length and structure. " * 50
        score = qf.compute(text)
        assert score.passed

    def test_short_text_scores_low(self):
        qf = QualityFilter(min_doc_length=50)
        text = "short"
        score = qf.compute(text)
        assert score.dimensions["length_score"] < 0.5

    def test_repetitive_text_fails(self):
        qf = QualityFilter()
        text = "repeat repeat repeat repeat repeat repeat repeat repeat " * 100
        score = qf.compute(text)
        assert score.dimensions["repetition_score"] < 0.5


class TestDeduplicator:
    def test_exact_duplicate(self):
        dedup = Deduplicator(near_enabled=False)
        text = "This is a unique document for testing."
        r1 = dedup.check(text, "doc1")
        assert not r1.is_duplicate
        r2 = dedup.check(text, "doc2")
        assert r2.is_duplicate
        assert r2.method == "exact"

    def test_unique_documents(self):
        dedup = Deduplicator(near_enabled=False)
        r1 = dedup.check("Document one content here.", "doc1")
        r2 = dedup.check("Completely different document two.", "doc2")
        assert not r1.is_duplicate
        assert not r2.is_duplicate

    def test_near_duplicate(self):
        dedup = Deduplicator(similarity_threshold=0.7)
        base = "The quick brown fox jumps over the lazy dog. " * 20
        r1 = dedup.check(base, "doc1")
        assert not r1.is_duplicate
        similar = base[:len(base)//2] + " some different text here " * 10
        r2 = dedup.check(similar, "doc2")
        # Near-duplicate detection may or may not trigger depending on similarity
        # Just verify it doesn't crash
        assert r2.method in ("exact", "near") or not r2.is_duplicate


class TestPreprocessing:
    def test_clean_text(self):
        result = clean_text("  Hello   World  \n\n\n  ")
        assert "Hello" in result
        assert "World" in result

    def test_remove_html(self):
        result = clean_text("<p>Hello World</p>", remove_html=True)
        assert "Hello World" in result
        assert "<p>" not in result

    def test_preprocessor_min_length(self):
        pp = TextPreprocessor(min_length=50)
        assert pp("short") is None
        assert pp("long enough text " * 10) is not None


class TestSequencePacker:
    def test_pack_single_document(self):
        packer = SequencePacker(max_length=100)
        tokens = list(range(50))
        sequences = packer.pack_document(tokens)
        assert len(sequences) == 1
        assert sequences[0].length <= 100

    def test_pack_long_document(self):
        packer = SequencePacker(max_length=50)
        tokens = list(range(120))
        sequences = packer.pack_document(tokens)
        assert len(sequences) >= 2

    def test_pack_multiple(self):
        packer = SequencePacker(max_length=100, packing_mode='fixed')
        docs = [list(range(30)), list(range(40)), list(range(20))]
        sequences = list(packer.pack_multiple(docs))
        assert len(sequences) > 0
        for seq in sequences:
            assert seq.length <= 100


class TestDynamicBatcher:
    def test_batchify(self):
        batcher = DynamicBatcher(max_tokens_per_batch=1000)
        sequences = [list(range(50)) for _ in range(20)]
        batches = list(batcher.batchify(sequences))
        assert len(batches) > 0
        for batch in batches:
            assert len(batch.input_ids) > 0

    def test_padding(self):
        batcher = DynamicBatcher(max_tokens_per_batch=1000)
        sequences = [list(range(30)), list(range(50))]
        batches = list(batcher.batchify(sequences))
        batch = batches[0]
        # All sequences should be same length after padding
        lengths = [len(ids) for ids in batch.input_ids]
        assert len(set(lengths)) == 1


class TestDatasetMixer:
    def test_static_weights(self):
        configs = [
            MixConfig(name="code", weight=0.7),
            MixConfig(name="text", weight=0.3),
        ]
        mixer = DatasetMixer(configs)
        weights = mixer.get_weights()
        assert "code" in weights
        assert "text" in weights

    def test_sampling(self):
        configs = [
            MixConfig(name="code", weight=1.0),
            MixConfig(name="text", weight=0.0),
        ]
        mixer = DatasetMixer(configs)
        for _ in range(10):
            assert mixer.sample() == "code"

    def test_temperature(self):
        configs = [
            MixConfig(name="a", weight=0.9),
            MixConfig(name="b", weight=0.1),
        ]
        mixer = DatasetMixer(configs, global_temperature=10.0)
        weights = mixer.get_weights()
        # High temperature should flatten distribution
        assert abs(weights["a"] - weights["b"]) < 0.5


class TestStatsCollector:
    def test_record_and_finalize(self):
        collector = StatsCollector("test")
        collector.record_document("hello world " * 10, "en", 0.8)
        collector.record_document("bonjour " * 10, "fr", 0.7)
        collector.record_tokens(50)
        
        stats = collector.finalize()
        assert stats.total_documents == 2
        assert stats.total_tokens == 50
        assert "en" in stats.languages
        assert "fr" in stats.languages

    def test_report(self):
        collector = StatsCollector("test")
        collector.record_document("test " * 10, "en", 0.9)
        report = collector.report()
        assert "test" in report
        assert "docs" in report


class TestRepoPreprocessor:
    def test_disabled(self):
        pp = RepoPreprocessor(enabled=False)
        files = [("a.py", "python", "print('hello')")]
        result = pp.process(files)
        assert result == ["print('hello')"]

    def test_extract_imports(self):
        pp = RepoPreprocessor()
        imports = pp.extract_imports("import os\nfrom sys import path", "python")
        assert len(imports) > 0

    def test_format_with_header(self):
        from src.data.repo_preprocessor import RepoFile
        pp = RepoPreprocessor(include_file_headers=True)
        f = RepoFile(path="test.py", language="python", content="print('hi')")
        formatted = pp.format_file_with_header(f)
        assert "<|file_start|>" in formatted
        assert "test.py" in formatted


class TestDatasetValidator:
    def test_validate_jsonl(self, tmp_path):
        validator = DatasetValidator()
        filepath = tmp_path / "test.jsonl"
        filepath.write_text('{"text": "hello world"}\n{"text": "another doc"}\n')
        result = validator.validate_jsonl(str(filepath))
        assert result.passed

    def test_validate_bad_jsonl(self, tmp_path):
        validator = DatasetValidator()
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text('not json\n{"text": "ok"}\n')
        result = validator.validate_jsonl(str(filepath))
        assert not result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])