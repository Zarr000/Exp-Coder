"""
Tests for Tokenizer Training System.

Tests:
- TokenizerTrainer
- StreamingTrainer
- IncrementalTrainer
- VocabularyPruner
- CodeAwareTokenizer
"""

import pytest
from src.tokenizer import (
    BPETokenizer,
    TokenizerTrainer,
    StreamingTrainer,
    IncrementalTrainer,
    VocabularyPruner,
    CodeAwareTokenizer,
    CodeTokenConfig,
    PROGRAMMING_KEYWORDS,
)


class TestTokenizerTrainer:
    """Test tokenizer trainer."""

    def test_train_basic(self):
        """Test basic training."""
        texts = [
            "hello world",
            "the quick brown fox",
            "hello world again",
            "testing a simple tokenizer",
        ]

        trainer = TokenizerTrainer(vocab_size=1000)
        tokenizer = trainer.train(texts, verbose=False)

        assert len(tokenizer.vocab) > 256

    def test_train_from_list(self):
        """Test training from list."""
        texts = ["hello world"] * 100 + ["test data"] * 50

        trainer = TokenizerTrainer(vocab_size=500)
        tokenizer = trainer.train(texts, verbose=False)

        assert tokenizer is not None

    def test_train_with_special_tokens(self):
        """Test training with custom special tokens."""
        special = {
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
        }

        texts = ["test text for training"] * 20
        trainer = TokenizerTrainer(vocab_size=500, special_tokens=special)
        tokenizer = trainer.train(texts, verbose=False)

        assert "<|bos|>" in tokenizer.vocab
        assert "<|eos|>" in tokenizer.vocab


class TestStreamingTrainer:
    """Test streaming trainer."""

    def test_feed_single(self):
        """Test feeding single text."""
        trainer = StreamingTrainer(vocab_size=500, buffer_size=100)

        trainer.feed("hello world")
        trainer.feed("test data")

        assert trainer._docs_processed == 2

    def test_feed_batch(self):
        """Test feeding batch."""
        trainer = StreamingTrainer(vocab_size=500, buffer_size=100)

        texts = ["hello world", "test data", "more text"]
        trainer.feed_batch(texts)

        assert trainer._docs_processed == 3

    def test_train_streaming(self):
        """Test streaming training."""
        texts = ["hello world"] * 50 + ["test data example"] * 30

        trainer = StreamingTrainer(vocab_size=500, buffer_size=50)
        for text in texts:
            trainer.feed(text)
        tokenizer = trainer.train(verbose=False)

        assert tokenizer is not None
        assert len(tokenizer.vocab) > 256


class TestIncrementalTrainer:
    """Test incremental trainer."""

    def test_expand_basic(self):
        """Test basic expansion."""
        # Create base tokenizer
        base = BPETokenizer(vocab_size=300)
        base.train(["hello world test data"] * 50, num_merges=20, verbose=False)

        # Expand
        trainer = IncrementalTrainer(base, target_vocab_size=500)
        trainer.add_texts(["new tokens to learn"] * 20)
        expanded = trainer.expand(num_merges=10)

        assert len(expanded.vocab) >= len(base.vocab)

    def test_get_new_tokens(self):
        """Test getting new tokens."""
        base = BPETokenizer(vocab_size=300)
        base.train(["hello world"] * 30, num_merges=20, verbose=False)

        trainer = IncrementalTrainer(base, target_vocab_size=400)
        trainer.add_texts(["completely new text here"] * 10)

        new_tokens = trainer.get_new_tokens()
        assert len(new_tokens) > 0


class TestVocabularyPruner:
    """Test vocabulary pruner."""

    def test_prune_basic(self):
        """Test basic pruning."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(
            ["test text for pruning"] * 50,
            num_merges=100,
            verbose=False
        )

        pruner = VocabularyPruner(tokenizer, min_usage=1)
        pruner.analyze_usage(["test text for pruning"] * 10)
        pruned = pruner.prune()

        assert pruned is not None

    def test_usage_analysis(self):
        """Test usage analysis."""
        tokenizer = BPETokenizer(vocab_size=500)
        tokenizer.train(["hello world"] * 50, num_merges=50, verbose=False)

        pruner = VocabularyPruner(tokenizer)
        usage = pruner.analyze_usage(["hello world hello"] * 5)

        assert isinstance(usage, dict)


class TestCodeAwareTokenizer:
    """Test code-aware tokenizer."""

    def test_tokenize_code_python(self):
        """Test Python code tokenization."""
        tokenizer = BPETokenizer(vocab_size=5000)
        tokenizer.train(
            [
                "def hello(): return 1",
                "class Test: pass",
                "import os",
            ] * 20,
            num_merges=500,
            verbose=False
        )

        code_tokenizer = CodeAwareTokenizer(tokenizer)

        code = "def createUser(): return User()"
        tokens = code_tokenizer.tokenize_code(code, "python")

        assert len(tokens) > 0

    def test_split_camel_case(self):
        """Test CamelCase splitting."""
        tokenizer = BPETokenizer(vocab_size=5000)
        tokenizer.train(["test"] * 10, num_merges=10, verbose=False)

        code_tokenizer = CodeAwareTokenizer(tokenizer)

        result = code_tokenizer.split_compound("createUser")
        assert len(result) >= 1

    def test_split_snake_case(self):
        """Test snake_case splitting."""
        tokenizer = BPETokenizer(vocab_size=5000)
        tokenizer.train(["test"] * 10, num_merges=10, verbose=False)

        code_tokenizer = CodeAwareTokenizer(tokenizer)

        result = code_tokenizer.split_compound("my_function")
        assert "my" in result or "function" in result

    def test_extract_identifiers(self):
        """Test identifier extraction."""
        tokenizer = BPETokenizer(vocab_size=5000)
        tokenizer.train(["test"] * 10, num_merges=10, verbose=False)

        code_tokenizer = CodeAwareTokenizer(tokenizer)

        code = "def createUser(): return user_id"
        ids = code_tokenizer.extract_identifiers(code)

        assert "createUser" in ids or "user_id" in ids

    def test_is_keyword(self):
        """Test keyword detection."""
        tokenizer = BPETokenizer(vocab_size=5000)
        tokenizer.train(["test"] * 10, num_merges=10, verbose=False)

        code_tokenizer = CodeAwareTokenizer(tokenizer)

        assert code_tokenizer.is_keyword("def") == True
        assert code_tokenizer.is_keyword("hello") == False
        assert code_tokenizer.is_keyword("def", "python") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])