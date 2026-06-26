"""
Vocabulary Builder for Expera AI Tokenizer
Utilities for building and analyzing vocabularies
"""

import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class VocabularyBuilder:
    """
    Utility class for building and analyzing tokenizer vocabularies.
    
    Provides tools for:
    - Analyzing token frequency distributions
    - Generating vocabulary statistics
    - Visualizing vocabulary coverage
    - Testing tokenization quality
    """
    
    def __init__(self):
        """Initialize vocabulary builder."""
        self.token_frequencies: Counter = Counter()
        self.stats: Dict = {}
        
    def analyze_corpus(
        self,
        texts: List[str],
        tokenizer,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Analyze a corpus with a given tokenizer.
        
        Args:
            texts: List of texts to analyze
            tokenizer: Tokenizer to use
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary of statistics
        """
        if max_samples:
            texts = texts[:max_samples]
            
        print(f"Analyzing {len(texts)} texts...")
        
        # Tokenize all texts
        all_tokens = []
        total_chars = 0
        total_tokens = 0
        
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            total_chars += len(text)
            total_tokens += len(tokens)
            
        # Calculate statistics
        self.token_frequencies = Counter(all_tokens)
        
        self.stats = {
            "num_texts": len(texts),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "avg_chars_per_text": total_chars / len(texts),
            "avg_tokens_per_text": total_tokens / len(texts),
            "compression_ratio": total_chars / total_tokens,
            "unique_tokens": len(self.token_frequencies),
            "vocab_coverage": len(self.token_frequencies) / len(tokenizer),
            "most_common_tokens": self.token_frequencies.most_common(20),
        }
        
        return self.stats
    
    def print_statistics(self) -> None:
        """Print vocabulary statistics."""
        if not self.stats:
            print("No statistics available. Run analyze_corpus first.")
            return
            
        print("\n" + "="*60)
        print("VOCABULARY STATISTICS")
        print("="*60)
        print(f"Number of texts: {self.stats['num_texts']:,}")
        print(f"Total characters: {self.stats['total_characters']:,}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        print(f"Average chars per text: {self.stats['avg_chars_per_text']:.2f}")
        print(f"Average tokens per text: {self.stats['avg_tokens_per_text']:.2f}")
        print(f"Compression ratio: {self.stats['compression_ratio']:.2f} chars/token")
        print(f"Unique tokens used: {self.stats['unique_tokens']:,}")
        print(f"Vocabulary coverage: {self.stats['vocab_coverage']:.2%}")
        print("\nMost common tokens:")
        for token_id, freq in self.stats['most_common_tokens'][:10]:
            print(f"  Token {token_id}: {freq:,} occurrences")
        print("="*60 + "\n")
        
    def save_statistics(self, path: str) -> None:
        """
        Save statistics to JSON file.
        
        Args:
            path: Path to save statistics
        """
        # Convert Counter to dict for JSON serialization
        stats_copy = self.stats.copy()
        stats_copy['most_common_tokens'] = [
            {"token_id": tid, "frequency": freq}
            for tid, freq in stats_copy['most_common_tokens']
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, indent=2)
            
        print(f"Statistics saved to {path}")
        
    def plot_token_distribution(
        self,
        tokenizer,
        top_n: int = 50,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot token frequency distribution.
        
        Args:
            tokenizer: Tokenizer for decoding tokens
            top_n: Number of top tokens to show
            save_path: Path to save plot (if None, display only)
        """
        if not self.token_frequencies:
            print("No token frequencies available. Run analyze_corpus first.")
            return
            
        # Get top N tokens
        top_tokens = self.token_frequencies.most_common(top_n)
        
        # Decode tokens for labels
        labels = []
        frequencies = []
        for token_id, freq in top_tokens:
            token_str = tokenizer.id_to_token(token_id)
            if token_str:
                # Truncate long tokens
                label = token_str[:20] + "..." if len(token_str) > 20 else token_str
                labels.append(f"{token_id}: {label}")
            else:
                labels.append(f"Token {token_id}")
            frequencies.append(freq)
            
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=frequencies, y=labels)
        plt.xlabel('Frequency')
        plt.ylabel('Token')
        plt.title(f'Top {top_n} Most Frequent Tokens')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def test_tokenization(
        self,
        tokenizer,
        test_samples: Optional[List[str]] = None,
    ) -> None:
        """
        Test tokenization on sample texts.
        
        Args:
            tokenizer: Tokenizer to test
            test_samples: List of test samples (if None, use defaults)
        """
        if test_samples is None:
            test_samples = [
                "def hello_world():\n    print('Hello, World!')",
                "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
                "SELECT * FROM users WHERE age > 18 ORDER BY name;",
                "The quick brown fox jumps over the lazy dog.",
                "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):",
            ]
            
        print("\n" + "="*60)
        print("TOKENIZATION TESTS")
        print("="*60)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nTest {i}:")
            print(f"Input: {sample[:100]}{'...' if len(sample) > 100 else ''}")
            
            # Encode
            tokens = tokenizer.encode(sample, add_special_tokens=False)
            print(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            
            # Decode
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"Decoded: {decoded[:100]}{'...' if len(decoded) > 100 else ''}")
            
            # Check round-trip
            if decoded == sample:
                print("✓ Round-trip successful")
            else:
                print("✗ Round-trip failed")
                print(f"  Expected: {sample}")
                print(f"  Got: {decoded}")
                
        print("="*60 + "\n")
        
    def compare_tokenizers(
        self,
        tokenizers: Dict[str, any],
        test_texts: List[str],
    ) -> Dict:
        """
        Compare multiple tokenizers on the same texts.
        
        Args:
            tokenizers: Dictionary of {name: tokenizer}
            test_texts: List of texts to test
            
        Returns:
            Comparison statistics
        """
        print("\n" + "="*60)
        print("TOKENIZER COMPARISON")
        print("="*60)
        
        results = {}
        
        for name, tokenizer in tokenizers.items():
            total_chars = sum(len(text) for text in test_texts)
            total_tokens = 0
            
            for text in test_texts:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
                
            compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            
            results[name] = {
                "total_tokens": total_tokens,
                "compression_ratio": compression_ratio,
                "vocab_size": len(tokenizer),
            }
            
            print(f"\n{name}:")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Compression ratio: {compression_ratio:.2f} chars/token")
            print(f"  Vocabulary size: {len(tokenizer):,}")
            
        print("="*60 + "\n")
        
        return results
    
    def estimate_dataset_tokens(
        self,
        tokenizer,
        sample_texts: List[str],
        total_dataset_size: int,
    ) -> Dict:
        """
        Estimate total tokens in a dataset based on samples.
        
        Args:
            tokenizer: Tokenizer to use
            sample_texts: Sample texts from dataset
            total_dataset_size: Total size of dataset in characters
            
        Returns:
            Estimation statistics
        """
        # Tokenize samples
        sample_chars = sum(len(text) for text in sample_texts)
        sample_tokens = 0
        
        for text in sample_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            sample_tokens += len(tokens)
            
        # Calculate compression ratio
        compression_ratio = sample_chars / sample_tokens if sample_tokens > 0 else 0
        
        # Estimate total tokens
        estimated_tokens = int(total_dataset_size / compression_ratio)
        
        results = {
            "sample_size_chars": sample_chars,
            "sample_size_tokens": sample_tokens,
            "compression_ratio": compression_ratio,
            "total_dataset_size_chars": total_dataset_size,
            "estimated_total_tokens": estimated_tokens,
        }
        
        print("\n" + "="*60)
        print("DATASET TOKEN ESTIMATION")
        print("="*60)
        print(f"Sample size: {sample_chars:,} characters, {sample_tokens:,} tokens")
        print(f"Compression ratio: {compression_ratio:.2f} chars/token")
        print(f"Total dataset size: {total_dataset_size:,} characters")
        print(f"Estimated total tokens: {estimated_tokens:,}")
        print("="*60 + "\n")
        
        return results
    
    @staticmethod
    def create_sample_corpus(
        output_path: str,
        num_samples: int = 1000,
        include_code: bool = True,
        include_text: bool = True,
    ) -> None:
        """
        Create a sample corpus for tokenizer training.
        
        Args:
            output_path: Path to save corpus
            num_samples: Number of samples to generate
            include_code: Whether to include code samples
            include_text: Whether to include text samples
        """
        samples = []
        
        if include_code:
            code_samples = [
                "def function_name(param1, param2):\n    return param1 + param2\n",
                "class ClassName:\n    def __init__(self):\n        self.value = 0\n",
                "for i in range(10):\n    print(i)\n",
                "if condition:\n    do_something()\nelse:\n    do_something_else()\n",
                "import numpy as np\nimport pandas as pd\n",
            ]
            samples.extend(code_samples * (num_samples // (len(code_samples) * 2)))
            
        if include_text:
            text_samples = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a high-level programming language.",
                "Data structures and algorithms are fundamental to computer science.",
                "Natural language processing enables computers to understand human language.",
            ]
            samples.extend(text_samples * (num_samples // (len(text_samples) * 2)))
            
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples[:num_samples]:
                f.write(sample + '\n')
                
        print(f"Sample corpus created at {output_path} with {num_samples} samples")
