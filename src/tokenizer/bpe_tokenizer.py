"""
Byte-Pair Encoding (BPE) Tokenizer for Expera AI
Implemented from scratch following GPT-2 style byte-level BPE

This implementation:
1. Operates on UTF-8 bytes (handles any Unicode)
2. Uses byte-level pre-tokenization
3. Learns merge operations from training corpus
4. Supports special tokens for code and structure
"""

import json
import regex as re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import pickle
from pathlib import Path


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer with byte-level encoding.
    
    This tokenizer converts text to bytes, then applies BPE merges
    to create subword tokens. It can handle any Unicode text without
    unknown tokens.
    
    Attributes:
        vocab: Dictionary mapping tokens to IDs
        merges: List of merge operations (pair -> merged token)
        special_tokens: Dictionary of special tokens
        byte_encoder: Mapping from bytes to Unicode characters
        byte_decoder: Reverse mapping
    """
    
    def __init__(
        self,
        vocab_size: int = 50304,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            special_tokens: Dictionary of special tokens (pad, unk, bos, eos, etc.)
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        if special_tokens is None:
            self.special_tokens = {
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
            }
        else:
            self.special_tokens = special_tokens
            
        # Additional special tokens for code
        self.additional_special_tokens = [
            "<|code|>", "<|/code|>",
            "<|python|>", "<|javascript|>", "<|java|>", "<|cpp|>",
            "<|html|>", "<|css|>", "<|sql|>", "<|bash|>",
            "<|comment|>", "<|/comment|>",
            "<|function|>", "<|/function|>",
            "<|class|>", "<|/class|>",
            "<|image|>", "<|/image|>",
            "<|user|>", "<|assistant|>", "<|system|>",
        ]
        
        # Initialize vocabularies
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        
        # Byte-level encoding (GPT-2 style)
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Pre-tokenization pattern (GPT-2 style)
        # This pattern splits on whitespace and keeps punctuation separate
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )
        
        # Cache for encoding
        self.cache: Dict[str, List[str]] = {}
        
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Create a mapping from bytes to Unicode characters.
        
        This is the GPT-2 byte encoder that maps bytes to printable Unicode
        characters, avoiding control characters and whitespace.
        
        Returns:
            Dictionary mapping byte values (0-255) to Unicode characters
        """
        # Printable ASCII characters
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        
        # Map remaining bytes to unused Unicode characters
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
                
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs of tokens in a word.
        
        Args:
            word: Tuple of tokens
            
        Returns:
            Set of adjacent token pairs
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(
        self,
        texts: List[str],
        num_merges: Optional[int] = None,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.
        
        This implements the core BPE algorithm:
        1. Start with character-level vocabulary
        2. Iteratively merge the most frequent pair of tokens
        3. Continue until desired vocabulary size is reached
        
        Args:
            texts: List of training texts
            num_merges: Number of merge operations (if None, calculated from vocab_size)
            min_frequency: Minimum frequency for a pair to be merged
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(texts)} texts...")
            
        # Calculate number of merges needed
        if num_merges is None:
            # Base vocabulary: 256 bytes + special tokens
            base_vocab_size = 256 + len(self.special_tokens) + len(self.additional_special_tokens)
            num_merges = self.vocab_size - base_vocab_size
            
        # Initialize vocabulary with byte-level characters (GPT-2 style).
        # BPE operates on byte_encoder output characters (e.g. space -> U+0120),
        # so the base vocabulary must contain those chars -- not raw ``chr(i)``.
        # Encoding raw ``chr(i)`` here silently mapped single bytes that are
        # not in the printable-ASCII range to <|unk|> at encode time.
        vocab = {self.byte_encoder[b]: b for b in range(256)}
        
        # Add special tokens
        current_id = 256
        for token in self.special_tokens.values():
            vocab[token] = current_id
            current_id += 1
            
        for token in self.additional_special_tokens:
            vocab[token] = current_id
            current_id += 1
            
        # Convert texts to byte-level representation
        if verbose:
            print("Converting texts to byte-level representation...")
            
        word_freqs = Counter()
        for text in texts:
            # Pre-tokenize using regex pattern
            words = re.findall(self.pat, text)
            for word in words:
                # Convert to bytes, then to Unicode characters
                byte_encoded = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))
                word_freqs[byte_encoded] += 1
                
        # Convert words to tuples of characters
        splits = {word: tuple(word) for word in word_freqs.keys()}
        
        # Perform BPE merges
        if verbose:
            print(f"Performing {num_merges} merge operations...")
            
        for i in range(num_merges):
            # Count pair frequencies
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                pairs = self._get_pairs(splits[word])
                for pair in pairs:
                    pair_freqs[pair] += freq
                    
            # Find most frequent pair
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])
            if best_pair[1] < min_frequency:
                break
                
            best_pair = best_pair[0]
            
            # Merge the best pair
            self.merges[best_pair] = ''.join(best_pair)
            self.merge_ranks[best_pair] = i
            
            # Update splits
            new_splits = {}
            for word in splits:
                new_word = self._merge_pair(splits[word], best_pair)
                new_splits[word] = new_word
            splits = new_splits
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in vocab:
                vocab[merged_token] = current_id
                current_id += 1
                
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1}/{num_merges} merges, vocab size: {len(vocab)}")
                
        # Store final vocabulary
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        
        if verbose:
            print(f"Training complete! Final vocabulary size: {len(self.vocab)}")
            
    def _merge_pair(
        self,
        word: Tuple[str, ...],
        pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
        """
        Merge all occurrences of a pair in a word.
        
        Args:
            word: Tuple of tokens
            pair: Pair to merge
            
        Returns:
            New word with pairs merged
        """
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(''.join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def _bpe(self, token: str) -> List[str]:
        """
        Apply BPE merges to a token.
        
        Args:
            token: Token to encode
            
        Returns:
            List of BPE tokens
        """
        if token in self.cache:
            return self.cache[token]
            
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return [token]
            
        while True:
            # Find the pair with the lowest merge rank (earliest merge)
            bigram = min(
                pairs,
                key=lambda pair: self.merge_ranks.get(pair, float('inf'))
            )
            
            if bigram not in self.merges:
                break
                
            word = self._merge_pair(word, bigram)
            
            if len(word) == 1:
                break
                
            pairs = self._get_pairs(word)
            
        # Cache result
        result = list(word)
        self.cache[token] = result
        return result
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Pre-tokenize
        tokens = re.findall(self.pat, text)
        
        # Convert to BPE tokens
        bpe_tokens = []
        for token in tokens:
            # Convert to bytes, then to Unicode characters
            byte_encoded = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # Apply BPE
            bpe_tokens.extend(self._bpe(byte_encoded))
            
        # Convert to IDs
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.special_tokens["bos_token"]])
            
        for token in bpe_tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.special_tokens["unk_token"]])
                
        if add_special_tokens:
            ids.append(self.vocab[self.special_tokens["eos_token"]])
            
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # Convert IDs to tokens
        tokens = []
        special_token_ids = {self.vocab[t] for t in self.special_tokens.values()}
        special_token_ids.update({self.vocab[t] for t in self.additional_special_tokens})
        
        for id in ids:
            if skip_special_tokens and id in special_token_ids:
                continue
            if id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[id])
            else:
                tokens.append(self.special_tokens["unk_token"])
                
        # Join tokens and decode bytes
        text = ''.join(tokens)
        
        # Convert Unicode characters back to bytes
        try:
            byte_array = bytearray([self.byte_decoder[c] for c in text])
            decoded = byte_array.decode('utf-8', errors='replace')
        except (KeyError, UnicodeDecodeError):
            decoded = text
            
        return decoded
    
    def save(self, path: str) -> None:
        """
        Save tokenizer to disk.
        
        Args:
            path: Directory path to save tokenizer
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        # Save merges
        with open(path / "merges.txt", "w", encoding="utf-8") as f:
            for (a, b), rank in sorted(self.merge_ranks.items(), key=lambda x: x[1]):
                f.write(f"{a} {b}\n")
                
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "additional_special_tokens": self.additional_special_tokens,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        print(f"Tokenizer saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load tokenizer from disk.
        
        Args:
            path: Directory path containing tokenizer files
            
        Returns:
            Loaded tokenizer
        """
        path = Path(path)
        
        # Load config
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Create tokenizer
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            special_tokens=config["special_tokens"],
        )
        tokenizer.additional_special_tokens = config["additional_special_tokens"]
        
        # Load vocabulary
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
            
        # Load merges
        with open(path / "merges.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                a, b = line.strip().split()
                tokenizer.merges[(a, b)] = a + b
                tokenizer.merge_ranks[(a, b)] = i
                
        print(f"Tokenizer loaded from {path}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        return self.vocab.get(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """Convert ID to token."""
        return self.inverse_vocab.get(id)

    # ------------------------------------------------------------------
    # Compatibility API used by the Exp-Coder inference pipeline.
    # (generator.py / pipeline.py expect eos_id() style helpers.)
    # ------------------------------------------------------------------
    def eos_id(self) -> int:
        return self.vocab.get(self.special_tokens["eos_token"], -1)

    def bos_id(self) -> int:
        return self.vocab.get(self.special_tokens["bos_token"], -1)

    def pad_id(self) -> int:
        return self.vocab.get(self.special_tokens["pad_token"], -1)

    def unk_id(self) -> int:
        return self.vocab.get(self.special_tokens["unk_token"], -1)

    def id_to_piece(self, token_id: int) -> str:
        return self.inverse_vocab.get(token_id, self.special_tokens["unk_token"])

    def decode_ids(self, ids: List[int]) -> str:
        """Decode a list of IDs (alias of :meth:`decode`)."""
        return self.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self.eos_id()

    @property
    def bos_token_id(self) -> int:
        return self.bos_id()

    @property
    def pad_token_id(self) -> int:
        return self.pad_id()

    @property
    def unk_token_id(self) -> int:
        return self.unk_id()
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BPETokenizer(vocab_size={len(self.vocab)})"
