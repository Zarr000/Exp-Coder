"""
Exp-Coder — train the canonical byte-level BPE tokenizer.

Usage:
    python scripts/train_tokenizer.py \
        --corpus <dir-or-file> \
        --output tokenizer/exp-coder \
        [--vocab-size 50304]

The default ``--vocab-size`` is the canonical Exp-Coder model vocabulary
(50304, aligned with ``configs/exp_coder_120m.yaml``).

Notes:
- Training is deterministic for a fixed corpus and seed.
- The base BPE vocabulary is 256 bytes + special tokens; a ``vocab-size``
  smaller than ~281 yields zero merge operations (harmless but pointless).
- A full 50304-vocab tokenizer on a large corpus is slow in pure Python BPE;
  start small (e.g. ``--vocab-size 2000``) for smoke runs.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_model_config
from src.data import load_text_corpus
from src.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Exp-Coder BPE tokenizer")
    parser.add_argument("--corpus", required=True, help="File or directory of training text/code")
    parser.add_argument("--output", required=True, help="Output directory (vocab.json/merges.txt/config.json)")
    parser.add_argument("--vocab-size", type=int, default=None, help="Target vocab size (default: canonical 50304)")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit documents used (debug)")
    args = parser.parse_args()

    random.seed(args.seed)
    texts = load_text_corpus(args.corpus)
    if args.max_samples:
        texts = texts[: args.max_samples]

    if not texts:
        raise ValueError(f"No text documents found in {args.corpus}")

    if args.vocab_size is None:
        model_config = load_model_config(
            Path(__file__).resolve().parent.parent / "configs/exp_coder_120m.yaml"
        )
        args.vocab_size = model_config.vocab_size

    print(f"[Tokenizer] {len(texts)} documents, target vocab {args.vocab_size}")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts, min_frequency=args.min_frequency, verbose=True)
    tokenizer.save(args.output)
    print(
        f"[Tokenizer] saved to {args.output}: "
        f"{len(tokenizer.vocab)} tokens (target {args.vocab_size})"
    )


if __name__ == "__main__":
    main()