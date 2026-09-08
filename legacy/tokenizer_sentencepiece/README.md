# Legacy: SentencePiece tokenizer artifacts

These files are from the old SentencePiece (unigram) tokenizer world that is
**not** the canonical Exp-Coder tokenizer.

- `tokenizer_stats.json` — reports an actual vocab of 200 (limited by a
  "sentencepiece Windows limitation") against a target of 50304. A 200-token
  tokenizer is not a functional Exp-Coder tokenizer.
- `tokenizer.vocab` — the 200-entry SentencePiece vocab.
- `corpus.txt` — the raw token-training corpus (may be reused to train the
  canonical byte-level BPE tokenizer via `scripts/train_tokenizer.py`).

The canonical tokenizer is `src/tokenizer/bpe_tokenizer.py` (GPT-2 style
byte-level BPE, `vocab.json` / `merges.txt` / `config.json` serialization).