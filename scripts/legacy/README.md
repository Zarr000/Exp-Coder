# Legacy scripts archive

These scripts are **deprecated** and kept for reference only. They are not
tracks through the canonical Exp-Coder pipeline.

- `pretrain.py` — imported a `SentencePieceTokenizer` class that does not
  exist in `src.tokenizer`, passed non-existent `TrainerConfig` fields, and
  called `get_scheduler` with the wrong arity. Broken end to end.
- `train_120m.py` / `train_350m.py` / `train_1b.py` — standalone character-
  level models with a different architecture from the canonical
  `ExperaModel`, hard-wired to CUDA (`torch.cuda.current_device()`), so they
  crash on CPU machines.

Use instead:

- `scripts/train.py`            — canonical training entry point (CPU-first)
- `scripts/train_tokenizer.py`  — train the canonical BPE tokenizer
- `python -m exp_coder.generate` — canonical inference entry point