# Exp-Coder

**Original language model from Exp Works. Author: Zarr.**

Exp-Coder is an original, decoder-only GPT-style transformer for
software development, built from scratch in Python + PyTorch.

Current development model: **Exp-Coder 120M**
(`configs/exp_coder_120m.yaml`):

```yaml
model:
  vocab_size: 50304
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_position_embeddings: 2048
  activation: gelu
```

Architecture features: pre-normalization, RoPE positional embeddings, GQA
(grouped-query attention) support, tied input/output embeddings, causal
masking, optional KV cache for fast generation.

Measured parameters for the canonical config: **~123.6M**.
(small deviation from "120M" is expected; the name is approximate.
`python -c "from src.config import load_model_config; ..."` verifies.)

## What is implemented and verified

This is **Alpha**. The following pipeline is real, tested, and runs on CPU:

```
Dataset (text/code)
  -> BPE tokenizer (src/tokenizer/bpe_tokenizer.py, GPT-2 style byte BPE)
  -> CausalLMDataset + DataLoader (src/data/causal_lm_dataset.py)
  -> ExperaModel (src/model/architecture/)
  -> LanguageModelingLoss (src/training/loss.py)
  -> backward / optimizer / scheduler (src/training/)
  -> checkpoint (exp-coder-v1 format)
  -> load/resume -> inference (src/inference/ + CLI)
```

Tokenizer:<br>
- 256 byte-level base tokens + special tokens (pad/unk/bos/eos + code tokens).
- Serialized as `vocab.json` + `merges.txt` + `config.json` in one directory.
- **Model vocab and tokenizer vocab must match** — enforced by
  `src.config.check_vocab_consistency` (fails loudly, never resizes silently).

Checkpoints (`exp-coder-v1`):<br>
- model / optimizer / scheduler state, step, epoch, metrics, RNG state,
  config + tokenizer metadata.

Training:<br>
- CPU-first (no CUDA required). CUDA is auto-detected and enables
  bf16/fp16 mixed precision when present.
- Gradient accumulation, clipping, cosine-with-warmup schedule, EMA optional.

## Deferred / experimental subsystems (NOT production)

The repository also contains many newer/experimental folders. They exist but
are **not** part of the verified core and are not required to build, train,
or run Exp-Coder:

`agents/`, `rag/`, `memory/`, `multimodal/`, `image/`, `quantization/`,
`runtime/`, `server/`, `alignment/`, `tools/`, `deploy/`,
plus experimental model modules (`moe.py`, `speculative.py`,
`flash_attention.py`, `grouped_ffn.py`, `rope.py` variants, `cache.py`
pre-allocated managers).

The core model does not import any of these.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.0+. Everything works with CPU-only PyTorch.
CUDA-enabled PyTorch is recommended for real training.

## Tests

```bash
pytest -q          # 200+ tests, runs on CPU, ~12s
```

## Train a tiny (debug) model on CPU

1. Train the canonical BPE tokenizer on a text/code corpus:
   ```bash
   python scripts/train_tokenizer.py \
       --corpus <dir-or-file> \
       --output tokenizer/exp-coder \
       --vocab-size 50304
   ```
   (For a quick smoke test use a small corpus and let merges run out; the
   tokenizer still saves fine. Tokenizer training is pure-Python BPE — slow on
   large corpora/50304 vocab.)

2. Train the model:
   ```bash
   python scripts/train.py \
       --config configs/exp_coder_tiny.yaml \
       --tokenizer tokenizer/exp-coder \
       --data <corpus dir> \
       --output checkpoints/exp-coder-tiny \
       --device cpu \
       --max-steps 100 --batch-size 4 --sequence-length 128
   ```

3. Generate:
   ```bash
   python -m exp_coder.generate \
       --tokenizer tokenizer/exp-coder \
       --checkpoint checkpoints/exp-coder-tiny/final.pt \
       --config configs/exp_coder_tiny.yaml \
       --prompt "def fibonacci" \
       --device cpu --strategy greedy
   ```

An untrained/lightly-trained model typically emits the EOS token immediately
and produces empty output — that is expected, not a bug.

## Train Exp-Coder 120M

```bash
python scripts/train_tokenizer.py --corpus <data> --output tokenizer/exp-coder
python scripts/train.py \
    --config configs/exp_coder_120m.yaml \
    --training configs/training.yaml \
    --tokenizer tokenizer/exp-coder \
    --data <data> \
    --output checkpoints/exp-coder-120m
```

Resume training:

```bash
python scripts/train.py ... --resume checkpoints/exp-coder-120m/step_1000.pt
```

## Inference

```bash
python -m exp_coder.generate \
    --checkpoint checkpoints/exp-coder-120m/final.pt \
    --tokenizer tokenizer/exp-coder \
    --prompt "Write a Python function that calculates Fibonacci numbers." \
    --strategy greedy | sampling | beam \
    --max-new-tokens 256 --temperature 0.7 --top-k 40 --top-p 0.9
```

No network access, no downloads. Greedy and sampling use the incremental KV
cache (verified equivalent to uncached generation).

## Hardware

- CPU: fully supported (all core tests and smoke training run CPU-only).
- CUDA: auto-detected; bf16/fp16 AMP on GPU only.
- This repo was developed/validated on an RTX 4050 laptop (≈6 GB VRAM) class
  machine — keep `--sequence-length`/batch modest for 120M on 6 GB.

## Legacy code

`configs/legacy/`, `scripts/legacy/`, and `legacy/tokenizer_sentencepiece/`
hold archived pre-Exp-Coder material for reference. They are not canonical
and several reference missing/obsolete components (e.g. a
`SentencePieceTokenizer` class that never existed in `src.tokenizer`,
`LlamaTokenizer`/`LlamaForCausalLM` from an unrelated framework).

## License

MIT (or your preferred license).