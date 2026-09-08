# Exp-Coder Core Architecture

> Status: implementation of record for the current core pipeline.
> Verified by the test suite (`pytest -q`) on CPU.

## Pipeline

```
Dataset (plain text/code)
   │
   ▼
BPE Tokenizer  (src/tokenizer/bpe_tokenizer.py)
   │   raw text -> byte-level BPE token ids
   ▼
CausalLMDataset  (src/data/causal_lm_dataset.py)
   │   fixed-length blocks; input_ids == labels (causal shift is inside the loss)
   ▼
DataLoader  (torch DataLoader, batch_size / sequence_length)
   │
   ▼
ExperaModel  (src/model/architecture/)
   │   decoder-only transformer: pre-norm, RoPE, GQA, tied embeddings
   ▼
LanguageModelingLoss  (src/training/loss.py)
   │   cross-entropy shifted by one position: logits[:, :-1] vs labels[:, 1:]
   ▼
backward -> optimizer (AdamW) -> scheduler (cosine+linear warmup)
   │   gradient accumulation + clipping, bf16/fp16 AMP on CUDA only
   ▼
Checkpoint  (exp-coder-v1: model/optimizer/scheduler/RNG/step/config/tokenizer)
   │
   ├── resume training
   ▼
Inference  (src/inference/generator.py + exp_coder.generate CLI)
   │   greedy / sampling / beam(exp) / contrastive(exp), optional KV cache
   ▼
Text generation
```

## Canonical configuration

`configs/exp_coder_120m.yaml` is the single source of truth (loaded only
through `src.config.load_model_config`). Details:

- `vocab_size: 50304` (must equal the tokenizer target vocab; enforced)
- `hidden_size: 768`, `num_layers: 12`, `num_heads: 12` (head_dim = 64)
- `intermediate_size: 3072` (4× hidden_size, GELU activation)
- `max_position_embeddings: 2048`, `rope_theta: 10000`
- tied embeddings (no separate LM-head parameters)
- measured: ~123.6M parameters

`configs/exp_coder_tiny.yaml` is a debug-only config for fast iteration.
`configs/training.yaml` holds the canonical training-run settings.

## Tokenizer

`src/tokenizer/bpe_tokenizer.py` = GPT-2 style byte-level BPE:

- base vocabulary: the 256 byte-encoder characters (byte -> printable unicode)
- special tokens: pad/unk/bos/eos (`<|pad|>`, `<|unk|>`, `<|startoftext|>`,
  `<|endoftext|>`) plus code/structure tokens
- serialized as `vocab.json`, `merges.txt`, `config.json` in one directory
- round-trip `text -> ids -> text` is stable modulo GPT-2 whitespace
  pre-tokenization rules; `add_special_tokens=True` pins BOS/EOS.
- the canonical model/tokenizer vocab are both 50304; divergence raises a
  clear `ValueError` (`src.config.check_vocab_consistency`).

Other tokenizers referenced in legacy dirs (`tokenizer_sentencepiece/`,
the old `SentencePieceTokenizer` import) are obsolete and archived.

## Model

`src/model/architecture/`:

- `expera_model.py` — `ExperaModel`: embeddings -> N transformer blocks ->
  final LayerNorm -> tied LM head. Implements causal mask construction and the
  autoregressive `generate()` loop (KV cache aware). Causal mask semantics:
  query row `i` (global position `past+i`) attends columns `j <= past+i`.
- `attention.py` — scaled dot-product attention, RoPE integration, GQA
  (`num_kv_heads <= num_heads`, KV heads repeated per group), incremental KV
  cache via `past_key_values`.
- `embeddings.py` — `TokenEmbedding`, `EmbeddingModule` (token embedding +
  LayerNorm + dropout), `RotaryPositionalEmbedding`.
- `transformer_block.py` — pre-norm block with residuals.
- `feedforward.py` — GELU/ReLU/SiLU/SwiGLU FFN.

The module name `ExperaModel` is retained for compatibility; it is the
canonical Exp-Coder model (aliased conceptually to "ExpCoderModel").

## Training

`src/training/`:

- `trainer.py` — `Trainer` + `TrainerConfig`: forward/backward, gradient
  accumulation and clipping, AMP (CUDA-only; CPU runs fp32), optional EMA,
  periodic validation + checkpointing, checkpoint save/load with RNG restore.
- `loss.py` — `LanguageModelingLoss` (causal shift + `ignore_index=-100`).
- `optimizer.py` / `scheduler.py` — AdamW with weight-decay grouping,
  cosine-with-warmup scheduler.
- `checkpoint_manager.py` — the canonical `exp-coder-v1` checkpoint writer
  (embedded `config` and `tokenizer_metadata`, RNG states).
- `src/utils/checkpoint.py` — deprecated legacy manager; do not use in new
  code (kept for historical scripts).

Entry points:

- `scripts/train_tokenizer.py` — train the canonical BPE tokenizer.
- `scripts/train.py` — canonical training entry (CPU-first).
- `python -m exp_coder.generate` — canonical generation CLI.

## Checkpoint format (`exp-coder-v1`)

```python
{
  "format": "exp-coder-v1",
  "step": int, "epoch": int, "metrics": {...},
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "scheduler_state_dict": ...,   # optional
  "scaler_state_dict": ...,      # optional
  "rng_state": {"python":..., "torch":..., "cuda":...},
  "config": {...},               # model/training config metadata
  "tokenizer_metadata": {...},   # vocab size / special tokens
}
```

`Trainer.save_checkpoint` and `CheckpointManager.save` write the same format.
`Trainer` restores RNG state on resume (documented guarantee: state vectors
for python/torch/CUDA RNG are restored; bit-exact continuation depends on the
same PyTorch build and single-thread determinism).

## Inference

`src/inference/generator.py`:

- `GreedySearch`, `Sampling` — both support the incremental KV cache; cached
  and uncached runs are asserted equal for the same prompt/seed.
- `BeamSearch` — simple arithmetic-correct implementation (experimental; no
  extra finished-beam masking).
- `ContrastiveSearch` — experimental, minimally repaired.
- `Generator` facade handles BPE tokenizer encode/decode, padding of ragged
  prompts, and single/list prompts.

`src/inference/pipeline.py` — `InferencePipeline`; loads BPETokenizer
directories first, SentencePiece `tokenizer.model` as a legacy fallback.

## Deferred / experimental (not required for the core)

The following subsystems exist in the repo but are intentionally left
untouched by the core pipeline and are **not** verified:

- `agents/`, `rag/`, `memory/`, `multimodal/`, `image/`
- `quantization/`, `runtime/`, `server/`, `alignment/`, `tools/`
- `deploy/`
- model experiments: `moe.py`, `speculative.py`, `flash_attention.py`,
  `grouped_ffn.py`, `rope.py` (YaRN/NTK/linear variants), `cache.py`
  (pre-allocated / paged / streaming KV managers)

They may be revisited once the core model is stable and the actual
bottlenecks are measured (no native/CUDA work before profiling).

## Test strategy

`tests/` runs on CPU in ~12s:

- tokenizer round-trip + save/load + vocab consistency
- causal mask values (with/without past)
- RoPE shape/determinism/rotation
- GQA equivalence to full attention
- loss/backward/optimizer step
- checkpoint round-trip + resume + train→save→load→inference
- greedy/sampling cached == uncached (multiple prompt lengths)
- end-to-end CPU smoke (tokenizer → dataset → train → checkpoint → inference)
- canonical 120M parameter count