# Exp-Coder Benchmarks (Session 3A)

Scientific validation & hardware measurement. Benchmark code lives here, kept
separate from production `src/`. Generated data and trained benchmark tokenizers
are NOT committed (see `.gitignore`); regenerate everything deterministically.

## Regenerate

```bash
# 1. deterministic multilingual code corpus (31 KB, 28 docs)
python benchmarks/make_code_corpus.py

# 2. train benchmark BPE tokenizer + measure metrics
python benchmarks/tokenizer_benchmark.py          # -> results/tokenizer.json

# 3. tiny-model overfit (canonical 800 steps x 2 seeds)
python benchmarks/overfit.py --tokenizer benchmarks/data/tokenizer \
    --data benchmarks/data/code_sample --max-steps 800 --seed 0

# 4. optional: learning-rate micro comparison (400 steps x 3 LRs)
python benchmarks/overfit.py --tokenizer benchmarks/data/tokenizer \
    --data benchmarks/data/code_sample --seq-len 64 --batch-size 4 \
    --lrs "3e-4,1e-3,3e-3"                       # -> results/overfit_lr_sweep.json

# 5. 120M hardware micro-benchmark (CPU-first; CUDA auto when available)
python benchmarks/train_benchmark.py --device cpu \
    --tokenizer benchmarks/data/tokenizer --data benchmarks/data/code_sample
```

## Result files

- `results/tokenizer.json` — tokenizer efficiency/speed/fragmentation
- `results/overfit.json` — canonical tiny-model overfit (800 steps × 2 seeds)
- `results/overfit_lr_sweep.json` — LR comparison (same stack)
- `results/cpu_120m.json` — CPU throughput/memory matrix for the canonical 120M
- `results/gpu_120m.json` — written only when CUDA is available (not in env)
- `results/overfit_curve_*.csv` — per-step loss/grad_norm/param_norm/accuracy

Every JSON embeds `meta.commit` and `meta.timestamp` (see `common.py`).

## Notes on the benchmark tokenizer

The benchmark tokenizer is trained on the 31 KB synthetic corpus with the
canonical 50304 target. On this small corpus merges exhaust early (actual
vocab ≈1600 tokens). This is intentional: it keeps tokenizer training fast
(<1 s) and reproducible, and it exercises the byte-level BPE mechanics.
Real-vocab (50304) fragmentation will be more efficient than measured here;
use these numbers as a lower bound, not the final 50304 result.