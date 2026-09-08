# Exp-Coder Benchmark Results (Session 3A)

Machine: Windows, Python 3.13.5, torch 2.12.1+cpu (10 threads). **CUDA not
available in this environment** — GPU rows not measured.

## Tokenizer (byte-level BPE, 50304 target trained on 31 KB synthetic corpus)

| Metric            | Result   |
| ----------------- | -------: |
| vocab size target | 50304    |
| vocab size actual | 1607     |
| total chars       | 31,569   |
| total tokens      | 10,261   |
| chars/token       | 3.08     |
| tokens/1k chars   | 325.0    |
| UNK rate          | 0.0      |
| encode chars/sec  | 3,116,750 |
| encode tokens/sec | 1,014,848 |
| decode tokens/sec | 6,381,095 |

Per-language chars/token: python 3.32, js 2.92, cpp 2.94, rust 3.09, go 3.20,
json 2.84, yaml 3.20, markdown 3.35, shell 2.90.

Caveat: small trained vocab inflates fragmentation; treat as a lower bound for
a real 50304-vocab tokenizer.

## Tiny overfit (7.5M params, seq128/bs8, 800 steps, seed 0)

initial loss 10.7994 -> final 0.0066 (reduction 99.94%), token accuracy 1.0.
Seed 1: 10.83 -> 0.0109, accuracy 0.992. No NaN/Inf; grad & param norms finite.

LR sweep (seq64/bs4, 400 steps, seed 0):
lr 3e-4 -> 2.2114, lr 1e-3 -> 0.1004, lr 3e-3 -> 0.0414.

## 120M CPU micro-benchmark (fp32, 10-thread CPU)

| seq_len | batch | step_s | tokens/s | RAM(ws,MB) |
| ------: | ----: | -----: | -------: | ---------: |
| 128     | 1     | 0.76   | 168      | 2413       |
| 256     | 1     | 1.14   | 224      | 3000       |
| 512     | 1     | 2.16   | 237      | 3402       |
| 128     | 2     | 1.08   | 237      | 3503       |
| 256     | 2     | 1.84   | 278      | 3295       |

Model RAM alone (fp32 params): ~494 MB. Working set includes grads, Adam
states, activations, and PyTorch allocator overhead (~2.4-3.5 GB observed).

## Conclusions

- Tokenizer: technically correct (UNK 0, round-trip exact); efficiency is a
  lower bound at this vocab size — full 50304 training is recommended but the
  byte BPE mechanics are sound.
- Training stack: numerically healthy; tiny model memorizes the tiny corpus.
- 120M: runs on CPU at ~170-280 tokens/s (fp32); ~0.2-0.3 tok/s/… per step at
  these sizes. GPU (RTX 4050) numbers pending a CUDA-enabled environment.
- Safe CPU config: seq 1024, batch ~4-8, grad-accum as needed (see report).