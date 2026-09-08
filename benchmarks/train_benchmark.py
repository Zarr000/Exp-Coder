"""Exp-Coder 120M hardware (throughput / memory) micro-benchmark.

MEASUREMENT, not training. Runs a small number of optimizer steps on the
canonical 120M model and records, per (seq_len, batch) cell:

- init time, first-forward time, average step time, tokens/sec, loss
- float32 CPU path: process working-set RAM
- CUDA path (auto-skipped when unavailable): allocated/reserved/peak VRAM

Writes results/cpu_120m.json and (if CUDA) results/gpu_120m.json.

Usage:
    python benchmarks/train_benchmark.py \
        --tokenizer benchmarks/data/tokenizer \
        --data benchmarks/data/code_sample \
        --device cpu --steps 12 \
        --matrix "128,1 256,1 128,2"
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_model_config
from src.data import CausalLMDataset
from src.model.architecture import ExperaModel
from src.tokenizer import BPETokenizer
from src.training import LanguageModelingLoss
from benchmarks.common import (
    CORPUS_DIR,
    RESULT_DIR,
    cuda_info,
    force_seed,
    memory_mb,
    write_result,
)

DEFAULT_TOKENIZER = Path(__file__).resolve().parent / "data" / "tokenizer"
DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "configs" / "exp_coder_120m.yaml")


def parse_matrix(text: str) -> List[Tuple[int, int]]:
    cells = []
    for token in text.split():
        if not token:
            continue
        seq, batch = token.split(",")
        cells.append((int(seq), int(batch)))
    return cells


def benchmark_cell(
    model,
    ids,
    *,
    device: torch.device,
    steps: int,
    lr: float,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = LanguageModelingLoss()

    # warmup (excluded from timing)
    logits = model(ids, use_cache=False, return_dict=True)["logits"]
    loss = crit(logits, ids)
    opt.zero_grad()
    loss.backward()
    opt.step()

    step_times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        logits = model(ids, use_cache=False, return_dict=True)["logits"]
        loss = crit(logits, ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step_times.append(time.perf_counter() - t0)

    avg = sum(step_times) / len(step_times)
    tokens = ids.shape[0] * ids.shape[1]
    return {
        "avg_step_sec": round(avg, 4),
        "tokens_per_sec": round(tokens / avg, 1),
        "loss": round(loss.item(), 4),
        "finite": bool(torch.isfinite(loss)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp-Coder 120M hardware benchmark")
    ap.add_argument("--config", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--data", default=str(CORPUS_DIR))
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matrix", default="128,1 256,1 512,1 128,2 256,2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    info = cuda_info()
    is_cuda = info["cuda_available"] and args.device in ("cuda", "auto")

    model_config = load_model_config(args.config)
    model_config.validate()
    tokenizer = BPETokenizer.load(args.tokenizer)
    dataset = CausalLMDataset(args.data, tokenizer, sequence_length=2048)

    force_seed(args.seed)
    device = torch.device("cuda" if is_cuda else "cpu")
    cells = parse_matrix(args.matrix)

    # ---- model init timing ----
    t0 = time.perf_counter()
    model = ExperaModel(**model_config.to_model_kwargs()).to(device)
    init_sec = time.perf_counter() - t0

    # ---- first forward timing ----
    sample = dataset[0]["input_ids"].unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(sample[:, :64].to(device), use_cache=False)  # warm
    warm_t = time.perf_counter() - t0

    results = []
    for seq_len, batch in cells:
        tokens = dataset.tokens
        block = tokens[: seq_len * batch]
        if len(block) < seq_len * batch:
            block = block + [0] * (seq_len * batch - len(block))
        ids = torch.tensor(block, dtype=torch.long).view(batch, seq_len).to(device)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        mem0 = memory_mb()
        row = benchmark_cell(model, ids, device=device, steps=args.steps, lr=args.lr)
        row["seq_len"] = seq_len
        row["batch"] = batch
        row["ram_mb"] = memory_mb()
        row["ram_delta_mb"] = round((row["ram_mb"] or 0) - (mem0 or 0), 0) if row["ram_mb"] and mem0 else None
        if is_cuda:
            row["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 0)
            row["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 0)
            row["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 0)
            row["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 0)
        print(
            f"seq={seq_len} batch={batch} step={row['avg_step_sec']}s "
            f"tok/s={row['tokens_per_sec']} loss={row['loss']}"
            + (f" peak={row.get('gpu_peak_allocated_mb')}MB" if is_cuda else f" ram={row['ram_mb']}MB")
        )
        results.append(row)

    payload = {
        "run_type": "hardware_microbenchmark",
        "device": "cuda" if is_cuda else "cpu",
        "cuda_available": info["cuda_available"],
        "model_config_path": args.config,
        "model_config": model_config.to_model_kwargs(),
        "params_measured": sum(p.numel() for p in model.parameters()),
        "tokenizer": {"target_vocab": tokenizer.vocab_size, "actual_vocab": len(tokenizer.vocab)},
        "dataset": {"blocks": len(dataset), "tokens": dataset.num_tokens},
        "init_sec": round(init_sec, 2),
        "warm_forward_sec": round(warm_t, 3),
        "steps_per_cell": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "cells": results,
    }

    out = args.out or str(RESULT_DIR / ("gpu_120m.json" if is_cuda else "cpu_120m.json"))
    write_result(out, payload)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()