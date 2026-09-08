"""Tiny Exp-Coder overfit experiment.

Proves the training stack can drive a TINY model to MEMORIZE a tiny dataset
(generic LM pre-training sanity, not quality). Measures:

- loss curve (initial/final/reduction)
- gradient and parameter norms
- token accuracy
- NaN/Inf detection
- optional small learning-rate comparison

Benchmark-only model config (does not modify production 120M config).

Usage:
    python benchmarks/overfit.py \
        --tokenizer benchmarks/data/tokenizer \
        --data benchmarks/data/code_sample \
        --lr 1e-3 --max-steps 800 --seed 0 \
        --lrs "1e-3,5e-4,2e-3" --sweep-steps 400
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.data import CausalLMDataset
from src.model.architecture import ExperaModel
from src.tokenizer import BPETokenizer
from src.training import LanguageModelingLoss
from benchmarks.common import RESULT_DIR, force_seed, write_result

DEFAULT_TOKENIZER = Path(__file__).resolve().parent / "data" / "tokenizer"
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "code_sample"

TINY_MODEL = dict(
    hidden_size=128,
    num_layers=3,
    num_heads=4,
    num_kv_heads=4,
    intermediate_size=512,
    max_position_embeddings=512,
    activation="gelu",
    dropout=0.0,
    attention_dropout=0.0,
)


def build_model(vocab_size: int) -> ExperaModel:
    return ExperaModel(vocab_size=vocab_size, **TINY_MODEL)


def run(
    tokenizer,
    dataset,
    *,
    lr: float,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    report_every: int = 5,
) -> dict:
    force_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model = build_model(tokenizer.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = LanguageModelingLoss()

    curve = []
    losses = []
    it = iter(loader)
    unstable = {"nan_step": None, "inf_grad": None, "inf_param": None}

    t0 = time.perf_counter()
    step = 0
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        ids = batch["input_ids"]
        logits = model(ids, use_cache=False, return_dict=True)["logits"]

        loss = crit(logits, ids)
        opt.zero_grad()
        loss.backward()

        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        grad_norm = grads.norm().item()
        param_norm = grad_norm  # placeholder replaced below

        if not torch.isfinite(loss):
            unstable["nan_step"] = step
            break
        if not torch.isfinite(grads).all():
            unstable["inf_grad"] = step
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            param_norm = math.sqrt(
                sum((p.data.norm() ** 2).item() for p in model.parameters())
            )
            if step == max_steps - 1 or step % 10 == 0:
                # loss supervises logits[:,t] -> labels[:,t+1]; align accuracy likewise
                pred = logits.argmax(-1)
                aligned = (pred[:, :-1] == ids[:, 1:])
                acc = aligned.float().mean().item()
            else:
                acc = None
            if step % 10 == 0:
                param_finite = all(torch.isfinite(p.data).all() for p in model.parameters())
                if not param_finite:
                    unstable["inf_param"] = step
                    break

        record = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": round(grad_norm, 4),
            "param_norm": round(param_norm, 1),
            "accuracy": acc,
        }
        curve.append(record)
        losses.append(loss.item())
        step += 1

    elapsed = time.perf_counter() - t0
    final = losses[-1] if losses else float("nan")
    first = losses[0] if losses else float("nan")
    return {
        "lr": lr,
        "seed": seed,
        "steps_run": step,
        "max_steps": max_steps,
        "initial_loss": round(first, 4),
        "final_loss": round(final, 4),
        "loss_reduction_frac": round((1 - final / max(first, 1e-9)), 4),
        "final_accuracy": curve[-1]["accuracy"] if curve else None,
        "elapsed_sec": round(elapsed, 1),
        "unstable": unstable,
        "curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp-Coder tiny overfit experiment")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lrs", default=None, help="comma list, e.g. '1e-3,5e-4,1e-2'")
    ap.add_argument("--sweep-steps", type=int, default=400)
    ap.add_argument("--out", default=None,
                    help="override result path (default: results/overfit.json "
                         "or results/overfit_lr_sweep.json when --lrs given)")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(
            RESULT_DIR / ("overfit_lr_sweep.json" if args.lrs else "overfit.json")
        )

    tokenizer = BPETokenizer.load(args.tokenizer)
    dataset = CausalLMDataset(args.data, tokenizer, sequence_length=args.seq_len)
    print(
        f"tiny overfit: vocab={tokenizer.vocab_size}(actual {len(tokenizer.vocab)}), "
        f"blocks={len(dataset)}, tokens={dataset.num_tokens}, seq={args.seq_len}, "
        f"bs={args.batch_size}"
    )

    lrs = (
        [float(x) for x in args.lrs.split(",")]
        if args.lrs else [args.lr]
    )

    runs = {}
    for lr in lrs:
        steps = args.sweep_steps if len(lrs) > 1 else args.max_steps
        res = run(
            tokenizer, dataset,
            lr=lr, max_steps=steps, batch_size=args.batch_size,
            seq_len=args.seq_len, seed=args.seed,
        )
        tag = f"lr_{lr:g}"
        runs[tag] = res
        print(
            f"[{tag}] loss {res['initial_loss']} -> {res['final_loss']} "
            f"acctok={res['final_accuracy']} steps={res['steps_run']} "
            f"unstable={res['unstable']} {res['elapsed_sec']}s"
        )
        # CSV curve for each run
        csv_out = RESULT_DIR / f"overfit_curve_{lr:g}_s{args.seed}.csv"
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res["curve"][0].keys()))
            w.writeheader()
            w.writerows(res["curve"])

    # one canonical 2-seed replication of the primary LR
    if len(lrs) == 1:
        res2 = run(
            tokenizer, dataset,
            lr=args.lr, max_steps=args.max_steps, batch_size=args.batch_size,
            seq_len=args.seq_len, seed=args.seed + 1,
        )
        runs["lr_%g_seed_%d" % (args.lr, args.seed + 1)] = res2
        print(
            f"[seed {args.seed+1}] loss {res2['initial_loss']} -> "
            f"{res2['final_loss']} steps={res2['steps_run']}"
        )

    result = {
        "run_type": "tiny_overfit",
        "model": {"vocab_size": tokenizer.vocab_size, "actual_vocab": len(tokenizer.vocab), **TINY_MODEL},
        "dataset": {
            "path": str(Path(args.data).resolve()),
            "blocks": len(dataset),
            "tokens": dataset.num_tokens,
            "sequence_length": args.seq_len,
            "batch_size": args.batch_size,
        },
        "runs": runs,
    }
    write_result(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()