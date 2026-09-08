"""Session 4A — 120M learning-rate sweep on the real code corpus (CPU).

Same dataset, seed, architecture, batch, and scheduler for every LR.
Only the LR differs. Bounded: ``--steps`` optimizer steps per LR.

Records per-step loss/grad/param metrics, then evaluates validation
perplexity (seq 512, deterministic, full val split) at the end of each run.

Outputs:
  results/session_4a/lr_sweep.json
  results/session_4a/lr_sweep.csv
"""

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import load_model_config
from src.data import CausalLMDataset
from src.model.architecture import ExperaModel
from src.training import LanguageModelingLoss, get_scheduler
from benchmarks.common import force_seed, write_result

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "benchmarks" / "data" / "session4_corpus"
TOK_DIR = CORPUS / "tokenizer_50304"
RESULTS = ROOT / "benchmarks" / "results" / "session_4a"

LRS_DEFAULT = [3e-4, 7e-4, 1e-3]


def evaluate(model, tokenizer, seq_len: int, device: torch.device) -> dict:
    """Deterministic full-val loss (no shuffle). Returns loss + stable ppl."""
    val = CausalLMDataset(str(CORPUS / "val"), tokenizer, sequence_length=seq_len)
    loader = DataLoader(val, batch_size=4, shuffle=False)
    crit = LanguageModelingLoss()
    model.eval()
    total, n, batches = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            logits = model(ids, use_cache=False, return_dict=True)["logits"]
            loss = crit(logits, ids)
            total += loss.item() * ids.shape[0]
            n += ids.shape[0]
            batches += 1
    model.train()
    mean_loss = total / max(1, n)
    return {"val_loss": round(mean_loss, 4),
            "val_perplexity": round(math.exp(mean_loss), 2),
            "val_batches": batches}


def run_lr(tokenizer, lr, steps, batch, seq_len, warmup, seed, device) -> dict:
    force_seed(seed)
    mc = load_model_config(ROOT / "configs" / "exp_coder_120m.yaml")
    model = ExperaModel(**mc.to_model_kwargs()).to(device)

    train = CausalLMDataset(str(CORPUS / "train"), tokenizer, sequence_length=seq_len)
    loader = DataLoader(train, batch_size=batch, shuffle=True)
    it = iter(loader)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = get_scheduler(opt, {
        "name": "cosine_with_warmup",
        "warmup_steps": warmup,
        "num_training_steps": steps,
        "min_lr_ratio": 0.1,
    })
    crit = LanguageModelingLoss()

    curve = []
    t0 = time.perf_counter()
    unstable = {"nan_step": None, "inf_grad": None, "inf_param": None}
    step = 0
    while step < steps:
        try:
            batch_ids = next(it)["input_ids"]
        except StopIteration:
            it = iter(loader)
            batch_ids = next(it)["input_ids"]

        logits = model(batch_ids, use_cache=False, return_dict=True)["logits"]
        loss = crit(logits, batch_ids)
        opt.zero_grad()
        loss.backward()

        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        if not torch.isfinite(loss):
            unstable["nan_step"] = step
            break
        if not torch.isfinite(grads).all():
            unstable["inf_grad"] = step
            break
        if step % 20 == 0:
            if not all(torch.isfinite(p.data).all() for p in model.parameters()):
                unstable["inf_param"] = step
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                gnorm = grads.norm().item()
                pnorm = math.sqrt(sum((p.data.norm() ** 2).item() for p in model.parameters()))
            curve.append({"step": step, "loss": round(loss.item(), 4),
                          "grad_norm": round(gnorm, 3),
                          "param_norm": round(pnorm, 1),
                          "lr": sched.get_last_lr()[0]})
        step += 1

    elapsed = time.perf_counter() - t0

    # init loss reference (fresh-model val) computed once outside per-LR loop
    val = evaluate(model, tokenizer, seq_len=512, device=device)

    # loss at 25/50/75% sampled from curve (nearest recorded step)
    sample = {}
    for frac, name in ((0.25, "q25"), (0.5, "q50"), (0.75, "q75")):
        target = int(steps * frac)
        best = min(curve, key=lambda c: abs(c["step"] - target))
        sample[name] = best["loss"]

    run = {
        "lr": lr,
        "seed": seed,
        "steps_run": step,
        "steps_planned": steps,
        "seq_len": seq_len,
        "batch": batch,
        "initial_loss": curve[0]["loss"] if curve else None,
        "q25_loss": sample.get("q25"),
        "q50_loss": sample.get("q50"),
        "q75_loss": sample.get("q75"),
        "final_train_loss": curve[-1]["loss"] if curve else None,
        "elapsed_sec": round(elapsed, 1),
        "tokens_per_sec": round(step * seq_len * batch / elapsed, 1),
        "unstable": unstable,
        "validation": val,
        "curve": curve,
    }
    return run


def _write_results(args, lrs, runs, tokenizer, out=None):
    path = str(out or RESULTS / "lr_sweep.json")
    write_result(path, {
        "run_type": "120m_lr_sweep_real_corpus",
        "lrs": lrs,
        "steps_per_lr": args.steps,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "warmup": args.warmup,
        "seed": args.seed,
        "tokenizer_vocab": tokenizer.vocab_size,
        "runs": runs,
    })
    with open(RESULTS / "lr_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lr", "step", "train_loss", "grad_norm", "lr_step"])
        for key, run in runs.items():
            for c in run["curve"]:
                w.writerow([key, c["step"], c["loss"], c["grad_norm"], c["lr"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", default=",".join(map(str, LRS_DEFAULT)))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")
    from src.tokenizer import BPETokenizer
    tokenizer = BPETokenizer.load(str(TOK_DIR))
    print(f"tokenizer 50304 loaded (actual vocab {len(tokenizer.vocab)})")

    lrs = [float(x) for x in args.lrs.split(",")]
    runs = {}
    for lr in lrs:
        gc.collect()
        print(f"=== LR {lr:g} ({args.steps} steps, seq {args.seq_len}, batch {args.batch}) ===")
        run = run_lr(tokenizer, lr, args.steps, args.batch, args.seq_len,
                     args.warmup, args.seed, device)
        runs[f"lr_{lr:g}"] = run
        v = run["validation"]
        print(f"[lr {lr:g}] train {run.get('final_train_loss')} -> val {v['val_loss']} "
              f"ppl {v['val_perplexity']} | {run['elapsed_sec']}s | unstable={run['unstable']}")
        # durable partial writes so a timeout never loses completed LR runs
        _write_results(args, lrs, runs, tokenizer)

    _write_results(args, lrs, runs, tokenizer)
    best = min(runs.items(), key=lambda kv: kv[1]["validation"]["val_loss"])
    print(f"BEST LR by validation loss: {best[0]} ({best[1]['validation']['val_loss']})")
    print("wrote lr_sweep.json / lr_sweep.csv")


if __name__ == "__main__":
    main()