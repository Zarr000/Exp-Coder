"""Session 4A — direct measurement of the intended 120M config.

Config under test: seq_len=1024, batch_size=1, grad_accum=4 (fp32 CPU).

Measures: init time, forward/backward/optimizer step split, avg step time,
tokens/sec, working-set RAM. No dataset/training: feeds fixed synthetic ids so
per-stage timings are clean.

Writes results/session_4a/performance_profile.json
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import load_model_config
from src.model.architecture import ExperaModel
from src.training import LanguageModelingLoss
from benchmarks.common import memory_mb, write_result

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "session_4a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "exp_coder_120m.yaml"))
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    mc = load_model_config(args.config)
    mc.validate()

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    model = ExperaModel(**mc.to_model_kwargs())
    init_sec = time.perf_counter() - t0

    ids = torch.randint(0, mc.vocab_size, (args.batch, args.seq_len))
    crit = LanguageModelingLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # warmup step
    logits = model(ids, use_cache=False, return_dict=True)["logits"]
    crit(logits, ids).backward()
    opt.step()
    opt.zero_grad()

    fwd_t, bwd_t, opt_t, step_t = [], [], [], []
    for _ in range(args.steps):
        t = time.perf_counter()
        logits = model(ids, use_cache=False, return_dict=True)["logits"]
        t1 = time.perf_counter()
        loss = crit(logits, ids)
        loss.backward()
        t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        t3 = time.perf_counter()
        fwd_t.append(t1 - t); bwd_t.append(t2 - t1); opt_t.append(t3 - t2)
        step_t.append(t3 - t)

    n = len(step_t)
    row = {
        "seq_len": args.seq_len,
        "batch": args.batch,
        "tokens_per_step": args.seq_len * args.batch,
        "init_sec": round(init_sec, 2),
        "forward_avg_sec": round(sum(fwd_t) / n, 4),
        "backward_avg_sec": round(sum(bwd_t) / n, 4),
        "optimizer_avg_sec": round(sum(opt_t) / n, 4),
        "avg_step_sec": round(sum(step_t) / n, 4),
        "tokens_per_sec": round(args.seq_len * args.batch / (sum(step_t) / n), 1),
        "final_loss": round(loss.item(), 4),
        "ram_ws_mb": memory_mb(),
        "finite": bool(torch.isfinite(loss)),
        "cuda": torch.cuda.is_available(),
    }
    print(row)
    write_result(str(RESULTS / "performance_profile.json"), {
        "run_type": "intended_config_120m_seq1024_ga4",
        "model_config": mc.to_model_kwargs(),
        "measurement": row,
        "setup": {"lr": args.lr, "seed": args.seed, "steps": args.steps},
    })
    print(f"wrote {RESULTS/'performance_profile.json'}")


if __name__ == "__main__":
    main()