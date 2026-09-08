"""
Exp-Coder — canonical training entry point.

Trains ``ExperaModel`` (decoder-only GPT-style transformer) on plain
text/code data with the canonical byte-level BPE tokenizer.

Runs on CPU by default; CUDA is auto-detected (``--device auto``) and
enables bf16/fp16 mixed precision.

Usage:
    python scripts/train_tokenizer.py --corpus <dir> --output tokenizer/exp-coder
    python scripts/train.py \
        --config configs/exp_coder_120m.yaml \
        --training configs/training.yaml \
        --tokenizer tokenizer/exp-coder \
        --data <corpus dir or file> \
        --output checkpoints/exp-coder-120m

    # resume training
    python scripts/train.py ... --resume checkpoints/exp-coder-120m/step_1000.pt

    # override any training.yaml value on the command line
    python scripts/train.py ... --max-steps 100 --batch-size 2 \
        --sequence-length 64 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root so `src.*` and `exp_coder.*` are importable without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import (
    load_model_config,
    load_training_config,
    check_vocab_consistency,
)
from src.data import CausalLMDataset
from src.model.architecture import ExperaModel
from src.tokenizer import BPETokenizer
from src.training import LanguageModelingLoss, Trainer, TrainerConfig, get_optimizer, get_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp-Coder model on CPU/CUDA")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs/exp_coder_120m.yaml"))
    parser.add_argument("--training", default=str(Path(__file__).resolve().parent.parent / "configs/training.yaml"))
    parser.add_argument("--tokenizer", required=True, help="Path to trained BPETokenizer directory")
    parser.add_argument("--data", required=True, help="Corpus file or directory (txt/py/md/jsonl)")
    parser.add_argument("--eval-data", default=None, help="Optional eval corpus file/directory")
    parser.add_argument("--output", default=None, help="Override checkpoint output_dir")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="Override device (default: auto)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint .pt file to resume from")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use --device cpu or install a "
            "CUDA-enabled PyTorch build."
        )
    return requested


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    model_config = load_model_config(args.config)
    model_config.validate()
    train_cfg = load_training_config(args.training)

    for key, value in (
        ("max_steps", args.max_steps),
        ("batch_size", args.batch_size),
        ("sequence_length", args.sequence_length),
        ("learning_rate", args.learning_rate),
        ("seed", args.seed),
    ):
        if value is not None:
            train_cfg[key] = value
    if args.output:
        train_cfg["output_dir"] = args.output

    device = args.device or train_cfg.get("device", "auto")
    device = resolve_device(device)
    use_amp = bool(train_cfg.get("use_amp", True)) and device == "cuda"

    if "seed" in train_cfg and train_cfg["seed"] is not None:
        torch.manual_seed(int(train_cfg["seed"]))
        import random
        random.seed(int(train_cfg["seed"]))

    # ------------------------------------------------------------------
    # Tokenizer (canonical byte-level BPE)
    # ------------------------------------------------------------------
    tokenizer_dir = Path(args.tokenizer)
    if not (tokenizer_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_dir}. Train it first:\n"
            f"  python scripts/train_tokenizer.py --corpus <data> --output {tokenizer_dir}"
        )
    tokenizer = BPETokenizer.load(str(tokenizer_dir))
    check_vocab_consistency(tokenizer, model_config)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    seq_len = int(train_cfg.get("sequence_length", 1024))
    batch_size = int(train_cfg.get("batch_size", 8))

    train_set = CausalLMDataset(args.data, tokenizer, sequence_length=seq_len)
    if len(train_set) == 0:
        raise ValueError("Training corpus produced no sequence blocks.")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    val_loader = None
    if args.eval_data:
        val_set = CausalLMDataset(args.eval_data, tokenizer, sequence_length=seq_len)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    model = ExperaModel(**model_config.to_model_kwargs()).to(torch.device(device))
    params = model.get_num_params()
    print(
        f"[Exp-Coder] model {params['total']:,} params "
        f"({params['total'] / 1e6:.1f}M), device={device}, "
        f"vocab={model_config.vocab_size}"
    )

    max_steps = int(train_cfg.get("max_steps", 1000))
    optimizer = get_optimizer(
        model.named_parameters(),
        {
            "name": "adamw",
            "learning_rate": float(train_cfg.get("learning_rate", 3e-4)),
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
    )
    warmup_steps = int(train_cfg.get("warmup_steps", max(1, int(max_steps * 0.05))))
    warmup_steps = max(1, min(warmup_steps, max_steps))
    scheduler = get_scheduler(
        optimizer,
        {
            "name": "cosine_with_warmup",
            "warmup_steps": warmup_steps,
            "num_training_steps": max_steps,
            "min_lr_ratio": 0.1,
        },
    )

    eval_interval = int(train_cfg.get("eval_interval", 0))
    save_interval = int(train_cfg.get("save_interval", max(100, max_steps // 10)))
    log_interval = int(train_cfg.get("logging_interval", 10))

    trainer_config = TrainerConfig(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=LanguageModelingLoss(),
        device=device,
        max_steps=max_steps,
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        use_amp=use_amp,
        amp_dtype=torch.bfloat16 if train_cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16,
        save_every=save_interval,
        save_dir=args.output or str(train_cfg.get("output_dir", "checkpoints/exp-coder")),
        keep_last_n=3,
        val_dataloader=val_loader,
        val_every=eval_interval if eval_interval > 0 and val_loader else max_steps + 1,
        use_ema=bool(train_cfg.get("use_ema", False)),
        log_every=log_interval,
        checkpoint_meta={
            "model_config_path": str(Path(args.config).resolve()),
            "training_config_path": str(Path(args.training).resolve()),
            "tokenizer_path": str(tokenizer_dir.resolve()),
            "model_config": model_config.to_model_kwargs(),
            "training": {k: v for k, v in train_cfg.items()},
        },
    )

    trainer = Trainer(trainer_config)

    if args.resume:
        if not Path(args.resume).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"[Exp-Coder] resumed from {args.resume} at step {trainer.global_step}")

    summary = trainer.train()

    print(f"[Exp-Coder] training complete in {summary['elapsed_time']:.1f}s "
          f"(step {summary['global_step']}, epoch {summary['epoch']})")


if __name__ == "__main__":
    main()