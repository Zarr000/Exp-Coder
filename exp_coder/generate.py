"""Exp-Coder inference CLI.

Usage:
    python -m exp_coder.generate \
        --checkpoint checkpoints/exp-coder-120m/final.pt \
        --tokenizer tokenizer/exp-coder \
        --prompt "Write a Python function that calculates Fibonacci numbers."

Requirements: a trained BPETokenizer directory and an Exp-Coder checkpoint.
No downloads, no network.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_model_config
from src.inference import GenerationConfig, Generator
from src.inference.generator import DecodingStrategy
from src.model.architecture import ExperaModel
from src.tokenizer import BPETokenizer

_CANONICAL_MODEL = str(
    Path(__file__).resolve().parent.parent / "configs" / "exp_coder_120m.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp-Coder text generation")
    parser.add_argument("--config", default=_CANONICAL_MODEL,
                        help="Model YAML (default: canonical Exp-Coder 120M)")
    parser.add_argument("--tokenizer", required=True,
                        help="Trained BPETokenizer directory (vocab.json/merges.txt/config.json)")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint .pt file (model_state_dict)")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--strategy", choices=["greedy", "sampling", "beam"], default="greedy")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; use --device cpu or install a "
            "CUDA-enabled PyTorch build."
        )
    device_name = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)

    model_config = load_model_config(args.config)

    tokenizer_dir = Path(args.tokenizer)
    if not (tokenizer_dir / "config.json").exists() or not (tokenizer_dir / "vocab.json").exists():
        raise SystemExit(
            f"Tokenizer not found or unsupported format at {tokenizer_dir}. "
            "Expected a BPETokenizer directory with vocab.json/merges.txt/config.json."
        )
    tokenizer = BPETokenizer.load(str(tokenizer_dir))
    if getattr(tokenizer, "vocab_size", None) != model_config.vocab_size:
        print(
            f"Warning: tokenizer target vocab {tokenizer.vocab_size} differs "
            f"from model vocab {model_config.vocab_size}; generation may be invalid.",
            file=sys.stderr,
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = ExperaModel(**model_config.to_model_kwargs()).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"Warning: checkpoint keys diverged (missing={len(missing)}, "
            f"unexpected={len(unexpected)}); loaded strictly-defined weights.",
            file=sys.stderr,
        )
    model.eval()

    strategy = DecodingStrategy.SAMPLING if args.strategy == "sampling" else (
        DecodingStrategy.BEAM if args.strategy == "beam" else DecodingStrategy.GREEDY
    )
    gen_config = GenerationConfig(
        strategy=strategy,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        num_beams=args.num_beams,
        use_cache=args.strategy == "greedy",
    )

    generator = Generator(model, tokenizer, gen_config)
    text = generator.generate(args.prompt)
    print(text)


if __name__ == "__main__":
    main()