"""Exp-Coder tokenizer scientific benchmark.

Trains (or loads) the canonical byte-level BPE tokenizer on the benchmark
code corpus and measures:

- chars/token, tokens per 1000 chars, unknown-token rate
- per-language fragmentation (Python, JS, C++, Rust, Go, JSON, YAML, MD, shell)
- encode/decode throughput (chars/s, tokens/s)
- representative token-piece inspection

Results (JSON) written to benchmarks/results/tokenizer.json.

Usage:
    python benchmarks/tokenizer_benchmark.py [--vocab-size 50304] [--out results/tokenizer.json]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import BPETokenizer
from benchmarks.common import CORPUS_DIR, RESULT_DIR, write_result
from benchmarks.make_code_corpus import corpus_files

TOKENIZER_DIR_DEFAULT = Path(__file__).resolve().parent / "data" / "tokenizer"


def token_pieces(tokenizer: BPETokenizer, text: str):
    """Return the visible BPE pieces (subwords) for a text."""
    import regex as re

    pieces: list = []
    for word in re.findall(tokenizer.pat, text):
        byte_encoded = "".join(
            tokenizer.byte_encoder[b] for b in word.encode("utf-8")
        )
        pieces.extend(tokenizer._bpe(byte_encoded))
    return [p for p in pieces if p in tokenizer.vocab or True]


def measure_language(tokenizer: BPETokenizer, lang: str, text: str) -> dict:
    chars = len(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = len(ids)
    unk = tokenizer.unk_id()
    unk_rate = sum(1 for i in ids if i == unk) / max(1, tokens)
    return {
        "language": lang,
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": round(chars / max(1, tokens), 3),
        "tokens_per_1k_chars": round(tokens / max(1, chars) * 1000, 2),
        "unk_rate": unk_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp-Coder tokenizer benchmark")
    parser.add_argument("--tokenizer", default=str(TOKENIZER_DIR_DEFAULT))
    parser.add_argument("--corpus", default=str(CORPUS_DIR))
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--speed-docs", type=int, default=40)
    parser.add_argument("--out", default=str(RESULT_DIR / "tokenizer.json"))
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    tokenizer_dir = Path(args.tokenizer)

    cfg = tokenizer_dir / "config.json"
    if cfg.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_dir))
        print(f"Loaded existing tokenizer from {tokenizer_dir}")
    else:
        paths = corpus_files(corpus_dir)
        texts = [p.read_text(encoding="utf-8") for p in paths]
        print(f"Training tokenizer on {len(paths)} docs from {corpus_dir}")
        t0 = time.perf_counter()
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(texts, min_frequency=args.min_frequency, verbose=False)
        train_s = time.perf_counter() - t0
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_dir))
        print(f"Trained in {train_s:.1f}s -> actual vocab {len(tokenizer.vocab)}")

    # ---- overall + per-language ----
    languages = {}
    totals = {"chars": 0, "tokens": 0, "unk": 0}
    for path in corpus_files(corpus_dir):
        lang = path.stem.split("_v")[0]
        text = path.read_text(encoding="utf-8")
        row = measure_language(tokenizer, lang, text)
        languages[lang] = row
        totals["chars"] += row["chars"]
        totals["tokens"] += row["tokens"]
        totals["unk"] += int(row["unk_rate"] * row["tokens"])

    overall = {
        "chars": totals["chars"],
        "tokens": totals["tokens"],
        "chars_per_token": round(totals["chars"] / max(1, totals["tokens"]), 3),
        "tokens_per_1k_chars": round(totals["tokens"] / max(1, totals["chars"]) * 1000, 2),
        "unk_rate": round(totals["unk"] / max(1, totals["tokens"]), 6),
    }

    # ---- throughput ----
    corpus_text = "\n".join(
        p.read_text(encoding="utf-8") for p in corpus_files(corpus_dir)
    )
    chars = len(corpus_text)

    t0 = time.perf_counter()
    for _ in range(args.speed_docs):
        ids = tokenizer.encode(corpus_text, add_special_tokens=False)
    t1 = time.perf_counter()
    enc_s = (t1 - t0) / args.speed_docs
    speed = {
        "encode_chars_per_sec": round(chars / max(1e-9, enc_s), 0),
        "encode_tokens_per_sec": round(len(ids) / max(1e-9, enc_s), 0),
    }

    t0 = time.perf_counter()
    for _ in range(args.speed_docs):
        tokenizer.decode(ids)
    t1 = time.perf_counter()
    dec_s = (t1 - t0) / args.speed_docs
    speed["decode_tokens_per_sec"] = round(len(ids) / max(1e-9, dec_s), 0)

    # ---- qualitative ----
    samples = {}
    for lang in ["python", "cpp", "rust", "go"]:
        path = corpus_dir / f"{lang}.py" if lang == "python" else corpus_dir / f"{lang}.{ 'cpp' if lang=='cpp' else 'rs' if lang=='rust' else 'go'}"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")[:220]
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = token_pieces(tokenizer, text)
        samples[lang] = {
            "chars": len(text),
            "tokens": len(ids),
            "roundtrip_ok": tokenizer.decode(ids) == text,
            "pieces": pieces[:60],
        }

    result = {
        "tokenizer": {
            "target_vocab_size": tokenizer.vocab_size,
            "actual_vocab_size": len(tokenizer.vocab),
            "special_tokens": tokenizer.special_tokens,
        },
        "overall": overall,
        "per_language": languages,
        "speed": speed,
        "qualitative": samples,
    }
    write_result(args.out, result)
    print(f"Wrote {args.out}")
    print("overall:", overall)
    print("speed:", speed)


if __name__ == "__main__":
    main()