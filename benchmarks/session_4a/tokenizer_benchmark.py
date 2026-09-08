"""Session 4A — 50304-target BPE tokenizer validation.

Loads the tokenizer trained by the corpus pipeline, then measures:

- overall + per-language chars/token, tokens/1k chars, UNK rate
- encode/decode throughput
- quality gates: round-trip exactness, UNK==0, no accidental specials,
  vocab consistency vs canonical model
- qualitative examples with visible token pieces

Results:
  results/session_4a/tokenizer_benchmark.json
  results/session_4a/tokenizer_by_language.csv
  results/session_4a/tokenizer_examples.md
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import regex as re

from src.config import load_model_config
from src.tokenizer import BPETokenizer
from benchmarks.common import write_result

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_TRAIN = ROOT / "benchmarks" / "data" / "session4_corpus" / "train"
TOK_DIR = ROOT / "benchmarks" / "data" / "session4_corpus" / "tokenizer_50304"
RESULTS = ROOT / "benchmarks" / "results" / "session_4a"
CANONICAL_MODEL = ROOT / "configs" / "exp_coder_120m.yaml"

LANGS = ["python", "javascript", "typescript", "cpp", "rust", "go", "json",
         "yaml", "shell", "markdown", "text"]


def pieces(tokenizer, text):
    out = []
    for word in re.findall(tokenizer.pat, text):
        b = "".join(tokenizer.byte_encoder[x] for x in word.encode("utf-8"))
        out.extend(tokenizer._bpe(b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(TOK_DIR))
    ap.add_argument("--corpus", default=str(CORPUS_TRAIN))
    ap.add_argument("--speed-docs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    corpus = Path(args.corpus)
    docs = sorted(p for p in corpus.iterdir() if p.is_file())

    # ---- per-language + overall ----
    agg = {"chars": 0, "tokens": 0, "unk": 0, "docs": 0}
    per_lang = defaultdict(lambda: {"docs": 0, "chars": 0, "tokens": 0, "unk": 0})
    for p in docs:
        text = p.read_text(encoding="utf-8")
        lang = p.stem.split("_", 1)[1].split(".", 1)[0] if "_" in p.stem else "text"
        lang = lang if lang in LANGS else "text"
        ids = tokenizer.encode(text, add_special_tokens=False)
        n = len(ids)
        u = sum(1 for i in ids if i == tokenizer.unk_id())
        r = per_lang[lang]
        r["docs"] += 1
        r["chars"] += len(text)
        r["tokens"] += n
        r["unk"] += u
        agg["chars"] += len(text)
        agg["tokens"] += n
        agg["unk"] += u
        agg["docs"] += 1

    def finish(d):
        d["chars_per_token"] = round(d["chars"] / max(1, d["tokens"]), 3)
        d["tokens_per_1k_chars"] = round(d["tokens"] / max(1, d["chars"]) * 1000, 2)
        d["unk_rate"] = round(d["unk"] / max(1, d["tokens"]), 6)
        return d

    overall = finish(dict(agg))
    per = {k: finish(v) for k, v in sorted(per_lang.items())}

    # ---- throughput ----
    big = "\n".join(p.read_text(encoding="utf-8") for p in docs[:50])
    chars = len(big)
    t0 = time.perf_counter()
    for _ in range(args.speed_docs):
        ids = tokenizer.encode(big, add_special_tokens=False)
    enc_s = (time.perf_counter() - t0) / args.speed_docs
    t0 = time.perf_counter()
    for _ in range(args.speed_docs):
        tokenizer.decode(ids)
    dec_s = (time.perf_counter() - t0) / args.speed_docs
    speed = {
        "encode_chars_per_sec": round(chars / enc_s, 0),
        "encode_tokens_per_sec": round(len(ids) / enc_s, 0),
        "decode_tokens_per_sec": round(len(ids) / dec_s, 0),
    }

    # ---- quality gates ----
    special_ids = set([tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()])
    roundtrip_bad = 0
    special_insertions = 0
    sampled = 0
    for p in docs[:40]:
        text = p.read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False)
        sampled += 1
        if tokenizer.decode(ids) != text:
            roundtrip_bad += 1
        special_insertions += sum(1 for i in ids if i in special_ids)
    gates = {
        "roundtrip_exact": roundtrip_bad == 0,
        "roundtrip_tested_docs": sampled,
        "roundtrip_failures": roundtrip_bad,
        "special_insertions_without_request": special_insertions,
        "no_accidental_specials": special_insertions == 0,
        "unk_rate_zero": overall["unk_rate"] == 0.0,
        "canonical_model_vocab": load_model_config(CANONICAL_MODEL).vocab_size,
        "tokenizer_requested_vocab": tokenizer.vocab_size,
        "vocab_consistent_with_model": tokenizer.vocab_size == load_model_config(CANONICAL_MODEL).vocab_size,
    }

    # ---- qualitative examples ----
    examples = []
    want = ["python", "cpp", "rust", "go", "javascript", "json", "yaml", "shell", "markdown"]
    seen = set()
    for p in docs:
        lang = p.stem.split("_", 1)[1].split(".", 1)[0] if "_" in p.stem else "text"
        if lang not in want or lang in seen:
            continue
        text = p.read_text(encoding="utf-8")[:600]
        ids = tokenizer.encode(text, add_special_tokens=False)
        ps = pieces(tokenizer, text)
        seen.add(lang)
        examples.append({
            "language": lang,
            "file": p.name,
            "source_chars": len(text),
            "tokens": len(ids),
            "roundtrip_ok": tokenizer.decode(ids) == text,
            "pieces": ps[:80],
        })

    # ---- write outputs ----
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_result(str(RESULTS / "tokenizer_benchmark.json"), {
        "run_type": "tokenizer_50304_real_corpus",
        "requested_vocab": tokenizer.vocab_size,
        "actual_vocab": len(tokenizer.vocab),
        "n_merges": len(tokenizer.merge_ranks),
        "corpus_train_chars": agg["chars"],
        "train_docs": agg["docs"],
        "overall": overall,
        "per_language": per,
        "speed": speed,
        "gates": gates,
        "qualitative": examples,
    })

    with open(RESULTS / "tokenizer_by_language.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "docs", "chars", "tokens", "chars_per_token",
                    "tokens_per_1k_chars", "unk_rate"])
        for lang, d in per.items():
            w.writerow([lang, d["docs"], d["chars"], d["tokens"],
                        d["chars_per_token"], d["tokens_per_1k_chars"], d["unk_rate"]])

    with open(RESULTS / "tokenizer_examples.md", "w", encoding="utf-8") as f:
        f.write("# Session 4A tokenizer qualitative audit\n\n")
        for ex in examples:
            f.write(f"## {ex['language']} ({ex['file']})\n")
            f.write(f"- source chars: {ex['source_chars']}, tokens: {ex['tokens']}, "
                    f"roundtrip: {'PASS' if ex['roundtrip_ok'] else 'FAIL'}\n")
            f.write("```text\npieces: " + " | ".join(ex["pieces"]) + "\n```\n\n")

    print("overall:", overall)
    print("speed:", speed)
    print("gates:", {k: v for k, v in gates.items() if k != "roundtrip_failures"})
    print("per-language chars/token:", {k: v["chars_per_token"] for k, v in per.items()})


if __name__ == "__main__":
    main()