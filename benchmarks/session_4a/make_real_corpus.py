"""Session 4A — deterministic real-code corpus.

Builds ``benchmarks/data/session4_corpus/{train,val}`` from LOCAL sources
(no network):

- the Exp-Coder repository's own Python source (src/, scripts/, tests/,
  benchmarks/) and docs/configuration (markdown, yaml, json)
- the Session 3A synthetic multilingual code sample (js/cpp/rust/go/shell/
  json/yaml/md/text) as an explicit, separately-labelled source

Document-level deterministic split (no random line shuffling):
for each language, sorted documents are partitioned ~90/10 train/val by
stride. Content-exact duplicates are removed (kept in train when they exist
in both, otherwise dropped from the duplicate set) and the check
train ∩ val == ∅ is asserted by sha256.

Document separators are handled by the tokenizer layer: the canonical
CausalLMDataset appends the tokenizer EOS id at the end of every document.

Outputs:
  benchmarks/data/session4_corpus/{train,val}/*.<ext>
  benchmarks/results/session_4a/corpus_manifest.json
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "session4_corpus"
CODE_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "code_sample"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "session_4a"

EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".h": "cpp", ".hxx": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".json": "json",
    ".jsonl": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".sh": "shell", ".bash": "shell",
    ".md": "markdown",
    ".dockerfile": "text", ".txt": "text", ".cfg": "text", ".ini": "text",
}

MAX_BYTES = 512 * 1024

EXCLUDE_PARTS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "test_checkpoints", "checkpoints", "logs",
}
EXCLUDE_SUFFIX_FILES = {".pyc", ".pyo"}

# dirs that are allowed as *documents* sources (everything else under these
# trees is scanned); results/ and generated benchmark outputs are excluded.
SOURCE_DIRS = ["src", "scripts", "tests", "benchmarks", "docs", "configs", "datasets"]
EXTRA_ROOT_FILES = ["README.md"]

# explicit overrides: (relative path) -> language
PATHS_OVERRIDE = {
    Path("benchmarks/data/code_sample") / p.name: name
    for p in []
}  # code_sample handled separately as synthetic source


def collect_source_docs() -> list:
    docs = []
    for sub in SOURCE_DIRS:
        base = ROOT / sub
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(ROOT)
            if not path.is_file():
                continue
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            if path.suffix in EXCLUDE_SUFFIX_FILES:
                continue
            if "results" in path.parts or "data" in path.parts or "legacy" in path.parts:
                continue
            lang = EXT_TO_LANG.get(path.suffix)
            if lang is None:
                continue
            docs.append((path, lang, "repo"))
    for name in EXTRA_ROOT_FILES:
        path = ROOT / name
        if path.exists():
            docs.append((path, "markdown", "repo"))
    # synthetic multilingual sample from Session 3A (explicitly labelled)
    if CODE_SAMPLE.exists():
        for path in sorted(CODE_SAMPLE.glob("*")):
            if path.is_file() and path.suffix != ".txt":
                lang = EXT_TO_LANG.get(path.suffix)
                if lang:
                    docs.append((path, lang, "synthetic_multilingual"))
    return docs


def read_doc(path: Path):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> None:
    docs = collect_source_docs()
    print(f"discovered {len(docs)} candidate documents")

    by_lang = defaultdict(list)
    for path, lang, source in docs:
        text = read_doc(path)
        if text is None:
            continue
        by_lang[lang].append({"rel": path.relative_to(ROOT), "lang": lang,
                              "source": source, "text": text, "sha": hashlib.sha256(text.encode()).hexdigest()})

    # ---- exact-duplicate removal ----
    seen = {}
    kept, dup_removed = [], 0
    for lang in sorted(by_lang):
        for doc in sorted(by_lang[lang], key=lambda d: str(d["rel"])):
            if doc["sha"] in seen:
                dup_removed += 1
                # if duplicate exists in train and this would land in val, drop it
                continue
            seen[doc["sha"]] = doc
            kept.append(doc)

    # ---- deterministic per-language ~90/10 split (document level) ----
    train, val = [], []
    for lang in sorted(by_lang):
        group = [d for d in kept if d["lang"] == lang]
        group.sort(key=lambda d: str(d["rel"]))
        for i, doc in enumerate(group):
            (val if i % 10 == 9 else train).append(doc)

    assert not (set(d["sha"] for d in train) & set(d["sha"] for d in val)), "train/val overlap!"

    # ---- write docs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train").mkdir(exist_ok=True)
    (OUT_DIR / "val").mkdir(exist_ok=True)
    written = 0
    for split_name, split in (("train", train), ("val", val)):
        for n, doc in enumerate(sorted(split, key=lambda d: str(d["rel"]))):
            target = OUT_DIR / split_name / f"{n:04d}_{doc['lang']}{doc['rel'].suffix}"
            target.write_text(doc["text"], encoding="utf-8")
            written += 1

    def stats(split, name):
        by = defaultdict(lambda: {"docs": 0, "chars": 0, "bytes": 0})
        for d in split:
            b = by[d["lang"]]
            b["docs"] += 1
            b["chars"] += len(d["text"])
            b["bytes"] += len(d["text"].encode("utf-8"))
        return {k: dict(v) for k, v in by.items()}

    manifest = {
        "session": "4a",
        "sources": {
            "repo": "Exp-Coder repository source/docs (src, scripts, tests, docs, configs, datasets, README)",
            "synthetic_multilingual": "Session 3A synthetic code sample (benchmarks/data/code_sample)",
        },
        "snapshot_commit": None,  # filled by caller
        "preprocessing": {
            "encoding": "utf-8 (strict; non-utf8 skipped as binary)",
            "max_bytes_per_doc": MAX_BYTES,
            "duplicates": "exact sha256 dedup (kept first occurrence)",
            "split": "deterministic per-language stride (every 10th sorted doc -> val)",
            "document_separator": "tokenizer EOS id appended per document (CausalLMDataset add_eos=True)",
        },
        "exact_duplicates_removed": dup_removed,
        "documents_written": written,
        "train": stats(train, "train"),
        "val": stats(val, "val"),
        "train_chars": sum(len(d["text"]) for d in train),
        "val_chars": sum(len(d["text"]) for d in val),
        "train_docs": len(train),
        "val_docs": len(val),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "corpus_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"train docs={len(train)} chars={manifest['train_chars']}")
    print(f"val   docs={len(val)} chars={manifest['val_chars']}")
    print(f"duplicates removed={dup_removed}")
    print(f"wrote {written} docs -> {OUT_DIR/'train','val'}")
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()