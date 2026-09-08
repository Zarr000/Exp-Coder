"""Deterministic causal-LM dataset for Exp-Coder.

Minimal but real dataset pipeline:

    text documents -> BPE tokenizer -> integer token stream
                     -> fixed-length blocks -> input_ids / labels

``labels`` equals ``input_ids``; the causal next-token shift is applied
inside the loss (:class:`src.training.loss.LanguageModelingLoss` shifts
``logits[:, :-1]`` against ``labels[:, 1:]``). This matches the GPT-2
convention and keeps the dataset trivially verifiable.

Behavior is deterministic for a fixed corpus, tokenizer and seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

_TEXT_EXTENSIONS = (".txt", ".py", ".md", ".jsonl", ".yml", ".yaml")


def load_text_corpus(
    path: Union[str, Path],
    extensions: Sequence[str] = _TEXT_EXTENSIONS,
) -> List[str]:
    """Load text/code documents from ``path`` (file or directory).

    - ``.jsonl`` lines are treated as JSON objects and their ``text`` /
      ``content`` / ``code`` field used when present, else the raw line.
    - other files read as plain UTF-8 text.
    """
    path = Path(path)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
    else:
        raise FileNotFoundError(f"Corpus path not found: {path}")

    documents: List[str] = []
    for file in candidates:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            if file.suffix.lower() == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        documents.append(line)
                        continue
                    if isinstance(record, str):
                        documents.append(record)
                    elif isinstance(record, dict):
                        documents.append(
                            record.get("text")
                            or record.get("content")
                            or record.get("code")
                            or ""
                        )
            else:
                text = f.read()
                if text.strip():
                    documents.append(text)
    return [d for d in documents if d and d.strip()]


class CausalLMDataset(Dataset):
    """Fixed-length causal language modeling blocks from tokenized text.

    Args:
        texts: list of raw documents (already-loaded) or a corpus path.
        tokenizer: tokenizer exposing ``encode(text, add_special_tokens=False)``.
        sequence_length: block length (tokens per sample).
        add_eos: append the tokenizer EOS id at the end of every document.
    """

    def __init__(
        self,
        texts: Union[str, Path, Sequence[str]],
        tokenizer: Any,
        sequence_length: int = 512,
        add_eos: bool = True,
    ):
        if isinstance(texts, (str, Path)):
            texts = load_text_corpus(texts)
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.add_eos = add_eos

        tokens: List[int] = []
        for text in self.texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if add_eos:
                ids = ids + [tokenizer.eos_id()]
            tokens.extend(ids)
        self.tokens = tokens

        self.num_blocks = max(1, len(tokens) // self.sequence_length)

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.sequence_length
        block = self.tokens[start : start + self.sequence_length]
        if len(block) < self.sequence_length:
            # Only reachable for corpora shorter than one block.
            block = block + [0] * (self.sequence_length - len(block))
        input_ids = torch.tensor(block, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)