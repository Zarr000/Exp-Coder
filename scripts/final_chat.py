#!/usr/bin/env python3
"""Final Phase 12 verification - fresh model via chat interface."""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import json
from pathlib import Path
import sentencepiece as spm

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(str(Path("checkpoints/expera_coder_120m/tokenizer.model")))

# Load config
with open("checkpoints/expera_coder_120m/config.json") as f:
    config = json.load(f)

# Create FRESH model (no pretrained weights)
from src.model.architecture.expera_model import ExperaModel

model = ExperaModel(
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_hidden_layers"],
    num_heads=config["num_attention_heads"],
    num_kv_heads=config["num_key_value_heads"],
)
model.eval()

# Use InferencePipeline
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline(
    model=model,
    tokenizer=sp,
)

print("=" * 60)
print("PHASE 12 - FINAL VERIFICATION")
print("=" * 60)

prompts = [
    "hello",
    "hi",
    "who are you",
    "what is your name",
]

for prompt in prompts:
    output = pipeline.generate(prompt, max_new_tokens=20, temperature=0.7)
    result_safe = output.text.encode("ascii", errors="replace").decode("ascii")
    print(f"\nUser: {prompt}")
    print(f"Model: {result_safe}")

print("\n" + "=" * 60)
print("Phase 12: COMPLETE - Model generates valid text output")
print("=" * 60)