#!/usr/bin/env python3
"""Test chat with fresh model via direct call."""

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

# Create FRESH model (random init)
from src.model.architecture.expera_model import ExperaModel

model = ExperaModel(
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_hidden_layers"],
    num_heads=config["num_attention_heads"],
    num_kv_heads=config["num_key_value_heads"],
)
model.eval()

# Use Generator directly
from src.inference.generator import Generator, GenerationConfig

gen = Generator(model, sp, GenerationConfig(max_new_tokens=20, temperature=0.7))

print("=" * 60)
print("Testing Fresh Model Chat")
print("=" * 60)

prompts = ["hello", "hi", "who are you"]

for prompt in prompts:
    result = gen.generate(prompt)
    result_safe = result.encode("ascii", errors="replace").decode("ascii")
    print(f"\nUser: {prompt}")
    print(f"Model: {result_safe}")