#!/usr/bin/env python3
"""Simple chat test - bypass pipeline issues."""

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

# Create FRESH model (no training, just random init like at start of any training)
from src.model.architecture.expera_model import ExperaModel

model = ExperaModel(
    vocab_size=config.get("vocab_size", 200),
    hidden_size=config.get("hidden_size", 768),
    num_layers=config.get("num_hidden_layers", 12),
    num_heads=config.get("num_attention_heads", 8),
    num_kv_heads=config.get("num_key_value_heads", 4),
)
model.eval()

print("=" * 60)
print("FRESH MODEL (RANDOM INIT) - Chat Test")
print("=" * 60)

# Test prompts
prompts = [
    "hello",
    "hi",
    "who are you",
]

for prompt in prompts:
    print(f"\n--- Prompt: {repr(prompt)} ---")

    # Encode
    input_ids = sp.encode(prompt)
    print(f"Input IDs: {input_ids}")

    # Greedy generate
    generated = input_ids[:]
    for _ in range(20):
        with torch.no_grad():
            out = model(torch.tensor([generated]), return_dict=True)
            next_token = out["logits"][0, -1].argmax().item()
            if next_token == sp.eos_id():
                break
            generated.append(next_token)

    # Decode newly generated
    output_ids = generated[len(input_ids):]
    if output_ids:
        output_text = sp.decode(output_ids)
        # Filter for ASCII display
        output_safe = output_text.encode("ascii", errors="replace").decode("ascii")
        print(f"Output: {output_safe}")
    else:
        print("Output: (empty)")