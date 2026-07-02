#!/usr/bin/env python3
"""Test trained chat model."""

import sys
import os

# Force UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import json
from pathlib import Path

import sentencepiece as spm

# Paths
model_dir = Path("checkpoints/expera_coder_120m")
release_dir = Path("release/expera_chat_120m")

# Load tokenizer
print("Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load(str(model_dir / "tokenizer.model"))
print(f"Tokenizer vocab size: {sp.GetPieceSize()}")

# Load config
with open(model_dir / "config.json") as f:
    config = json.load(f)

# Load base model
from src.model.architecture.expera_model import ExperaModel

model = ExperaModel(
    vocab_size=config.get("vocab_size", 200),
    hidden_size=config.get("hidden_size", 768),
    num_layers=config.get("num_hidden_layers", 12),
    num_heads=config.get("num_attention_heads", 8),
    num_kv_heads=config.get("num_key_value_heads", 4),
    max_position_embeddings=config.get("max_position_embeddings", 2048),
)

# Try to load trained weights from release
print("\nLoading trained weights from release...")
trained_path = release_dir / "expera_chat_final.pt"
if trained_path.exists():
    print(f"Loading: {trained_path}")
    state_dict = torch.load(trained_path, map_location="cpu")

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Fix key names if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[9:]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("Loaded trained weights!")
else:
    print(f"No trained checkpoint found at {trained_path}")

model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Test prompts
print("\n" + "=" * 60)
print("GENERATION TEST")
print("=" * 60)

from src.inference.generator import Generator, GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=30,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)

generator = Generator(model, sp, gen_config)

prompts = [
    "hello",
    "User: hello\nAssistant:",
    "Hi",
]

for prompt in prompts:
    print(f"\n--- Prompt: {repr(prompt)} ---")

    # Encode
    input_ids = sp.EncodeAsIds(prompt)
    print(f"Input IDs: {input_ids}")

    # Generate
    result = generator.generate(prompt)
    # Print safe output
    result_safe = result.encode('ascii', errors='replace').decode('ascii')
    print(f"Output (ascii): {result_safe}")
    print(f"Output len: {len(result)}")

    # Also try via pipeline
    from src.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        model=model,
        tokenizer=sp,
    )

    output = pipeline.generate(prompt, max_new_tokens=30, temperature=0.7)
    print(f"Pipeline output: {repr(output.text)}")