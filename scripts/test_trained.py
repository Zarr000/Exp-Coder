#!/usr/bin/env python3
"""Quick test of trained chat model."""

import sys
sys.path.insert(0, '.')

import torch
import json
from pathlib import Path
import sentencepiece as spm

# Use same model path as training
MODEL_PATH = "checkpoints/expera_coder_120m"
OUTPUT_DIR = "release/expera_chat_120m"

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(str(Path(MODEL_PATH) / "tokenizer.model"))
print(f"Loaded tokenizer: {tokenizer.GetPieceSize()} tokens")

# Load config
with open(Path(MODEL_PATH) / "config.json") as f:
    model_config = json.load(f)
print(f"Model config: {model_config}")

# Create model with SAME config as training
from src.model.architecture.expera_model import ExperaModel
model = ExperaModel(
    vocab_size=model_config.get("vocab_size", 200),
    hidden_size=model_config.get("hidden_size", 768),
    num_layers=model_config.get("num_hidden_layers", 12),
    num_heads=model_config.get("num_attention_heads", 8),
    num_kv_heads=model_config.get("num_key_value_heads", 4),
)
model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Load pretrained weights first
if (Path(MODEL_PATH) / "model.pt").exists():
    state = torch.load(Path(MODEL_PATH) / "model.pt", map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    # Fix keys
    new_state = state
    if list(state.keys())[0].startswith("_orig_mod."):
        new_state = {}
        for k, v in state.items():
            key = k[9:] if k.startswith("_orig_mod.") else k
            new_state[key] = v
    model.load_state_dict(new_state, strict=False)
    print("Loaded pretrained weights")
else:
    print("No pretrained weights found")

# Load trained chat weights (overlaying on top)
best_pt = Path(OUTPUT_DIR) / "best.pt"
if best_pt.exists():
    chat_state = torch.load(best_pt, map_location="cpu", weights_only=False)
    if "model_state_dict" in chat_state:
        chat_state = chat_state["model_state_dict"]
    # Fix keys
    new_chat_state = chat_state
    if list(chat_state.keys())[0].startswith("_orig_mod."):
        new_chat_state = {}
        for k, v in chat_state.items():
            key = k[9:] if k.startswith("_orig_mod.") else k
            new_chat_state[key] = v
    model.load_state_dict(new_chat_state, strict=False)
    print("Loaded trained chat weights!")
else:
    print("No trained weights found")

# Test prompts
prompts = ["hi", "hello", "who are you"]

for prompt in prompts:
    print(f"\n--- Prompt: {prompt} ---")

    # Encode with chat format (matching training)
    full_prompt = f"<|system|>\nYou are Expera AI, a helpful assistant.<|user|>\nUser: {prompt}<|assistant|>\nAssistant:"

    input_ids = tokenizer.encode(full_prompt)
    print(f"Input: {input_ids}")

    # Generate
    generated = input_ids[:]
    for i in range(64):
        with torch.no_grad():
            out = model(torch.tensor([generated]), return_dict=True)
            logits = out["logits"][0, -1]
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == tokenizer.eos_id():
                break
            if i > 50 and next_token == input_ids[-1]:  # Stop if repeating
                break

    # Decode
    output = tokenizer.decode(generated[len(input_ids):])
    print(f"Output: {repr(output)}")