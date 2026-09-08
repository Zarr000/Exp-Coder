#!/usr/bin/env python3
"""Trace generation step by step."""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import json
from pathlib import Path

import sentencepiece as spm

model_dir = Path("checkpoints/expera_coder_120m")
release_dir = Path("release/expera_chat_120m")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(str(model_dir / "tokenizer.model"))

# Load config and model
with open(model_dir / "config.json") as f:
    config = json.load(f)

from src.model.architecture.expera_model import ExperaModel

model = ExperaModel(
    vocab_size=config.get("vocab_size", 200),
    hidden_size=config.get("hidden_size", 768),
    num_layers=config.get("num_hidden_layers", 12),
    num_heads=config.get("num_attention_heads", 8),
    num_kv_heads=config.get("num_key_value_heads", 4),
    max_position_embeddings=config.get("max_position_embeddings", 2048),
)

# Load trained weights
state_dict = torch.load(release_dir / "expera_chat_final.pt", map_location="cpu")
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

# Fix keys
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_key = k[9:]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

print("Model loaded. Tracing generation...")

# Test prompt
prompt = "hello"
input_ids = sp.EncodeAsIds(prompt)
input_tensor = torch.tensor([input_ids])

print(f"Input prompt: {repr(prompt)}")
print(f"Input IDs: {input_ids}")
print(f"Input decoded: {sp.DecodeIds(input_ids)}")

# Forward pass
print("\n--- Forward pass ---")
with torch.no_grad():
    output = model(input_tensor, return_dict=True)
    logits = output["logits"]

print(f"Logits shape: {logits.shape}")  # [batch, seq, vocab]
print(f"Last position logits shape: {logits[0, -1].shape}")

# Get top tokens at last position
last_logits = logits[0, -1]
top_k = 10
top_probs, top_ids = torch.topk(last_logits, top_k)

print(f"\nTop {top_k} predicted token IDs at last position:")
for i, (prob, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
    piece = sp.IdToPiece(tid)
    decoded = sp.DecodeIds([tid])
    print(f"  {i+1}. ID={tid:3d} prob={prob:.4f} piece={piece!r:15s} decode={decoded!r}")

# Now do manual generation - what does generator do?
print("\n--- Manual generation step ---")

# Step 1: apply temperature
temp = 0.7
scaled_logits = last_logits / temp
print(f"After temperature: mean={scaled_logits.mean():.4f}, std={scaled_logits.std():.4f}")

# Step 2: top-k
top_k_val = 50
top_k_indices = torch.topk(scaled_logits, top_k_val).indices
mask = torch.full_like(scaled_logits, float("-inf"))
mask.scatter_(-1, top_k_indices, 0.0)
filtered_logits = scaled_logits + mask

# Step 3: top-p
top_p_val = 0.9
sorted_logits, sorted_indices = torch.sort(filtered_logits, dim=-1, descending=True)
probs = torch.softmax(sorted_logits, dim=-1)
cumsum = torch.cumsum(probs, dim=-1)
mask_p = cumsum > top_p_val
mask_p[..., 1:] = mask_p[..., :-1].clone()
mask_p[..., 0] = False
mask_p = mask_p.scatter(-1, sorted_indices, mask_p)
final_logits = filtered_logits.masked_fill(mask_p, float("-inf"))

# Sample
probs_final = torch.softmax(final_logits, dim=-1)
next_token = torch.multinomial(probs_final, num_samples=1)
print(f"\nSampled next token ID: {next_token.item()}")
print(f"Sampled piece: {sp.IdToPiece(next_token.item())}")
print(f"Sampled decode: {sp.DecodeIds([next_token.item()])}")