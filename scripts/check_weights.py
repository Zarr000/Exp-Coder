#!/usr/bin/env python3
"""Check weight validity."""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import json
from pathlib import Path

model_dir = Path("checkpoints/expera_coder_120m")
release_dir = Path("release/expera_chat_120m")

# Load checkpoint
trained_path = release_dir / "expera_chat_final.pt"
state_dict = torch.load(trained_path, map_location="cpu")

if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

# Check weights
print("Checking weight validity...")

issues = []
for key, tensor in list(state_dict.items())[:20]:
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()

    if has_nan or has_inf:
        issues.append((key, has_nan, has_inf, mean_val, std_val))
        print(f"ISSUE: {key}: has_nan={has_nan}, has_inf={has_inf}, mean={mean_val:.4f}, std={std_val:.4f}")
    else:
        print(f"OK: {key}: mean={mean_val:.4f}, std={std_val:.4f}")

if issues:
    print(f"\n!!! Found {len(issues)} problematic tensors !!!")
else:
    print("\nAll weights look valid.")

# Check embedding layer specifically
print("\n--- Embedding layer stats ---")
emb_key = None
for key in state_dict.keys():
    if "embedding" in key and "weight" in key:
        print(f"{key}: shape={state_dict[key].shape}, mean={state_dict[key].mean():.4f}, std={state_dict[key].std():.4f}")
        # Check first 10 tokens
        print(f"  First 10 token embeddings (L2 norm):")
        for i in range(min(10, state_dict[key].shape[0])):
            norm = state_dict[key][i].norm().item()
            print(f"    token {i}: norm={norm:.4f}")