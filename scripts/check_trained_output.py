#!/usr/bin/env python3
"""Check trained model outputs."""

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
sp.Load(str(Path('checkpoints/expera_coder_120m/tokenizer.model')))

# Load config
with open('checkpoints/expera_coder_120m/config.json') as f:
    config = json.load(f)

from src.model.architecture.expera_model import ExperaModel

# Create model
model = ExperaModel(
    vocab_size=config.get('vocab_size', 200),
    hidden_size=config.get('hidden_size', 768),
    num_layers=config.get('num_hidden_layers', 12),
    num_heads=config.get('num_attention_heads', 8),
    num_kv_heads=config.get('num_key_value_heads', 4)
)

# Load checkpoint
state = torch.load(Path('release/expera_chat_120m/expera_chat_final.pt'), map_location='cpu')
if 'model_state_dict' in state:
    state = state['model_state_dict']

model.load_state_dict(state, strict=False)
model.eval()

# Test generation
prompt = "hello"
input_ids = sp.EncodeAsIds(prompt)
input_tensor = torch.tensor([input_ids])

with torch.no_grad():
    output = model(input_tensor, return_dict=True)
    logits = output["logits"]
    last_logits = logits[0, -1]

    # Get probabilities
    probs = torch.softmax(last_logits, dim=-1)

    # Show entropy (random = high, confident = low)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    print(f"Entropy: {entropy:.4f}")

    # Show probability distribution
    print("Top 10 probs:")
    top_probs, top_ids = torch.topk(probs, 10)
    for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
        piece = sp.IdToPiece(tid)
        print(f"  {tid}: {piece!r} prob={prob:.4f}")