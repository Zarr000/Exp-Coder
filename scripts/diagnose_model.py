#!/usr/bin/env python3
"""
Full Model Diagnostic Script.

Prints:
- Model vocab size
- Tokenizer vocab size, eos_id, bos_id, pad_id, unk_id
- Top 20 predicted token ids for given prompts
- id_to_piece() decode for every top token
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Force UTF-8 output
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
from pathlib import Path

import torch
import sentencepiece as spm


def main():
    # Paths
    model_dir = Path("checkpoints/expera_coder_120m")

    # Load tokenizer
    tokenizer_path = model_dir / "tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    print("=" * 60)
    print("TOKENIZER DIAGNOSTIC")
    print("=" * 60)
    print(f"Tokenizer vocab size: {sp.GetPieceSize()}")
    print(f"eos_id: {sp.eos_id()}")
    print(f"bos_id: {sp.bos_id()}")
    print(f"pad_id: {sp.pad_id()}")
    print(f"unk_id: {sp.unk_id()}")

    # Print all special IDs
    print(f"\nAll special tokens:")
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        if piece.startswith("<") and piece.endswith(">"):
            print(f"  ID {i}: {piece}")

    print(f"\nTokenizer decode test:")
    # Test encode/decode
    test_text = "hello"
    ids = sp.EncodeAsIds(test_text)
    decoded = sp.DecodeIds(ids)
    print(f"  '{test_text}' -> {ids} -> '{decoded}'")

    # Load model config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model_vocab_size = config.get("vocab_size", 0)
    print(f"\nModel vocab size (from config): {model_vocab_size}")

    print("=" * 60)
    print("MODEL LOADING")
    print("=" * 60)

    # Load model
    from src.model.architecture.expera_model import ExperaModel

    model = ExperaModel(
        vocab_size=config.get("vocab_size", 200),
        hidden_size=config.get("hidden_size", 768),
        num_layers=config.get("num_hidden_layers", 12),
        num_heads=config.get("num_attention_heads", 8),
        num_kv_heads=config.get("num_key_value_heads", 4),
        intermediate_size=config.get("intermediate_size", 1536),
        max_position_embeddings=config.get("max_position_embeddings", 2048),
    )

    # Load checkpoint
    checkpoint_path = model_dir / "model.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle different key formats
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[9:]  # Remove "_orig_mod." prefix
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Get actual embedding size
    actual_vocab_size = model.lm_head.out_features
    print(f"Model output vocab size (lm_head): {actual_vocab_size}")

    # Check for mismatch
    tokenizer_vocab = sp.GetPieceSize()
    if actual_vocab_size != tokenizer_vocab:
        print(f"\n!!! TOKENIZER MISMATCH !!!")
        print(f"  Model expects: {actual_vocab_size}")
        print(f"  Tokenizer has: {tokenizer_vocab}")
        print(f"  Need to resize embeddings")
    else:
        print(f"\n[OK] No mismatch detected")

    print("=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    # Test prompts
    prompts = [
        "hello",
        "User: hello\nAssistant:",
    ]

    from src.inference.generator import Generator, GenerationConfig

    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
    )

    generator = Generator(model, sp, gen_config)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Encode prompt
        input_ids = sp.EncodeAsIds(prompt)
        input_tensor = torch.tensor([input_ids])

        print(f"  Input IDs: {input_ids}")

        # Generate
        with torch.no_grad():
            output = model(input_tensor)
            logits = output["logits"]

            # Get top 20 tokens for last position
            last_logits = logits[0, -1, :]
            top_probs, top_ids = torch.topk(last_logits, 20)

            print(f"  Top 20 predicted token IDs:")
            for i, (prob, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
                piece = sp.IdToPiece(tid)
                # Filter to ASCII-printable for display
                piece_clean = piece.encode('ascii', errors='replace').decode('ascii')
                print(f"    {i+1:2d}. ID={tid:3d} prob={prob:.4f} piece={piece_clean}")

        # Full generation with debug - trace step by step
        print(f"  Manual generation trace:")
        model.eval()
        input_tensor = torch.tensor([sp.EncodeAsIds(prompt)])
        print(f"    Start input_ids: {input_tensor.tolist()}")

        with torch.no_grad():
            # First forward
            outputs = model(input_tensor, use_cache=True, return_dict=True)
            logits = outputs["logits"]

            # Get first predicted token
            first_logits = logits[0, -1, :]
            first_token = first_logits.argmax().item()
            first_prob = first_logits[first_token].item()
            first_piece = sp.IdToPiece(first_token)

            print(f"    First prediction: id={first_token} prob={first_prob:.4f} piece={first_piece}")

        # Try using generator - walk through the decoder
        # Create our own generation trace
        from src.inference.generator import Sampling

        decoder = Sampling(eos_token_id=sp.eos_id())

        from src.inference.generator import GenerationConfig
        config = GenerationConfig(max_new_tokens=20, temperature=0.7, top_k=50, top_p=0.9)

        print(f"    Running Sampling decoder...")
        generated = decoder.decode(model, input_tensor, config=config)

        print(f"    Generated IDs: {generated.tolist()}")
        print(f"    Generated IDs length: {generated.shape[1]}")

        # Decode the output
        output_text = sp.DecodeIds(generated[0].tolist())
        print(f"    Raw decoded: {repr(output_text)[:100]}")

        # Strip input
        output_only = generated[0, input_tensor.shape[1]:].tolist()
        output_text2 = sp.DecodeIds(output_only)
        print(f"    After stripping input: {repr(output_text2)[:100]}")

    print("=" * 60)


if __name__ == "__main__":
    main()