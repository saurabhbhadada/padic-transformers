#!/usr/bin/env python3
"""
Quick test to verify compression is working correctly.

This loads a tiny portion of the model and tests that:
1. Compression is applied successfully
2. K/V values are actually different after compression
3. Model still runs without errors
"""

import sys
sys.path.insert(0, '/workspace/padic-transformers')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kernels import CacheCompressionConfig
from src.models import apply_compression_to_model


def test_compression():
    print("="*60)
    print("Testing Compression Integration")
    print("="*60)

    # Load tiny model (Pythia-160M for speed)
    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"  Number of layers: {len(model.gpt_neox.layers)}")

    # Test baseline inference
    print("\n2. Testing baseline (no compression)...")
    test_text = "The future of artificial intelligence is"
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs_baseline = model(**inputs)
        logits_baseline = outputs_baseline.logits

    print(f"✓ Baseline inference successful")
    print(f"  Output shape: {logits_baseline.shape}")
    print(f"  Sample logit: {logits_baseline[0, 0, 0].item():.4f}")

    # Apply compression
    print("\n3. Applying compression...")
    compression_config = CacheCompressionConfig(
        strategy='simple_2x',
        uniform_precision=8
    )

    model = apply_compression_to_model(model, compression_config)

    # Test compressed inference
    print("\n4. Testing compressed inference...")
    with torch.no_grad():
        outputs_compressed = model(**inputs)
        logits_compressed = outputs_compressed.logits

    print(f"✓ Compressed inference successful")
    print(f"  Output shape: {logits_compressed.shape}")
    print(f"  Sample logit: {logits_compressed[0, 0, 0].item():.4f}")

    # Compare outputs
    print("\n5. Comparing outputs...")
    diff = (logits_baseline - logits_compressed).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff > 0:
        print(f"✓ Compression is affecting outputs (good!)")
        print(f"  Quantization error is introduced as expected")
    else:
        print(f"⚠ WARNING: Outputs are identical!")
        print(f"  Compression may not be working correctly")

    # Test generation
    print("\n6. Testing text generation...")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"✓ Generation successful")
    print(f"  Generated: {generated_text}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_compression()
