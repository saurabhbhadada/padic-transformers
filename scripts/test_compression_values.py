#!/usr/bin/env python3
"""
Debug script to check if compression/decompression is working correctly.
"""

import sys
sys.path.insert(0, '/workspace/padic-transformers')

import torch
from src.kernels import float_to_2adic, _2adic_to_float

def test_roundtrip():
    """Test if compression/decompression roundtrip preserves values reasonably."""

    print("=" * 60)
    print("Testing Compression Roundtrip")
    print("=" * 60)

    # Create test tensor with various magnitudes (similar to K/V states)
    test_values = torch.tensor([
        [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        [-0.1, -0.5, -1.0, -2.0, -5.0, -10.0],
    ], dtype=torch.float16)

    print(f"\n1. Original values:")
    print(f"   Shape: {test_values.shape}")
    print(f"   Values:\n{test_values}")
    print(f"   Min: {test_values.min():.4f}, Max: {test_values.max():.4f}")
    print(f"   Mean: {test_values.mean():.4f}, Std: {test_values.std():.4f}")

    # Compress with 8-bit precision
    precision = 8
    print(f"\n2. Compressing with {precision}-bit precision...")
    compressed, scale = float_to_2adic(test_values, precision=precision)

    print(f"   Compressed dtype: {compressed.dtype}")
    print(f"   Compressed shape: {compressed.shape}")
    print(f"   Scale factor: {scale.item():.4f}")
    print(f"   Compressed values:\n{compressed}")

    # Decompress
    print(f"\n3. Decompressing...")
    decompressed = _2adic_to_float(compressed, precision=precision, scale=scale)

    print(f"   Decompressed dtype: {decompressed.dtype}")
    print(f"   Decompressed shape: {decompressed.shape}")
    print(f"   Values:\n{decompressed}")
    print(f"   Min: {decompressed.min():.4f}, Max: {decompressed.max():.4f}")
    print(f"   Mean: {decompressed.mean():.4f}, Std: {decompressed.std():.4f}")

    # Compute error
    error = (test_values.float() - decompressed).abs()
    print(f"\n4. Error Analysis:")
    print(f"   Absolute error:\n{error}")
    print(f"   Max error: {error.max():.6f}")
    print(f"   Mean error: {error.mean():.6f}")
    print(f"   Relative error: {(error / (test_values.abs().float() + 1e-8)).mean():.6f}")

    # Check if values are reasonable
    if error.max() > 1.0:
        print(f"\n⚠️  WARNING: Maximum error {error.max():.4f} is too large!")
        print(f"   Compression may be corrupting values")
        return False
    elif torch.isnan(decompressed).any() or torch.isinf(decompressed).any():
        print(f"\n⚠️  WARNING: Decompressed values contain NaN or Inf!")
        return False
    else:
        print(f"\n✓ Compression roundtrip looks reasonable")
        return True

def test_realistic_kv():
    """Test with realistic K/V state shapes and magnitudes."""

    print("\n" + "=" * 60)
    print("Testing with Realistic K/V State")
    print("=" * 60)

    # Simulate realistic K/V state: [batch=1, num_heads=12, seq_len=10, head_dim=64]
    batch, num_heads, seq_len, head_dim = 1, 12, 10, 64

    # Generate random values with typical magnitude (around -5 to 5)
    key_states = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16) * 2.0

    print(f"\n1. Original K states:")
    print(f"   Shape: {key_states.shape}")
    print(f"   Min: {key_states.min():.4f}, Max: {key_states.max():.4f}")
    print(f"   Mean: {key_states.mean():.4f}, Std: {key_states.std():.4f}")

    # Compress
    precision = 8
    compressed, scale = float_to_2adic(key_states, precision=precision)

    print(f"\n2. Compressed:")
    print(f"   Scale: {scale.item():.4f}")
    print(f"   Compressed min: {compressed.min().item()}, max: {compressed.max().item()}")

    # Decompress
    decompressed = _2adic_to_float(compressed, precision=precision, scale=scale).to(torch.float16)

    print(f"\n3. Decompressed:")
    print(f"   Min: {decompressed.min():.4f}, Max: {decompressed.max():.4f}")
    print(f"   Mean: {decompressed.mean():.4f}, Std: {decompressed.std():.4f}")

    # Error
    error = (key_states - decompressed).abs()
    print(f"\n4. Error:")
    print(f"   Max error: {error.max():.6f}")
    print(f"   Mean error: {error.mean():.6f}")
    print(f"   95th percentile error: {torch.quantile(error.float(), 0.95):.6f}")

    # Check for corruption
    if error.max() > 0.5:  # For 8-bit, errors should be small
        print(f"\n⚠️  WARNING: Errors are too large for 8-bit quantization!")
        return False
    else:
        print(f"\n✓ K/V compression looks good")
        return True

if __name__ == "__main__":
    success1 = test_roundtrip()
    success2 = test_realistic_kv()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All compression tests passed!")
    else:
        print("✗ Compression tests FAILED - values are being corrupted")
    print("=" * 60)
