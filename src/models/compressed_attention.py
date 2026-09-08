"""
Compressed attention layer for GPT-NeoX models.

This modifies the attention computation to use compressed (quantized) K/V values,
giving real perplexity impact from compression.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
sys.path.insert(0, '/workspace/padic-transformers')

from src.kernels import float_to_2adic, _2adic_to_float, CacheCompressionConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward
)


def create_compressed_attention_forward(compression_config: CacheCompressionConfig):
    """
    Create a modified forward function that compresses K/V in attention.

    Based on actual GPTNeoX implementation:
    - QKV are computed together and chunked
    - Rotary embeddings are applied
    - Then attention is computed via ALL_ATTENTION_FUNCTIONS interface

    We intercept AFTER rotary embeddings to compress K/V.
    """

    def compressed_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_past = None,
        position_embeddings = None,
        **kwargs
    ):
        """
        Modified GPTNeoXAttention forward with K/V compression.

        This follows the exact flow of the original but compresses
        key_states and value_states after rotary embeddings.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        # 1. Compute Q, K, V
        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        # 2. Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. Update cache if needed
        if layer_past is not None:
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx)

        # ===== COMPRESSION HAPPENS HERE =====
        if compression_config.strategy != 'none':
            # Store original dtype to preserve it after decompression
            original_dtype = key_states.dtype

            # Get precision
            if compression_config.strategy == 'simple_2x':
                precision = compression_config.uniform_precision
            elif compression_config.strategy == 'adaptive':
                precision = compression_config.medium_precision
            else:
                precision = 16

            # Compress K and V (quantize)
            key_compressed = float_to_2adic(key_states, precision=precision)
            value_compressed = float_to_2adic(value_states, precision=precision)

            # Decompress (introduces quantization error!)
            key_states = _2adic_to_float(key_compressed, precision=precision).to(original_dtype)
            value_states = _2adic_to_float(value_compressed, precision=precision).to(original_dtype)
        # ===== END COMPRESSION =====

        # 4. Compute attention using original interface
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,  # Compressed!
            value_states,  # Compressed!
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        # 5. Reshape and project output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights

    return compressed_forward


def apply_compression_to_model(model, compression_config: CacheCompressionConfig):
    """
    Apply K/V compression to all GPTNeoXAttention layers.

    This replaces the forward method of each attention layer to:
    1. Compute Q, K, V normally
    2. Apply rotary embeddings
    3. Compress K and V to 2-adic representation
    4. Decompress (with quantization loss)
    5. Use lossy K/V in attention computation

    Args:
        model: HuggingFace GPT-NeoX model
        compression_config: Compression configuration

    Returns:
        Modified model (in-place)
    """
    print(f"\nApplying {compression_config.strategy} compression to attention layers...")

    # Find all GPTNeoXAttention layers
    attention_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'GPTNeoXAttention':
            attention_layers.append((name, module))

    print(f"Found {len(attention_layers)} GPTNeoXAttention layers")

    if len(attention_layers) == 0:
        print("ERROR: No GPTNeoXAttention layers found!")
        print("Model architecture:", model.__class__.__name__)
        print("Available modules:", [n for n, _ in model.named_modules()][:10])
        return model

    # Apply compression to each layer
    modified_count = 0
    for layer_name, attn_module in attention_layers:
        try:
            # Verify required attributes exist
            required_attrs = ['query_key_value', 'dense', 'head_size', 'scaling', 'config']
            if not all(hasattr(attn_module, attr) for attr in required_attrs):
                print(f"  Skipping {layer_name}: missing required attributes")
                continue

            # Store original forward (for potential restoration)
            attn_module._original_forward = attn_module.forward

            # Create compressed version
            compressed_forward = create_compressed_attention_forward(compression_config)

            # Replace forward method (bind to instance)
            attn_module.forward = compressed_forward.__get__(attn_module, attn_module.__class__)

            modified_count += 1

        except Exception as e:
            print(f"  Error modifying {layer_name}: {e}")
            continue

    print(f"✓ Successfully applied compression to {modified_count}/{len(attention_layers)} layers")

    if compression_config.strategy == 'simple_2x':
        print(f"  Precision: {compression_config.uniform_precision}-bit")
    elif compression_config.strategy == 'adaptive':
        print(f"  Precision: adaptive (16/8/4 bits)")

    return model


def remove_compression_from_model(model):
    """
    Remove compression from model (restore original attention).

    Note: This only works if original forward was stored.
    """
    # For now, just reload the model fresh
    # In production, we'd store original forwards
    print("To remove compression, reload the model from checkpoint")
    return model
