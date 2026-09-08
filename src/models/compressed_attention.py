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

from src.kernels import float_to_2adic, _2adic_to_float, CacheCompressionConfig, CompressedCache
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

        # 3. Update cache (CompressedCache handles compression internally)
        # If using CompressedCache, K/V are compressed, stored as uint8, then decompressed
        # If using standard cache, K/V pass through unchanged
        if layer_past is not None:
            original_dtype = key_states.dtype
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx)
            # Ensure dtype matches (CompressedCache returns float32, need to match query)
            key_states = key_states.to(original_dtype)
            value_states = value_states.to(original_dtype)

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


def create_compressed_cache(compression_config: CacheCompressionConfig):
    """
    Create a CompressedCache for KV cache compression.

    This cache stores K/V in compressed format (uint8) instead of float16,
    achieving 2x memory reduction.

    Args:
        compression_config: Compression configuration

    Returns:
        CompressedCache instance
    """
    print(f"\nCreating CompressedCache with {compression_config.strategy} strategy...")

    if compression_config.strategy == 'simple_2x':
        print(f"  Precision: {compression_config.uniform_precision}-bit")
    elif compression_config.strategy == 'adaptive':
        print(f"  Precision: adaptive (16/8/4 bits)")

    cache = CompressedCache(compression_config)
    print(f"✓ CompressedCache created")

    return cache


def apply_compression_to_model(model, compression_config: CacheCompressionConfig):
    """
    Prepare model for compressed inference.

    NOTE: This function is now simplified - compression is handled by CompressedCache.
    To use compression, create a CompressedCache and pass it via past_key_values parameter.

    Args:
        model: HuggingFace GPT-NeoX model
        compression_config: Compression configuration

    Returns:
        tuple: (model, compressed_cache)
    """
    print(f"\nSetting up compression with {compression_config.strategy} strategy...")

    # Create compressed cache
    cache = create_compressed_cache(compression_config)

    print(f"✓ Model ready for compressed inference")
    print(f"  Pass the cache via: model(..., past_key_values=cache)")

    return model, cache


def remove_compression_from_model(model):
    """
    Remove compression from model (restore original attention).

    Note: This only works if original forward was stored.
    """
    # For now, just reload the model fresh
    # In production, we'd store original forwards
    print("To remove compression, reload the model from checkpoint")
    return model
