"""
Compressed KV cache implementation for transformers.

Stores K/V in compressed 2-adic format (uint8) instead of float16,
achieving 2x memory reduction for KV cache.
"""

import torch
from typing import List, Optional, Tuple, Any
import sys
sys.path.insert(0, '/workspace/padic-transformers')

from transformers.cache_utils import Cache
from src.kernels import float_to_2adic, _2adic_to_float, CacheCompressionConfig


class CompressedCache(Cache):
    """
    Compressed KV cache using 2-adic quantization.

    Stores keys and values in compressed format (uint8 for 8-bit precision)
    instead of float16, reducing memory footprint by 2x.

    Memory savings:
    - float16: 2 bytes per value
    - uint8: 1 byte per value
    - Scale factor: negligible overhead (1 float per layer)

    Compression ratio: ~2x for KV cache
    """

    def __init__(self, compression_config: CacheCompressionConfig):
        # Initialize base Cache with empty layers list
        # We manage our own compressed storage instead of using layer objects
        super().__init__(layers=[])

        self.compression_config = compression_config

        # Storage format: List of (compressed_tensor, scale_factor) per layer
        self.key_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.value_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Track sequence length per layer
        self.seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states (compressed).

        Args:
            key_states: New key states [batch, num_heads, seq_len, head_dim]
            value_states: New value states [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index

        Returns:
            Full key and value states (decompressed) including cache
        """
        # Get precision
        if self.compression_config.strategy == 'simple_2x':
            precision = self.compression_config.uniform_precision
        elif self.compression_config.strategy == 'adaptive':
            precision = self.compression_config.medium_precision
        else:
            precision = 16

        # Compress new K/V
        key_compressed, key_scale = float_to_2adic(key_states, precision=precision)
        value_compressed, value_scale = float_to_2adic(value_states, precision=precision)

        # Initialize layer cache if needed
        if layer_idx >= len(self.key_cache):
            self.key_cache.append((key_compressed, key_scale))
            self.value_cache.append((value_compressed, value_scale))
        else:
            # Concatenate with existing cache along sequence dimension (dim=2)
            old_key_compressed, old_key_scale = self.key_cache[layer_idx]
            old_value_compressed, old_value_scale = self.value_cache[layer_idx]

            # Use maximum scale to preserve range of both old and new values
            key_scale_combined = torch.max(old_key_scale, key_scale)
            value_scale_combined = torch.max(old_value_scale, value_scale)

            # Re-quantize old values with new combined scale if needed
            if key_scale_combined > old_key_scale:
                # Decompress old, re-compress with new scale
                old_key_decompressed = _2adic_to_float(old_key_compressed, precision, old_key_scale)
                old_key_compressed, _ = float_to_2adic(old_key_decompressed, precision=precision)

            if value_scale_combined > old_value_scale:
                old_value_decompressed = _2adic_to_float(old_value_compressed, precision, old_value_scale)
                old_value_compressed, _ = float_to_2adic(old_value_decompressed, precision=precision)

            # Re-quantize new values if needed
            if key_scale_combined > key_scale:
                key_decompressed = _2adic_to_float(key_compressed, precision, key_scale)
                key_compressed, _ = float_to_2adic(key_decompressed, precision=precision)

            if value_scale_combined > value_scale:
                value_decompressed = _2adic_to_float(value_compressed, precision, value_scale)
                value_compressed, _ = float_to_2adic(value_decompressed, precision=precision)

            # Concatenate compressed tensors
            key_compressed = torch.cat([old_key_compressed, key_compressed], dim=2)
            value_compressed = torch.cat([old_value_compressed, value_compressed], dim=2)

            # Update cache with concatenated compressed values
            self.key_cache[layer_idx] = (key_compressed, key_scale_combined)
            self.value_cache[layer_idx] = (value_compressed, value_scale_combined)

            key_scale = key_scale_combined
            value_scale = value_scale_combined

        # Decompress for attention computation
        # This is where quantization error is introduced!
        key_states = _2adic_to_float(key_compressed, precision=precision, scale=key_scale)
        value_states = _2adic_to_float(value_compressed, precision=precision, scale=value_scale)

        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length of cached keys."""
        if len(self.key_cache) <= layer_idx:
            return 0
        key_compressed, _ = self.key_cache[layer_idx]
        return key_compressed.shape[2]  # Sequence dimension

    def get_max_length(self) -> Optional[int]:
        """Get maximum cache length (None for dynamic)."""
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """
        Get the usable length of the cache for a given layer and new sequence length.

        Args:
            new_seq_length: The new sequence length being added
            layer_idx: The layer index

        Returns:
            The usable length (same as current sequence length for our cache)
        """
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        Reorder cache for beam search (not implemented yet).

        Args:
            beam_idx: Beam indices for reordering
        """
        # For now, raise error - beam search not supported with compression
        raise NotImplementedError(
            "Beam search is not yet supported with CompressedCache. "
            "Use standard cache for beam search operations."
        )

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """
        Get the query offset for a given layer.

        For standard autoregressive generation, this is 0.

        Args:
            layer_idx: The layer index

        Returns:
            Query offset (0 for standard generation)
        """
        return 0

    def reset(self):
        """Reset/clear all cached states."""
        self.key_cache = []
        self.value_cache = []
        self.seen_tokens = 0

    @property
    def batch_size(self) -> int:
        """Get the batch size from the cache."""
        if len(self.key_cache) == 0:
            return 0
        key_compressed, _ = self.key_cache[0]
        return key_compressed.shape[0]  # Batch dimension

    def get_memory_footprint(self) -> dict:
        """
        Calculate memory footprint of compressed cache.

        Returns:
            Dict with memory statistics in MB
        """
        key_memory = 0
        value_memory = 0
        scale_memory = 0

        for (key_compressed, key_scale), (value_compressed, value_scale) in zip(
            self.key_cache, self.value_cache
        ):
            # Compressed tensors (uint8 = 1 byte)
            key_memory += key_compressed.numel() * key_compressed.element_size()
            value_memory += value_compressed.numel() * value_compressed.element_size()

            # Scale factors (float32 = 4 bytes, but only 1 per layer)
            scale_memory += key_scale.numel() * 4
            scale_memory += value_scale.numel() * 4

        total_memory = key_memory + value_memory + scale_memory

        # Calculate what uncompressed would be (float16 = 2 bytes)
        uncompressed_memory = 0
        for key_compressed, _ in self.key_cache:
            uncompressed_memory += key_compressed.numel() * 2  # float16
        for value_compressed, _ in self.value_cache:
            uncompressed_memory += value_compressed.numel() * 2  # float16

        return {
            "compressed_mb": total_memory / (1024 ** 2),
            "uncompressed_mb": uncompressed_memory / (1024 ** 2),
            "compression_ratio": uncompressed_memory / total_memory if total_memory > 0 else 1.0,
            "key_mb": key_memory / (1024 ** 2),
            "value_mb": value_memory / (1024 ** 2),
            "scale_mb": scale_memory / (1024 ** 2),
        }
