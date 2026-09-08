"""
Wrapper for HuggingFace models with KV-cache compression integrated.

This actually uses compressed cache during inference to measure real impact.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedModel
import sys
sys.path.insert(0, '/workspace/padic-transformers')

from src.kernels import KVCacheCompressor, CacheCompressionConfig


class CompressedModelWrapper:
    """
    Wraps a HuggingFace model to use compressed KV-cache.

    This hooks into the attention layers to compress/decompress cache
    during actual inference, giving real perplexity and memory measurements.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        compression_config: CacheCompressionConfig,
    ):
        self.model = model
        self.compression_config = compression_config
        self.compressor = KVCacheCompressor(compression_config)
        self.current_position = 0

        # Statistics
        self.compression_stats = {
            'total_compressions': 0,
            'total_compression_time': 0.0,
            'total_decompression_time': 0.0,
            'memory_saved_bytes': 0,
        }

    def __call__(self, *args, **kwargs):
        """Forward pass with compression."""
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values = None,
        **kwargs
    ):
        """
        Forward pass with KV-cache compression.

        Strategy:
        1. Run model normally to get outputs and cache
        2. Compress the cache
        3. Decompress for next forward pass

        This simulates using compressed cache in real inference.
        """
        import time

        # Decompress past cache if provided
        if past_key_values is not None and self.compression_config.strategy != 'none':
            if self.compression_stats['total_compressions'] == 0:
                print(f"[DEBUG] Starting decompression for {len(past_key_values)} layers")
            t0 = time.time()
            past_key_values = self._decompress_cache(past_key_values)
            self.compression_stats['total_decompression_time'] += time.time() - t0

        # Run model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs
        )

        # Compress new cache if using cache
        if use_cache and hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            if self.compression_config.strategy != 'none':
                if self.compression_stats['total_compressions'] == 0:
                    print(f"[DEBUG] First compression happening with strategy: {self.compression_config.strategy}")
                t0 = time.time()
                compressed_cache = self._compress_cache(outputs.past_key_values)
                comp_time = time.time() - t0
                self.compression_stats['total_compression_time'] += comp_time
                self.compression_stats['total_compressions'] += 1

                if self.compression_stats['total_compressions'] == 1:
                    print(f"[DEBUG] First compression took {comp_time*1000:.2f}ms")

                # Replace cache with compressed version
                outputs.past_key_values = compressed_cache
                self.current_position += input_ids.shape[1]

        return outputs

    def _compress_cache(self, cache):
        """Compress KV cache."""
        if cache is None:
            return None

        # Handle DynamicCache
        if hasattr(cache, '__class__') and 'DynamicCache' in cache.__class__.__name__:
            compressed_cache = []

            for layer_idx in range(len(cache.key_cache)):
                keys = cache.key_cache[layer_idx]
                values = cache.value_cache[layer_idx]

                # Compress
                k_comp, v_comp = self.compressor.compress(
                    keys, values, position=self.current_position
                )

                # Track memory savings
                original_size = keys.element_size() * keys.numel() * 2  # K + V
                compressed_size = k_comp.element_size() * k_comp.numel() * 2
                self.compression_stats['memory_saved_bytes'] += (original_size - compressed_size)

                compressed_cache.append((k_comp, v_comp))

            return compressed_cache

        # Handle tuple cache
        elif isinstance(cache, (tuple, list)):
            compressed_cache = []

            for layer_cache in cache:
                if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                    keys, values = layer_cache[0], layer_cache[1]

                    # Compress
                    k_comp, v_comp = self.compressor.compress(
                        keys, values, position=self.current_position
                    )

                    # Track memory
                    original_size = keys.element_size() * keys.numel() * 2
                    compressed_size = k_comp.element_size() * k_comp.numel() * 2
                    self.compression_stats['memory_saved_bytes'] += (original_size - compressed_size)

                    compressed_cache.append((k_comp, v_comp))
                else:
                    compressed_cache.append(layer_cache)

            return tuple(compressed_cache)

        return cache

    def _decompress_cache(self, compressed_cache):
        """Decompress KV cache back to float for computation."""
        if compressed_cache is None:
            return None

        decompressed_cache = []

        for layer_idx, layer_cache in enumerate(compressed_cache):
            if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                k_comp, v_comp = layer_cache[0], layer_cache[1]

                # Decompress
                if self.compression_config.strategy == "adaptive":
                    keys, values = self.compressor.decompress_adaptive(
                        k_comp, v_comp, current_position=self.current_position
                    )
                else:
                    keys, values = self.compressor.decompress(k_comp, v_comp)

                decompressed_cache.append((keys, values))
            else:
                decompressed_cache.append(layer_cache)

        return tuple(decompressed_cache)

    def get_compression_stats(self):
        """Get compression statistics."""
        stats = self.compression_stats.copy()
        if stats['total_compressions'] > 0:
            stats['avg_compression_time_ms'] = (
                stats['total_compression_time'] * 1000 / stats['total_compressions']
            )
            stats['avg_decompression_time_ms'] = (
                stats['total_decompression_time'] * 1000 / stats['total_compressions']
            )
            stats['memory_saved_mb'] = stats['memory_saved_bytes'] / (1024 ** 2)
        return stats

    def reset_stats(self):
        """Reset compression statistics."""
        self.compression_stats = {
            'total_compressions': 0,
            'total_compression_time': 0.0,
            'total_decompression_time': 0.0,
            'memory_saved_bytes': 0,
        }
        self.current_position = 0

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device

    @property
    def config(self):
        """Get model config."""
        return self.model.config
