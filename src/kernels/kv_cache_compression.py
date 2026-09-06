"""
KV-Cache compression using 2-adic representations.

Implements three compression strategies:
1. Simple 2x: Uniform 8-bit compression
2. Adaptive: Position-based precision (recent=16bit, old=4bit)
3. Ultrametric clustering: Hierarchical compression (future work)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from .padic_ops import (
    float_to_2adic,
    _2adic_to_float,
    float_to_2adic_signed,
    _2adic_to_float_signed,
    ultrametric_distance,
)


@dataclass
class CacheCompressionConfig:
    """Configuration for KV-cache compression."""

    # Compression strategy: 'none', 'simple_2x', 'adaptive', 'ultrametric'
    strategy: str = 'simple_2x'

    # For simple_2x
    uniform_precision: int = 8  # bits

    # For adaptive
    recent_threshold: int = 128  # tokens
    medium_threshold: int = 1024  # tokens
    recent_precision: int = 16  # bits (full precision)
    medium_precision: int = 8   # bits (2x compression)
    old_precision: int = 4      # bits (4x compression)

    # General
    use_signed: bool = False  # Use signed vs unsigned 2-adic


class KVCacheCompressor:
    """
    Compresses KV-cache using 2-adic representations.

    Usage:
        compressor = KVCacheCompressor(config)

        # During inference
        k_compressed, v_compressed = compressor.compress(k, v, position)

        # When needed for attention
        k_decompressed = compressor.decompress_keys(k_compressed)
        v_decompressed = compressor.decompress_values(v_compressed)
    """

    def __init__(self, config: CacheCompressionConfig):
        self.config = config
        self.current_position = 0

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress key and value tensors.

        Args:
            keys: [batch, n_heads, seq_len, head_dim] or [batch, seq_len, dim]
            values: Same shape as keys
            position: Current sequence position (for adaptive compression)

        Returns:
            Compressed keys and values (int tensors with reduced precision)
        """
        if self.config.strategy == 'none':
            return keys, values

        elif self.config.strategy == 'simple_2x':
            return self._compress_simple_2x(keys, values)

        elif self.config.strategy == 'adaptive':
            if position is None:
                position = self.current_position
            return self._compress_adaptive(keys, values, position)

        elif self.config.strategy == 'ultrametric':
            return self._compress_ultrametric(keys, values)

        else:
            raise ValueError(f"Unknown compression strategy: {self.config.strategy}")

    def decompress(
        self,
        keys_compressed: torch.Tensor,
        values_compressed: torch.Tensor,
        precision: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress keys and values back to float.

        Args:
            keys_compressed: Compressed keys (int tensor)
            values_compressed: Compressed values (int tensor)
            precision: Bit precision used (if None, use config)

        Returns:
            Decompressed float tensors
        """
        if precision is None:
            precision = self.config.uniform_precision

        convert_fn = _2adic_to_float_signed if self.config.use_signed else _2adic_to_float

        keys_float = convert_fn(keys_compressed, precision)
        values_float = convert_fn(values_compressed, precision)

        return keys_float, values_float

    def _compress_simple_2x(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple uniform compression: all positions get same precision.

        Memory: 2x reduction (16bit -> 8bit)
        """
        precision = self.config.uniform_precision
        convert_fn = float_to_2adic_signed if self.config.use_signed else float_to_2adic

        keys_compressed = convert_fn(keys, precision)
        values_compressed = convert_fn(values, precision)

        return keys_compressed, values_compressed

    def _compress_adaptive(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        current_position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive compression: recent tokens get more precision.

        Args:
            keys: [batch, n_heads, seq_len, head_dim]
            values: Same shape
            current_position: Current generation position

        Returns:
            Compressed tensors with mixed precision
        """
        seq_len = keys.shape[-2] if keys.dim() == 4 else keys.shape[1]
        device = keys.device

        # Determine precision for each position based on age
        positions = torch.arange(seq_len, device=device)
        ages = current_position - positions

        # Create precision map
        precisions = torch.full((seq_len,), self.config.old_precision, device=device)
        precisions[ages < self.config.medium_threshold] = self.config.medium_precision
        precisions[ages < self.config.recent_threshold] = self.config.recent_precision

        # Compress with position-dependent precision
        convert_fn = float_to_2adic_signed if self.config.use_signed else float_to_2adic

        # For simplicity in this version, compress by tier
        # More efficient implementation would use custom kernel

        keys_compressed = torch.zeros_like(keys, dtype=torch.int32)
        values_compressed = torch.zeros_like(values, dtype=torch.int32)

        # Recent positions (full precision)
        recent_mask = ages < self.config.recent_threshold
        if recent_mask.any():
            if keys.dim() == 4:
                keys_compressed[:, :, recent_mask] = convert_fn(
                    keys[:, :, recent_mask], self.config.recent_precision
                )
                values_compressed[:, :, recent_mask] = convert_fn(
                    values[:, :, recent_mask], self.config.recent_precision
                )
            else:
                keys_compressed[:, recent_mask] = convert_fn(
                    keys[:, recent_mask], self.config.recent_precision
                )
                values_compressed[:, recent_mask] = convert_fn(
                    values[:, recent_mask], self.config.recent_precision
                )

        # Medium positions
        medium_mask = (ages >= self.config.recent_threshold) & (ages < self.config.medium_threshold)
        if medium_mask.any():
            if keys.dim() == 4:
                keys_compressed[:, :, medium_mask] = convert_fn(
                    keys[:, :, medium_mask], self.config.medium_precision
                )
                values_compressed[:, :, medium_mask] = convert_fn(
                    values[:, :, medium_mask], self.config.medium_precision
                )
            else:
                keys_compressed[:, medium_mask] = convert_fn(
                    keys[:, medium_mask], self.config.medium_precision
                )
                values_compressed[:, medium_mask] = convert_fn(
                    values[:, medium_mask], self.config.medium_precision
                )

        # Old positions (maximum compression)
        old_mask = ages >= self.config.medium_threshold
        if old_mask.any():
            if keys.dim() == 4:
                keys_compressed[:, :, old_mask] = convert_fn(
                    keys[:, :, old_mask], self.config.old_precision
                )
                values_compressed[:, :, old_mask] = convert_fn(
                    values[:, :, old_mask], self.config.old_precision
                )
            else:
                keys_compressed[:, old_mask] = convert_fn(
                    keys[:, old_mask], self.config.old_precision
                )
                values_compressed[:, old_mask] = convert_fn(
                    values[:, old_mask], self.config.old_precision
                )

        # Store precision metadata for decompression
        self._adaptive_precisions = precisions

        return keys_compressed, values_compressed

    def decompress_adaptive(
        self,
        keys_compressed: torch.Tensor,
        values_compressed: torch.Tensor,
        current_position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress adaptively compressed cache.

        Needs to know the precision used for each position.
        """
        seq_len = keys_compressed.shape[-2] if keys_compressed.dim() == 4 else keys_compressed.shape[1]
        device = keys_compressed.device

        convert_fn = _2adic_to_float_signed if self.config.use_signed else _2adic_to_float

        # Reconstruct position-based precision
        positions = torch.arange(seq_len, device=device)
        ages = current_position - positions

        keys_float = torch.zeros_like(keys_compressed, dtype=torch.float32)
        values_float = torch.zeros_like(values_compressed, dtype=torch.float32)

        # Decompress each tier
        recent_mask = ages < self.config.recent_threshold
        if recent_mask.any():
            if keys_compressed.dim() == 4:
                keys_float[:, :, recent_mask] = convert_fn(
                    keys_compressed[:, :, recent_mask], self.config.recent_precision
                )
                values_float[:, :, recent_mask] = convert_fn(
                    values_compressed[:, :, recent_mask], self.config.recent_precision
                )
            else:
                keys_float[:, recent_mask] = convert_fn(
                    keys_compressed[:, recent_mask], self.config.recent_precision
                )
                values_float[:, recent_mask] = convert_fn(
                    values_compressed[:, recent_mask], self.config.recent_precision
                )

        medium_mask = (ages >= self.config.recent_threshold) & (ages < self.config.medium_threshold)
        if medium_mask.any():
            if keys_compressed.dim() == 4:
                keys_float[:, :, medium_mask] = convert_fn(
                    keys_compressed[:, :, medium_mask], self.config.medium_precision
                )
                values_float[:, :, medium_mask] = convert_fn(
                    values_compressed[:, :, medium_mask], self.config.medium_precision
                )
            else:
                keys_float[:, medium_mask] = convert_fn(
                    keys_compressed[:, medium_mask], self.config.medium_precision
                )
                values_float[:, medium_mask] = convert_fn(
                    values_compressed[:, medium_mask], self.config.medium_precision
                )

        old_mask = ages >= self.config.medium_threshold
        if old_mask.any():
            if keys_compressed.dim() == 4:
                keys_float[:, :, old_mask] = convert_fn(
                    keys_compressed[:, :, old_mask], self.config.old_precision
                )
                values_float[:, :, old_mask] = convert_fn(
                    values_compressed[:, :, old_mask], self.config.old_precision
                )
            else:
                keys_float[:, old_mask] = convert_fn(
                    keys_compressed[:, old_mask], self.config.old_precision
                )
                values_float[:, old_mask] = convert_fn(
                    values_compressed[:, old_mask], self.config.old_precision
                )

        return keys_float, values_float

    def _compress_ultrametric(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ultrametric clustering-based compression (advanced).

        TODO: Implement hierarchical clustering using 2-adic distance.
        For now, falls back to simple_2x.
        """
        # Placeholder for future implementation
        return self._compress_simple_2x(keys, values)

    def get_memory_stats(
        self,
        keys: torch.Tensor,
        compressed_keys: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate memory statistics for compression.

        Returns:
            Dictionary with memory info (in MB)
        """
        def tensor_mb(t):
            return t.element_size() * t.numel() / (1024 ** 2)

        original_mem = tensor_mb(keys)
        compressed_mem = tensor_mb(compressed_keys)
        compression_ratio = original_mem / compressed_mem if compressed_mem > 0 else 0

        return {
            'original_mb': original_mem,
            'compressed_mb': compressed_mem,
            'compression_ratio': compression_ratio,
            'savings_mb': original_mem - compressed_mem,
        }


# Convenience functions

def compress_kv_cache_simple(
    keys: torch.Tensor,
    values: torch.Tensor,
    precision: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quick function for simple 2x compression.

    Args:
        keys, values: Float tensors
        precision: Bits for 2-adic representation

    Returns:
        Compressed int tensors
    """
    config = CacheCompressionConfig(strategy='simple_2x', uniform_precision=precision)
    compressor = KVCacheCompressor(config)
    return compressor.compress(keys, values)


def decompress_kv_cache_simple(
    keys_compressed: torch.Tensor,
    values_compressed: torch.Tensor,
    precision: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quick function to decompress simple 2x compressed cache.
    """
    config = CacheCompressionConfig(strategy='simple_2x', uniform_precision=precision)
    compressor = KVCacheCompressor(config)
    return compressor.decompress(keys_compressed, values_compressed, precision)
