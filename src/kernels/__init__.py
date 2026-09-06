"""2-adic kernel operations for neural networks."""

from .padic_ops import (
    float_to_2adic,
    _2adic_to_float,
    float_to_2adic_signed,
    _2adic_to_float_signed,
    ultrametric_distance,
    _2adic_norm,
    _2adic_valuation,
    float_to_2adic_differentiable,
    visualize_2adic,
    PadicConfig,
)

from .kv_cache_compression import (
    KVCacheCompressor,
    CacheCompressionConfig,
    compress_kv_cache_simple,
    decompress_kv_cache_simple,
)

__all__ = [
    # Core ops
    'float_to_2adic',
    '_2adic_to_float',
    'float_to_2adic_signed',
    '_2adic_to_float_signed',
    'ultrametric_distance',
    '_2adic_norm',
    '_2adic_valuation',
    'float_to_2adic_differentiable',
    'visualize_2adic',
    'PadicConfig',
    # KV cache compression
    'KVCacheCompressor',
    'CacheCompressionConfig',
    'compress_kv_cache_simple',
    'decompress_kv_cache_simple',
]
