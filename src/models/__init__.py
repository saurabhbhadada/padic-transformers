"""Model utilities for compressed inference."""

from .compressed_attention import apply_compression_to_model, remove_compression_from_model

__all__ = ['apply_compression_to_model', 'remove_compression_from_model']
