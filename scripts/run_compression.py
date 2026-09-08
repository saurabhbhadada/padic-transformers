#!/usr/bin/env python3
"""
Run evaluation with KV-cache compression.

Usage:
    python scripts/run_compression.py \\
        --model pythia-1b \\
        --compression simple_2x \\
        --precision 8 \\
        --context-lengths 2048,4096,8192 \\
        --output results/compression_2x.json
"""

import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/padic-transformers')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from src.kernels import (
    KVCacheCompressor,
    CacheCompressionConfig,
)


def load_model_and_tokenizer(model_name: str, cache_dir: Path):
    """Load pretrained model and tokenizer."""
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_name}",
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model, tokenizer


def load_eval_dataset(dataset_name: str, split: str = "test"):
    """Load evaluation dataset."""
    print(f"\nLoading dataset: {dataset_name}")

    if dataset_name == "wikitext":
        # Use Hugging Face dataset path format
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        text_key = "text"
    elif dataset_name == "ptb":
        dataset = load_dataset("ptb_text_only", split=split)
        text_key = "sentence"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"✓ Dataset loaded: {len(dataset)} examples")
    return dataset, text_key


def setup_compression_config(strategy: str, precision: int = 8):
    """Create compression configuration."""
    if strategy == "none":
        config = CacheCompressionConfig(strategy="none")
    elif strategy == "simple_2x":
        config = CacheCompressionConfig(
            strategy="simple_2x",
            uniform_precision=precision,
        )
    elif strategy == "adaptive":
        config = CacheCompressionConfig(
            strategy="adaptive",
            recent_threshold=128,
            medium_threshold=1024,
            recent_precision=16,
            medium_precision=8,
            old_precision=4,
        )
    else:
        raise ValueError(f"Unknown compression strategy: {strategy}")

    return config


def compute_perplexity_with_compression(
    model,
    tokenizer,
    dataset,
    text_key: str,
    compression_config: CacheCompressionConfig,
    context_length: int = 2048,
    stride: int = 512,
    max_samples: int = None,
):
    """
    Compute perplexity with KV-cache compression.

    NOTE: This is a simplified version that simulates compression overhead.
    For full integration, would need to modify attention mechanism directly.
    """
    model.eval()
    device = next(model.parameters()).device

    # Create compressor
    compressor = KVCacheCompressor(compression_config)

    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Collect all text
    texts = []
    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        text = example[text_key]
        if text and text.strip():
            texts.append(text)

    # Tokenize
    print(f"Tokenizing {len(texts)} examples...")
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    print(f"Total tokens: {input_ids.shape[1]:,}")
    print(f"Context length: {context_length}, Stride: {stride}")
    print(f"Compression: {compression_config.strategy}")

    # Compute perplexity with sliding window
    nlls = []
    num_tokens = 0
    compression_times = []
    decompression_times = []
    cache_sizes_original = []
    cache_sizes_compressed = []

    max_length = min(input_ids.shape[1], context_length * 10)  # Limit for memory

    with torch.no_grad():
        for begin_loc in tqdm(range(0, max_length, stride), desc="Computing perplexity"):
            end_loc = min(begin_loc + context_length, input_ids.shape[1])
            trg_len = end_loc - begin_loc

            input_batch = input_ids[:, begin_loc:end_loc]
            target_ids = input_batch.clone()

            # Forward pass
            outputs = model(input_batch, labels=target_ids, use_cache=True)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            num_tokens += trg_len

            # Simulate KV-cache compression (after generation)
            # In real implementation, this happens inside attention
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                # Get first layer's cache as example
                layer_cache = outputs.past_key_values[0]
                keys, values = layer_cache[0], layer_cache[1]

                # Measure compression
                t0 = time.time()
                keys_comp, values_comp = compressor.compress(keys, values, position=begin_loc)
                compression_time = time.time() - t0

                # Measure decompression
                t0 = time.time()
                if compression_config.strategy == "adaptive":
                    keys_decomp, values_decomp = compressor.decompress_adaptive(
                        keys_comp, values_comp, current_position=begin_loc
                    )
                else:
                    keys_decomp, values_decomp = compressor.decompress(
                        keys_comp, values_comp
                    )
                decompression_time = time.time() - t0

                # Track memory
                original_size = keys.element_size() * keys.numel()
                compressed_size = keys_comp.element_size() * keys_comp.numel()

                compression_times.append(compression_time)
                decompression_times.append(decompression_time)
                cache_sizes_original.append(original_size)
                cache_sizes_compressed.append(compressed_size)

            if end_loc == input_ids.shape[1]:
                break

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens)

    # Memory stats
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0

    # Compression stats
    if cache_sizes_original:
        avg_original_mb = np.mean(cache_sizes_original) / (1024 ** 2)
        avg_compressed_mb = np.mean(cache_sizes_compressed) / (1024 ** 2)
        compression_ratio = avg_original_mb / avg_compressed_mb if avg_compressed_mb > 0 else 1.0
        avg_compression_time_ms = np.mean(compression_times) * 1000
        avg_decompression_time_ms = np.mean(decompression_times) * 1000
    else:
        avg_original_mb = 0
        avg_compressed_mb = 0
        compression_ratio = 1.0
        avg_compression_time_ms = 0
        avg_decompression_time_ms = 0

    results = {
        "perplexity": ppl.item(),
        "num_tokens": num_tokens,
        "peak_memory_mb": peak_memory_mb,
        "context_length": context_length,
        "compression_strategy": compression_config.strategy,
        "cache_original_mb": avg_original_mb,
        "cache_compressed_mb": avg_compressed_mb,
        "compression_ratio": compression_ratio,
        "avg_compression_time_ms": avg_compression_time_ms,
        "avg_decompression_time_ms": avg_decompression_time_ms,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run compression evaluation")
    parser.add_argument("--model", type=str, default="pythia-1b", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "ptb"])
    parser.add_argument("--compression", type=str, default="simple_2x",
                       choices=["none", "simple_2x", "adaptive"],
                       help="Compression strategy")
    parser.add_argument("--precision", type=int, default=8, help="Precision for simple_2x (bits)")
    parser.add_argument("--context-lengths", type=str, default="2048", help="Comma-separated context lengths")
    parser.add_argument("--cache-dir", type=str, default="/workspace/padic-transformers/checkpoints/pretrained")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")

    args = parser.parse_args()

    # Parse context lengths
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]

    print("="*60)
    print(f"KV-CACHE COMPRESSION EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Compression: {args.compression}")
    if args.compression == "simple_2x":
        print(f"Precision: {args.precision} bits")
    print(f"Context lengths: {context_lengths}")
    print(f"Output: {args.output}")
    print("="*60)

    # Setup compression config
    compression_config = setup_compression_config(args.compression, args.precision)

    # Load model and dataset
    cache_dir = Path(args.cache_dir)
    model, tokenizer = load_model_and_tokenizer(args.model, cache_dir)
    dataset, text_key = load_eval_dataset(args.dataset)

    # Run evaluation for each context length
    all_results = {
        "model": args.model,
        "dataset": args.dataset,
        "compression": args.compression,
        "precision": args.precision if args.compression == "simple_2x" else "adaptive",
        "results": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Evaluating at context length: {ctx_len}")
        print(f"{'='*60}")

        start_time = time.time()

        results = compute_perplexity_with_compression(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            text_key=text_key,
            compression_config=compression_config,
            context_length=ctx_len,
            stride=args.stride,
            max_samples=args.max_samples,
        )

        elapsed_time = time.time() - start_time
        results["elapsed_time_sec"] = elapsed_time

        all_results["results"][str(ctx_len)] = results

        print(f"\n{'='*60}")
        print(f"Results for context length {ctx_len}:")
        print(f"  Perplexity: {results['perplexity']:.4f}")
        print(f"  Peak memory: {results['peak_memory_mb']:.2f} MB")
        print(f"  Cache compression: {results['compression_ratio']:.2f}x")
        print(f"  Cache size: {results['cache_original_mb']:.2f} MB → {results['cache_compressed_mb']:.2f} MB")
        print(f"  Compression overhead: {results['avg_compression_time_ms']:.3f} ms")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"{'='*60}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ctx_len, results in all_results["results"].items():
        print(f"Context {ctx_len}: PPL={results['perplexity']:.4f}, "
              f"Compression={results['compression_ratio']:.2f}x, "
              f"Memory={results['peak_memory_mb']:.2f}MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
