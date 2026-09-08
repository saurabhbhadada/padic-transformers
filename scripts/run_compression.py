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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run evaluation with KV-cache compression in attention layers.

This applies compression directly in attention computation,
giving real perplexity impact from quantization.
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
from tqdm import tqdm

from src.kernels import CacheCompressionConfig
from src.models import apply_compression_to_model


def load_model_and_tokenizer(model_name: str, cache_dir: Path, compression_config: CacheCompressionConfig):
    """Load pretrained model and tokenizer, apply compression to attention."""
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

    # Apply compression to attention layers
    if compression_config.strategy != "none":
        model = apply_compression_to_model(model, compression_config)

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
    Compute perplexity with K/V compression in attention.

    Compression is applied in the attention layers, so we just run
    standard perplexity evaluation. The quantization error from
    compressed K/V will affect the perplexity.
    """
    model.eval()
    device = next(model.parameters()).device

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
    # Compression happens inside attention layers automatically
    nlls = []
    num_tokens = 0

    max_length = min(input_ids.shape[1], context_length * 10)  # Limit for memory

    with torch.no_grad():
        for begin_loc in tqdm(range(0, max_length, stride), desc="Computing perplexity"):
            end_loc = min(begin_loc + context_length, input_ids.shape[1])
            trg_len = end_loc - begin_loc

            input_batch = input_ids[:, begin_loc:end_loc]
            target_ids = input_batch.clone()

            # Forward pass - K/V compression happens in attention layers
            outputs = model(input_batch, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            num_tokens += trg_len

            if end_loc == input_ids.shape[1]:
                break

    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens)

    # Memory stats
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0

    # Compression ratio based on strategy
    if compression_config.strategy == "simple_2x":
        compression_ratio = 2.0
        precision_bits = compression_config.uniform_precision
    elif compression_config.strategy == "adaptive":
        compression_ratio = 3.5  # Approximate: mix of 16/8/4 bits
        precision_bits = "adaptive"
    else:
        compression_ratio = 1.0
        precision_bits = 16

    results = {
        "perplexity": ppl.item(),
        "num_tokens": num_tokens,
        "peak_memory_mb": peak_memory_mb,
        "context_length": context_length,
        "compression_strategy": compression_config.strategy,
        "precision_bits": precision_bits,
        "compression_ratio": compression_ratio,
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
    model, tokenizer = load_model_and_tokenizer(args.model, cache_dir, compression_config)
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
