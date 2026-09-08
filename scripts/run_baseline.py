#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline evaluation (FP16, no compression) on pretrained model.

Usage:
    python scripts/run_baseline.py \\
        --model pythia-1b \\
        --dataset wikitext \\
        --context-lengths 2048,4096,8192 \\
        --output results/baseline.json
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


def compute_perplexity(
    model,
    tokenizer,
    dataset,
    text_key: str,
    context_length: int = 2048,
    stride: int = 512,
    max_samples: int = None,
):
    """
    Compute perplexity on dataset.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        dataset: Dataset with text
        text_key: Key for text field in dataset
        context_length: Context window size
        stride: Stride for sliding window
        max_samples: Max number of samples to evaluate (None = all)

    Returns:
        Dictionary with perplexity and memory stats
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

    # Compute perplexity with sliding window
    nlls = []
    num_tokens = 0

    max_length = min(input_ids.shape[1], context_length * 10)  # Limit for memory

    with torch.no_grad():
        for begin_loc in tqdm(range(0, max_length, stride), desc="Computing perplexity"):
            end_loc = min(begin_loc + context_length, input_ids.shape[1])
            trg_len = end_loc - begin_loc

            input_batch = input_ids[:, begin_loc:end_loc]
            target_ids = input_batch.clone()

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

    results = {
        "perplexity": ppl.item(),
        "num_tokens": num_tokens,
        "peak_memory_mb": peak_memory_mb,
        "context_length": context_length,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--model", type=str, default="pythia-1b", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "ptb"])
    parser.add_argument("--context-lengths", type=str, default="2048", help="Comma-separated context lengths")
    parser.add_argument("--cache-dir", type=str, default="/workspace/padic-transformers/checkpoints/pretrained")
    parser.add_argument("--output", type=str, default="results/baseline.json", help="Output JSON file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")

    args = parser.parse_args()

    # Parse context lengths
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]

    print("="*60)
    print("BASELINE EVALUATION (FP16, No Compression)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Context lengths: {context_lengths}")
    print(f"Output: {args.output}")
    print("="*60)

    # Load model and dataset
    cache_dir = Path(args.cache_dir)
    model, tokenizer = load_model_and_tokenizer(args.model, cache_dir)
    dataset, text_key = load_eval_dataset(args.dataset)

    # Run evaluation for each context length
    all_results = {
        "model": args.model,
        "dataset": args.dataset,
        "compression": "none",
        "precision": "fp16",
        "results": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Evaluating at context length: {ctx_len}")
        print(f"{'='*60}")

        start_time = time.time()

        results = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            text_key=text_key,
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
        print(f"Context {ctx_len}: PPL={results['perplexity']:.4f}, Memory={results['peak_memory_mb']:.2f}MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
