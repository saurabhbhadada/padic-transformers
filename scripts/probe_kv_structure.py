#!/usr/bin/env python3
"""
Probe for p-adic/ultrametric structure in transformer KV cache.

This is the MAIN RESEARCH EXPERIMENT - determines if p-adic compression is viable.

Research Question:
    Do transformer KV representations exhibit meaningful ultrametric structure
    where p-adic distance correlates with transformer behavior better than
    Euclidean distance?

Usage:
    python scripts/probe_kv_structure.py \\
        --model pythia-1b \\
        --dataset wikitext \\
        --num-samples 1000 \\
        --output results/structure_probe.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/workspace/padic-transformers')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.kernels import _2adic_valuation


def extract_kv_states(model, tokenizer, texts, max_length=512):
    """
    Extract KV cache states from real model inference.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of input texts
        max_length: Maximum sequence length

    Returns:
        dict: {
            'keys': List of key tensors per layer,
            'values': List of value tensors per layer,
            'attention_weights': Attention weights for analysis,
            'tokens': Token IDs
        }
    """
    model.eval()
    device = next(model.parameters()).device

    all_keys = []
    all_values = []
    all_attention = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting KV states"):
            # Skip empty texts
            if not text or not text.strip():
                continue

            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

            # Skip if tokenization failed
            if inputs['input_ids'].shape[1] == 0:
                continue

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass with output_attentions=True
            outputs = model(**inputs, output_attentions=True, use_cache=True)

            # Extract KV cache from past_key_values
            past_kv = outputs.past_key_values
            if past_kv is None or len(past_kv) == 0:
                continue

            # Store keys, values, and attentions
            layer_keys = [kv[0].cpu() for kv in past_kv]
            layer_values = [kv[1].cpu() for kv in past_kv]
            layer_attentions = [attn.cpu() for attn in outputs.attentions]

            all_keys.append(layer_keys)
            all_values.append(layer_values)
            all_attention.append(layer_attentions)

    return {
        'keys': all_keys,
        'values': all_values,
        'attention_weights': all_attention,
    }


def compute_euclidean_distance(k1, k2):
    """Compute L2 distance between two tensors."""
    return torch.norm(k1 - k2, p=2).item()


def compute_cosine_similarity(k1, k2):
    """Compute cosine similarity between two tensors."""
    k1_flat = k1.flatten()
    k2_flat = k2.flatten()
    return torch.nn.functional.cosine_similarity(k1_flat, k2_flat, dim=0).item()


def compute_padic_distance(k1, k2, precision=16):
    """
    Compute 2-adic distance between two tensors.

    Steps:
    1. Quantize to integers: k_int = round(k * 2^precision)
    2. Compute difference: diff = k1_int - k2_int
    3. Compute 2-adic valuation: v_2(diff) = max{n : 2^n | diff}
    4. P-adic distance: d_2 = 2^(-v_2(diff))

    Higher v_2 = more shared binary structure = closer in p-adic metric
    """
    # Quantize to integers
    scale = 2 ** precision
    k1_int = torch.round(k1 * scale).to(torch.int64)
    k2_int = torch.round(k2 * scale).to(torch.int64)

    # Compute difference
    diff = k1_int - k2_int

    # Compute 2-adic valuation (how many times 2 divides diff)
    # For tensors, take minimum valuation across all elements
    diff_flat = diff.flatten()
    valuations = []

    for val in diff_flat:
        if val == 0:
            valuations.append(precision)  # Infinite valuation, cap at precision
        else:
            valuations.append(_2adic_valuation(val.item()))

    # Use minimum valuation (weakest link determines distance)
    min_valuation = min(valuations) if valuations else 0

    # P-adic distance: 2^(-v_2)
    padic_dist = 2 ** (-min_valuation)

    return padic_dist, min_valuation


def analyze_distance_correlations_cpu(keys, attentions, num_pairs=500):
    """CPU version: Sample pairs and compute distances one by one."""
    euclidean_dists = []
    cosine_sims = []
    padic_valuations = []
    attention_sims = []

    print(f"  Using CPU implementation (sampling {num_pairs} pairs)...")

    for _ in tqdm(range(num_pairs), desc="Sampling pairs"):
        # Sample random sequence and two positions
        seq_idx = np.random.randint(len(keys))
        key_tensor = keys[seq_idx][0]  # [heads, seq, dim] - take first batch

        seq_len = key_tensor.shape[1]
        if seq_len < 2:
            continue

        # Sample two positions
        i, j = np.random.choice(seq_len, size=2, replace=False)

        # Extract key vectors (average across heads)
        ki = key_tensor[:, i, :].mean(dim=0)  # [dim]
        kj = key_tensor[:, j, :].mean(dim=0)  # [dim]

        # Compute distances
        euclidean_dists.append(compute_euclidean_distance(ki, kj))
        cosine_sims.append(compute_cosine_similarity(ki, kj))
        _, padic_val = compute_padic_distance(ki, kj, precision=16)
        padic_valuations.append(padic_val)

        # Compute attention similarity
        attn = attentions[seq_idx][0]  # [heads, seq, seq]
        attn_i = attn[:, i, :].mean(dim=0)
        attn_j = attn[:, j, :].mean(dim=0)
        attn_sim = torch.nn.functional.cosine_similarity(attn_i, attn_j, dim=0).item()
        attention_sims.append(attn_sim)

    return euclidean_dists, cosine_sims, padic_valuations, attention_sims


def analyze_distance_correlations_gpu(keys, attentions, num_pairs=500):
    """GPU version: Pre-compute all distances, then sample."""
    euclidean_dists = []
    cosine_sims = []
    padic_valuations = []
    attention_sims = []

    print(f"  Using GPU implementation (batch distance computation)...")

    # Get GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process each sequence
    sequences_to_process = min(len(keys), 20)  # Sample from first 20 sequences
    pairs_per_seq = num_pairs // sequences_to_process

    for seq_idx in tqdm(range(sequences_to_process), desc="Processing sequences"):
        key_tensor = keys[seq_idx][0].to(device)  # [heads, seq, dim] - move to GPU
        attn_tensor = attentions[seq_idx][0].to(device)  # [heads, seq, seq] - move to GPU

        seq_len = key_tensor.shape[1]
        if seq_len < 2:
            continue

        # Average across heads, now on GPU
        K = key_tensor.mean(dim=0)  # [seq, dim] on GPU
        A = attn_tensor.mean(dim=0)  # [seq, seq] on GPU

        # Compute ALL pairwise distances on GPU at once (fast!)
        dists_euc = torch.cdist(K.unsqueeze(0), K.unsqueeze(0)).squeeze(0)  # [seq, seq]

        # Cosine similarity matrix
        K_norm = torch.nn.functional.normalize(K, dim=1)
        cosine_matrix = torch.mm(K_norm, K_norm.t())  # [seq, seq]

        # Attention similarity matrix
        A_norm = torch.nn.functional.normalize(A, dim=1)
        attn_sim_matrix = torch.mm(A_norm, A_norm.t())  # [seq, seq]

        # Sample pairs from this sequence
        actual_pairs = min(pairs_per_seq, seq_len * (seq_len - 1) // 2)

        for _ in range(actual_pairs):
            i, j = np.random.choice(seq_len, size=2, replace=False)

            # Lookup from pre-computed matrices (instant!)
            euclidean_dists.append(dists_euc[i, j].item())
            cosine_sims.append(cosine_matrix[i, j].item())
            attention_sims.append(attn_sim_matrix[i, j].item())

            # P-adic valuation (CPU only, but just for sampled pairs)
            ki = K[i].cpu()
            kj = K[j].cpu()
            _, padic_val = compute_padic_distance(ki, kj, precision=16)
            padic_valuations.append(padic_val)

    return euclidean_dists, cosine_sims, padic_valuations, attention_sims


def analyze_distance_correlations(kv_states, layer_idx=6, num_pairs=500, use_gpu=True):
    """
    Analyze correlation between distance metrics and attention behavior.

    Args:
        kv_states: Extracted KV cache states
        layer_idx: Layer to analyze
        num_pairs: Number of pairs to sample
        use_gpu: Try GPU-accelerated version (fallback to CPU on error)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Layer {layer_idx}")
    print(f"{'='*80}")

    # Extract keys and attention for this layer
    keys = [sample[layer_idx] for sample in kv_states['keys']]
    attentions = [sample[layer_idx] for sample in kv_states['attention_weights']]

    # Try GPU version first, fallback to CPU
    if use_gpu and torch.cuda.is_available():
        try:
            euclidean_dists, cosine_sims, padic_valuations, attention_sims = \
                analyze_distance_correlations_gpu(keys, attentions, num_pairs)
        except Exception:
            euclidean_dists, cosine_sims, padic_valuations, attention_sims = \
                analyze_distance_correlations_cpu(keys, attentions, num_pairs)
    else:
        euclidean_dists, cosine_sims, padic_valuations, attention_sims = \
            analyze_distance_correlations_cpu(keys, attentions, num_pairs)

    # Convert to numpy
    euclidean_dists = np.array(euclidean_dists)
    cosine_sims = np.array(cosine_sims)
    padic_valuations = np.array(padic_valuations)
    attention_sims = np.array(attention_sims)

    # Compute correlations
    print(f"\n{'='*80}")
    print("CORRELATION WITH ATTENTION SIMILARITY")
    print('='*80)

    # Euclidean distance vs attention (expect negative correlation)
    euc_corr, euc_p = pearsonr(-euclidean_dists, attention_sims)
    print(f"\nEuclidean distance vs Attention:")
    print(f"  Pearson r = {euc_corr:.4f}, p-value = {euc_p:.6f}")

    # Cosine similarity vs attention (expect positive correlation)
    cos_corr, cos_p = pearsonr(cosine_sims, attention_sims)
    print(f"\nCosine similarity vs Attention:")
    print(f"  Pearson r = {cos_corr:.4f}, p-value = {cos_p:.6f}")

    # P-adic valuation vs attention (expect positive correlation if structure exists)
    padic_corr, padic_p = pearsonr(padic_valuations, attention_sims)
    print(f"\nP-adic valuation v_2(K_i - K_j) vs Attention:")
    print(f"  Pearson r = {padic_corr:.4f}, p-value = {padic_p:.6f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)

    if padic_corr > max(euc_corr, cos_corr) * 1.1:
        print("✅ P-ADIC STRUCTURE DETECTED!")
        print(f"   P-adic correlation ({padic_corr:.4f}) is stronger than Euclidean ({euc_corr:.4f})")
        print("   This suggests KV cache has exploitable ultrametric structure")
    else:
        print("❌ NO CLEAR P-ADIC ADVANTAGE")
        print(f"   P-adic correlation ({padic_corr:.4f}) not significantly better than baselines")
        print(f"   Euclidean: {euc_corr:.4f}, Cosine: {cos_corr:.4f}")
        print("   Consider abandoning p-adic KV compression")

    return {
        'euclidean_corr': euc_corr,
        'euclidean_p': euc_p,
        'cosine_corr': cos_corr,
        'cosine_p': cos_p,
        'padic_corr': padic_corr,
        'padic_p': padic_p,
        'num_pairs': len(euclidean_dists),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe for p-adic structure in KV cache")
    parser.add_argument("--model", type=str, default="pythia-1b", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset shorthand (wikitext/code/math)")
    parser.add_argument("--dataset-path", type=str, help="Full dataset path (optional, overrides --dataset)")
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split")
    parser.add_argument("--text-field", type=str, default="text", help="Text field name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of text samples")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--layer", type=int, default=6, help="Layer to analyze")
    parser.add_argument("--num-pairs", type=int, default=500, help="Number of pairs to sample for correlation")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Use GPU-accelerated distance computation")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU-only computation")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--cache-dir", type=str, default="/workspace/padic-transformers/checkpoints/pretrained")

    args = parser.parse_args()

    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
        dataset_name = args.dataset_path.split('/')[-1]
    else:
        # Shortcuts for common datasets
        dataset_shortcuts = {
            'wikitext': ('Salesforce/wikitext', 'wikitext-103-raw-v1', 'text'),
            'code': ('bigcode/the-stack-dedup', 'data/python', 'content'),
            'math': ('hendrycks/competition_math', 'train', 'problem'),
        }
        if args.dataset in dataset_shortcuts:
            dataset_path, config, field = dataset_shortcuts[args.dataset]
            args.text_field = field
            dataset_name = args.dataset
        else:
            dataset_path = args.dataset
            dataset_name = args.dataset

    print("="*80)
    print("P-ADIC STRUCTURE PROBE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Analyzing layer: {args.layer}")
    print(f"Estimated tokens: {args.num_samples * args.max_length:,}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/{args.model}",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Required for output_attentions to work
    )
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{args.model}", cache_dir=args.cache_dir)
    print("✓ Model loaded")

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}...")
    try:
        if args.dataset == 'wikitext':
            dataset = load_dataset(dataset_path, 'wikitext-103-raw-v1', split=args.dataset_split)
        elif args.dataset == 'code':
            dataset = load_dataset(dataset_path, data_dir='data/python', split=args.dataset_split, streaming=True)
            # Take first N samples from streaming dataset
            dataset = list(dataset.take(args.num_samples))
        elif args.dataset == 'math':
            dataset = load_dataset(dataset_path, split=args.dataset_split)
        else:
            dataset = load_dataset(dataset_path, split=args.dataset_split)

        texts = [ex[args.text_field] for ex in dataset if ex[args.text_field].strip()][:args.num_samples]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying simple load...")
        dataset = load_dataset(dataset_path, split=args.dataset_split)
        texts = [ex[args.text_field] for ex in dataset if ex[args.text_field].strip()][:args.num_samples]

    # Extract KV states
    print(f"\nExtracting KV states from {len(texts)} samples...")
    kv_states = extract_kv_states(model, tokenizer, texts, max_length=args.max_length)

    # Determine GPU usage
    use_gpu = args.use_gpu and not args.use_cpu

    # Analyze correlations
    results = analyze_distance_correlations(
        kv_states,
        layer_idx=args.layer,
        num_pairs=args.num_pairs,
        use_gpu=use_gpu
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        'model': args.model,
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'layer_analyzed': args.layer,
        'correlations': results,
    }

    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
