#!/usr/bin/env python3
"""
Probe for p-adic/ultrametric structure in transformer KEYS (K only).

SCOPE: This script probes KEY structure. Values (V) are extracted but NOT analyzed.
A separate V probe is needed to test if values exhibit p-adic structure.

This is the MAIN RESEARCH EXPERIMENT - determines if p-adic compression is viable.

Research Question:
    Do transformer KEY representations exhibit meaningful ultrametric structure
    where p-adic distance between K[i] and K[j] predicts their behavioral
    similarity better than Euclidean distance?

Behavioral similarity:
    How similarly do future queries attend to K[i] vs K[j]?
    Measured as correlation between attention columns A[:,i] and A[:,j]

Usage:
    python scripts/probe_kv_structure.py \\
        --model pythia-1b \\
        --dataset wikitext \\
        --num-samples 2000 \\
        --layer 6 \\
        --output results/K_probe_layer6.json

TODO: Create probe_V_structure.py to analyze value similarity vs output similarity
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
from scipy.stats import pearsonr, spearmanr, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.kernels import _padic_valuation


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


def compute_padic_statistics(k1, k2, precision=16, prime=2, scale_factor=None):
    """
    Compute p-adic statistics between two tensors.

    CRITICAL: Traditional min(v_p) is degenerate in high dimensions!
    For 256-D vectors, P(min > 0) ≈ (1/2)^256 ≈ 0, making the metric flat.

    We compute robust S_k statistics instead:
    S_k = fraction of dimensions where v_p(K[i,d] - K[j,d]) >= k

    Interpretation:
    - S_1: % of dims sharing ≥1 low-order binary digit
    - S_2: % of dims sharing ≥2 low-order binary digits (most robust)
    - High S_k => many dimensions have shared p-adic structure

    Args:
        k1, k2: Tensors to compare (shape: [dim])
        precision: Bit precision for quantization
        prime: Prime for p-adic metric (2, 3, 5, etc.)
        scale_factor: Optional fixed scale (for control experiments)

    Returns:
        dict with:
        - min_valuation: min v_p across dimensions (degenerate but traditional)
        - mean_valuation: mean v_p (robust alternative)
        - S_k: fraction of dims with v_p >= k for k=1,2,3,4
        - valuations: full per-dimension valuations
    """
    # Convert to float32 to avoid FP16 overflow
    k1 = k1.float()
    k2 = k2.float()

    # Symmetric quantization: map to [-2^(p-1), 2^(p-1)]
    max_abs = max(k1.abs().max().item(), k2.abs().max().item())

    if max_abs < 1e-8:
        # Both tensors are nearly zero
        dim = k1.numel()
        return {
            'min_valuation': precision,
            'mean_valuation': precision,
            'S1': 1.0,
            'S2': 1.0,
            'S3': 1.0,
            'S4': 1.0,
            'valuations': torch.full((dim,), precision),
        }

    # Use provided scale or compute from precision
    if scale_factor is None:
        int_range = 2 ** (precision - 1)
        scale = int_range / max_abs
    else:
        scale = scale_factor

    # Quantize to integers
    k1_int = torch.round(k1 * scale).to(torch.int64)
    k2_int = torch.round(k2 * scale).to(torch.int64)

    # Compute difference
    diff = k1_int - k2_int

    # Compute p-adic valuation per dimension
    diff_flat = diff.flatten()
    valuations = _padic_valuation(diff_flat, prime=prime, precision=precision)

    # Traditional min-based metric (degenerate in high dims)
    min_val = valuations.min().item()

    # Robust statistics
    mean_val = valuations.float().mean().item()

    # S_k: fraction of dimensions with v_p >= k
    dim = len(valuations)
    S1 = (valuations >= 1).float().sum().item() / dim
    S2 = (valuations >= 2).float().sum().item() / dim
    S3 = (valuations >= 3).float().sum().item() / dim
    S4 = (valuations >= 4).float().sum().item() / dim

    return {
        'min_valuation': min_val,
        'mean_valuation': mean_val,
        'S1': S1,
        'S2': S2,
        'S3': S3,
        'S4': S4,
        'valuations': valuations,
    }


def generate_random_baseline(keys):
    """
    Generate random tensors with same marginal distribution as real keys.

    This is a critical control: if p-adic structure exists in transformers,
    it should be STRONGER than random tensors with the same statistics.
    """
    random_keys = []
    for key_tensor in keys:
        # key_tensor shape: [batch=1, heads, seq, dim]
        shape = key_tensor.shape

        # Match mean and std of real keys
        mean = key_tensor.mean().item()
        std = key_tensor.std().item()

        # Generate random tensor with same distribution
        random_tensor = torch.randn(shape) * std + mean
        random_keys.append(random_tensor)

    return random_keys


def analyze_distance_correlations_cpu(keys, attentions, num_pairs=500, prime=2, precision=16, scale_factor=None):
    """CPU version: Analyze each attention head separately."""
    euclidean_dists = []
    cosine_sims = []
    padic_min = []
    padic_mean = []
    padic_S1 = []
    padic_S2 = []
    padic_S3 = []
    padic_S4 = []
    attention_sims = []
    head_ids = []

    print(f"  Using CPU implementation (p={prime}, precision={precision})")
    print(f"  Computing robust S_k statistics (fraction of dims with v_p >= k)")

    # Get number of heads
    num_heads = keys[0][0].shape[0]

    for _ in tqdm(range(num_pairs), desc="Sampling pairs"):
        # Sample random sequence and two positions
        seq_idx = np.random.randint(len(keys))
        key_tensor = keys[seq_idx][0]  # [heads, seq, dim] - take first batch

        seq_len = key_tensor.shape[1]
        if seq_len < 2:
            continue

        # Sample two positions
        i, j = np.random.choice(seq_len, size=2, replace=False)

        # Only consider future positions (causal masking)
        future_start = max(i, j) + 1
        if future_start >= seq_len:
            continue  # No future positions

        attn = attentions[seq_idx][0]  # [heads, seq, seq]

        # Analyze EACH HEAD separately
        for head_idx in range(num_heads):
            # Extract key vectors for this head
            ki = key_tensor[head_idx, i, :]  # [dim]
            kj = key_tensor[head_idx, j, :]  # [dim]

            # Compute distances
            euclidean_dists.append(compute_euclidean_distance(ki, kj))
            cosine_sims.append(compute_cosine_similarity(ki, kj))

            # Compute p-adic statistics (robust + traditional)
            padic_stats = compute_padic_statistics(ki, kj, precision=precision, prime=prime, scale_factor=scale_factor)
            padic_min.append(padic_stats['min_valuation'])
            padic_mean.append(padic_stats['mean_valuation'])
            padic_S1.append(padic_stats['S1'])
            padic_S2.append(padic_stats['S2'])
            padic_S3.append(padic_stats['S3'])
            padic_S4.append(padic_stats['S4'])

            # Compute attention similarity for this head (COLUMNS)
            # A[head, future_start:, i] = how future queries attend to key i in this head
            attn_col_i = attn[head_idx, future_start:, i]
            attn_col_j = attn[head_idx, future_start:, j]

            if len(attn_col_i) > 0 and len(attn_col_j) > 0:
                attn_sim = torch.nn.functional.cosine_similarity(
                    attn_col_i, attn_col_j, dim=0
                ).item()
                attention_sims.append(attn_sim)
                head_ids.append(head_idx)
            else:
                # Remove the measurements we just added
                euclidean_dists.pop()
                cosine_sims.pop()
                padic_min.pop()
                padic_mean.pop()
                padic_S1.pop()
                padic_S2.pop()
                padic_S3.pop()
                padic_S4.pop()

    return {
        'euclidean': euclidean_dists,
        'cosine': cosine_sims,
        'padic_min': padic_min,
        'padic_mean': padic_mean,
        'padic_S1': padic_S1,
        'padic_S2': padic_S2,
        'padic_S3': padic_S3,
        'padic_S4': padic_S4,
        'attention': attention_sims,
        'heads': head_ids,
    }


def analyze_distance_correlations_gpu(keys, attentions, num_pairs=500, prime=2, precision=16, scale_factor=None):
    """GPU version: Analyze each attention head separately."""
    euclidean_dists = []
    cosine_sims = []
    padic_min = []
    padic_mean = []
    padic_S1 = []
    padic_S2 = []
    padic_S3 = []
    padic_S4 = []
    attention_sims = []
    head_ids = []

    print(f"  Using GPU implementation (p={prime}, precision={precision})")
    print(f"  Computing robust S_k statistics (fraction of dims with v_p >= k)")

    # Get GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get number of heads
    num_heads = keys[0][0].shape[0]

    # Process each sequence
    sequences_to_process = min(len(keys), 20)  # Sample from first 20 sequences
    pairs_per_seq = num_pairs // (sequences_to_process * num_heads)

    for seq_idx in tqdm(range(sequences_to_process), desc="Processing sequences"):
        key_tensor = keys[seq_idx][0].to(device)  # [heads, seq, dim]
        attn_tensor = attentions[seq_idx][0].to(device)  # [heads, seq, seq]

        seq_len = key_tensor.shape[1]
        if seq_len < 2:
            continue

        # Process EACH HEAD separately
        for head_idx in range(num_heads):
            K = key_tensor[head_idx]  # [seq, dim] for this head
            A = attn_tensor[head_idx]  # [seq, seq] for this head

            # Compute ALL pairwise distances on GPU (cdist requires float32)
            K_float32 = K.to(torch.float32)
            dists_euc = torch.cdist(K_float32.unsqueeze(0), K_float32.unsqueeze(0)).squeeze(0)

            # Cosine similarity matrix (for keys)
            K_norm = torch.nn.functional.normalize(K, dim=1)
            cosine_matrix = torch.mm(K_norm, K_norm.t())

            # Sample pairs from this sequence/head
            actual_pairs = min(pairs_per_seq, seq_len * (seq_len - 1) // 2)

            for _ in range(actual_pairs):
                i, j = np.random.choice(seq_len, size=2, replace=False)

                # Only use pairs where we have future context
                future_start = max(i, j) + 1
                if future_start >= seq_len:
                    continue

                # Lookup key similarity
                euclidean_dists.append(dists_euc[i, j].item())
                cosine_sims.append(cosine_matrix[i, j].item())

                # Compute attention column similarity (how future queries attend to these keys)
                attn_col_i = A[future_start:, i]
                attn_col_j = A[future_start:, j]

                if len(attn_col_i) > 0:
                    attn_col_i_norm = torch.nn.functional.normalize(attn_col_i.unsqueeze(0), dim=1)
                    attn_col_j_norm = torch.nn.functional.normalize(attn_col_j.unsqueeze(0), dim=1)
                    attn_sim = torch.mm(attn_col_i_norm, attn_col_j_norm.t()).item()
                    attention_sims.append(attn_sim)
                else:
                    continue

                # P-adic statistics (CPU only, but just for sampled pairs)
                ki = K[i].cpu()
                kj = K[j].cpu()
                padic_stats = compute_padic_statistics(ki, kj, precision=precision, prime=prime, scale_factor=scale_factor)
                padic_min.append(padic_stats['min_valuation'])
                padic_mean.append(padic_stats['mean_valuation'])
                padic_S1.append(padic_stats['S1'])
                padic_S2.append(padic_stats['S2'])
                padic_S3.append(padic_stats['S3'])
                padic_S4.append(padic_stats['S4'])
                head_ids.append(head_idx)

    return {
        'euclidean': euclidean_dists,
        'cosine': cosine_sims,
        'padic_min': padic_min,
        'padic_mean': padic_mean,
        'padic_S1': padic_S1,
        'padic_S2': padic_S2,
        'padic_S3': padic_S3,
        'padic_S4': padic_S4,
        'attention': attention_sims,
        'heads': head_ids,
    }


def bootstrap_correlation_ci(x, y, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence interval for correlation.

    CRITICAL LIMITATION: Measurements aren't independent - many pairs share
    sequences/tokens. This bootstrap uses naive resampling, which may
    underestimate variance.

    IDEAL: Resample at sequence level, recompute pairs, then correlations.
    But that requires refactoring the analysis pipeline.

    For publication: Consider this a lower bound on CI width. Real uncertainty
    is likely larger due to dependencies.
    """
    correlations = []
    n = len(x)

    for _ in range(n_bootstrap):
        # Resample with replacement (naive - ignores dependencies)
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]

        # Compute correlation on bootstrap sample
        r_boot, _ = pearsonr(x_boot, y_boot)
        correlations.append(r_boot)

    # Compute confidence interval
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower = np.percentile(correlations, lower_percentile)
    ci_upper = np.percentile(correlations, upper_percentile)

    return ci_lower, ci_upper


def run_single_analysis(keys, attentions, num_pairs, prime, precision, scale_factor, use_gpu):
    """Run analysis with specific parameters, returns per-head results."""
    if use_gpu and torch.cuda.is_available():
        try:
            return analyze_distance_correlations_gpu(keys, attentions, num_pairs, prime, precision, scale_factor)
        except Exception:
            return analyze_distance_correlations_cpu(keys, attentions, num_pairs, prime, precision, scale_factor)
    else:
        return analyze_distance_correlations_cpu(keys, attentions, num_pairs, prime, precision, scale_factor)


def analyze_K_structure(kv_states, layer_idx=6, num_pairs=500, use_gpu=True):
    """
    Probe for p-adic structure in KEYS only (not values).

    SCOPE: This probes K similarity vs K behavior, NOT KV jointly.
    A separate probe for V is needed to test value structure.

    Research question:
    Does p-adic distance between keys K[i] and K[j] predict
    how similarly they are attended to by future queries?

    Specifically:
    - Key distance: ||K[i] - K[j]|| or d_p(K[i], K[j])
    - Key behavior: similarity of A[:, i] and A[:, j]
      (how future queries attend to key i vs key j)

    NOT comparing Q[i] vs Q[j] (that would be attention ROWS, wrong!)

    ANALYSIS PER ATTENTION HEAD:
    - Each attention head operates in its own learned subspace
    - Averaging across heads is mathematically questionable
    - We analyze each head separately and aggregate statistics
    - This can reveal head-specific ultrametric structure

    INCLUDES CRITICAL CONTROLS FOR FP16 CONFOUNDS:
    - Random baseline (same distribution as real keys)
    - Multiple primes (p=2, 3, 5)
    - Multiple precisions (8, 12, 16 bit)

    Args:
        kv_states: Extracted KV cache states
        layer_idx: Layer to analyze
        num_pairs: Number of pairs to sample
        use_gpu: Try GPU-accelerated version (fallback to CPU on error)

    Returns:
        dict with correlation results, CIs, per-head stats
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Layer {layer_idx} (Per-Head, S_k Statistics)")
    print(f"{'='*80}")

    # Extract keys and attention for this layer
    keys = [sample[layer_idx] for sample in kv_states['keys']]
    attentions = [sample[layer_idx] for sample in kv_states['attention_weights']]

    # Generate random baseline
    print("\nGenerating random baseline (same distribution as real keys)...")
    random_keys = generate_random_baseline(keys)

    # Run analysis for REAL keys
    print(f"\n{'='*80}")
    print("REAL TRANSFORMER KEYS")
    print(f"{'='*80}")

    results = {}

    # Test multiple primes
    for prime in [2, 3, 5]:
        print(f"\n--- Prime p={prime} ---")
        stats = run_single_analysis(keys, attentions, num_pairs, prime, 16, None, use_gpu)

        euc_arr = np.array(stats['euclidean'])
        cos_arr = np.array(stats['cosine'])
        att_arr = np.array(stats['attention'])
        heads_arr = np.array(stats['heads'])

        # P-adic statistics (robust + traditional)
        padic_min_arr = np.array(stats['padic_min'])
        padic_mean_arr = np.array(stats['padic_mean'])
        S1_arr = np.array(stats['padic_S1'])
        S2_arr = np.array(stats['padic_S2'])
        S3_arr = np.array(stats['padic_S3'])
        S4_arr = np.array(stats['padic_S4'])

        # Compute both Pearson and Spearman (v_p is discrete and heavily tied)
        # Pearson for Euclidean/Cosine
        euc_corr_p, _ = pearsonr(-euc_arr, att_arr)
        cos_corr_p, _ = pearsonr(cos_arr, att_arr)

        # Spearman for discrete p-adic statistics
        min_corr_s, _ = spearmanr(padic_min_arr, att_arr)
        mean_corr_s, _ = spearmanr(padic_mean_arr, att_arr)
        S1_corr_s, _ = spearmanr(S1_arr, att_arr)
        S2_corr_s, _ = spearmanr(S2_arr, att_arr)
        S3_corr_s, _ = spearmanr(S3_arr, att_arr)
        S4_corr_s, _ = spearmanr(S4_arr, att_arr)

        # Bootstrap CI for key comparison: S2 vs Cosine
        print(f"Overall (Pearson for continuous, Spearman for discrete):")
        print(f"  Euclidean: r_p={euc_corr_p:.4f}")
        print(f"  Cosine:    r_p={cos_corr_p:.4f}")
        print(f"  P-adic min(v_p):  r_s={min_corr_s:.4f} (degenerate in 256-D)")
        print(f"  P-adic mean(v_p): r_s={mean_corr_s:.4f}")
        print(f"  S_1 (≥1 digits):  r_s={S1_corr_s:.4f}")
        print(f"  S_2 (≥2 digits):  r_s={S2_corr_s:.4f}")
        print(f"  S_3 (≥3 digits):  r_s={S3_corr_s:.4f}")
        print(f"  S_4 (≥4 digits):  r_s={S4_corr_s:.4f}")

        # Bootstrap confidence interval for S2 vs Cosine difference
        S2_ci_lower, S2_ci_upper = bootstrap_correlation_ci(S2_arr, att_arr, n_bootstrap=1000)
        cos_ci_lower, cos_ci_upper = bootstrap_correlation_ci(cos_arr, att_arr, n_bootstrap=1000)

        print(f"  S_2 bootstrap 95% CI: [{S2_ci_lower:.4f}, {S2_ci_upper:.4f}]")
        print(f"  Cosine bootstrap 95% CI: [{cos_ci_lower:.4f}, {cos_ci_upper:.4f}]")

        # Report difference with approximate CI
        diff = S2_corr_s - cos_corr_p
        diff_ci_lower = S2_ci_lower - cos_ci_upper  # Conservative estimate
        diff_ci_upper = S2_ci_upper - cos_ci_lower
        print(f"  S_2 - Cosine: {diff:.4f}, 95% CI ≈ [{diff_ci_lower:.4f}, {diff_ci_upper:.4f}]")

        # Per-head breakdown (using best robust statistic)
        num_heads = int(heads_arr.max()) + 1
        per_head_S2 = []
        for h in range(num_heads):
            mask = heads_arr == h
            if mask.sum() > 10:  # Need at least 10 samples
                h_S2_corr, _ = pearsonr(S2_arr[mask], att_arr[mask])
                per_head_S2.append(h_S2_corr)
            else:
                per_head_S2.append(np.nan)

        # Show head-specific patterns
        valid_heads = [r for r in per_head_S2 if not np.isnan(r)]
        if valid_heads:
            print(f"  Per-head S_2: mean={np.mean(valid_heads):.4f}, "
                  f"std={np.std(valid_heads):.4f}, "
                  f"range=[{np.min(valid_heads):.4f}, {np.max(valid_heads):.4f}]")

        results[f'real_p{prime}'] = {
            'euclidean_corr_p': euc_corr_p,
            'cosine_corr_p': cos_corr_p,
            'padic_min_corr_s': min_corr_s,
            'padic_mean_corr_s': mean_corr_s,
            'padic_S1_corr_s': S1_corr_s,
            'padic_S2_corr_s': S2_corr_s,
            'padic_S3_corr_s': S3_corr_s,
            'padic_S4_corr_s': S4_corr_s,
            'S2_ci': (S2_ci_lower, S2_ci_upper),
            'cosine_ci': (cos_ci_lower, cos_ci_upper),
            'diff_S2_cosine': diff,
            'diff_ci_approx': (diff_ci_lower, diff_ci_upper),
            'per_head_S2': per_head_S2,
        }

    # Test multiple precisions (p=2 only)
    print(f"\n--- Multiple Precisions (p=2) ---")
    for precision in [8, 12, 16]:
        stats = run_single_analysis(keys, attentions, num_pairs, 2, precision, None, use_gpu)
        S2_arr = np.array(stats['padic_S2'])
        att_arr = np.array(stats['attention'])
        S2_corr_s, _ = spearmanr(S2_arr, att_arr)
        print(f"{precision}-bit: S_2 r_s={S2_corr_s:.4f}")
        results[f'real_prec{precision}'] = {'padic_S2_corr_s': S2_corr_s}

    # Run analysis for RANDOM baseline
    print(f"\n{'='*80}")
    print("RANDOM BASELINE (control for FP16 artifact)")
    print(f"{'='*80}")

    for prime in [2, 3, 5]:
        print(f"\n--- Prime p={prime} ---")
        stats = run_single_analysis(random_keys, attentions, num_pairs, prime, 16, None, use_gpu)

        euc_arr = np.array(stats['euclidean'])
        cos_arr = np.array(stats['cosine'])
        att_arr = np.array(stats['attention'])
        S2_arr = np.array(stats['padic_S2'])

        euc_corr_p, _ = pearsonr(-euc_arr, att_arr)
        cos_corr_p, _ = pearsonr(cos_arr, att_arr)
        S2_corr_s, _ = spearmanr(S2_arr, att_arr)

        print(f"Overall: Euclidean r_p={euc_corr_p:.4f}, Cosine r_p={cos_corr_p:.4f}, S_2 r_s={S2_corr_s:.4f}")

        results[f'random_p{prime}'] = {
            'euclidean_corr_p': euc_corr_p,
            'cosine_corr_p': cos_corr_p,
            'padic_S2_corr_s': S2_corr_s,
        }

    # Final verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    # Use S_2 as the primary statistic (robust in high dimensions)
    real_p2_S2 = results['real_p2']['padic_S2_corr_s']
    real_p2_cos = results['real_p2']['cosine_corr_p']
    real_p2_diff = results['real_p2']['diff_S2_cosine']
    real_p2_diff_ci = results['real_p2']['diff_ci_approx']

    rand_p2 = results['random_p2']['padic_S2_corr_s']
    real_p3 = results['real_p3']['padic_S2_corr_s']
    real_p5 = results['real_p5']['padic_S2_corr_s']

    print(f"\nUsing S_2 (Spearman) vs Cosine (Pearson):")
    print(f"Real transformer (p=2):")
    print(f"  S_2:    {real_p2_S2:.4f}")
    print(f"  Cosine: {real_p2_cos:.4f}")
    print(f"  Diff:   {real_p2_diff:.4f}, 95% CI ≈ [{real_p2_diff_ci[0]:.4f}, {real_p2_diff_ci[1]:.4f}]")
    print(f"Random baseline (p=2): S_2={rand_p2:.4f}")
    print(f"Real transformer (p=3): S_2={real_p3:.4f}")
    print(f"Real transformer (p=5): S_2={real_p5:.4f}")

    # Interpret results based on CIs and multi-prime consistency
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print('='*80)
    print("\nNOTE: CIs use naive bootstrap (not sequence-level resampling).")
    print("True uncertainty likely larger due to pair dependencies.\n")

    # Check if CI excludes zero (statistically significant difference)
    ci_excludes_zero = (real_p2_diff_ci[0] > 0) or (real_p2_diff_ci[1] < 0)

    # Check if real >> random
    real_stronger_than_random = real_p2_S2 > rand_p2 + 0.05  # At least 0.05 better

    # Check multi-prime consistency
    multi_prime_consistent = (real_p3 > 0.3) and (real_p5 > 0.3)

    if ci_excludes_zero and real_p2_diff > 0.1 and real_stronger_than_random and multi_prime_consistent:
        print("✅ STRONG P-ADIC SIGNAL")
        print(f"   - S_2 significantly better than Cosine (CI excludes 0)")
        print(f"   - Real >> random baseline (+{real_p2_S2 - rand_p2:.3f})")
        print(f"   - Signal across multiple primes (not just FP16)")
        print("   → Proceed with PadicKV implementation")
    elif ci_excludes_zero and real_p2_diff > 0:
        print("⚠️  MODERATE P-ADIC SIGNAL")
        print(f"   - S_2 statistically better than Cosine, but small effect")
        print(f"   - May be partially FP16 artifact (check p=3, p=5)")
        print("   → Investigate further before implementation")
    elif real_stronger_than_random:
        print("⚠️  WEAK SIGNAL - Likely FP16 artifact")
        print(f"   - S_2 no better than Cosine (CI includes 0)")
        print(f"   - Real > random but not by much")
        print("   → Probably detecting binary encoding, not structure")
    else:
        print("❌ NO P-ADIC ADVANTAGE")
        print(f"   - S_2 ≈ Cosine ≈ Random")
        print(f"   - No evidence of ultrametric structure")
        print("   → Abandon p-adic KV compression")

    return results


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
    print("P-ADIC STRUCTURE PROBE FOR KEYS (K)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Analyzing layer: {args.layer}")
    print(f"Estimated tokens: {args.num_samples * args.max_length:,}")
    print("\nScope:")
    print("  - Probing KEY structure only (not VALUES)")
    print("  - Values extracted but not analyzed")
    print("  - Separate V probe needed for complete picture")
    print("\nFeatures:")
    print("  ✓ Per-head analysis (no averaging across subspaces)")
    print("  ✓ Robust S_k statistics (not degenerate min in 256-D)")
    print("  ✓ Attention columns (key behavior, not query behavior)")
    print("\nControls:")
    print("  ✓ Random baseline (same distribution)")
    print("  ✓ Multiple primes (p=2, 3, 5)")
    print("  ✓ Multiple precisions (8, 12, 16 bit)")
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

    # Analyze K structure (values probed separately)
    print(f"\n{'='*80}")
    print("PROBING KEY (K) STRUCTURE ONLY")
    print("NOTE: Values (V) should be probed separately")
    print(f"{'='*80}")

    results = analyze_K_structure(
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
