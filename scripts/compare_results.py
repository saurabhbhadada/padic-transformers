#!/usr/bin/env python3
"""
Compare baseline vs compressed results from JSON outputs.

Usage:
    python scripts/compare_results.py \\
        results/baseline.json \\
        results/compression_2x.json
"""

import argparse
import json
from pathlib import Path


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(baseline_path, compressed_path):
    """Compare baseline and compressed results."""
    baseline = load_results(baseline_path)
    compressed = load_results(compressed_path)

    print("=" * 80)
    print("BASELINE VS COMPRESSED COMPARISON")
    print("=" * 80)
    print(f"\nBaseline:   {baseline_path}")
    print(f"Compressed: {compressed_path}")
    print(f"\nModel:    {baseline['model']}")
    print(f"Dataset:  {baseline['dataset']}")
    if 'precision' in compressed:
        print(f"Precision: {compressed['precision']} bits")
    print("=" * 80)

    # Compare each context length
    for ctx_len in baseline['results'].keys():
        if ctx_len not in compressed['results']:
            print(f"\n⚠ Context length {ctx_len} not found in compressed results")
            continue

        base_res = baseline['results'][ctx_len]
        comp_res = compressed['results'][ctx_len]

        print(f"\n{'=' * 80}")
        print(f"CONTEXT LENGTH: {ctx_len}")
        print(f"{'=' * 80}")
        print(f"\n{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Change':<15}")
        print("-" * 80)

        # Perplexity
        ppl_base = base_res['perplexity']
        ppl_comp = comp_res['perplexity']
        ppl_change = ((ppl_comp - ppl_base) / ppl_base) * 100
        print(f"{'Perplexity':<30} {ppl_base:<15.4f} {ppl_comp:<15.4f} {ppl_change:+.2f}%")

        # Peak memory
        mem_base = base_res['peak_memory_mb']
        mem_comp = comp_res['peak_memory_mb']
        mem_change = ((mem_comp - mem_base) / mem_base) * 100
        mem_saved = mem_base - mem_comp
        print(f"{'Peak Memory (MB)':<30} {mem_base:<15.2f} {mem_comp:<15.2f} {mem_change:+.2f}% ({mem_saved:+.0f}MB)")

        # Cache compression
        if 'cache_compressed_mb' in comp_res:
            cache_comp = comp_res['cache_compressed_mb']
            cache_uncomp = comp_res['cache_uncompressed_mb']
            ratio = comp_res['compression_ratio']
            print(f"{'Cache Memory (MB)':<30} {cache_uncomp:<15.2f} {cache_comp:<15.2f} {ratio:.2f}x")

        # Time (if available)
        if 'elapsed_time_sec' in base_res and 'elapsed_time_sec' in comp_res:
            time_base = base_res['elapsed_time_sec']
            time_comp = comp_res['elapsed_time_sec']
            time_change = ((time_comp - time_base) / time_base) * 100
            print(f"{'Time (sec)':<30} {time_base:<15.2f} {time_comp:<15.2f} {time_change:+.2f}%")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for ctx_len in baseline['results'].keys():
        if ctx_len in compressed['results']:
            base_res = baseline['results'][ctx_len]
            comp_res = compressed['results'][ctx_len]
            ppl_change = ((comp_res['perplexity'] - base_res['perplexity']) / base_res['perplexity']) * 100
            mem_change = ((comp_res['peak_memory_mb'] - base_res['peak_memory_mb']) / base_res['peak_memory_mb']) * 100

            print(f"\nContext {ctx_len}:")
            print(f"  PPL:    {base_res['perplexity']:.4f} → {comp_res['perplexity']:.4f} ({ppl_change:+.2f}%)")
            print(f"  Memory: {base_res['peak_memory_mb']:.0f}MB → {comp_res['peak_memory_mb']:.0f}MB ({mem_change:+.2f}%)")

            if 'compression_ratio' in comp_res:
                print(f"  Cache:  {comp_res['compression_ratio']:.2f}x compression")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs compressed results")
    parser.add_argument("baseline", type=str, help="Path to baseline results JSON")
    parser.add_argument("compressed", type=str, help="Path to compressed results JSON")

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    compressed_path = Path(args.compressed)

    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        return

    if not compressed_path.exists():
        print(f"Error: Compressed file not found: {compressed_path}")
        return

    compare_results(baseline_path, compressed_path)


if __name__ == "__main__":
    main()
