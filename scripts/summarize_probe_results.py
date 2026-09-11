#!/usr/bin/env python3
"""
Summarize results from multi-scale p-adic structure probe.

Usage:
    python scripts/summarize_probe_results.py results/probe_*.json \\
        --output results/probe_summary.md
"""

import argparse
import json
from pathlib import Path
import sys


def load_results(filepaths):
    """Load all probe results."""
    results = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append((Path(filepath).stem, data))
    return results


def generate_summary(results, output_path):
    """Generate markdown summary of all results."""

    with open(output_path, 'w') as f:
        f.write("# P-adic Structure Probe Results\n\n")
        f.write("Multi-scale analysis across datasets and token counts.\n\n")
        f.write("---\n\n")

        # Group by dataset
        datasets = {}
        for name, data in results:
            # Parse name: probe_{dataset}_{tokens}
            parts = name.split('_')
            dataset = parts[1] if len(parts) > 1 else 'unknown'
            tokens = parts[2] if len(parts) > 2 else '0'

            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append((int(tokens), data))

        # Sort each dataset by token count
        for dataset in datasets:
            datasets[dataset].sort(key=lambda x: x[0])

        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Dataset | Tokens | Euclidean r | Cosine r | P-adic r | Winner | Advantage |\n")
        f.write("|---------|--------|-------------|----------|----------|--------|------------|\n")

        for dataset, runs in datasets.items():
            for tokens, data in runs:
                corr = data['correlations']
                euc_r = corr['euclidean_corr']
                cos_r = corr['cosine_corr']
                pad_r = corr['padic_corr']

                # Determine winner
                if pad_r > max(euc_r, cos_r):
                    winner = "**P-adic**"
                    advantage = f"+{(pad_r / max(euc_r, cos_r) - 1) * 100:.1f}%"
                elif cos_r > max(euc_r, pad_r):
                    winner = "Cosine"
                    advantage = f"+{(cos_r / max(euc_r, pad_r) - 1) * 100:.1f}%"
                else:
                    winner = "Euclidean"
                    advantage = f"+{(euc_r / max(cos_r, pad_r) - 1) * 100:.1f}%"

                tokens_str = f"{tokens/1e6:.1f}M"
                f.write(f"| {dataset} | {tokens_str} | {euc_r:.4f} | {cos_r:.4f} | {pad_r:.4f} | {winner} | {advantage} |\n")

        # Detailed results per dataset
        for dataset, runs in datasets.items():
            f.write(f"\n---\n\n")
            f.write(f"## Dataset: {dataset.upper()}\n\n")

            for tokens, data in runs:
                corr = data['correlations']
                tokens_str = f"{tokens/1e6:.1f}M"

                f.write(f"### {tokens_str} Tokens\n\n")
                f.write(f"**Correlation with Attention Similarity:**\n\n")
                f.write(f"- Euclidean distance: `r = {corr['euclidean_corr']:.4f}` (p = {corr['euclidean_p']:.6f})\n")
                f.write(f"- Cosine similarity:  `r = {corr['cosine_corr']:.4f}` (p = {corr['cosine_p']:.6f})\n")
                f.write(f"- **P-adic valuation: `r = {corr['padic_corr']:.4f}` (p = {corr['padic_p']:.6f})**\n\n")

                # Verdict
                euc_r = corr['euclidean_corr']
                pad_r = corr['padic_corr']

                if pad_r > euc_r * 1.3:  # 30% stronger
                    f.write(f"**Verdict:** ✅ **STRONG P-ADIC SIGNAL** ({(pad_r/euc_r - 1)*100:.1f}% stronger)\n\n")
                elif pad_r > euc_r * 1.1:  # 10% stronger
                    f.write(f"**Verdict:** ⚠️  **WEAK P-ADIC SIGNAL** ({(pad_r/euc_r - 1)*100:.1f}% stronger)\n\n")
                else:
                    f.write(f"**Verdict:** ❌ **NO P-ADIC ADVANTAGE** (Euclidean is better)\n\n")

        # Overall conclusion
        f.write(f"\n---\n\n")
        f.write(f"## Overall Conclusion\n\n")

        # Count strong signals
        strong_signals = 0
        weak_signals = 0
        no_signals = 0

        for dataset, runs in datasets.items():
            for tokens, data in runs:
                corr = data['correlations']
                euc_r = corr['euclidean_corr']
                pad_r = corr['padic_corr']

                if pad_r > euc_r * 1.3:
                    strong_signals += 1
                elif pad_r > euc_r * 1.1:
                    weak_signals += 1
                else:
                    no_signals += 1

        total = strong_signals + weak_signals + no_signals

        f.write(f"**Results across {total} experiments:**\n\n")
        f.write(f"- ✅ Strong p-adic signal (>30% advantage): {strong_signals}/{total}\n")
        f.write(f"- ⚠️  Weak p-adic signal (10-30% advantage): {weak_signals}/{total}\n")
        f.write(f"- ❌ No p-adic advantage: {no_signals}/{total}\n\n")

        if strong_signals >= total * 0.5:
            f.write(f"### 🎯 **RECOMMENDATION: PROCEED WITH PADICEV**\n\n")
            f.write(f"P-adic structure is consistently detected across datasets and scales. ")
            f.write(f"Strong evidence for exploitable ultrametric structure in KV cache.\n\n")
            f.write(f"**Next steps:**\n")
            f.write(f"1. Implement PadicKV hierarchical cache\n")
            f.write(f"2. Compare against INT4/INT8 baselines at same memory budget\n")
            f.write(f"3. Write paper: 'Ultrametric Structure in Transformer KV Cache'\n")
        elif weak_signals + strong_signals >= total * 0.5:
            f.write(f"### ⚠️  **RECOMMENDATION: INVESTIGATE FURTHER**\n\n")
            f.write(f"Weak p-adic signal detected. Consider:\n")
            f.write(f"1. Test on more diverse datasets\n")
            f.write(f"2. Try different layers (early vs late)\n")
            f.write(f"3. Investigate specific domains where signal is strong\n")
        else:
            f.write(f"### ❌ **RECOMMENDATION: PIVOT AWAY FROM P-ADIC KV COMPRESSION**\n\n")
            f.write(f"No consistent p-adic advantage detected. The ultrametric structure ")
            f.write(f"does not appear to be exploitable for KV cache compression.\n\n")
            f.write(f"**Alternative directions:**\n")
            f.write(f"1. Standard INT4 quantization with learned codebooks\n")
            f.write(f"2. Attention-aware pruning/eviction\n")
            f.write(f"3. Explore p-adic for other components (embeddings, weights)\n")

    print(f"\n✓ Summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize probe results")
    parser.add_argument("results", nargs='+', help="Probe result JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output markdown file")

    args = parser.parse_args()

    print(f"Loading {len(args.results)} result files...")
    results = load_results(args.results)

    print(f"Generating summary...")
    generate_summary(results, args.output)


if __name__ == "__main__":
    main()
