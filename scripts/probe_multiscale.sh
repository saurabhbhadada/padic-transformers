#!/bin/bash
# Multi-scale p-adic structure probe
# Tests across different datasets and token counts

set -e

MODEL="pythia-1b"
LAYER=6  # Middle layer

# Token counts to test
TOKENS=(1000000 5000000 10000000)

# Datasets to test
DATASETS=(
    "wikitext:Salesforce/wikitext:wikitext-103-raw-v1:text"
    "code:bigcode/the-stack-dedup:data/python:content"
    "math:hendrycks/competition_math:train:problem"
)

echo "=================================================="
echo "MULTI-SCALE P-ADIC STRUCTURE PROBE"
echo "=================================================="
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "Token counts: ${TOKENS[@]}"
echo "Datasets: WikiText, Code (Python), Math"
echo "=================================================="

for DATASET_SPEC in "${DATASETS[@]}"; do
    IFS=':' read -r NAME PATH SPLIT FIELD <<< "$DATASET_SPEC"

    echo ""
    echo "=================================================="
    echo "DATASET: $NAME"
    echo "=================================================="

    for NUM_TOKENS in "${TOKENS[@]}"; do
        # Calculate number of samples (assuming ~500 tokens per sample)
        NUM_SAMPLES=$((NUM_TOKENS / 500))

        OUTPUT="results/probe_${NAME}_${NUM_TOKENS}.json"

        echo ""
        echo "Testing $NUM_TOKENS tokens ($NUM_SAMPLES samples)..."
        echo "Output: $OUTPUT"

        python3 scripts/probe_kv_structure.py \
            --model "$MODEL" \
            --dataset-path "$PATH" \
            --dataset-split "$SPLIT" \
            --text-field "$FIELD" \
            --num-samples "$NUM_SAMPLES" \
            --layer "$LAYER" \
            --output "$OUTPUT"

        # Print quick summary
        echo "Results:"
        python3 -c "
import json
with open('$OUTPUT') as f:
    r = json.load(f)
    c = r['correlations']
    print(f\"  Euclidean: {c['euclidean_corr']:.4f}\")
    print(f\"  P-adic:    {c['padic_corr']:.4f}\")
    if c['padic_corr'] > c['euclidean_corr'] * 1.1:
        print(f\"  ✅ P-adic wins by {(c['padic_corr']/c['euclidean_corr']-1)*100:.1f}%\")
    else:
        print(f\"  ❌ No p-adic advantage\")
"
    done
done

echo ""
echo "=================================================="
echo "GENERATING SUMMARY REPORT"
echo "=================================================="

python3 scripts/summarize_probe_results.py results/probe_*.json \
    --output results/probe_summary.md

echo ""
echo "✓ All probes complete!"
echo "See: results/probe_summary.md"
