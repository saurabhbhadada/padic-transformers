# P-adic Transformers

Research project exploring 2-adic number representations in transformer architectures for improved memory efficiency and hierarchical reasoning.

**Target**: 1B parameter hybrid model | **Timeline**: 3-4 months to publication

## Quick Start

**→ See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed remote GPU setup**

```bash
# On your remote GPU machine:
git clone <repo> padic-transformers && cd padic-transformers
make build              # Build Docker (10-15 mins, one-time)
make download-pythia    # Download Pythia-1B (~2GB)

# Run first experiment (baseline)
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_baseline.py --output results/baseline.json"

# Run with compression
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_compression.py --compression simple_2x --output results/compression_2x.json"

# Visualize results
make jupyter  # Open notebooks/01_visualize_compression_results.ipynb
```

## Documentation

📚 **Core Documentation**
- **[Experimental Plan](docs/EXPERIMENTAL_PLAN.md)** - Complete research roadmap, both tracks, timeline ⭐ START HERE
- [Project Overview & Architecture](docs/PROJECT.md) - Full project details, datasets, benchmarks
- [Leaderboard Submission Guide](docs/LEADERBOARDS.md) - Publication strategy, evaluation methodology

🔧 **Development**
- [Makefile Commands](Makefile) - All available commands
- [Docker Setup](Dockerfile) - Environment configuration

## Project Structure

```
padic-transformers/
├── src/                # Source code
│   ├── models/         # P-adic + hybrid architectures
│   ├── kernels/        # Custom Triton/CUDA kernels
│   ├── data/           # Data loading & preprocessing
│   ├── training/       # Training loops
│   └── evaluation/     # Benchmark evaluation
├── datasets/           # Downloaded datasets
├── configs/            # Experiment configs
├── scripts/            # Training/eval scripts
└── docs/               # Documentation
```

## Key Features

- **2-adic embeddings** - Binary-aligned for hardware efficiency
- **Memory compression** - KV-cache compression for long context
- **Hybrid architecture** - Mix of 2-adic and standard transformer layers
- **Custom kernels** - Optimized Triton/CUDA implementations

## Research Goals

1. Match baseline 1B models on standard benchmarks
2. Achieve 2-4x memory compression with <5% accuracy drop
3. Excel at hierarchical reasoning (math, code)
4. Publish at ICLR/NeurIPS/ICML

## License

MIT
