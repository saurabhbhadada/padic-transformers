# P-adic Transformers: Neural Networks with Ultrametric Structure

Research project exploring p-adic (specifically 2-adic) number representations in transformer architectures for improved memory efficiency and hierarchical reasoning.

## Project Overview

**Goal**: Develop hybrid transformer models using 2-adic embeddings and operations to achieve:
- Memory compression (especially KV-cache for long context)
- Better hierarchical/compositional reasoning
- Hardware-efficient computation (binary-aligned)

**Model Scale**: 1B parameter hybrid architecture
**Timeline**: 3-4 months to publication-ready results

## Project Structure

```
padic-transformers/
├── src/
│   ├── models/          # Model architectures (hybrid p-adic + standard)
│   ├── kernels/         # Custom Triton/CUDA/JAX kernels for p-adic ops
│   ├── data/            # Data loading, preprocessing, tokenization
│   ├── training/        # Training loops, optimization, distributed training
│   ├── evaluation/      # Benchmark evaluation, metrics, leaderboards
│   └── utils/           # Helper functions, logging, config management
├── datasets/            # Downloaded and preprocessed datasets
├── configs/             # Experiment configurations (YAML/JSON)
├── scripts/             # Training and evaluation scripts
├── notebooks/           # Research notebooks and exploration
├── tests/               # Unit tests for p-adic operations
├── results/             # Experiment results, plots, analysis
└── checkpoints/         # Model checkpoints and artifacts
```

## Datasets for Pretraining

**Total**: ~150-200B tokens across:
- The Pile / RedPajama (50% - general text)
- The Stack v2 (25% - code, hierarchical structure)
- Proof-Pile-2 + OpenWebMath (15% - mathematics)
- C4 (10% - web diversity)

## Evaluation Benchmarks

### Language Understanding
- **Open LLM Leaderboard** (Hugging Face) - primary leaderboard
  - MMLU, HellaSwag, ARC-Challenge, TruthfulQA, Winogrande, GSM8K

### Mathematical Reasoning
- MATH dataset
- GSM8K
- TheoremQA

### Code Generation
- HumanEval
- MBPP
- CodeContests

### Memory/Efficiency Metrics
- Long-context retrieval (RULER, LongBench)
- KV-cache compression ratio
- Perplexity vs memory footprint

### Compositional Reasoning
- SCAN
- CFQ
- COGS

## Leaderboards & Publication Strategy

### Official Leaderboards (Auto-Tracked)

**1. Hugging Face Open LLM Leaderboard** ⭐ Primary Target
- URL: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- Benchmarks: MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K
- Process: Upload model to HF Hub → Automated evaluation
- Ranking: Public, widely cited in papers

**2. Papers with Code**
- URL: https://paperswithcode.com/
- Auto-tracks: ArXiv papers + GitHub repos
- Manual submission: Upload evaluation results for any benchmark
- Creates comparison tables across all published methods

**3. AlpacaEval Leaderboard**
- URL: https://tatsu-lab.github.io/alpaca_eval/
- Focus: Instruction-following quality
- Evaluation: GPT-4 as judge

**4. BigCode Leaderboard**
- URL: https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard
- Focus: Code generation (HumanEval, MBPP)

**5. LMSys Chatbot Arena**
- URL: https://chat.lmsys.org/
- Human preference evaluation
- Good for visibility after initial results

### Systematic Evaluation Approach

**Use EleutherAI LM Evaluation Harness** for reproducibility:
```bash
make eval-all CKPT=checkpoints/best_model.pt
```
Generates standardized JSON outputs compatible with all leaderboards.

**Benchmark Coverage for Paper**:
- General: MMLU (57 tasks), HellaSwag, ARC
- Math: MATH, GSM8K, TheoremQA
- Code: HumanEval, MBPP
- Efficiency: Memory footprint, inference speed
- Novel: Your p-adic-specific benchmarks

### Conference Targets
- ICLR, NeurIPS, ICML (main ML venues)
- EMNLP, ACL (if language-focused)
- Workshop papers: Math-AI @ NeurIPS, Efficient NLP

## Getting Started

This project uses Docker + Makefile for reproducible development:

```bash
# Initial setup
make setup              # Create directories and .env file
make build              # Build Docker image (10-15 mins)

# Download datasets
make download           # Download all datasets
make download-pile      # Just The Pile
make download-math      # Math datasets only

# Development
make shell              # Open interactive shell
make jupyter            # Start Jupyter Lab
make tensorboard        # Monitor training

# Training
make train CONFIG=configs/1b_hybrid.yaml
make train-1b           # Shortcut for 1B model
make train-resume CKPT=checkpoints/step_1000.pt

# Evaluation
make eval CKPT=checkpoints/best_model.pt
make eval-mmlu CKPT=checkpoints/best_model.pt
make eval-math CKPT=checkpoints/best_model.pt

# Run tests
make test

# View all commands
make help
```

## Hardware Requirements

- Training: 8x H200 GPUs (or equivalent)
- Estimated time: 2-3 weeks for 150B tokens
- Storage: ~1TB for datasets + checkpoints

## Research Questions

1. Can 2-adic embeddings compress model memory without accuracy loss?
2. Do 2-adic operations provide inductive bias for hierarchical reasoning?
3. What is the optimal hybrid architecture (which layers benefit from p-adic)?
4. Can custom kernels make p-adic ops competitive with FP16/BF16 speed?

## License

MIT (or choose appropriate research license)
