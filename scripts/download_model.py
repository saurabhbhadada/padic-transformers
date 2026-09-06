#!/usr/bin/env python3
"""
Download pretrained models for evaluation.

Supports:
- Pythia models (1B, 1.4B, etc.)
- TinyLlama
- OPT models
"""

import argparse
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_CONFIGS = {
    'pythia-1b': {
        'model_name': 'EleutherAI/pythia-1b',
        'description': 'Pythia 1B (fully open, trained on The Pile)',
    },
    'pythia-1.4b': {
        'model_name': 'EleutherAI/pythia-1.4b',
        'description': 'Pythia 1.4B',
    },
    'pythia-160m': {
        'model_name': 'EleutherAI/pythia-160m',
        'description': 'Pythia 160M (smaller, for quick tests)',
    },
    'tinyllama': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'description': 'TinyLlama 1.1B',
    },
    'opt-1.3b': {
        'model_name': 'facebook/opt-1.3b',
        'description': 'OPT 1.3B from Meta',
    },
}


def download_model(model_key: str, cache_dir: Path, device: str = 'auto'):
    """
    Download a pretrained model and tokenizer.

    Args:
        model_key: Key from MODEL_CONFIGS
        cache_dir: Directory to cache the model
        device: Device to load model on ('cpu', 'cuda', 'auto')

    Returns:
        model, tokenizer
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_key]
    model_name = config['model_name']

    print(f"\n{'='*60}")
    print(f"Downloading: {config['description']}")
    print(f"Model name: {model_name}")
    print(f"Cache dir: {cache_dir}")
    print(f"{'='*60}\n")

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Download model
    print("Downloading model weights (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
        device_map=device if device != 'cpu' else None,
    )

    print("\n✓ Download complete!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Model dtype: {model.dtype}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    return model, tokenizer


def list_models():
    """Print available models."""
    print("\nAvailable models:")
    print("-" * 60)
    for key, config in MODEL_CONFIGS.items():
        print(f"  {key:<20} - {config['description']}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Download pretrained models for evaluation')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        help='Model to download (e.g., pythia-1b, tinyllama)',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/workspace/padic-transformers/checkpoints/pretrained',
        help='Directory to cache downloaded models',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to load model on',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test model after downloading',
    )

    args = parser.parse_args()

    if args.list or args.model is None:
        list_models()
        if args.model is None:
            return

    # Download model
    cache_dir = Path(args.cache_dir)
    model, tokenizer = download_model(args.model, cache_dir, args.device)

    # Test if requested
    if args.test:
        print("\n" + "="*60)
        print("Testing model generation...")
        print("="*60 + "\n")

        test_prompt = "The future of artificial intelligence is"
        inputs = tokenizer(test_prompt, return_tensors='pt')

        if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {test_prompt}")
        print(f"Generated: {generated_text}")
        print("\n✓ Model is working correctly!")

    print(f"\nModel saved to: {cache_dir}")
    print("You can now use this model for evaluation.")


if __name__ == '__main__':
    main()
