#!/bin/bash

# P-adic Transformers Docker Environment Setup Script

set -e

echo "========================================"
echo "P-adic Transformers Environment Setup"
echo "========================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA Docker runtime not properly configured."
    echo "Please ensure nvidia-docker2 is installed and configured."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your API keys and configuration."
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p datasets checkpoints results logs

# Build Docker image
echo "Building Docker image (this may take 10-15 minutes)..."
docker-compose build

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Quick Start Commands:"
echo "  1. Start container:       ./run.sh"
echo "  2. Start with Jupyter:    ./run.sh jupyter"
echo "  3. Run training:          ./run.sh train configs/1b_hybrid.yaml"
echo "  4. Open shell:            docker-compose exec padic-dev bash"
echo ""
echo "Don't forget to:"
echo "  - Edit .env file with your API keys"
echo "  - Download datasets: python scripts/download_data.py"
echo ""
