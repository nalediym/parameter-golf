#!/bin/bash
# setup_runpod_experiments.sh - Initialize RunPod for parallel training
# Run this on your RunPod instance

set -e

echo "=========================================="
echo "SETUP: RunPod Parallel Experiment Runner"
echo "=========================================="

# Install tmux if not present
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# Setup workspace
WORKDIR="${WORKDIR:-/workspace/parameter-golf}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Clone YOUR FORK if not exists
USER_FORK="https://github.com/nalediym/parameter-golf.git"
if [ ! -d ".git" ]; then
    echo "Cloning YOUR FORK: $USER_FORK"
    git clone "$USER_FORK" .
fi

# Download full dataset (if not already present)
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading FineWeb dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# Create experiment directories
mkdir -p experiments/run_logs
mkdir -p experiments/run_artifacts
mkdir -p experiments/results

# Setup Python environment
pip install -q torch numpy sentencepiece huggingface-hub datasets tqdm 2>/dev/null || true

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Create tmux sessions: ./runpod_tmux_runner.sh"
echo "  2. Or run manually: ./experiments/run_experiment_A1.sh"
echo ""
