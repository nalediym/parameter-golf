#!/bin/bash
# run_experiment_A1.sh - IPA-only baseline (Experiment A1 from TODO)
# Tests: IPA base channel without features

set -e

WORKDIR="${WORKDIR:-/workspace/parameter-golf}"
cd "$WORKDIR"

echo "=========================================="
echo "EXPERIMENT A1: IPA-Only Base Channel"
echo "=========================================="

# Configuration for A1
export RUN_ID="A1_ipa_only"
export DATA_PATH="./data/datasets/fineweb10B_ipa/"  # Need to create this
export TOKENIZER_PATH="./data/tokenizers/ipa_47.model"  # Need to create
export VOCAB_SIZE=47
export NUM_LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export TIE_EMBEDDINGS=1
export TRAIN_BATCH_TOKENS=524288
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600

echo ""
echo "Configuration:"
echo "  Run ID: $RUN_ID"
echo "  Vocab: $VOCAB_SIZE (IPA symbols)"
echo "  Model: ${NUM_LAYERS}x${MODEL_DIM}, ${NUM_HEADS} heads"
echo "  Budget: 16MB artifact limit"
echo ""

# Check if IPA data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "⚠ IPA dataset not found at $DATA_PATH"
    echo "  Need to run: python3 convert_fineweb_ipa.py --shards 80"
    exit 1
fi

echo "Running training..."
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "experiments/run_logs/${RUN_ID}.log"

echo ""
echo "=========================================="
echo "A1 Complete! Check: experiments/run_logs/${RUN_ID}.log"
echo "=========================================="
