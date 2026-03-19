#!/bin/bash
# run_experiment_A3.sh - Morphological-only baseline (Experiment A3 from TODO)
# Tests: Morphological base channel without features

set -e

WORKDIR="${WORKDIR:-/workspace/parameter-golf}"
cd "$WORKDIR"

echo "=========================================="
echo "EXPERIMENT A3: Morphological-Only Base"
echo "=========================================="

# Configuration for A3
export RUN_ID="A3_morph_only"
export DATA_PATH="./data/datasets/fineweb10B_morph_v2/"  # Need to create
export TOKENIZER_PATH="./data/tokenizers/morph_1024.model"  # Use morph vocab
export VOCAB_SIZE=1024  # Morpheme vocabulary
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
echo "  Vocab: $VOCAB_SIZE (morphemes)"
echo "  Model: ${NUM_LAYERS}x${MODEL_DIM}, ${NUM_HEADS} heads"
echo "  Budget: 16MB artifact limit"
echo ""

# Check if morph data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "⚠ Morph dataset not found at $DATA_PATH"
    echo "  Need to run: cd experiments/morphological_tokenization && python3 convert_fineweb_to_morph.py"
    exit 1
fi

echo "Running training..."
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "experiments/run_logs/${RUN_ID}.log"

echo ""
echo "=========================================="
echo "A3 Complete! Check: experiments/run_logs/${RUN_ID}.log"
echo "=========================================="
