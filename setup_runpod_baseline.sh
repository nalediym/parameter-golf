#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash setup_runpod_baseline.sh --mode smoke
#   bash setup_runpod_baseline.sh --mode full --nproc 1 --run-id my_remote_run

MODE="smoke"
NPROC=1
RUN_ID=""
REPO_DIR="/workspace/parameter-golf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash setup_runpod_baseline.sh [--mode smoke|full] [--nproc N] [--run-id ID]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Invalid --mode '$MODE'. Use 'smoke' or 'full'."
  exit 1
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="runpod_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

echo "== Parameter Golf Runpod bootstrap =="
echo "Mode: $MODE"
echo "nproc_per_node: $NPROC"
echo "RUN_ID: $RUN_ID"

if [[ ! -d "/workspace" ]]; then
  echo "Expected /workspace to exist on the Runpod image."
  echo "If this is a different machine, adjust REPO_DIR in this script."
  exit 1
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Cloning repository into $REPO_DIR"
  git clone https://github.com/openai/parameter-golf.git "$REPO_DIR"
else
  echo "Repository already exists at $REPO_DIR; pulling latest changes"
  git -C "$REPO_DIR" pull --ff-only
fi

mkdir -p "$REPO_DIR/logs"

echo "Downloading cached FineWeb data/tokenizer..."
if [[ "$MODE" == "smoke" ]]; then
  python3 "$REPO_DIR/data/cached_challenge_fineweb.py" --variant sp1024 --train-shards 1
else
  python3 "$REPO_DIR/data/cached_challenge_fineweb.py" --variant sp1024
fi

TRAIN_LOG="$REPO_DIR/logs/${RUN_ID}.log"

echo "Starting training (log: $TRAIN_LOG)"
RUN_ID="$RUN_ID" \
DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp1024/" \
TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node="$NPROC" "$REPO_DIR/train_gpt.py" 2>&1 | tee "$TRAIN_LOG"

echo
echo "Done. Key files:"
echo "- Train log: $TRAIN_LOG"
echo "- Repo dir:  $REPO_DIR"
echo
echo "Tip: grep final metrics from the log:"
echo "  rg \"val_loss|val_bpb|final_int8_zlib_roundtrip\" '$TRAIN_LOG'"
