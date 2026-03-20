#!/bin/bash
# runpod_autorun.sh — Drop-in startup script for RunPod pods
# Clones repo, downloads data, runs training, pushes results, stops pod.
# No SSH needed — set this as the pod's startup command or run it once.
#
# Usage: curl -fsSL https://raw.githubusercontent.com/nalediym/parameter-golf/exp/competitive-stack/runpod_autorun.sh | bash

set -uo pipefail

BRANCH="${BRANCH:-exp/competitive-stack}"
NGPU="${NGPU:-8}"
RUN_ID="${RUN_ID:-bpe_competitive_$(date +%Y%m%d-%H%M%S)}"
LOG="/workspace/autorun_${RUN_ID}.log"

exec > >(tee "$LOG") 2>&1

echo "=== AUTORUN START $(date) ==="
echo "Branch: $BRANCH | GPUs: $NGPU | Run ID: $RUN_ID"

# Install gh CLI if missing
if ! command -v gh &>/dev/null; then
    echo "Installing gh CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update -qq && apt-get install -y -qq gh > /dev/null 2>&1
fi

# Git identity
git config --global user.name "nalediym"
git config --global user.email "nalediym@users.noreply.github.com"
git config --global pull.rebase false

# Clone repo
cd /workspace
if [ ! -d "parameter-golf/.git" ]; then
    git clone https://github.com/nalediym/parameter-golf.git --branch "$BRANCH"
fi
cd parameter-golf
git checkout "$BRANCH"
git pull origin "$BRANCH" || true

# Download data (standard BPE — no conversion needed)
echo "=== DOWNLOADING DATA ==="
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run training with competitive stack on standard BPE
echo "=== TRAINING START $(date) ==="
RUN_ID="$RUN_ID" \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=2048 \
MLP_MULT=3 \
QAT=1 \
ORTHO_INIT=1 \
USE_BIGRAM_HASH=1 \
USE_SMEAR_GATE=1 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
TRAIN_BATCH_TOKENS=786432 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py \
    2>&1 | tee "experiments/run_logs/train_${RUN_ID}.log" || true

echo "=== TRAINING DONE $(date) ==="

# Extract result
BPB=$(grep "final_int8_zlib_roundtrip_sliding_exact" "experiments/run_logs/train_${RUN_ID}.log" 2>/dev/null | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
echo "=== RESULT: val_bpb=${BPB:-unknown} ==="

# Push logs
gh auth setup-git 2>/dev/null || true
git add experiments/run_logs/ 2>/dev/null || true
git commit -m "results: 8xH100 BPE competitive val_bpb=${BPB:-unknown} $(date +%Y-%m-%d)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
git push origin "$BRANCH" 2>/dev/null || true

echo "=== AUTORUN COMPLETE $(date) ==="
echo "val_bpb: ${BPB:-unknown}"
echo "Log: $LOG"
echo ""
echo "STOP THE POD: runpodctl pod stop <pod-id>"
