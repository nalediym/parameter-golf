#!/bin/bash
# runpod_ipa_experiment.sh — Run all IPA experiment phases on RunPod
#
# Usage:
#   bash runpod_ipa_experiment.sh phase2       # Convert 1 shard (smoke test)
#   bash runpod_ipa_experiment.sh phase2-full   # Convert all 80 shards
#   bash runpod_ipa_experiment.sh phase3       # 50-step smoke test (1xH100)
#   bash runpod_ipa_experiment.sh phase4       # Full 10-min run (8xH100)
#   bash runpod_ipa_experiment.sh baseline     # BPE baseline + sliding window
#   bash runpod_ipa_experiment.sh all          # phase2 -> phase3 -> phase4
#   bash runpod_ipa_experiment.sh push         # Commit and push logs back

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/parameter-golf}"
LOGDIR="$WORKDIR/experiments/run_logs"
ARTIFACTDIR="$WORKDIR/experiments/run_artifacts"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$LOGDIR" "$ARTIFACTDIR"
cd "$WORKDIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# SETUP: Pull latest code, download data if needed
# ============================================================
setup() {
    log "=== SETUP ==="
    git fetch origin
    git checkout exp/ipa-baseline
    git pull origin exp/ipa-baseline

    if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
        log "Downloading FineWeb dataset..."
        python3 data/cached_challenge_fineweb.py --variant sp1024
    fi

    log "Running converter tests..."
    python3 tests/test_ipa_converter.py
    log "Setup complete."
}

# ============================================================
# PHASE 2: Convert BPE shards to IPA uint8
# ============================================================
run_phase2() {
    local SHARDS="${1:-1}"
    local LOGFILE="$LOGDIR/phase2_convert_${TIMESTAMP}.log"

    log "=== PHASE 2: Convert $SHARDS shard(s) to IPA ==="
    log "Log: $LOGFILE"

    python3 convert_fineweb_to_ipa_shards.py \
        --shards "$SHARDS" \
        --output-dir data/datasets/fineweb10B_ipa \
        2>&1 | tee "$LOGFILE"

    # Verify gate
    log "--- PHASE 2 GATE CHECK ---"
    python3 -c "
import numpy as np, json
from pathlib import Path

out = Path('data/datasets/fineweb10B_ipa')
shards = sorted(out.glob('fineweb_*.bin'))
print(f'Shards found: {len(shards)}')

for shard in shards:
    header = np.fromfile(shard, dtype='<i4', count=256)
    assert header[0] == 20240520, f'Bad magic in {shard.name}'
    assert header[1] == 2, f'Bad version in {shard.name}'
    num_tokens = header[2]
    orig_bytes = header[3]

    header_size = 256 * 4
    tokens = np.fromfile(shard, dtype=np.uint8, count=num_tokens, offset=header_size)
    max_id = tokens.max()

    # Check byte count sidecar
    bytes_file = str(shard).replace('.bin', '.bytes')
    sidecar_bytes = int(open(bytes_file).read().strip())
    assert sidecar_bytes == orig_bytes, f'Sidecar mismatch in {shard.name}'

    print(f'  {shard.name}: {num_tokens:,} tokens, max_id={max_id}, orig_bytes={orig_bytes:,} -- OK')

print('GATE PASSED: all shards valid')
" 2>&1 | tee -a "$LOGFILE"

    log "Phase 2 complete. Log: $LOGFILE"
}

# ============================================================
# PHASE 3: 50-step smoke test
# ============================================================
run_phase3() {
    local LOGFILE="$LOGDIR/phase3_smoke_${TIMESTAMP}.log"
    local NGPU="${NGPU:-1}"

    log "=== PHASE 3: Smoke test (50 steps, ${NGPU}xGPU) ==="
    log "Log: $LOGFILE"

    RUN_ID="ipa_smoke_${TIMESTAMP}" \
    DATA_PATH=./data/datasets/fineweb10B_ipa/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=127 \
    TRAIN_SEQ_LEN=2048 \
    ITERATIONS=50 \
    VAL_LOSS_EVERY=0 \
    MAX_WALLCLOCK_SECONDS=300 \
    TRAIN_LOG_EVERY=10 \
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py \
        2>&1 | tee "$LOGFILE"

    log "Phase 3 complete. Log: $LOGFILE"
}

# ============================================================
# PHASE 4: Full 10-minute training run
# ============================================================
run_phase4() {
    local LOGFILE="$LOGDIR/phase4_full_${TIMESTAMP}.log"
    local NGPU="${NGPU:-8}"

    log "=== PHASE 4: Full IPA training (10 min, ${NGPU}xH100) ==="
    log "Log: $LOGFILE"

    RUN_ID="ipa_full_${TIMESTAMP}" \
    DATA_PATH=./data/datasets/fineweb10B_ipa/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=127 \
    TRAIN_SEQ_LEN=2048 \
    EVAL_STRIDE=64 \
    EVAL_BATCH_SEQS=1024 \
    VAL_LOSS_EVERY=1000 \
    MAX_WALLCLOCK_SECONDS=600 \
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py \
        2>&1 | tee "$LOGFILE"

    # Copy final model artifact
    if [ -f "final_model.int8.ptz" ]; then
        cp final_model.int8.ptz "$ARTIFACTDIR/ipa_full_${TIMESTAMP}.int8.ptz"
        log "Model artifact saved to $ARTIFACTDIR"
    fi

    log "Phase 4 complete. Log: $LOGFILE"
}

# ============================================================
# BASELINE: BPE + sliding window eval (for comparison)
# ============================================================
run_baseline() {
    local LOGFILE="$LOGDIR/baseline_slide64_${TIMESTAMP}.log"
    local NGPU="${NGPU:-8}"

    log "=== BASELINE: BPE-1024 + sliding window eval (${NGPU}xH100) ==="
    log "Log: $LOGFILE"

    RUN_ID="baseline_slide64_${TIMESTAMP}" \
    EVAL_STRIDE=64 \
    EVAL_BATCH_SEQS=1024 \
    VAL_LOSS_EVERY=1000 \
    MAX_WALLCLOCK_SECONDS=600 \
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py \
        2>&1 | tee "$LOGFILE"

    if [ -f "final_model.int8.ptz" ]; then
        cp final_model.int8.ptz "$ARTIFACTDIR/baseline_slide64_${TIMESTAMP}.int8.ptz"
    fi

    log "Baseline complete. Log: $LOGFILE"
}

# ============================================================
# PUSH: Commit logs + artifacts and push back to fork
# ============================================================
push_results() {
    log "=== PUSHING RESULTS ==="
    cd "$WORKDIR"

    git add experiments/run_logs/ experiments/run_artifacts/ \
          data/datasets/fineweb10B_ipa/conversion_log.json \
          data/datasets/fineweb10B_ipa/total_original_bytes.txt \
          2>/dev/null || true

    # Don't add the actual shard .bin files (too large for git)
    git reset -- 'data/datasets/fineweb10B_ipa/*.bin' 2>/dev/null || true
    git reset -- 'data/datasets/fineweb10B_ipa/*.bytes' 2>/dev/null || true
    git reset -- 'experiments/run_artifacts/*.ptz' 2>/dev/null || true

    if git diff --cached --quiet; then
        log "No new results to push."
        return
    fi

    git commit -m "results(ipa): RunPod experiment logs $(date +%Y-%m-%d)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

    git push origin exp/ipa-baseline
    log "Results pushed to origin/exp/ipa-baseline"
    log "Pull locally with: git pull origin exp/ipa-baseline"
}

# ============================================================
# MAIN
# ============================================================
case "${1:-help}" in
    setup)
        setup
        ;;
    phase2)
        setup
        run_phase2 1
        ;;
    phase2-full)
        setup
        run_phase2 80
        ;;
    phase3)
        setup
        run_phase2 1
        run_phase3
        ;;
    phase4)
        setup
        run_phase2 80
        run_phase4
        ;;
    baseline)
        setup
        run_baseline
        ;;
    all)
        setup
        run_phase2 1
        run_phase3
        log "=== Smoke test passed. Converting all shards... ==="
        run_phase2 80
        run_phase4
        push_results
        ;;
    push)
        push_results
        ;;
    *)
        echo "Usage: bash runpod_ipa_experiment.sh {setup|phase2|phase2-full|phase3|phase4|baseline|all|push}"
        echo ""
        echo "  setup        Pull code + download data + run tests"
        echo "  phase2       Convert 1 shard to IPA (smoke test)"
        echo "  phase2-full  Convert all 80 shards to IPA"
        echo "  phase3       50-step training smoke test"
        echo "  phase4       Full 10-min training + sliding window eval"
        echo "  baseline     BPE baseline + sliding window eval (comparison)"
        echo "  all          Run phase2 -> phase3 -> phase2-full -> phase4 -> push"
        echo "  push         Commit and push logs back to fork"
        echo ""
        echo "Env vars:"
        echo "  NGPU=8       Number of GPUs (default: 8 for phase4/baseline, 1 for phase3)"
        ;;
esac
