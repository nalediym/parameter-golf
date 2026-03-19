#!/bin/bash
# runpod_tmux_runner.sh - Launch parallel experiments in tmux sessions
# Usage: ./runpod_tmux_runner.sh [experiment_set]
#   experiment_set: all | ipa | morph | sign | baseline

set -e

WORKDIR="${WORKDIR:-/workspace/parameter-golf}"
cd "$WORKDIR"

EXPERIMENT_SET="${1:-all}"

echo "=========================================="
echo "LAUNCHING: $EXPERIMENT_SET experiments"
echo "=========================================="

# Kill existing sessions for clean start (optional)
# tmux kill-server 2>/dev/null || true

launch_session() {
    local session_name="$1"
    local command="$2"
    
    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "  ⚠ Session '$session_name' already exists. Skipping."
        return
    fi
    
    echo "  → Creating tmux session: $session_name"
    tmux new-session -d -s "$session_name" -c "$WORKDIR"
    tmux send-keys -t "$session_name" "echo '=== $session_name ===' && $command" Enter
}

case "$EXPERIMENT_SET" in
    baseline)
        echo "Launching baseline experiments..."
        launch_session "baseline_1024" "RUN_ID=baseline_sp1024 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee experiments/run_logs/baseline_1024.log"
        ;;
    
    ipa)
        echo "Launching IPA experiments..."
        launch_session "ipa_A1" "./experiments/run_experiment_A1.sh 2>&1 | tee experiments/run_logs/ipa_A1.log"
        launch_session "ipa_A2" "./experiments/run_experiment_A2.sh 2>&1 | tee experiments/run_logs/ipa_A2.log"
        launch_session "ipa_A5" "./experiments/run_experiment_A5.sh 2>&1 | tee experiments/run_logs/ipa_A5.log"
        ;;
    
    morph)
        echo "Launching Morphological experiments..."
        launch_session "morph_A3" "./experiments/run_experiment_A3.sh 2>&1 | tee experiments/run_logs/morph_A3.log"
        launch_session "morph_A4" "./experiments/run_experiment_A4.sh 2>&1 | tee experiments/run_logs/morph_A4.log"
        ;;
    
    sign)
        echo "Launching Sign-language experiments..."
        launch_session "sign_base" "./experiments/run_sign_base.sh 2>&1 | tee experiments/run_logs/sign_base.log"
        ;;
    
    all)
        echo "Launching ALL experiments in parallel..."
        launch_session "baseline_1024" "RUN_ID=baseline_sp1024 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee experiments/run_logs/baseline_1024.log"
        launch_session "ipa_A1" "./experiments/run_experiment_A1.sh 2>&1 | tee experiments/run_logs/ipa_A1.log"
        launch_session "morph_A3" "./experiments/run_experiment_A3.sh 2>&1 | tee experiments/run_logs/morph_A3.log"
        launch_session "sign_base" "./experiments/run_sign_base.sh 2>&1 | tee experiments/run_logs/sign_base.log"
        ;;
    
    *)
        echo "Unknown experiment set: $EXPERIMENT_SET"
        echo "Usage: $0 {all|ipa|morph|sign|baseline}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "ACTIVE TMUX SESSIONS:"
echo "=========================================="
tmux list-sessions 2>/dev/null || echo "  (No active sessions)"

echo ""
echo "Commands to interact:"
echo "  tmux ls                    # List all sessions"
echo "  tmux attach -t <session>   # Attach to a session"
echo "  tmux detach                # Detach from session (Ctrl+b then d)"
echo "  tmux kill-session -t <session>  # Kill a session"
echo ""
echo "Monitor logs:"
echo "  tail -f experiments/run_logs/*.log"
echo ""
