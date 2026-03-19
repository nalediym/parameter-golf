#!/bin/bash
# push_logs.sh — Simple script to push RunPod logs back to fork.
# Run on RunPod: bash push_logs.sh
set -e
cd "$(dirname "$0")"

echo "Adding log files..."
git add -f experiments/run_logs/ 2>/dev/null || true
git add data/datasets/fineweb10B_ipa/conversion_log.json 2>/dev/null || true
git add data/datasets/fineweb10B_ipa/total_original_bytes.txt 2>/dev/null || true

echo "Staged files:"
git diff --cached --name-only

if git diff --cached --quiet; then
    echo "Nothing new to push."
    exit 0
fi

git commit -m "results: RunPod logs $(date +%Y-%m-%d-%H%M)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git push origin exp/ipa-baseline
echo "Done! Logs pushed."
