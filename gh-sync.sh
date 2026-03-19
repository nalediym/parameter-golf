#!/bin/bash
# gh-sync.sh - Quick sync with upstream using gh CLI

set -e

echo "=========================================="
echo "SYNC WITH UPSTREAM (gh CLI)"
echo "=========================================="

# The one-liner magic
echo "Syncing nalediym/parameter-golf with openai/parameter-golf..."
gh repo sync nalediym/parameter-golf --source openai/parameter-golf --force

echo ""
echo "✓ Sync complete!"

# Verify
echo ""
echo "Last 3 commits:"
git log --oneline -3
