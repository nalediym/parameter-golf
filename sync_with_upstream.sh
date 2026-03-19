#!/bin/bash
# sync_with_upstream.sh - Keep your fork in sync with openai/parameter-golf
# Run this periodically to get latest changes from upstream

set -e

echo "=========================================="
echo "SYNCING FORK WITH UPSTREAM"
echo "=========================================="

# Fetch upstream changes
echo "Fetching upstream (openai/parameter-golf)..."
git fetch upstream

# Checkout main and merge upstream
echo "Updating main branch..."
git checkout main
git merge upstream/main --no-edit

# Push to your fork
echo "Pushing to your fork..."
git push origin main

echo ""
echo "✓ Sync complete!"
echo "  Your fork is now up-to-date with openai/parameter-golf"
echo ""
echo "Current remotes:"
git remote -v

# Using gh CLI (alternative):
# gh repo sync nalediym/parameter-golf --source openai/parameter-golf
