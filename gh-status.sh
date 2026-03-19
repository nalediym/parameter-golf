#!/bin/bash
# gh-status.sh - Quick status of your repo

set -e

echo "=========================================="
echo "REPO STATUS (gh CLI)"
echo "=========================================="

# Fork relationship
echo ""
echo "📋 Fork Info:"
gh repo view nalediym/parameter-golf --json parent --jq '.parent | "Parent: \(.owner.login)/\(.name)"'

# Current branch
echo ""
echo "🌿 Current Branch:"
git branch --show-current
git log --oneline -1

# Your open PRs
echo ""
echo "📬 Your Open PRs:"
gh pr list --author @me --limit 5

# Open issues
echo ""
echo "📮 Open Issues:"
gh issue list --limit 5

# Sync status
echo ""
echo "🔄 Sync Status:"
git fetch upstream --quiet
echo "  Local:  $(git rev-parse --short HEAD)"
echo "  Upstream: $(git rev-parse --short upstream/main)"
