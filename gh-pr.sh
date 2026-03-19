#!/bin/bash
# gh-pr.sh - Quick PR creation with gh CLI

set -e

# Get current branch
BRANCH=$(git branch --show-current)

# Don't create PR from main
if [ "$BRANCH" = "main" ]; then
    echo "❌ You're on main branch. Create a feature branch first:"
    echo "   git checkout -b feature/my-change"
    exit 1
fi

echo "=========================================="
echo "CREATE PULL REQUEST"
echo "=========================================="
echo "Branch: $BRANCH"
echo ""

# Check if PR already exists
EXISTING_PR=$(gh pr list --head "$BRANCH" --json number -q '.[0].number')

if [ -n "$EXISTING_PR" ]; then
    echo "⚠ PR already exists for this branch: #$EXISTING_PR"
    echo "   View it: gh pr view $EXISTING_PR"
    exit 0
fi

# Create PR with options
echo "Creating PR..."

# Option 1: Non-interactive (use commit messages)
# gh pr create --fill

# Option 2: Web browser (more control)
gh pr create --web

echo ""
echo "✓ PR created!"
