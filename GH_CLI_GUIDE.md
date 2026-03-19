# GitHub CLI (gh) Workflow Guide

## Setup gh CLI

```bash
# Install gh (if not already installed)
# macOS:
brew install gh

# Linux:
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Login
gh auth login
```

## Daily gh Commands

### Sync with Upstream (One-liner!)
```bash
# The easiest way - syncs your fork with upstream
gh repo sync nalediym/parameter-golf --source openai/parameter-golf
```

### View Repository Status
```bash
gh repo view                    # View your fork
gh repo view openai/parameter-golf # View upstream
gh repo list nalediym           # List your repos
```

### Work with Issues & PRs
```bash
gh issue list                   # List open issues
gh issue create                 # Create new issue
gh issue view <number>          # View specific issue

gh pr list                      # List pull requests
gh pr create                    # Create PR (from current branch)
gh pr checkout <number>         # Checkout a PR locally
gh pr view <number>             # View PR details
gh pr merge <number>            # Merge a PR
```

### Create PR from Current Branch
```bash
# Push branch and create PR in one go
git checkout -b feature/my-experiment
# ... make changes ...
git add -A
git commit -m "feat: add experiment"
git push origin feature/my-experiment

# Create PR with gh
gh pr create --title "Add experiment A1" --body "IPA baseline experiment"

# Or with web editor
gh pr create --web
```

### Check PR Status
```bash
gh pr status                    # Status of PRs related to current branch
gh pr checks                    # Check CI status
gh pr diff                      # View diff
```

### Fork Management
```bash
# View fork relationship
gh repo view nalediym/parameter-golf --json parent

# Set default repo (save typing!)
gh repo set-default nalediym/parameter-golf

# After setting default, just use:
gh pr list
gh issue list
```

## RunPod + gh CLI

When SSH'd into RunPod:

```bash
# Login to gh (one-time)
gh auth login --with-token <<< $GITHUB_TOKEN
# Or copy your ~/.config/gh/config.yml from local machine

# Quick sync before experiments
gh repo sync nalediym/parameter-golf --source openai/parameter-golf

# Push results back
git add experiments/run_logs/
git commit -m "results: RunPod batch 1"
git push origin runpod-batch-1

# Create PR if needed
gh pr create --title "Results: RunPod Experiments" --body "Baseline, IPA, and Morph results"
```

## Quick Reference

| Task | Git | GitHub CLI |
|------|-----|-----------|
| Clone | `git clone <url>` | `gh repo clone nalediym/parameter-golf` |
| Sync fork | `git fetch upstream && git merge` | `gh repo sync nalediym/parameter-golf` |
| Create PR | Go to GitHub | `gh pr create` |
| View PRs | Open browser | `gh pr list` |
| Checkout PR | Manual | `gh pr checkout <num>` |
| Create issue | Open browser | `gh issue create` |

## Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Quick sync
alias sync-params='gh repo sync nalediym/parameter-golf --source openai/parameter-golf'

# Quick PR
alias pr-create='gh pr create --fill'

# View my PRs
alias my-prs='gh pr list --author @me'

# Quick status
alias ghst='gh pr status && gh issue list'
```

## Advanced: Automation

```bash
#!/bin/bash
# runpod-experiment-and-push.sh

echo "Running experiments and pushing results..."

# Sync first
gh repo sync nalediym/parameter-golf --source openai/parameter-golf

# Run experiments
./runpod_tmux_runner.sh all

# Wait for completion
echo "Waiting for experiments..."
sleep 600

# Commit and push
git add experiments/run_logs/
git commit -m "results: $(date +%Y-%m-%d) RunPod experiments"
git push origin $(git branch --show-current)

# Create PR
gh pr create --title "Experiment Results $(date +%Y-%m-%d)" --fill

echo "✓ Done! Check: $(gh pr view --json url -q .url)"
```

## Authentication on RunPod

Option 1: Environment variable
```bash
export GITHUB_TOKEN="ghp_xxxxxxxx"
gh auth login --with-token <<< $GITHUB_TOKEN
```

Option 2: Copy config from local
```bash
# On local machine
cat ~/.config/gh/config.yml

# On RunPod, create the file
mkdir -p ~/.config/gh
cat > ~/.config/gh/config.yml << 'EOF'
github.com:
    user: nalediym
    oauth_token: ghp_xxxxxxxx
    git_protocol: https
EOF
chmod 600 ~/.config/gh/config.yml
```
