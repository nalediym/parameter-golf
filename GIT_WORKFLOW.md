# Git Workflow Guide - Your Parameter Golf Fork

## Current Setup (Verified ✓)

Your remotes are already configured correctly:
```
origin              https://github.com/nalediym/parameter-golf.git (your fork)
private-experiments https://github.com/nalediym/parameter-golf-private-experiments.git
upstream            https://github.com/openai/parameter-golf.git (official repo)
```

## Daily Workflow

### 1. Start Your Day - Sync with Upstream
```bash
./sync_with_upstream.sh
# Or manually:
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

### 2. Create Feature Branch
```bash
git checkout main
git pull origin main
git checkout -b feature/experiment-A1-ipa
```

### 3. Make Changes & Commit
```bash
# Edit files...
git add -A
git commit -m "feat: add IPA-only baseline experiment A1"
git push origin feature/experiment-A1-ipa
```

### 4. Create Pull Request (if needed)
```bash
# Push to your fork, then open PR on GitHub if you want upstream to see it
# For private experiments, just push to private-experiments remote
git push private-experiments feature/experiment-A1-ipa
```

## RunPod Git Workflow

When on RunPod, your repo will clone YOUR FORK:

```bash
ssh root@<your-pod-ip>
cd /workspace

# This clones YOUR FORK automatically
git clone https://github.com/nalediym/parameter-golf.git
cd parameter-golf

# Work on your branch
git checkout -b runpod/experiment-batch-1

# ... run experiments ...

# Push results back to your fork
git add experiments/run_logs/
git commit -m "results: baseline + IPA experiments on RunPod"
git push origin runpod/experiment-batch-1
```

## Clean Git Commands

```bash
# See all branches
git branch -a

# See recent commits
git log --oneline -10

# Check what's changed
git status
git diff

# Undo uncommitted changes
git checkout -- <file>      # single file
git checkout -- .            # all files

# Fix last commit
git commit --amend

# Stash work temporarily
git stash
git stash pop
```

## Smooth Git Setup - One Time

```bash
# Set your identity (if not already set)
git config --global user.name "nalediym"
git config --global user.email "your@email.com"

# Set default branch name
git config --global init.defaultBranch main

# Enable helpful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.lg "log --oneline --graph --decorate"
```

## Troubleshooting

### "fatal: refusing to merge unrelated histories"
```bash
git merge upstream/main --allow-unrelated-histories
```

### "Your branch is behind origin/main"
```bash
git checkout main
git pull origin main
git checkout your-branch
git rebase main
```

### "Permission denied" when pushing
```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:nalediym/parameter-golf.git

# Or generate a Personal Access Token on GitHub
```

### Want to start fresh on RunPod
```bash
ssh root@<your-pod-ip>
rm -rf /workspace/parameter-golf
git clone https://github.com/nalediym/parameter-golf.git /workspace/parameter-golf
```

## Current Branch Status

You're on: `private-share-update`

Main branches:
- `main` - Your fork's main branch (sync with upstream)
- `private-share-update` - Your current working branch

Remote branches available:
- `origin/main` - Your fork on GitHub
- `upstream/main` - OpenAI's official repo
- `private-experiments/main` - Your private experiments repo

## Next Steps

1. **Sync with upstream**: Run `./sync_with_upstream.sh`
2. **Commit your new files**: 
   ```bash
   git add -A
   git commit -m "setup: add RunPod experiment scripts"
   git push origin private-share-update
   ```
3. **Switch to main**: `git checkout main`
4. **Create feature branches** for each experiment

All scripts are now configured to use YOUR FORK (nalediym/parameter-golf) ✓
