# RunPod + tmux Quick Reference

## Setup Steps (One-time on RunPod)

```bash
# SSH into your RunPod instance
ssh root@<your-pod-ip>

# Run setup with YOUR FORK
cd /workspace
curl -fsSL https://raw.githubusercontent.com/nalediym/parameter-golf/main/setup_runpod_experiments.sh | bash

# Or manually (clones your fork, not openai's):
git clone https://github.com/nalediym/parameter-golf.git /workspace/parameter-golf
cd /workspace/parameter-golf
bash setup_runpod_experiments.sh
```

## Running Experiments with tmux

### Option 1: Use the runner script (Recommended)
```bash
# Launch all experiments in parallel
./runpod_tmux_runner.sh all

# Or launch specific sets
./runpod_tmux_runner.sh baseline    # Just baseline BPE-1024
./runpod_tmux_runner.sh ipa         # IPA experiments A1, A2, A5
./runpod_tmux_runner.sh morph       # Morphological A3, A4
./runpod_tmux_runner.sh sign        # Sign-language experiments
```

### Option 2: Manual tmux sessions
```bash
# Create a new session
tmux new-session -d -s baseline_sp1024
tmux send-keys -t baseline_sp1024 'RUN_ID=baseline torchrun --standalone --nproc_per_node=8 train_gpt.py' Enter

# Create another for IPA
tmux new-session -d -s ipa_A1
tmux send-keys -t ipa_A1 './experiments/run_experiment_A1.sh' Enter
```

## Monitoring Sessions

```bash
# List all sessions
tmux ls

# Attach to a session
tmux attach -t baseline_sp1024

# Detach from session (while attached)
Ctrl+b, then d

# View logs without attaching
tail -f experiments/run_logs/*.log

# Check all running experiments
watch -n 5 'tmux ls && echo "--- Logs ---" && ls -lh experiments/run_logs/'
```

## Killing Sessions

```bash
# Kill specific session
tmux kill-session -t <session-name>

# Kill all sessions
tmux kill-server
```

## Expected Timeline

| Experiment | GPUs | Duration | Output |
|------------|------|----------|--------|
| baseline_sp1024 | 8xH100 | ~10 min | ~1.22 bpb |
| A1 (IPA-only) | 8xH100 | ~10 min | TBD |
| A3 (morph-only) | 8xH100 | ~10 min | TBD |
| A5 (hybrid) | 8xH100 | ~10 min | TBD |

## Data Preparation (Run once)

Before running experiments, you need to convert FineWeb:

```bash
# For IPA experiments
python3 convert_fineweb_ipa.py --shards 80

# For morph experiments  
cd experiments/morphological_tokenization
python3 convert_fineweb_to_morph.py
```

## File Locations on RunPod

```
/workspace/parameter-golf/
├── experiments/
│   ├── run_logs/          # All training logs
│   ├── run_artifacts/     # Model checkpoints
│   └── results/            # Summary JSONs
├── data/
│   ├── datasets/
│   │   ├── fineweb10B_sp1024/     # Standard BPE
│   │   ├── fineweb10B_ipa/        # IPA converted
│   │   └── fineweb10B_morph_v2/   # Morpheme converted
│   └── tokenizers/
└── records/               # Your submissions go here
```

## tmux Cheatsheet

| Command | Action |
|---------|--------|
| `Ctrl+b c` | Create new window |
| `Ctrl+b n` | Next window |
| `Ctrl+b p` | Previous window |
| `Ctrl+b "` | Split horizontally |
| `Ctrl+b %` | Split vertically |
| `Ctrl+b arrow` | Navigate panes |
| `Ctrl+b d` | Detach |
| `Ctrl+b [` | Scroll mode (q to exit) |
| `Ctrl+b z` | Zoom pane |

## Troubleshooting

**Problem**: "No space left on device" during training
**Solution**: 
```bash
docker system prune -a  # Clean Docker
du -sh /workspace/* | sort -hr  # Find large files
```

**Problem**: Training killed unexpectedly
**Solution**: Check OOM, reduce batch size or model size
```bash
dmesg | tail -50  # Check kernel messages
nvidia-smi        # Check GPU memory
```

**Problem**: Can't attach to tmux session
**Solution**: Session may have died, check logs
```bash
tmux ls
cat experiments/run_logs/<session>.log | tail -100
```
