# What's Left To Do - Parameter Golf Experiments

## 🎯 Immediate Actions Required

### 1. Data Pipeline Completion
- [ ] Convert FineWeb to IPA format (80 shards) - `convert_fineweb_ipa.py`
- [ ] Convert FineWeb to morphological format - `experiments/morphological_tokenization/convert_fineweb_to_morph.py`
- [ ] Build compact IPA tokenizer (47 vocab) for official training
- [ ] Validate conversion quality on validation set

### 2. Run Pod Experiments
Once data is ready, run these in parallel with tmux:

```bash
# On your RunPod instance:
./runpod_tmux_runner.sh all
```

Experiments to run:
- [ ] **A0**: Baseline BPE-1024 (already exists, verify on your pod)
- [ ] **A1**: IPA-only (47 vocab) 
- [ ] **A3**: Morphological-only (morpheme vocab)
- [ ] **A5**: Hybrid - IPA base + morph features

### 3. Sign-Language-Inspired Tokenizer (Not Started)

From `TODO_sign_language_inspired_tokenization.md`:

**Phase 1 - Core Implementation:**
- [ ] Define token record with `base_id` + feature IDs
- [ ] Implement IPA base channel
- [ ] Implement `ORTH_HINT` feature channel
- [ ] Implement `MORPH_ROLE` feature channel
- [ ] Build preprocessing pipeline

**Phase 2 - Model Integration:**
- [ ] Implement channel embeddings
- [ ] Additive fusion: `e_total = e_base + e_features`
- [ ] Test concat+projection fusion
- [ ] Parameter budget tracking

**Phase 3 - Experiments:**
- [ ] A1: IPA-only base
- [ ] A2: IPA + ORTH_HINT
- [ ] A3: morph-only base
- [ ] A4: morph + MORPH_ROLE
- [ ] A5: hybrid (IPA + MORPH_ROLE + ORTH_HINT)
- [ ] A6: full feature bundle

### 4. Evaluation & Probes
- [ ] Build homophone probe set (night/knight, to/too/two)
- [ ] Build morphological-family probe (run/running/runner/ran)
- [ ] Build clause-function probes (questions, negation)
- [ ] Track bits-per-byte on validation
- [ ] Measure embedding memory overhead

### 5. Submission Prep
- [ ] Compress to 16MB (code + model)
- [ ] Verify <10min runtime on 8xH100
- [ ] Create README.md for best result
- [ ] Create submission.json
- [ ] Submit PR to /records folder

## 📊 Experiment Matrix

| Exp | Base Channel | Features | Vocab Size | Expected Impact |
|-----|--------------|----------|------------|-----------------|
| A0 | BPE-1024 | None | 1024 | Baseline: ~1.22 bpb |
| A1 | IPA | None | ~47 | 92% embedding savings |
| A2 | IPA | ORTH_HINT | ~47 + hints | Disambiguation help |
| A3 | Morph | None | ~1000 | Structured tokens |
| A4 | Morph | MORPH_ROLE | ~1000 + roles | Syntax awareness |
| A5 | IPA | MORPH_ROLE + ORTH_HINT | ~47 + features | Best of both |

## 🚀 Next Steps (Priority Order)

1. **Today**: 
   - SSH into RunPod
   - Run `setup_runpod_experiments.sh`
   - Start converting FineWeb to IPA

2. **Tomorrow**:
   - Finish data conversion
   - Launch A0 (baseline) to verify setup
   - Launch A1 (IPA) experiment

3. **This Week**:
   - Complete A3 (morph)
   - Analyze results
   - Decide if A5 (hybrid) is worth building

4. **Next Week**:
   - Build sign-language features if A5 promising
   - Run full ablation study (A6, A7)
   - Prepare best submission

## 🔧 Using Your RunPod

```bash
# Connect
ssh root@<your-pod-ip>

# Quick start
cd /workspace/parameter-golf
./runpod_tmux_runner.sh all

# Monitor
tmux ls
tail -f experiments/run_logs/*.log
```

See `RUNPOD_TMUX_GUIDE.md` for full reference.
