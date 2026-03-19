# What's Left To Do - Parameter Golf

## Current Best: 1.1925 bpb (sliding window eval on BPE baseline)

## Strategy (from eng review 2026-03-19)

Reduced scope: prove ONE tokenizer experiment end-to-end before expanding.
Go/no-go: IPA must beat 1.19 bpb to justify further work.

## Phase 0: Prep (on main) -- DONE
- [x] Port sliding window eval into main `train_gpt.py`
- [x] Consolidate duplicate IPA scripts
- [x] Update this file

## Phase 1: IPA Vocab + Converter (branch: exp/ipa-baseline)
- [ ] Build exact vocab inventory (~94 chars measured on FineWeb)
- [ ] Expand G2P exceptions to ~2000 words (from CMUdict, BSD licensed)
- [ ] Add passthrough logic: alpha words -> G2P, everything else -> char-by-char
- [ ] Tests T1-T5 (converter correctness)
- GATE: converter handles 10K random FineWeb words without crash

## Phase 2: Shard Conversion
- [ ] Convert 1 shard to uint8 IPA + byte_count sidecar
- [ ] Assertion: sum(byte_counts) == original shard byte count
- [ ] Tests T6-T9 (shard integrity)
- GATE: shard loads, all token IDs < vocab_size, byte counts match

## Phase 3: Training Smoke Test
- [ ] Parameterize `train_gpt.py` (TOKENIZER_TYPE=ipa)
- [ ] uint8 loader, vocab=~94, seq_len=2048
- [ ] bpb calc: total_nats / (original_bytes * ln(2))
- [ ] 50-step smoke test
- [ ] Tests T10-T14
- GATE: training completes, loss decreasing, no NaN

## Phase 4: Full Run + Eval
- [ ] Convert all 80 shards
- [ ] Full 10-min training on 8xH100
- [ ] Sliding window eval (EVAL_STRIDE=64)
- [ ] Tests T15-T16 (bpb sanity)
- GATE: bpb < 1.19 -> expand to A2 | bpb >= 1.19 -> stop and analyze

## Key Parameters (from measurement on real FineWeb data)
- IPA chars / BPE tokens: 2.30x
- IPA chars / original bytes: 0.94x
- Unique IPA chars (with passthrough): 94
- IPA seq=2048 context: ~2176 bytes (~870 words)
- Architecture: 9 layers, 512 dim, tied embeddings (same as baseline)

## Deferred (only if A1 beats 1.19 bpb)
- A2: IPA + ORTH_HINT feature channel
- A3: Morphological-only tokenizer
- A5: Hybrid IPA + morph
- Sign-language-inspired multi-channel embeddings
- Probe evaluation suites (homophone, morphological-family)
- Full ablation study (A6, A7)

## Submission Prep (after best experiment identified)
- [ ] Compress to 16MB (code + model)
- [ ] Verify <10min runtime on 8xH100
- [ ] Create README.md for best result
- [ ] Create submission.json
- [ ] Submit PR to /records folder

## RunPod Quick Reference
```bash
ssh root@<your-pod-ip>
cd /workspace/parameter-golf

# BPE baseline with sliding window eval
EVAL_STRIDE=64 EVAL_BATCH_SEQS=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# IPA experiment (after Phase 3)
TOKENIZER_TYPE=ipa EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

See `RUNPOD_TMUX_GUIDE.md` for full reference.
