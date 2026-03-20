# Research: int4 Quantization for Parameter Golf

**Date:** 2026-03-19
**Status:** Research only -- no code changes
**Goal:** Determine if int4 quantization can improve leaderboard score by fitting a larger model in the 16MB budget

---

## Current Baseline Numbers

| Metric | Value |
|---|---|
| Model params | 17,059,912 |
| int8 payload bytes | 17,178,912 |
| int8+zlib compressed | 15,816,489 bytes |
| Code size | ~58,340 bytes |
| Total artifact | 15,874,829 bytes |
| Headroom to 16MB cap | ~125 KB |
| Pre-quant val_bpb | 1.2196 |
| Post-quant val_bpb (int8) | 1.1925 (with sliding window) |
| int8 quant gap (non-sliding) | ~0.007 bpb (naive baseline), ~0.0005 bpb (fp16 embed trick) |

Architecture: 9 layers, d_model=512, 8 heads / 4 KV heads, MLP_MULT=2, vocab=1024, tied embeddings.

## The int4 Proposition

**Thesis:** int4 weights use 4 bits/param vs int8's 8 bits/param. If we switch to int4 post-training quantization:
- Payload drops from ~17MB to ~8.5MB (plus scale overhead)
- With zlib, compressed artifact might drop from ~15.8MB to ~8-9MB
- Freed ~7MB allows roughly doubling model size: d_model=768 or 18 layers
- **Question:** Does the quality gain from 2x parameters exceed the quality loss from coarser quantization?

### Size math (estimated)

| Config | Params | int8 payload | int4 payload* | int4+zlib est. |
|---|---|---|---|---|
| Current (9x512) | 17M | 17.2MB | 8.9MB | ~8-9MB |
| 2x wider (9x768) | ~38M | 38MB (too big) | ~19.5MB (too big) | ~16MB (borderline) |
| 1.5x wider (9x640) | ~26M | 26MB (too big) | ~13.5MB | ~12MB |
| Deeper (18x512) | ~33M | 33MB (too big) | ~17MB (too big) | ~14MB (borderline) |
| 12x576 | ~23M | 23MB (too big) | ~12MB | ~10-11MB |

*int4 payload = params/2 bytes (packed) + scale overhead (~2 bytes per group of 32-128 weights)

Sweet spot appears to be around 22-26M params with int4, fitting models that are impossible at int8.

---

## Technique 1: GPTQ (Post-Training Quantization with Calibration)

### How it works
GPTQ (Frantar et al., 2022) is an approximate second-order method for post-training quantization. It processes weight matrices column-by-column, using a Hessian approximation (from a small calibration set) to decide which values to round up vs down, minimizing the layer-wise reconstruction error. Typically achieves near-lossless int4 on 7B+ models.

### Feasibility for Parameter Golf

| Criterion | Assessment |
|---|---|
| Single-file implementation | Moderate difficulty. Core algorithm is ~100-150 lines, but needs calibration pass infrastructure. The Hessian computation requires running forward passes through the model on calibration data. |
| 10-min training window | GPTQ itself runs in seconds-to-minutes on small models. The concern is that it eats into training time. For a 17-26M param model, GPTQ would take <30 seconds -- negligible. |
| Quality vs int8 | For 7B+ models, GPTQ int4 loses ~0.01-0.05 perplexity points vs fp16. For our tiny 17M model, the picture is worse: small models have less redundancy, so quantization hurts more. Expect 0.02-0.10 bpb degradation vs int8. |
| PyTorch native support | No native GPTQ in PyTorch. `torch.ao.quantization` has some int4 support but not GPTQ specifically. The `auto-gptq` library exists but is a heavy dependency. |

### Key concern
GPTQ was designed for large models (7B+) where weight distributions are smooth and redundant. At 17-26M parameters, individual weights carry more information. The column-wise optimization may not compensate well enough.

### Verdict: POSSIBLE but risky for small models. Worth trying if combined with a larger architecture.

---

## Technique 2: QAT (Quantization-Aware Training)

### How it works
QAT inserts fake-quantization nodes during training so the model learns to be robust to quantization noise. Forward pass uses quantized-then-dequantized weights; backward pass uses straight-through estimator (STE) to pass gradients through the rounding.

### Existing infrastructure
The codebase already has int8 QAT implemented (disabled by default):
- `fake_quantize_int8_per_row()` in `train_gpt.py` (records version)
- `CastedLinear._qat` flag
- Toggle via `QAT=1` environment variable
- Previous finding (FP16Embed submission): "QAT: tried both full-training and late-stage. The overhead per step wasn't worth the small quant gap reduction."

### int4 QAT adaptation
Converting the existing int8 QAT to int4 is straightforward:
```python
def fake_quantize_int4_per_group(w, group_size=32):
    # Reshape to groups
    w_grouped = w.reshape(-1, group_size)
    scale = w_grouped.detach().abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp_(min=1.0 / 7.0)
    w_deq = (w_grouped / scale).round().clamp_(-8, 7) * scale
    return w + (w_deq.reshape_as(w) - w).detach()  # STE
```

### Feasibility for Parameter Golf

| Criterion | Assessment |
|---|---|
| Single-file implementation | Easy -- ~15 lines on top of existing QAT code. Just change the clamp range and add grouping. |
| 10-min training window | This is the critical issue. int8 QAT was already found to hurt net quality because the per-step overhead reduces total training steps. int4 QAT would be even worse -- the quantization noise is higher, so the model needs MORE steps to converge, not fewer. The 10-min cap is brutal here. |
| Quality vs int8 | With enough training time, QAT can close most of the int4 gap. But "enough time" for int4 QAT is typically 10-50% of original training -- we don't have that budget when we're already time-constrained. |
| PyTorch native support | `torch.ao.quantization` has `FakeQuantize` modules. `torchao` (PyTorch AO library) has int4 QAT support as of 2024. However, adding torchao as a dependency may not be available in the RunPod environment. |

### Late-stage QAT variant
Train normally for 9 minutes, then enable int4 QAT for the final 1 minute. This minimizes the step-count penalty while still adapting weights to int4 rounding. This is more promising than full-training QAT.

### Verdict: BEST APPROACH for this competition, specifically late-stage QAT. The existing infrastructure makes it low-effort to try.

---

## Technique 3: NF4 / FP4 (Normalized Float4)

### How it works
NF4 (Normal Float 4-bit, from QLoRA / Dettmers et al. 2023) uses a non-uniform 4-bit format where the 16 representable values are chosen to match the expected normal distribution of neural network weights. Instead of uniform spacing (int4: -8 to 7), NF4 places values at quantiles of N(0,1), giving better coverage of the weight distribution.

FP4 uses a 4-bit floating point format (1 sign, 2 exponent, 1 mantissa or similar).

### NF4 quantile values (for reference)
```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]
```

### Feasibility for Parameter Golf

| Criterion | Assessment |
|---|---|
| Single-file implementation | Moderate. NF4 encoding/decoding requires a lookup table (~30 lines). The tricky part is efficiently packing 4-bit values into bytes for storage. |
| 10-min training window | NF4 is a post-training format (used with QLoRA for fine-tuning frozen weights). No training overhead. Quantization takes seconds. |
| Quality vs int8 | NF4 is theoretically optimal for normally-distributed weights. For our model, expect ~0.5-2% better reconstruction error vs uniform int4 (per the QLoRA paper). This matters most for layers where weight distributions are approximately normal. |
| PyTorch native support | `bitsandbytes` library has NF4 support. `torchao` has some 4-bit support. No native PyTorch op. |

### Implementation note
The biggest challenge is the bit-packing for storage. Two int4 values must be packed into one byte. PyTorch has no native int4 dtype, so you store as uint8 with manual pack/unpack:
```python
packed = (high_nibble << 4) | (low_nibble & 0x0F)  # two int4 -> one uint8
```

### Verdict: GOOD post-training option. NF4 is slightly better than uniform int4 for quality, and the lookup-table approach is clean to implement.

---

## Technique 4: Mixed Precision (int4 weights + higher precision activations/key tensors)

### How it works
Not all model parameters benefit equally from higher precision. The idea is to use int4 for the bulk of weights (attention projections, MLP matrices) while keeping critical tensors at higher precision:
- Embedding/output head: fp16 (already done in FP16Embed submission)
- LayerNorm/RMSNorm parameters: fp32 (already done -- these are "control tensors")
- Attention scale parameters: fp32 (already done)
- First and last transformer layers: int8 instead of int4

### Feasibility for Parameter Golf

| Criterion | Assessment |
|---|---|
| Single-file implementation | Easy -- the existing `quantize_state_dict_int8` already has the infrastructure for per-tensor routing. Just add an int4 path alongside int8. |
| 10-min training window | No training overhead -- this is a post-training serialization choice. |
| Quality vs int8 | This is strictly better than uniform int4 everywhere. The question is how much headroom it uses vs pure int4. Typical overhead: keeping embeddings at fp16 costs ~1MB for a 1024x512 embedding. |
| PyTorch native support | Fully manual, same as current int8 approach. |

### Concrete size estimate for current architecture (9x512)

| Component | Params | int8 size | int4+scales size | fp16 size |
|---|---|---|---|---|
| tok_emb (1024x512) | 524K | 524KB | 262KB + 33KB = 295KB | 1,048KB |
| 9x attn QKV (512x384) | 1.77M | 1.77MB | 885KB + 55KB = 940KB | 3.54MB |
| 9x attn out (512x512) | 2.36M | 2.36MB | 1.18MB + 74KB = 1.25MB | 4.72MB |
| 9x MLP up (512x1024) | 4.72M | 4.72MB | 2.36MB + 148KB = 2.51MB | 9.44MB |
| 9x MLP down (1024x512) | 4.72M | 4.72MB | 2.36MB + 148KB = 2.51MB | 9.44MB |
| Small tensors (norms, etc.) | ~17K | fp16: ~34KB | fp16: ~34KB | ~34KB |
| **Total payload** | **~17M** | **~17.2MB** | **~9.6MB** | ~34MB |

With int4 weights + fp16 embedding: payload ~9.9MB, after zlib ~8-9MB.
This leaves ~7MB of headroom for a bigger model.

### Verdict: ESSENTIAL. Any int4 scheme should use mixed precision. The existing code structure makes this natural.

---

## Technique 5: Per-Group / Per-Channel Scaling

### How it works
The current int8 scheme uses per-row scales (one fp16 scale per output row of a weight matrix). For int4, per-row scaling is often too coarse because 4 bits can only represent 16 values -- if any part of the row has a very different range, the entire row suffers.

**Per-group scaling** divides each row into groups of G elements (typically G=32, 64, or 128) and assigns a separate scale factor per group. This captures local variations in weight magnitude.

Scale overhead: for a (M, N) matrix with group size G:
- int4 payload: M*N/2 bytes (packed)
- Scale payload: M * ceil(N/G) * 2 bytes (fp16 scales)
- Total: M*N/2 + 2*M*N/G bytes

### Group size tradeoff

| Group Size | Scale Overhead (% of int4) | Quality Impact |
|---|---|---|
| G=32 | +12.5% | Best quality, highest overhead |
| G=64 | +6.25% | Good balance |
| G=128 | +3.1% | Minimal overhead, more quality loss |
| Per-row (G=N=512) | +0.8% | Current int8 approach, worst for int4 |

### Feasibility for Parameter Golf

| Criterion | Assessment |
|---|---|
| Single-file implementation | Easy -- ~30 lines to reshape, quantize per group, and pack. |
| 10-min training window | No training overhead. |
| Quality vs int8 | Per-group int4 with G=64 typically achieves quality within 0.5-1.5% of int8 on models >1B params. For our tiny model, expect larger gaps. G=32 is safest. |
| PyTorch native support | No native int4 group quantization. Manual implementation required (same as now). |

### Implementation sketch
```python
def quantize_int4_per_group(t, group_size=64):
    # t: (M, N) float tensor
    M, N = t.shape
    N_padded = ((N + group_size - 1) // group_size) * group_size
    t_padded = F.pad(t, (0, N_padded - N))
    t_grouped = t_padded.reshape(M, -1, group_size)  # (M, num_groups, G)

    scale = t_grouped.abs().amax(dim=-1, keepdim=True) / 7.0  # (M, num_groups, 1)
    scale = scale.clamp_(min=1e-8)
    q = (t_grouped / scale).round().clamp_(-8, 7)  # int4 range

    # Pack two int4 values into one uint8
    q_uint8 = (q[..., 0::2].to(torch.uint8) << 4) | (q[..., 1::2].to(torch.uint8) & 0x0F)
    return q_uint8, scale.squeeze(-1).to(torch.float16)
```

### Verdict: ESSENTIAL complement to int4. G=64 is the recommended starting point for this model size.

---

## Comparison: What Existing Submissions/Projects Use

### Parameter Golf submissions
- **All current submissions use int8 + zlib.** No int4 submissions exist yet.
- **FP16Embed** (rank 2) keeps embedding in fp16 to reduce quant gap -- relevant insight for mixed-precision int4.
- **QAT was tried and rejected** for int8 because per-step overhead wasn't worth the small gap reduction. This is different for int4 where the gap is larger.

### modded-nanogpt project
- The modded-nanogpt speedrun optimizes for training time, not model size. No quantization is used in the competition itself.
- The codebase uses bf16 training throughout.

### Broader ML ecosystem
- **QLoRA** (Dettmers 2023): NF4 quantization of frozen base model + fp16 LoRA adapters. Very successful for fine-tuning. Not directly applicable (we're training from scratch).
- **BitNet** (Wang et al. 2023): 1-bit or 1.58-bit weights during training. Requires custom kernels and many more training steps.
- **AQLM / QuIP#**: Advanced vector quantization. Overkill for this competition and hard to implement in a single file.

---

## Recommended Strategy

### Phase 1: int4 Post-Training Quantization (lowest risk)
1. Implement `quantize_int4_per_group()` alongside existing int8 code
2. Use mixed precision: int4 for large weight matrices, fp16 for embeddings, fp32 for norms
3. Use group_size=64 for per-group scaling
4. Test on current 9x512 architecture first to measure the int4 quality gap
5. If gap is < 0.03 bpb, proceed to Phase 2

**Expected outcome:** ~0.01-0.03 bpb degradation vs int8, but model compresses to ~8-9MB.

### Phase 2: Scale Up Model
1. Increase model to use the freed space: try 12x576 (~23M params) or 9x640 (~26M params)
2. Deeper is generally better than wider for small models, so prefer more layers
3. Retrain from scratch with the larger architecture
4. Apply int4 post-training quantization
5. Compare val_bpb against the int8 baseline

**Expected outcome:** Larger model should improve pre-quant bpb by 0.05-0.15, int4 costs 0.01-0.03, net improvement of 0.02-0.12 bpb.

### Phase 3: Late-Stage QAT (if Phase 2 works)
1. Train the larger model normally for 9 minutes
2. Enable int4 fake quantization for the final 1 minute (~1000-1500 steps)
3. This should recover most of the int4 quality gap
4. Compare against Phase 2 results

**Expected outcome:** Recover 30-70% of the int4 quality gap, gaining 0.005-0.02 bpb.

### Phase 4: NF4 Exploration (optional)
1. Replace uniform int4 with NF4 lookup table
2. Test if the non-uniform distribution helps for this specific model
3. NF4 should be a small win over uniform int4

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| int4 quality loss exceeds parameter gain | HIGH | Test on current architecture first before scaling up. If int4 gap > 0.05 bpb on 17M model, abort. |
| zlib compresses int4 poorly (less entropy) | MEDIUM | int4 values have lower entropy (4 bits vs 8), so zlib should compress them well. But packed nibbles may have different compression characteristics. Test empirically. |
| Larger model doesn't converge in 10 min | MEDIUM | More parameters = slower steps + need more steps. 2x params might need 2x training time. May need to reduce batch size or use more aggressive LR scheduling. |
| Implementation bugs in bit-packing | LOW | The pack/unpack logic is finicky. Off-by-one in nibble ordering can silently corrupt weights. Need careful roundtrip tests. |
| Scale overhead eats the savings | LOW | With G=64, overhead is ~6% of int4 payload. Not a problem. |

---

## Quick Decision Framework

```
IF int4_gap_on_current_model < 0.03 bpb:
    -> Scale up model to fill 16MB budget with int4
    -> Expected net gain: 0.02-0.12 bpb (significant for leaderboard)

IF int4_gap_on_current_model is 0.03-0.06 bpb:
    -> Try late-stage QAT to close gap
    -> Scale up model moderately (1.3x, not 2x)
    -> Expected net gain: 0.00-0.05 bpb (marginal)

IF int4_gap_on_current_model > 0.06 bpb:
    -> Abort int4 approach for this model size
    -> int4 may only work at larger scales where there's more redundancy
    -> Stick with int8 + other improvements
```

---

## Next Step

Run a quick empirical test: take the current trained model checkpoint, apply int4 per-group quantization (G=64), dequantize, and evaluate. This gives us the actual int4 quality gap on our specific model without any training changes. If the gap is acceptable, proceed with Phase 2.

Implementation would be ~50-80 lines added to the quantization section of `train_gpt.py`, with no changes to the training loop.
