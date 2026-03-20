# TODOS

## Timer output clutter
**What:** The `rpod` countdown timer prints on new lines instead of overwriting in-place because training output on stdout interferes with `\r` carriage returns.
**Why:** Makes the terminal hard to read during training runs.
**Fix:** Move timer to terminal title bar (`\033]0;...\007`) or only print every 60s instead of every second.
**Priority:** Low — cosmetic, doesn't affect results.

## Phase 4: Full 8xH100 run
**What:** Run `./rpod train` on an 8xH100 pod for competition-grade bpb.
**Why:** 1xH100 gets ~1500 steps in 10 min. 8xH100 gets ~12,000 steps — needed for convergence.
**Depends on:** 1xH100 results showing IPA bpb trending in right direction.
**Cost:** ~$3.50 (10 min at $20/hr).

## Tests T6-T16
**What:** Build remaining test suite (shard integrity, training integration, bpb correctness).
**Why:** Competition scrutinizes tokenizer changes. Tests are proof of correctness.
**Status:** T1-T5 done (66/66 passing). T6-T16 not yet built.
**Priority:** Medium — needed before submission, not before experiments.

## BPE on IPA — hybrid tokenizer (HIGH PRIORITY)
**What:** Train a BPE/SentencePiece tokenizer on IPA-converted text instead of raw English. Common IPA sequences get merged into single tokens.
**Why:** Char-level IPA (current approach) has 2.3x sequence expansion which kills context. BPE merging would compress common IPA patterns back down:
```
Char IPA:  "ðə naɪt" → [ð, ə, ' ', n, a, ɪ, t]     7 tokens (too long)
BPE IPA:   "ðə naɪt" → [ðə, naɪt]                    2 tokens (compact!)
```
**Best of both worlds:**
- Short sequences (like BPE) — model sees more context
- Phonetic structure (like IPA) — "knight" and "night" share tokens
- Related words share subwords: "naɪt" / "naɪts" / "naɪtli"
- Common sounds merge: "ɪŋ" (-ing), "ʃən" (-tion), "ðə" (the)
**Implementation:**
1. Convert FineWeb to IPA text (already done)
2. Train SentencePiece on the IPA text with vocab_size=256 or 512
3. Use that tokenizer with train_gpt.py (standard BPE pipeline, just different alphabet)
**Depends on:** IPA conversion pipeline (done), SentencePiece training
**Priority:** HIGH — this is the most promising next experiment. Addresses the core weakness (long sequences) while keeping the core strength (phonetic structure).

## Research: int4 Quantization
**What:** Explore 4-bit quantization to fit larger models in 16MB budget.
**Why:** int4 = 4 bits/weight (16 values) vs int8 = 8 bits (256 values). Could free 6-8MB for 2x parameters or deeper architecture.
**Key techniques:**
- **GPTQ/AWQ** — post-training quantization with calibration
- **QAT** — quantization-aware training during the 10min window
- **NF4/FP4** — normalized/float4 formats (LLM.int8(), QLoRA)
- **Mixed precision** — int4 weights + int8/fp16 activations
- **Grouping** — per-channel or per-block scaling for better precision
**Papers:** LLM.int8() (Dettmers et al), GPTQ (Frantar et al), QLoRA (Dettmers et al)
**Risk:** 16 values might be too coarse for 10min training. Need to test if quality loss > parameter gain.
**Priority:** High — could be the differentiator for leaderboard.

## Research: int2 / Ternary / Binary Quantization
**What:** Explore extreme low-bit quantization (1-2 bits) to maximize parameter count in 16MB.
**Why:** int2 = 2 bits (4 values), binary = 1 bit (2 values). Could fit 20-40M parameters vs 10M with int8.
**Successful implementations:**
- **BinaryConnect / BinaryNet** (Courbariaux et al, 2015-2016) — First binary weights, {-1, +1}, trained MNIST/CIFAR successfully
- **Ternary Weight Networks** (Li et al, 2016) — {-1, 0, +1}, 2x compression vs binary, better accuracy
- **XNOR-Net** (Rastegari et al, 2016) — Binary weights + activations, efficient inference on mobile
- **BitNet** (Microsoft, 2023) — 1.58-bit (ternary with scaling), matches fp16 on large models, now in transformers
- **DBQ** (Differential Binary Quantization, 2023) — Learnable thresholds for binary/ternary
**Key insight:** Binary/ternary works better at scale. Small models (10M params) suffer more from quantization than large ones (1B+).
**For 16MB constraint:**
- Binary: ~80M parameters possible (but likely untrainable in 10min)
- Ternary: ~40M parameters (maybe viable?)
- int2 with learned thresholds: ~40M parameters, more flexible than ternary
**Risk:** Extreme quantization noise may prevent convergence in 10-minute training window.
**Priority:** Medium-High — high risk/high reward. Test after int4 experiments.
