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

## Research: Latent Space Language Model (HIGH RISK / HIGH REWARD)
**What:** Compress the token sequence into a smaller latent space before the transformer processes it. Borrowed from Stable Diffusion's approach: instead of attending over 2048 tokens, compress to ~256 latents, run the transformer on that, then decode back.
**Why:** Attention is O(n²). Processing 256 latents instead of 2048 tokens is 64x cheaper. This means you can see 4-8x more training data in 10 minutes OR use a much deeper model.
**Two approaches:**
```
FULL LATENT (like Stable Diffusion's VAE):
  2048 tokens → encoder (1-2 layers) → 256 latents →
  transformer (9 layers, cheap!) → decoder → 2048 predictions

FUNNEL TRANSFORMER (simpler, stays autoregressive):
  Layers 1-3:  2048 positions (full resolution)
  Layers 4-6:   512 positions (pool 4:1)
  Layers 7-9:   128 positions (pool 4:1)
  Then upsample back to 2048 for output
```
**Why this could win big:**
- 4-16x cheaper attention = more steps in 10 minutes
- OR: use the saved compute for much longer context (seq=8192?)
- Nobody in the competition is doing this — true innovation token
**Risks:**
- Autoregressive generation in latent space is unsolved for small models
- Encoder/decoder add parameters to the 16MB budget
- bpb calculation gets complicated (need to account for compression loss)
- Nobody has proven this works at 17M params with 10min training
**References:**
- Stable Diffusion / Latent Diffusion (Rombach et al, 2022) — proved latent space works for generation
- Funnel Transformer (Dai et al, 2020) — progressive pooling in transformers
- Perceiver (Jaegle et al, 2021) — cross-attention to fixed latent array
- MEGABYTE (Yu et al, 2023) — hierarchical transformer with local/global stages
**Implementation plan:**
1. Start with Funnel Transformer (simpler, no separate encoder/decoder)
2. Implement strided pooling between layer groups
3. Add upsampling before output projection
4. Test on IPA-BPE tokenized data
**Priority:** HIGH RISK / HIGH REWARD — the biggest potential differentiator. Try after seq=2048 + looping results are in. If those don't beat 1.19, this is the Hail Mary.

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

## Research: Evolutionary Architecture Search (Genetic NAS)
**What:** Use genetic algorithms to search hyperparameter space: depth, width, heads, LR, batch size, tie_weights.
**How it works:**
- Generation 0: Random architectures (population of 5-10)
- Train each for 10min, score by val_bpb
- Select parents → crossover + mutation → next generation
- Repeat for N generations overnight
**Search space:** Depth 6-12, Width 384-768, Heads 4-16, LR [1e-4, 1e-3], Batch size, Tie embeddings (bool)
**Pros:**
- No LLM API costs (pure compute)
- Parallelizable across GPUs (each individual = one GPU)
- Can find counter-intuitive combinations humans miss
- Fitness landscape is actual val_bpb, not proxy
**Cons:**
- **Expensive:** 10 experiments = 100 min = ~$33. 100 experiments overnight = ~$330
- **Novelty ceiling:** Only searches numeric hyperparams, can't invent new attention patterns, optimizers, or architectures
- **Local optima:** May converge to "safe" architectures, miss risky breakthroughs
- **Variance:** 10min runs have high variance — hard to distinguish signal from noise
- **Time vs autoresearch:** Same cost as autoresearch but narrower search space
- **Diminishing returns:** Baseline is already well-tuned by OpenAI; gains may be small
**Comparison to autoresearch:**
- Autoresearch: LLM agent can propose novel architectures (e.g., "try MoE layers with sliding window attention")
- Genetic NAS: Only tunes knobs within existing architecture family
**Hybrid approach:** Use autoresearch to explore architecture families, then genetic NAS to optimize within the best family.
**Priority:** Medium — interesting for hyperparameter tuning, but quant/tokenizer gains likely bigger.
