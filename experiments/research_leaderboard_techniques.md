# Parameter Golf: Leaderboard Techniques Research

**Date:** 2026-03-19
**Branch:** exp/ipa-baseline
**Sources:** PRs #122, #128, #135, #150, #152 from openai/parameter-golf

---

## Current Leaderboard (top entries, val_bpb)

| PR | Author | BPB | Key Stack |
|----|--------|-----|-----------|
| #135 | unnir | 1.1539 | OrthoInit + Int6 + MLP3x + BigramHash + SmearGate |
| #122 | mtybadger | 1.1585 | Vocab2048 + NorMuon + Int6 + SWA + FA3 |
| #150 | yahya010 | 1.1593 | Int6 QAT + BigramHash + MLP1344 + 10 layers |
| #128 | rsavitt | 1.1594 | Int6 + STE QAT + MLP3x + Sliding Window |
| #152 | timowhite88 | 1.1744 | TTT (test-time training on val set) |

Baseline: ~1.227 BPB (NaiveBaseline, int8+zlib)

---

## 1. Int6 Quantization + STE QAT

**What it is:** Replace the baseline int8 per-row quantization with int6 (6-bit, range [-32, 31]). This saves ~25% on weight storage vs int8. The savings buy room for a larger model (e.g., 3x MLP) while staying under 16MB.

**STE (Straight-Through Estimator):** During training, weights are "fake-quantized" in the forward pass — rounded to the nearest int6 level and scaled back — but gradients flow through unchanged (as if quantization never happened). This is QAT (Quantization-Aware Training): the model learns to be robust to the quantization it will face at export time.

**Implementation (from PR #135):**

```python
class _FakeQuantizeInt6STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor) -> Tensor:
        w32 = w.float()
        abs_max = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = abs_max / 31.0
        q = torch.clamp(torch.round(w32 / scale), -32, 31)
        return (q * scale).to(w.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output  # <-- Straight-through: gradient passes as-is
```

Applied in `CastedLinear.forward()`:
```python
class CastedLinear(nn.Linear):
    qat: bool = False
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self.qat and self.training and w.ndim == 2:
            w = fake_quantize_int6_ste(w)  # fake-quantize during training only
        return F.linear(x, w, self.bias)
```

**Export quantization** (per-row int6 + zstd-22):
```python
def quantize_int6_per_row(t):
    row_max = t.abs().amax(dim=1)
    scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
    q = torch.clamp(torch.round(t / scale.float()[:, None]), -32, 31).to(torch.int8)
    return q, scale
```

**Key details:**
- Embeddings (tok_emb) and last 2 layers' K projections kept as fp16 (quantization-sensitive)
- zstd level 22 compresses better than zlib-9 on int6 data
- QAT adds ~54% step overhead (PR #135 disables it by default, PR #128 keeps it on)
- Quant gap: ~0.005 BPB with QAT, ~0.01 without

**Effort:** Medium. ~100 lines for quantize/dequantize + ~20 lines for STE autograd function. Requires `zstandard` pip package.

---

## 2. BigramHash Embedding

**What it is:** A hash-based lookup that injects bigram (token-pair) context into the residual stream at position 0, before any transformer layers. It captures which token came before the current one, giving the model local n-gram information for free.

**How it works (from PR #135):**

```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        self.bigram_vocab_size = bigram_vocab_size  # 4096 buckets
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)  # 4096 x 128
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)  # 128 -> 512
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1  # 4095
        out = torch.empty_like(t)
        out[..., 0] = mod  # BOS gets special bucket
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))  # (bsz, seqlen, 128)
        h = self.proj(h)  # (bsz, seqlen, 512)
        return h * self.scale
```

**Used in GPT forward:**
```python
x = self.tok_emb(input_ids)
if self.bigram is not None:
    x = x + self.bigram(input_ids)  # add bigram signal to unigram embedding
x = F.rms_norm(x, (x.size(-1),))
```

**Mechanism:** Hash(prev_token XOR curr_token) maps each consecutive pair into one of 4096 buckets. The embedding is 128-dim, projected to 512-dim, initialized to zero and scaled by 0.05, so it starts as a small residual perturbation that grows during training.

**Parameter cost:** 4096 * 128 (embed) + 128 * 512 (proj) + 1 (scale) = ~590K params. At int6, that's roughly 440KB.

**Effort:** Low. ~40 lines of code. Clean addition, no changes to transformer blocks.

---

## 3. NorMuon Optimizer

**What it is:** A variant of Muon that adds per-row adaptive scaling via a second moment estimate. Standard Muon orthogonalizes the gradient via Newton-Schulz iteration. NorMuon adds an Adam-like second moment on top.

**Key difference from Muon (from PR #122):**

```python
def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # Step 1: Standard Muon momentum + Nesterov
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # Step 2: Standard Newton-Schulz orthogonalization (same as Muon)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Step 3: NorMuon addition — per-row adaptive scaling
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)  # EMA of squared updates per row
    step_size = 1 / second_momentum.sqrt().add_(1e-10)  # Adam-like 1/sqrt(v)
    update.mul_(step_size)
    # Renormalize to preserve original update norm (scale-only adjustment)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update
```

**What NorMuon adds over Muon:**
1. Tracks `second_momentum` — an EMA of the per-row squared update values (like Adam's v_t)
2. Scales each row by `1/sqrt(second_momentum)`, giving rows with consistently large updates a smaller step
3. Then renormalizes so the overall update magnitude is preserved
4. Net effect: rows that oscillate or have high variance get damped; stable rows get amplified. This is per-row preconditioning on top of the orthogonalized gradient.

**Extra state:** One extra buffer per parameter of shape `(..., 1)` — the per-row second moment.

**Effort:** Medium. ~30 lines to modify the Muon update function, plus a new `NorMuon` optimizer class. The distributed communication pattern changes slightly (uses `all_gather` instead of `all_reduce`).

---

## 4. OrthoInit / Spectral Embed Init

**What it is (from PR #135):** All large weight matrices (Q, K, V, fc, proj) are initialized with `nn.init.orthogonal_(gain=1.0)` instead of the default Kaiming/Xavier. Output projections (attn.proj, mlp.proj) are additionally scaled by `1/sqrt(2 * num_layers)` following muP (maximal update parameterization).

```python
def _init_weights(self):
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)  # 0.005
    num_layers = len(self.blocks)
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)  # output projections start at zero
            elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if ".proj." in name or name.endswith(".proj"):
                    module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```

**Why it helps:**
- Orthogonal init means the weight matrix starts with all singular values = 1 (perfectly conditioned)
- No gradient signal is wasted on correcting bad conditioning in early training
- Muon's Newton-Schulz iteration converges faster when starting from near-orthogonal matrices
- The `1/sqrt(2*layers)` scaling on projections prevents residual stream norm explosion with depth

**Tied embedding init:** Uses `std=0.005` (very small), separate from the orthogonal scheme.

**Effort:** Low. ~15 lines in `_init_weights()`. Zero runtime cost.

---

## 5. MLP 3x Expansion

**What it is:** MLP hidden dimension = 3 * model_dim (1536) instead of the baseline 2 * model_dim (1024).

**Tradeoff:** More parameters per layer, but the int6 quantization savings (vs int8) free enough byte budget to accommodate them. The MLP is a relu^2 activation:

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, mlp_hidden=0):
        hidden = mlp_hidden if mlp_hidden > 0 else int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())  # relu^2
```

**Budget math:**
- Baseline MLP per layer: 512 * 1024 * 2 = 1,048,576 params -> int8 = ~1MB/layer
- 3x MLP per layer: 512 * 1536 * 2 = 1,572,864 params -> int6 = ~1.18MB/layer (but 6/8 of int8 rate)
- Net: 3x MLP at int6 uses ~1.18MB vs baseline 2x MLP at int8 using ~1.0MB. Only 18% more bytes for 50% more capacity.

**Some entries use MLP_HIDDEN=1344 (PR #150) instead of 1536** — a compromise that allows 10 layers instead of 9 while staying under 16MB.

**Effort:** Trivial. Change one hyperparameter: `MLP_MULT=3`.

---

## 6. SmearGate

**What it is (from PR #135):** A learned gate that blends each token's embedding with the previous token's embedding, applied right after the token embedding layer and before the first transformer block.

```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Parameter(torch.zeros(dim))  # 512 params, starts at 0

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]  # (1, 1, dim)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```

**How it works:** Each dimension of the embedding independently learns how much to "smear" from the previous position. Initialized at zero (sigmoid(0) = 0.5), so it starts by averaging current and previous tokens equally. Over training, some dimensions learn to rely heavily on the previous token's embedding and others learn to keep the current one.

**Why it helps:** Gives the model a cheap unigram-to-bigram transition signal before attention even runs. Combined with BigramHash, it creates two complementary sources of local context: SmearGate operates in embedding space (smoothing), BigramHash operates via explicit pair hashing (discrete lookup).

**Parameter cost:** 512 floats = negligible.

**Effort:** Trivial. ~10 lines.

---

## 7. TTT (Test-Time Training)

**What it is (from PR #152):** After normal training completes and the model is quantized/exported, the eval phase re-trains the full model on the validation set before scoring it. This exploits the competition's 10-minute eval budget.

**Implementation:**

```python
def ttt_adapt(model, device, val_tokens, seq_len,
              lr=3e-4, momentum=0.95, epochs=1, batch_size=32, log_fn=None):
    """Adapt model weights on validation data via SGD."""
    total_seqs = (val_tokens.numel() - 1) // seq_len
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    model.train()
    for epoch in range(epochs):
        for batch_start in range(0, total_seqs, batch_size):
            batch_end = min(batch_start + batch_size, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
    for p in params:
        p.requires_grad_(False)
```

**Key details:**
- NOT LoRA. This is full-model SGD with momentum on ALL parameters.
- Runs 1-2 epochs over the entire validation set (next-token prediction loss).
- Aggressive config: lr=0.002, momentum=0.9, 2 epochs = 3% BPB improvement.
- Takes ~311s of the eval budget, leaving ~38s for sliding window eval.
- The model is essentially "fine-tuned" on the test distribution.
- Only runs on rank 0, then broadcasts weights to all ranks.

**Is this cheating?** The competition rules seem to allow it — the 10-minute budget is shared between training and eval, and the validation set is publicly known. It is analogous to adaptive compression (like how Lempel-Ziv adapts to the data being compressed). However, it may face scrutiny in future rule updates.

**Results:** Static BPB 1.2087 -> TTT BPB 1.1744 (2.84% gain at zero parameter cost).

**Effort:** Low-medium. ~40 lines for the adapt function + ~20 lines to wire it into the eval pipeline. But the BPB gain is modest compared to other techniques, and it is architecturally simpler than the top entries.

---

## 8. Vocab Size > 1024

**What it is (from PR #122):** Increase vocabulary from 1024 to 2048 (or 4096, tried in earlier PR #87) with a custom-trained SentencePiece tokenizer.

**Why it helps:**
- Larger vocab = more bytes per token on average = better BPB (bits per byte)
- BPB = (bits_per_token) * (tokens_per_byte). If tokens encode more bytes each, you need fewer bits per byte to represent the same information.
- The tokenizer is trained on the same FineWeb data distribution, so it captures common subwords efficiently.

**The tradeoff:**
- Embedding table: vocab_size * model_dim. At vocab=2048, that is 2048 * 512 = 1M params (vs 512K at vocab=1024).
- With tied embeddings in fp16, that is 2MB (vs 1MB). This extra 1MB must come from somewhere.
- PR #122 drops a layer (8 layers instead of 9) to compensate: losing ~2.5M MLP+attn params to gain ~0.5M embedding params.
- Embeddings are kept in fp16 (not quantized) because they are very sensitive to quantization noise.

**Custom tokenizer training:**
```bash
./data/download_hf_docs_and_tokenize.py \
  --output-root ./data \
  --tokenizer-config ./data/tokenizer_specs.json \
  --max-train-tokens 8000000000 \
  --tokenizer-train-docs 100000
```

Tokenizers available at: https://huggingface.co/sproos/parameter-golf-tokenizers

**Results at vocab=2048:** 1.1585 BPB (competitive, but the layer tradeoff is painful).

**Effort:** Medium-high. Requires training new tokenizers, re-tokenizing all data shards, and tuning model architecture to fit the budget. Not a drop-in change.

---

## Technique Interaction Matrix

| Technique | Saves Bytes | Costs Bytes | BPB Improvement | Composability |
|-----------|-------------|-------------|-----------------|---------------|
| Int6 + zstd | ~4MB vs int8+zlib | 0 | ~0.01 (quant gap) | Core enabler |
| STE QAT | 0 | 54% train time | ~0.005 (less quant gap) | Pairs with int6 |
| MLP 3x | 0 | ~2MB (offset by int6) | ~0.01-0.02 | Needs int6 savings |
| BigramHash | 0 | ~440KB | ~0.005-0.01 | Independent |
| SmearGate | 0 | ~2KB | ~0.002-0.005 | Independent |
| OrthoInit | 0 | 0 | ~0.005 (early convergence) | Independent |
| NorMuon | 0 | +state memory | ~0.005 | Replaces Muon |
| Sliding Window Eval | 0 | ~200s eval time | ~0.01-0.02 | Independent |
| TTT | 0 | ~300s eval time | ~0.03 | Competes for eval budget |
| Vocab 2048+ | ~0 | 1+ layers lost | ~0.005-0.01 | Complex tradeoff |
| SWA | 0 | Warmdown memory | ~0.002-0.005 | Independent |

---

## Implementation Priority (for our IPA baseline)

### Tier 1: Do First (high impact, low risk)
1. **Int6 + zstd-22** — Core byte savings, enables everything else
2. **MLP 3x** — One-line change, big capacity gain
3. **Sliding Window Eval** (stride=64) — Already have this, ~0.015 BPB free
4. **OrthoInit** — 15 lines, zero runtime cost

### Tier 2: Do Next (medium impact, medium effort)
5. **STE QAT** — Reduces quant gap, but costs train throughput
6. **BigramHash** — Clean ~40-line addition, small but consistent gain
7. **SmearGate** — Trivial to add, small gain

### Tier 3: Experimental (high effort or risky)
8. **NorMuon** — Replaces optimizer, needs careful tuning
9. **TTT** — Ethical gray area, may be disallowed
10. **Vocab 2048+** — Requires new tokenizer pipeline, arch changes

### Hyperparameter Consensus (from top entries)
- `matrix_lr=0.02` (halved from baseline 0.04)
- `scalar_lr=0.02`
- `muon_momentum=0.99` (up from 0.95)
- `warmdown_iters=3000` (up from 2500)
- `grad_clip_norm=0.3`
- `weight_decay=0.01`
- `train_batch_tokens=786432` (up from 524288)
- `train_seq_len=2048` (up from 1024)

---

## Open Questions

1. Can Int6 + BigramHash + SmearGate compose with our IPA tokenization approach?
2. Is there room for int4 on certain layers? (see research_int4_quantization.md)
3. Can TTT be combined with int6 + MLP3x for a dominant stack?
4. SWA (stochastic weight averaging during warmdown) — PR #122 uses it, ~7 checkpoints averaged. Worth investigating.
5. FlashAttention 3 — PR #122 reports ~10ms/step savings. Free if available.
