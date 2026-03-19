#!/usr/bin/env python3
"""
Train a small transformer on morpheme tokens using MLX.
Standalone script — no dependencies beyond mlx and numpy.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ==============================================================================
# CONFIG
# ==============================================================================

DATA_PATH = "data/morph_train_100k.bin"
VOCAB_PATH = "data/morph_vocab_v2.json"

NUM_LAYERS = 4
MODEL_DIM = 256
NUM_HEADS = 4
SEQ_LEN = 256
MLP_MULT = 2

LR = 3e-4
TRAIN_STEPS = 200
LOG_EVERY = 10

# ==============================================================================
# DATA
# ==============================================================================

def load_morpheme_data(path: str) -> np.ndarray:
    """Load raw uint16 morpheme tokens (no header)."""
    tokens = np.fromfile(path, dtype="<u2")
    print(f"Loaded {len(tokens):,} morpheme tokens from {path}")
    return tokens.astype(np.int32)


def make_batches(tokens: np.ndarray, seq_len: int):
    """Yield (x, y) pairs of shape [batch, seq_len] forever."""
    # Trim to multiple of seq_len+1 isn't quite right — we need
    # enough tokens for input + 1 target per sequence.
    n_seqs = (len(tokens) - 1) // seq_len
    usable = n_seqs * seq_len
    while True:
        # Shuffle sequence start positions each epoch
        indices = np.random.permutation(n_seqs)
        for idx in indices:
            start = idx * seq_len
            x = tokens[start : start + seq_len]
            y = tokens[start + 1 : start + seq_len + 1]
            yield mx.array(x[None, :], dtype=mx.int32), mx.array(y[None, :], dtype=mx.int32)

# ==============================================================================
# MODEL
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        hidden = dim * mult
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MorphGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int,
                 num_heads: int, mlp_mult: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = [Block(dim, num_heads, mlp_mult) for _ in range(num_layers)]
        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.tok_emb.weight

    def __call__(self, x: mx.array) -> mx.array:
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.lm_head(h)

    def loss(self, x: mx.array, y: mx.array) -> mx.array:
        logits = self(x)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
            y.reshape(-1),
            reduction="mean",
        )

# ==============================================================================
# TRAINING
# ==============================================================================

def count_params(model) -> int:
    """Count unique parameters (respecting tied weights)."""
    seen_ids = set()
    total = 0
    for name, p in tree_flatten(model.parameters()):
        pid = id(p)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        total += p.size
    return total


def main():
    print("=" * 60)
    print("Morpheme Token MLX Training")
    print("=" * 60)

    # Load vocab
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    vocab_size = vocab["vocab_size"]
    print(f"Vocab size: {vocab_size}")

    # Load data
    tokens = load_morpheme_data(DATA_PATH)
    print(f"Sequences available: {(len(tokens) - 1) // SEQ_LEN}")

    # Build model
    model = MorphGPT(
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS,
        dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        mlp_mult=MLP_MULT,
    )
    mx.eval(model.parameters())

    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")
    print(f"Estimated fp16 size: {n_params * 2 / 1024 / 1024:.2f} MB")
    print(f"Estimated int8+zlib size: ~{n_params / 1024 / 1024:.2f} MB")
    print(f"Config: layers={NUM_LAYERS} dim={MODEL_DIM} heads={NUM_HEADS} seq_len={SEQ_LEN}")
    print()

    # Optimizer
    lr_schedule = optim.cosine_decay(LR, TRAIN_STEPS)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    # Compile loss+grad
    loss_and_grad = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    # Data iterator
    batches = make_batches(tokens, SEQ_LEN)

    # Train
    print(f"Training for {TRAIN_STEPS} steps...")
    print("-" * 60)

    losses = []
    t_start = time.perf_counter()

    for step in range(1, TRAIN_STEPS + 1):
        x, y = next(batches)
        loss, grads = loss_and_grad(x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = float(loss.item())
        losses.append(loss_val)

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            avg_recent = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
            tok_s = step * SEQ_LEN / elapsed
            print(
                f"step {step:>4d}/{TRAIN_STEPS}  "
                f"loss={loss_val:.4f}  avg={avg_recent:.4f}  "
                f"elapsed={elapsed:.1f}s  tok/s={tok_s:.0f}"
            )

    total_time = time.perf_counter() - t_start

    # Report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final loss:             {losses[-1]:.4f}")
    print(f"Avg last 20 steps:      {sum(losses[-20:]) / 20:.4f}")
    print(f"Parameters:             {n_params:,}")
    print(f"Est. fp16 model size:   {n_params * 2 / 1024 / 1024:.2f} MB")
    print(f"Est. int8+zlib size:    ~{n_params / 1024 / 1024:.2f} MB")
    print(f"Training time:          {total_time:.1f}s")
    print(f"Tokens processed:       {TRAIN_STEPS * SEQ_LEN:,}")
    print(f"Throughput:             {TRAIN_STEPS * SEQ_LEN / total_time:.0f} tok/s")
    print(f"Vocab size:             {vocab_size}")
    print(f"Data tokens:            {len(tokens):,}")


if __name__ == "__main__":
    main()
