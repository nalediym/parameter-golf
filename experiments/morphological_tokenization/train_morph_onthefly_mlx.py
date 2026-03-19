#!/usr/bin/env python3
"""
On-the-fly morpheme tokenization training script.

Reads FineWeb BPE binary shards, decodes to text via SentencePiece,
segments into morphemes using AgglutinativeTokenizer, and trains a
small transformer — all in the training loop.

Usage:
    /Users/naledi/Projects/parameter-golf/.venv/bin/python3 train_morph_onthefly_mlx.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from morphological_tokenizer import AgglutinativeTokenizer

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = Path("../../data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
TOKENIZER_PATH = Path("../../data/tokenizers/fineweb_1024_bpe.model")
VOCAB_PATH = Path("data/morph_vocab_v2.json")

NUM_LAYERS = 4
MODEL_DIM = 256
NUM_HEADS = 4
SEQ_LEN = 256
LR = 3e-4
NUM_STEPS = 200
LOG_EVERY = 10
BATCH_SEQS = 8  # sequences per batch
BPE_CHUNK_SIZE = 2048  # BPE tokens to decode at a time per sequence


# =============================================================================
# DATA: on-the-fly morpheme conversion
# =============================================================================
class MorphDataLoader:
    """Loads BPE shards, decodes to text, segments into morphemes on-the-fly."""

    def __init__(
        self,
        shard_path: Path,
        sp_model_path: Path,
        vocab_path: Path,
        seq_len: int,
        batch_seqs: int,
        bpe_chunk_size: int,
    ):
        # Load BPE shard
        header = np.fromfile(shard_path, dtype="<i4", count=256)
        assert header[0] == 20240520 and header[1] == 1
        self.num_bpe_tokens = int(header[2])
        self.bpe_tokens = np.fromfile(
            shard_path, dtype="<u2", offset=256 * 4, count=self.num_bpe_tokens
        )
        print(f"Loaded {self.num_bpe_tokens:,} BPE tokens from {shard_path.name}")

        # SentencePiece decoder
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))

        # Morpheme vocab
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        self.token_to_id = vocab_data["token_to_id"]
        self.vocab_size = len(self.token_to_id)
        self.unk_id = self.token_to_id.get("<UNK>", 1)
        self.pad_id = self.token_to_id.get("<PAD>", 0)
        print(f"Morph vocab size: {self.vocab_size}")

        # Morpheme segmenter with cached word->morphemes
        self.segmenter = AgglutinativeTokenizer()
        self._word_cache: dict[str, list[int]] = {}

        self.seq_len = seq_len
        self.batch_seqs = batch_seqs
        self.bpe_chunk_size = bpe_chunk_size
        self.rng = np.random.default_rng(42)

    def _segment_word_cached(self, word: str) -> list[int]:
        """Segment a word into morpheme IDs, with caching."""
        if word in self._word_cache:
            return self._word_cache[word]
        result = self.segmenter.segment(word)
        ids = [self.token_to_id.get(m, self.unk_id) for m in result.morphemes]
        self._word_cache[word] = ids
        return ids

    def _text_to_morph_ids(self, text: str) -> list[int]:
        """Convert text to morpheme ID sequence."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        ids: list[int] = []
        for word in words:
            ids.extend(self._segment_word_cached(word))
        return ids

    def _sample_morph_sequence(self) -> np.ndarray:
        """Sample a random BPE chunk, decode, segment, return morph IDs."""
        max_start = self.num_bpe_tokens - self.bpe_chunk_size
        start = self.rng.integers(0, max(1, max_start))
        bpe_chunk = self.bpe_tokens[start : start + self.bpe_chunk_size]
        text = self.sp.decode(bpe_chunk.astype(int).tolist())
        morph_ids = self._text_to_morph_ids(text)

        # We need seq_len + 1 tokens (input + target shift)
        need = self.seq_len + 1
        if len(morph_ids) < need:
            # Pad with PAD tokens if too short (rare with 2048 BPE tokens)
            morph_ids.extend([self.pad_id] * (need - len(morph_ids)))
        # Take a random contiguous window
        if len(morph_ids) > need:
            offset = self.rng.integers(0, len(morph_ids) - need)
            morph_ids = morph_ids[offset : offset + need]
        return np.array(morph_ids[:need], dtype=np.int32)

    def next_batch(self) -> tuple[mx.array, mx.array]:
        """Get next training batch. Returns (x, y) each [batch, seq_len]."""
        seqs = np.stack([self._sample_morph_sequence() for _ in range(self.batch_seqs)])
        x = mx.array(seqs[:, :-1])  # [B, seq_len]
        y = mx.array(seqs[:, 1:])   # [B, seq_len]
        return x, y


# =============================================================================
# MODEL: simple causal transformer
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ln2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MorphTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_layers: int, seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = [TransformerBlock(dim, num_heads) for _ in range(num_layers)]
        self.ln_f = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        pos = mx.arange(T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def loss(self, x: mx.array, y: mx.array) -> mx.array:
        logits = self(x)  # [B, T, V]
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
            y.reshape(-1),
            reduction="mean",
        )


# =============================================================================
# TRAINING
# =============================================================================
def count_params(model: nn.Module) -> int:
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def main():
    print("=" * 60)
    print("Morpheme On-the-Fly Training (MLX)")
    print("=" * 60)

    # --- Data loader ---
    loader = MorphDataLoader(
        shard_path=DATA_PATH,
        sp_model_path=TOKENIZER_PATH,
        vocab_path=VOCAB_PATH,
        seq_len=SEQ_LEN,
        batch_seqs=BATCH_SEQS,
        bpe_chunk_size=BPE_CHUNK_SIZE,
    )

    # --- Model ---
    vocab_size = loader.vocab_size
    model = MorphTransformer(
        vocab_size=vocab_size,
        dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
    )
    mx.eval(model.parameters())
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")
    print(f"  vocab_size={vocab_size} dim={MODEL_DIM} heads={NUM_HEADS} layers={NUM_LAYERS} seq_len={SEQ_LEN}")

    # --- Optimizer ---
    optimizer = optim.AdamW(learning_rate=LR)

    # --- Compiled loss + grad ---
    loss_and_grad_fn = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    # --- Training loop ---
    print(f"\nTraining for {NUM_STEPS} steps, batch_size={BATCH_SEQS}, seq_len={SEQ_LEN}")
    print("-" * 60)

    losses: list[float] = []
    total_tokens = 0
    t_start = time.perf_counter()

    for step in range(1, NUM_STEPS + 1):
        x, y = loader.next_batch()
        loss, grads = loss_and_grad_fn(x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = float(loss.item())
        losses.append(loss_val)
        total_tokens += BATCH_SEQS * SEQ_LEN

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            tok_s = total_tokens / elapsed
            print(f"step {step:4d}/{NUM_STEPS}  loss={loss_val:.4f}  tok/s={tok_s:,.0f}  elapsed={elapsed:.1f}s")

    t_total = time.perf_counter() - t_start
    final_tok_s = total_tokens / t_total

    # --- Results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Parameters:     {n_params:,}")
    print(f"Final loss:     {losses[-1]:.4f}")
    print(f"Best loss:      {min(losses):.4f}")
    print(f"Throughput:     {final_tok_s:,.0f} tok/s")
    print(f"Total time:     {t_total:.1f}s")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Cache hits:     {len(loader._word_cache):,} unique words cached")

    # Loss curve (ASCII)
    print(f"\nLoss curve (sampled every {LOG_EVERY} steps):")
    sampled = losses[::LOG_EVERY]
    lo, hi = min(sampled), max(sampled)
    width = 40
    for i, l in enumerate(sampled):
        step_num = (i + 1) * LOG_EVERY
        bar_len = int((l - lo) / max(hi - lo, 1e-6) * width) if hi > lo else width // 2
        bar = "#" * max(bar_len, 1)
        print(f"  step {step_num:4d} | {l:.4f} |{bar}")


if __name__ == "__main__":
    main()
