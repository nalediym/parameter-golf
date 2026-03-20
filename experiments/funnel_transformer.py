"""
Funnel Transformer prototype for Parameter Golf.

The idea: progressively pool the sequence length through the transformer layers
so that deeper layers attend over shorter sequences (quadratically cheaper).
Then upsample back for the output projection.

Architecture:
  - Layers 1-3: full 2048 positions
  - Pool 4:1 -> 512 positions
  - Layers 4-6: 512 positions (4x cheaper attention)
  - Pool 4:1 -> 128 positions
  - Layers 7-9: 128 positions (16x cheaper attention)
  - Upsample back to 2048 for output projection

This is a standalone prototype — does not modify train_gpt.py.
"""

from __future__ import annotations

import math
import sys
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Reusable building blocks (copied from train_gpt.py to keep standalone)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Funnel pooling / upsampling helpers
# ---------------------------------------------------------------------------


def pool_sequence(x: Tensor, factor: int) -> Tensor:
    """Average-pool along the sequence dimension by `factor`.

    Args:
        x: (batch, seq_len, dim)
        factor: pooling factor (must evenly divide seq_len)
    Returns:
        (batch, seq_len // factor, dim)
    """
    bsz, seq_len, dim = x.shape
    assert seq_len % factor == 0, f"seq_len {seq_len} not divisible by factor {factor}"
    return x.reshape(bsz, seq_len // factor, factor, dim).mean(dim=2)


def upsample_sequence(x: Tensor, target_len: int) -> Tensor:
    """Nearest-neighbor upsample along the sequence dimension.

    Args:
        x: (batch, seq_len, dim)
        target_len: desired output sequence length
    Returns:
        (batch, target_len, dim)
    """
    # Transpose to (batch, dim, seq_len) for F.interpolate, then back
    return F.interpolate(
        x.transpose(1, 2),
        size=target_len,
        mode="nearest",
    ).transpose(1, 2)


# ---------------------------------------------------------------------------
# Funnel Transformer
# ---------------------------------------------------------------------------


class FunnelTransformer(nn.Module):
    """Funnel Transformer with progressive sequence pooling.

    Stage layout (default 9 layers, pool_factor=4):
      Stage 0 — layers 0..2:  full sequence (2048)
      Stage 1 — layers 3..5:  pooled 4:1  (512)
      Stage 2 — layers 6..8:  pooled 4:1  (128)
      Upsample back to 2048 for output projection.

    The model is a drop-in replacement for GPT: same forward(input_ids, target_ids)
    signature, returns cross-entropy loss.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        num_layers: int = 9,
        model_dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: int = 2,
        tie_embeddings: bool = True,
        tied_embed_init_std: float = 0.005,
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        # Funnel-specific
        layers_per_stage: int = 3,
        pool_factor: int = 4,
    ):
        super().__init__()
        assert num_layers % layers_per_stage == 0, (
            f"num_layers ({num_layers}) must be divisible by layers_per_stage ({layers_per_stage})"
        )
        self.num_stages = num_layers // layers_per_stage
        self.layers_per_stage = layers_per_stage
        self.pool_factor = pool_factor
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.model_dim = model_dim

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Transformer blocks — one per layer, organized into stages
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            )

        # Learned linear projection for upsampling back to full resolution.
        # This gives the model a chance to distribute information better than
        # pure nearest-neighbor repeat.
        self.upsample_proj = CastedLinear(model_dim, model_dim, bias=False)

        # Final norm + head
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _forward_features(self, input_ids: Tensor) -> Tensor:
        """Run embedding + funnel blocks + upsample. Returns (bsz, seq_len, dim)."""
        bsz, full_seq_len = input_ids.shape

        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0_full = x  # save full-resolution x0 for residual mixing

        current_len = full_seq_len
        layer_idx = 0

        for stage in range(self.num_stages):
            # Pool x0 to match current resolution
            if stage == 0:
                x0 = x0_full
            else:
                x0 = pool_sequence(x0_full, full_seq_len // current_len)

            # Run layers for this stage
            for _ in range(self.layers_per_stage):
                x = self.blocks[layer_idx](x, x0)
                layer_idx += 1

            # Pool after stage (except the last stage)
            if stage < self.num_stages - 1:
                x = pool_sequence(x, self.pool_factor)
                current_len = current_len // self.pool_factor

        # Upsample back to full resolution
        if current_len < full_seq_len:
            x = upsample_sequence(x, full_seq_len)
            x = self.upsample_proj(x)

        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Drop-in compatible with GPT.forward — returns cross-entropy loss."""
        x = self._forward_features(input_ids)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self._forward_features(input_ids)
        x = self.final_norm(x)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# ---------------------------------------------------------------------------
# Parameter count estimation
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return a breakdown of parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    by_module: dict[str, int] = {}
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n > 0:
            by_module[name] = n

    return {"total": total, "trainable": trainable, "by_module": by_module}


def estimate_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB assuming int8 quantization for large tensors."""
    total_bytes = 0
    for p in model.parameters():
        if p.numel() > 65_536:
            # int8 quantized: 1 byte per param + scale overhead (~negligible)
            total_bytes += p.numel()
        else:
            # Small tensors kept as fp16
            total_bytes += p.numel() * 2
    return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def test_forward_pass():
    """Create the model and verify a forward pass with random data."""
    print("=" * 60)
    print("Funnel Transformer — Forward Pass Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Default config matching Parameter Golf constraints
    model = FunnelTransformer(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        layers_per_stage=3,
        pool_factor=4,
    ).to(device)

    # Parameter count
    stats = count_parameters(model)
    est_mb = estimate_size_mb(model)
    print(f"\nParameter count: {stats['total']:,}")
    print(f"Trainable:       {stats['trainable']:,}")
    print(f"Estimated int8 size: {est_mb:.2f} MB  (limit: 16 MB)")
    print(f"\nBreakdown by module:")
    for name, count in stats["by_module"].items():
        print(f"  {name:20s}: {count:>10,}")

    budget_ok = est_mb <= 16.0
    print(f"\nWithin 16 MB budget: {'YES' if budget_ok else 'NO'}")

    # Forward pass test
    seq_len = 2048
    batch_size = 2
    input_ids = torch.randint(0, 1024, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, 1024, (batch_size, seq_len), device=device)

    print(f"\nRunning forward pass: batch={batch_size}, seq_len={seq_len} ...")
    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        loss = model(input_ids, target_ids)

    print(f"Loss: {loss.item():.4f}")
    assert loss.isfinite(), "Loss is not finite!"
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"

    # Also test forward_logits
    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        logits = model.forward_logits(input_ids)

    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 1024), f"Unexpected logits shape: {logits.shape}"
    assert logits.isfinite().all(), "Logits contain non-finite values!"

    # Verify sequence length math
    print(f"\nSequence length progression:")
    print(f"  Stage 0 (layers 0-2): {seq_len} positions")
    print(f"  Stage 1 (layers 3-5): {seq_len // 4} positions  (4x cheaper attention)")
    print(f"  Stage 2 (layers 6-8): {seq_len // 16} positions (16x cheaper attention)")
    print(f"  Output projection:    {seq_len} positions  (upsampled)")

    # Rough FLOPs comparison
    # Attention FLOPs ~ 2 * n^2 * d per layer (ignoring constants)
    d = 512
    full_cost = 3 * (2 * seq_len**2 * d)  # 3 layers at full
    mid_cost = 3 * (2 * (seq_len // 4)**2 * d)  # 3 layers at 1/4
    low_cost = 3 * (2 * (seq_len // 16)**2 * d)  # 3 layers at 1/16
    funnel_cost = full_cost + mid_cost + low_cost
    baseline_cost = 9 * (2 * seq_len**2 * d)
    savings_pct = (1 - funnel_cost / baseline_cost) * 100
    print(f"\nAttention FLOPs savings vs 9-layer full: {savings_pct:.1f}%")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_forward_pass()
