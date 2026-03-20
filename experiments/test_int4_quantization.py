#!/usr/bin/env python3
"""
int4 quantization feasibility test for Parameter Golf.

Takes the existing int8 quantization pipeline and compares it against
a new int4 per-group quantization scheme. Works on random tensors —
no GPU or trained checkpoint required.

Measures:
  - Roundtrip reconstruction error (MSE, max abs error)
  - Payload sizes: raw, int8, int4
  - Compressed sizes (zlib): int8+zlib vs int4+zlib
  - What larger model could fit in 16MB with int4
"""

from __future__ import annotations

import io
import sys
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Import the existing int8 helpers from train_gpt.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8

# ---------------------------------------------------------------------------
# int4 per-group quantization
# ---------------------------------------------------------------------------

def quantize_int4_per_group(t: Tensor, group_size: int = 64) -> tuple[Tensor, Tensor, int]:
    """Quantize a 2-D float tensor to symmetric int4 with per-group scales.

    Returns
    -------
    packed : uint8 tensor — two int4 values packed per byte
    scales : float16 tensor of shape (M, num_groups)
    orig_N : original number of columns (needed if N was padded)
    """
    assert t.ndim == 2, f"Expected 2-D tensor, got {t.ndim}-D"
    t32 = t.float()
    M, N = t32.shape
    orig_N = N

    # Pad columns to a multiple of group_size
    N_padded = ((N + group_size - 1) // group_size) * group_size
    if N_padded != N:
        t32 = F.pad(t32, (0, N_padded - N))

    num_groups = N_padded // group_size
    t_grouped = t32.reshape(M, num_groups, group_size)  # (M, G, gs)

    # Per-group absmax scale
    amax = t_grouped.abs().amax(dim=-1, keepdim=True)  # (M, G, 1)
    scale = (amax / 7.0).clamp_(min=1e-8)

    # Quantize to [-8, 7]
    q = (t_grouped / scale).round().clamp_(-8, 7).to(torch.int8)  # (M, G, gs)

    # Flatten back to (M, N_padded) then pack pairs of int4 into uint8
    q_flat = q.reshape(M, N_padded)
    # Ensure even number of columns (guaranteed since group_size is even)
    high = (q_flat[:, 0::2] + 8).to(torch.uint8)  # shift to [0, 15]
    low = (q_flat[:, 1::2] + 8).to(torch.uint8)
    packed = (high << 4) | low  # two nibbles per byte

    scales = scale.squeeze(-1).to(torch.float16)  # (M, num_groups)
    return packed.contiguous(), scales.contiguous(), orig_N


def dequantize_int4_per_group(
    packed: Tensor, scales: Tensor, orig_N: int, group_size: int = 64
) -> Tensor:
    """Reverse of quantize_int4_per_group — returns float32 tensor."""
    M = packed.shape[0]

    # Unpack uint8 -> two int4 values
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # back to [-8, 7]
    low = (packed & 0x0F).to(torch.int8) - 8

    # Interleave back
    N_padded = packed.shape[1] * 2
    q_flat = torch.empty(M, N_padded, dtype=torch.int8)
    q_flat[:, 0::2] = high
    q_flat[:, 1::2] = low

    num_groups = N_padded // group_size
    q_grouped = q_flat.reshape(M, num_groups, group_size)
    scales_expanded = scales.float().unsqueeze(-1)  # (M, G, 1)

    out = (q_grouped.float() * scales_expanded).reshape(M, N_padded)
    return out[:, :orig_N].contiguous()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_bytes(t: Tensor) -> int:
    return t.numel() * t.element_size()


def save_tensors_to_bytes(tensors: list[Tensor]) -> bytes:
    """Serialize a list of tensors into a bytes buffer (for measuring zlib size)."""
    buf = io.BytesIO()
    for t in tensors:
        buf.write(t.numpy().tobytes())
    return buf.getvalue()


def zlib_size(data: bytes) -> int:
    return len(zlib.compress(data, level=9))


# ---------------------------------------------------------------------------
# Build a fake state dict matching the current architecture
# ---------------------------------------------------------------------------

def make_fake_state_dict(
    n_layers: int = 9,
    d_model: int = 512,
    n_head: int = 8,
    n_kv_head: int = 4,
    mlp_mult: int = 2,
    vocab_size: int = 1024,
) -> dict[str, Tensor]:
    """Generate a state dict with random weights shaped like the real model."""
    sd: dict[str, Tensor] = {}
    head_dim = d_model // n_head
    qkv_dim = head_dim * (n_head + 2 * n_kv_head)
    mlp_dim = d_model * mlp_mult

    # Token embedding (tied with output head)
    sd["tok_emb.weight"] = torch.randn(vocab_size, d_model) * 0.02

    for i in range(n_layers):
        prefix = f"blocks.{i}."
        # Attention
        sd[prefix + "attn_norm.weight"] = torch.ones(d_model)
        sd[prefix + "attn.qkv_proj.weight"] = torch.randn(qkv_dim, d_model) * 0.02
        sd[prefix + "attn.out_proj.weight"] = torch.randn(d_model, d_model) * 0.02
        # MLP
        sd[prefix + "mlp_norm.weight"] = torch.ones(d_model)
        sd[prefix + "mlp.up_proj.weight"] = torch.randn(mlp_dim, d_model) * 0.02
        sd[prefix + "mlp.down_proj.weight"] = torch.randn(d_model, mlp_dim) * 0.02

    sd["final_norm.weight"] = torch.ones(d_model)
    return sd


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  int4 per-group quantization feasibility test")
    print("=" * 72)

    GROUP_SIZE = 64
    CODE_SIZE_ESTIMATE = 60_000  # ~58 KB for the submission script
    BUDGET = 16 * 1024 * 1024   # 16 MB

    sd = make_fake_state_dict()
    total_params = sum(t.numel() for t in sd.values())
    fp32_bytes = sum(tensor_bytes(t) for t in sd.values())

    print(f"\nFake model: 9 layers, d_model=512, 8/4 heads, MLP_MULT=2, vocab=1024")
    print(f"Total parameters: {total_params:,}")
    print(f"FP32 size:        {fp32_bytes:,} bytes ({fp32_bytes / 1e6:.2f} MB)")

    # ---- int8 quantization (existing) ----
    int8_obj, int8_stats = quantize_state_dict_int8(sd)
    int8_payload = int8_stats["int8_payload_bytes"]

    # Collect all int8 tensors for zlib measurement
    int8_blobs = []
    for t in int8_obj["quantized"].values():
        int8_blobs.append(t)
    for t in int8_obj["scales"].values():
        int8_blobs.append(t)
    for t in int8_obj["passthrough"].values():
        int8_blobs.append(t)
    int8_raw = save_tensors_to_bytes(int8_blobs)
    int8_compressed = zlib_size(int8_raw)

    # Roundtrip error for int8
    sd_int8_rt = dequantize_state_dict_int8(int8_obj)
    int8_mse_total = 0.0
    int8_maxerr = 0.0
    n_params_checked = 0
    for name in sd:
        orig = sd[name].float()
        recon = sd_int8_rt[name].float()
        diff = (orig - recon)
        int8_mse_total += (diff ** 2).sum().item()
        int8_maxerr = max(int8_maxerr, diff.abs().max().item())
        n_params_checked += orig.numel()
    int8_mse = int8_mse_total / n_params_checked

    print(f"\n--- int8 (existing) ---")
    print(f"Payload:          {int8_payload:,} bytes ({int8_payload / 1e6:.2f} MB)")
    print(f"zlib compressed:  {int8_compressed:,} bytes ({int8_compressed / 1e6:.2f} MB)")
    print(f"Roundtrip MSE:    {int8_mse:.2e}")
    print(f"Roundtrip MaxErr: {int8_maxerr:.6f}")

    # ---- int4 per-group quantization ----
    int4_blobs = []
    int4_payload = 0
    int4_mse_total = 0.0
    int4_maxerr = 0.0
    int4_params_checked = 0
    passthrough_payload = 0

    for name, tensor in sd.items():
        t = tensor.detach().float()
        if t.ndim == 2 and t.numel() > 65_536:
            # Quantize large 2-D tensors to int4
            packed, scales, orig_N = quantize_int4_per_group(t, group_size=GROUP_SIZE)
            recon = dequantize_int4_per_group(packed, scales, orig_N, group_size=GROUP_SIZE)

            int4_blobs.append(packed)
            int4_blobs.append(scales)
            int4_payload += tensor_bytes(packed) + tensor_bytes(scales)

            diff = t - recon
            int4_mse_total += (diff ** 2).sum().item()
            int4_maxerr = max(int4_maxerr, diff.abs().max().item())
            int4_params_checked += t.numel()
        else:
            # Small/non-2D tensors: keep as fp16 (same as int8 path)
            kept = t.to(torch.float16) if t.is_floating_point() else t
            int4_blobs.append(kept)
            passthrough_payload += tensor_bytes(kept)
            # These are exact (fp16 roundtrip)
            recon = kept.float()
            diff = t - recon
            int4_mse_total += (diff ** 2).sum().item()
            int4_maxerr = max(int4_maxerr, diff.abs().max().item())
            int4_params_checked += t.numel()

    int4_total_payload = int4_payload + passthrough_payload
    int4_raw = save_tensors_to_bytes(int4_blobs)
    int4_compressed = zlib_size(int4_raw)
    int4_mse = int4_mse_total / int4_params_checked

    print(f"\n--- int4 per-group (group_size={GROUP_SIZE}) ---")
    print(f"Payload (int4):   {int4_payload:,} bytes ({int4_payload / 1e6:.2f} MB)")
    print(f"Payload (pass):   {passthrough_payload:,} bytes ({passthrough_payload / 1e6:.2f} MB)")
    print(f"Total payload:    {int4_total_payload:,} bytes ({int4_total_payload / 1e6:.2f} MB)")
    print(f"zlib compressed:  {int4_compressed:,} bytes ({int4_compressed / 1e6:.2f} MB)")
    print(f"Roundtrip MSE:    {int4_mse:.2e}")
    print(f"Roundtrip MaxErr: {int4_maxerr:.6f}")

    # ---- Comparison ----
    savings_raw = int8_payload - int4_total_payload
    savings_compressed = int8_compressed - int4_compressed
    ratio = int4_compressed / int8_compressed if int8_compressed > 0 else 0

    print(f"\n{'=' * 72}")
    print(f"  COMPARISON")
    print(f"{'=' * 72}")
    print(f"Raw payload savings:  {savings_raw:,} bytes ({savings_raw / 1e6:.2f} MB)")
    print(f"Compressed savings:   {savings_compressed:,} bytes ({savings_compressed / 1e6:.2f} MB)")
    print(f"Compression ratio:    int4 is {ratio:.2%} of int8 (zlib)")
    print(f"MSE ratio:            int4/int8 = {int4_mse / int8_mse:.1f}x worse")

    # ---- What fits in 16MB with int4? ----
    int4_headroom = BUDGET - CODE_SIZE_ESTIMATE - int4_compressed
    int8_headroom = BUDGET - CODE_SIZE_ESTIMATE - int8_compressed

    print(f"\n{'=' * 72}")
    print(f"  BUDGET ANALYSIS (16 MB cap, ~{CODE_SIZE_ESTIMATE // 1000} KB code)")
    print(f"{'=' * 72}")
    print(f"int8 artifact:    {int8_compressed + CODE_SIZE_ESTIMATE:,} bytes  "
          f"(headroom: {int8_headroom:,} bytes = {int8_headroom / 1024:.0f} KB)")
    print(f"int4 artifact:    {int4_compressed + CODE_SIZE_ESTIMATE:,} bytes  "
          f"(headroom: {int4_headroom:,} bytes = {int4_headroom / 1024:.0f} KB)")

    # Estimate what larger model could fit
    # Rough heuristic: params scale quadratically with d_model (most params are in MLP+attn)
    # int4+zlib bytes per param (from this test)
    int4_bytes_per_param = int4_compressed / total_params
    int8_bytes_per_param = int8_compressed / total_params
    max_params_int4 = (BUDGET - CODE_SIZE_ESTIMATE) / int4_bytes_per_param
    max_params_int8 = (BUDGET - CODE_SIZE_ESTIMATE) / int8_bytes_per_param

    print(f"\nCompressed bytes/param:  int8={int8_bytes_per_param:.3f}  int4={int4_bytes_per_param:.3f}")
    print(f"Max params in budget:    int8={max_params_int8 / 1e6:.1f}M  int4={max_params_int4 / 1e6:.1f}M")
    print(f"Param increase with int4: {max_params_int4 / max_params_int8:.1f}x")

    # Suggest architectures
    print(f"\n--- Candidate int4 architectures ---")
    candidates = [
        ("12 layers, d=576",  12, 576, 8, 4, 2, 1024),
        ("9 layers, d=640",    9, 640, 8, 4, 2, 1024),
        ("12 layers, d=640",  12, 640, 8, 4, 2, 1024),
        ("15 layers, d=512",  15, 512, 8, 4, 2, 1024),
        ("9 layers, d=768",    9, 768, 8, 4, 2, 1024),
    ]
    for label, nl, dm, nh, nkv, mm, vs in candidates:
        cand_sd = make_fake_state_dict(nl, dm, nh, nkv, mm, vs)
        cand_params = sum(t.numel() for t in cand_sd.values())
        # Estimate int4+zlib size using bytes/param from test
        est_size = cand_params * int4_bytes_per_param + CODE_SIZE_ESTIMATE
        fits = "FITS" if est_size < BUDGET else "TOO BIG"
        margin = BUDGET - est_size
        print(f"  {label:24s}  params={cand_params / 1e6:5.1f}M  "
              f"est={est_size / 1e6:5.1f}MB  margin={margin / 1024:+7.0f}KB  [{fits}]")

    # ---- Roundtrip correctness check ----
    print(f"\n{'=' * 72}")
    print(f"  ROUNDTRIP CORRECTNESS (spot checks)")
    print(f"{'=' * 72}")

    # Pick a small tensor to visually inspect
    test = torch.tensor([[-0.05, 0.02, -0.01, 0.04]], dtype=torch.float32)
    test_padded = F.pad(test, (0, GROUP_SIZE - test.shape[1]))
    packed, scales, orig_N = quantize_int4_per_group(test_padded.reshape(1, -1), group_size=GROUP_SIZE)
    recon = dequantize_int4_per_group(packed, scales, orig_N, group_size=GROUP_SIZE)
    print(f"Original (first 4): {test[0, :4].tolist()}")
    print(f"Recon    (first 4): {recon[0, :4].tolist()}")
    print(f"Scale:              {scales[0, 0].item():.6f}")
    print(f"Max abs error:      {(test_padded.reshape(1, -1)[:, :orig_N].float() - recon).abs().max().item():.6f}")

    # Verify packing roundtrip on a range of values
    all_vals = torch.arange(-8, 8, dtype=torch.float32).unsqueeze(0)  # (1, 16)
    if all_vals.shape[1] < GROUP_SIZE:
        all_vals = F.pad(all_vals, (0, GROUP_SIZE - all_vals.shape[1]))
    packed_v, scales_v, orig_N_v = quantize_int4_per_group(all_vals, group_size=GROUP_SIZE)
    recon_v = dequantize_int4_per_group(packed_v, scales_v, orig_N_v, group_size=GROUP_SIZE)
    # For this input, scale = 8/7, so quantized values should be close to originals
    err = (all_vals[:, :16].float() - recon_v[:, :16]).abs().max().item()
    print(f"\nAll int4 values [-8..7] roundtrip max error: {err:.6f}")
    status = "PASS" if err < 1.0 else "FAIL"
    print(f"Packing correctness: {status}")

    print(f"\n{'=' * 72}")
    print(f"  DONE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
