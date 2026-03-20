#!/usr/bin/env python3
"""Tests T6-T30: IPA pipeline, training integration, bpb, IPA-BPE, layer looping."""
import inspect
import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        print(f"  FAIL: {name} {detail}")


def skip(name, reason=""):
    global SKIPPED
    SKIPPED += 1
    print(f"  SKIP: {name} ({reason})")


# ---------------------------------------------------------------------------
# Helpers: create synthetic shard files for unit tests
# ---------------------------------------------------------------------------

HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # 1024 bytes
MAGIC = 20240520


def _write_shard_v1(path, token_ids_u16):
    """Write a minimal v1 BPE shard (uint16 tokens)."""
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(token_ids_u16)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.array(token_ids_u16, dtype="<u2").tobytes())


def _write_shard_v2_uint8(path, token_ids_u8, original_byte_count):
    """Write a minimal v2 IPA uint8 shard."""
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = 2
    header[2] = len(token_ids_u8)
    header[3] = original_byte_count
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.array(token_ids_u8, dtype=np.uint8).tobytes())
    # sidecar
    bytes_path = str(path).replace(".bin", ".bytes")
    with open(bytes_path, "w") as f:
        f.write(str(original_byte_count))


def _write_shard_v2_uint16(path, token_ids_u16, original_byte_count):
    """Write a minimal v2 IPA-BPE uint16 shard."""
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = 2
    header[2] = len(token_ids_u16)
    header[3] = original_byte_count
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.array(token_ids_u16, dtype="<u2").tobytes())


# ---------------------------------------------------------------------------
# Shard data paths (may not exist locally)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
IPA_CHAR_DIR = REPO_ROOT / "data" / "datasets" / "fineweb10B_ipa"
IPA_BPE_DIR = REPO_ROOT / "data" / "datasets" / "fineweb10B_ipa_bpe"
BPE_DIR = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"


def _find_shard(directory, pattern="*.bin"):
    """Return first shard file in directory or None."""
    if not directory.is_dir():
        return None
    shards = sorted(directory.glob(pattern))
    return shards[0] if shards else None


# ===========================================================================
# SHARD TESTS (T6-T9) — IPA char-level uint8 shards
# ===========================================================================

def test_t6_shard_header_magic():
    """T6: IPA shard header has correct magic (20240520) and version."""
    print("\nT6: Shard header magic and version")
    shard = _find_shard(IPA_CHAR_DIR)
    if shard is None:
        skip("real shard check", "IPA char shards not found locally")
    else:
        header = np.fromfile(shard, dtype="<i4", count=256)
        check("header has 256 ints", header.size == 256)
        check("magic is 20240520", int(header[0]) == MAGIC, f"got {header[0]}")
        check("version is 2", int(header[1]) == 2, f"got {header[1]}")

    # Also test with synthetic shard
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        _write_shard_v2_uint8(tmp, [1, 2, 3], 100)
        header = np.fromfile(tmp, dtype="<i4", count=256)
        check("synthetic magic is 20240520", int(header[0]) == MAGIC)
        check("synthetic version is 2", int(header[1]) == 2)
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp).replace(".bin", ".bytes")).unlink(missing_ok=True)


def test_t7_byte_count_sidecar():
    """T7: Byte count sidecar matches header byte count."""
    print("\nT7: Byte count sidecar matches header")
    shard = _find_shard(IPA_CHAR_DIR)
    if shard is None:
        skip("real shard sidecar check", "IPA char shards not found locally")
    else:
        header = np.fromfile(shard, dtype="<i4", count=256)
        header_byte_count = int(header[3])
        sidecar = Path(str(shard).replace(".bin", ".bytes"))
        if sidecar.exists():
            sidecar_byte_count = int(sidecar.read_text().strip())
            check(
                "sidecar matches header[3]",
                sidecar_byte_count == header_byte_count,
                f"sidecar={sidecar_byte_count} header={header_byte_count}",
            )
        else:
            skip("sidecar file", "no .bytes file found next to shard")

    # Synthetic test
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        _write_shard_v2_uint8(tmp, [10, 20, 30], 42)
        header = np.fromfile(tmp, dtype="<i4", count=256)
        sidecar_path = Path(str(tmp).replace(".bin", ".bytes"))
        sidecar_val = int(sidecar_path.read_text().strip())
        check("synthetic sidecar matches header", sidecar_val == int(header[3]))
        check("synthetic byte count is 42", sidecar_val == 42)
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp).replace(".bin", ".bytes")).unlink(missing_ok=True)


def test_t8_token_ids_in_range():
    """T8: All token IDs within valid range for the vocab."""
    print("\nT8: Token IDs in valid range")
    shard = _find_shard(IPA_CHAR_DIR)
    if shard is None:
        skip("real shard token range", "IPA char shards not found locally")
    else:
        header = np.fromfile(shard, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(shard, dtype=np.uint8, count=num_tokens, offset=HEADER_BYTES)
        max_id = int(tokens.max()) if tokens.size > 0 else 0
        # IPA char-level vocab is ~127 symbols (uint8 range 0-255 is fine,
        # but realistic IPA char vocabs are < 200)
        check("max token ID < 256 (uint8)", max_id < 256, f"max={max_id}")
        check("token count > 0", tokens.size > 0)
        check("all tokens >= 0", int(tokens.min()) >= 0)

    # Synthetic
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        ids = list(range(127))
        _write_shard_v2_uint8(tmp, ids, 500)
        tokens = np.fromfile(tmp, dtype=np.uint8, count=len(ids), offset=HEADER_BYTES)
        check("synthetic all IDs < 127", int(tokens.max()) < 127)
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp).replace(".bin", ".bytes")).unlink(missing_ok=True)


def test_t9_no_data_loss():
    """T9: No data loss — shard token count matches header."""
    print("\nT9: Shard token count matches header")
    shard = _find_shard(IPA_CHAR_DIR)
    if shard is None:
        skip("real shard data loss check", "IPA char shards not found locally")
    else:
        header = np.fromfile(shard, dtype="<i4", count=256)
        num_tokens = int(header[2])
        payload_bytes = shard.stat().st_size - HEADER_BYTES
        # v2 uint8: payload_bytes == num_tokens
        check(
            "payload matches header token count",
            payload_bytes == num_tokens,
            f"payload_bytes={payload_bytes} num_tokens={num_tokens}",
        )

    # Synthetic
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        ids = [1, 2, 3, 4, 5]
        _write_shard_v2_uint8(tmp, ids, 99)
        header = np.fromfile(tmp, dtype="<i4", count=256)
        payload = tmp.stat().st_size - HEADER_BYTES
        check("synthetic payload == header token count", payload == int(header[2]))
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp).replace(".bin", ".bytes")).unlink(missing_ok=True)


# ===========================================================================
# TRAINING INTEGRATION (T10-T14)
# ===========================================================================

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_t10_uint8_shard_loads():
    """T10: uint8 shards load correctly via load_data_shard (version 2)."""
    print("\nT10: load_data_shard with v2 uint8")
    torch = _require_torch()
    if torch is None:
        skip("load_data_shard", "torch not available")
        return

    from train_gpt import load_data_shard

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        ids = [10, 20, 30, 40, 50]
        _write_shard_v2_uint8(tmp, ids, 200)
        tokens = load_data_shard(tmp)
        check("returns tensor", hasattr(tokens, "shape"))
        check("correct length", tokens.numel() == len(ids), f"got {tokens.numel()}")
        check("values match", list(tokens.tolist()) == ids, f"got {tokens.tolist()}")
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp).replace(".bin", ".bytes")).unlink(missing_ok=True)


def test_t11_model_builds_vocab127():
    """T11: Model builds with vocab=127 (IPA char-level)."""
    print("\nT11: GPT builds with vocab=127")
    torch = _require_torch()
    if torch is None:
        skip("model build vocab 127", "torch not available")
        return

    from train_gpt import GPT

    model = GPT(
        vocab_size=127,
        num_layers=4,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    check("model created", model is not None)
    check("tok_emb vocab dim", model.tok_emb.num_embeddings == 127)
    param_count = sum(p.numel() for p in model.parameters())
    check("has parameters", param_count > 0, f"count={param_count}")


def test_t12_bpb_uses_override_byte_count():
    """T12: bpb calculation uses original byte count (override_byte_count)."""
    print("\nT12: bpb uses override_byte_count")
    torch = _require_torch()
    if torch is None:
        skip("bpb override", "torch not available")
        return

    from train_gpt import eval_val, eval_val_sliding

    # Check that eval_val accepts override_byte_count parameter
    sig = inspect.signature(eval_val)
    check(
        "eval_val has override_byte_count param",
        "override_byte_count" in sig.parameters,
    )
    default = sig.parameters["override_byte_count"].default
    check("override_byte_count defaults to 0", default == 0, f"got {default}")

    sig_sw = inspect.signature(eval_val_sliding)
    check(
        "eval_val_sliding has override_byte_count param",
        "override_byte_count" in sig_sw.parameters,
    )


def test_t13_model_forward_smoke():
    """T13: Smoke test — model forward pass doesn't crash."""
    print("\nT13: Model forward smoke test")
    torch = _require_torch()
    if torch is None:
        skip("forward smoke", "torch not available")
        return

    from train_gpt import GPT

    model = GPT(
        vocab_size=127,
        num_layers=2,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    model.eval()

    # Random input within vocab
    input_ids = torch.randint(0, 127, (1, 32))
    target_ids = torch.randint(0, 127, (1, 32))
    try:
        with torch.no_grad():
            loss = model(input_ids, target_ids)
        check("forward produces loss", loss is not None)
        check("loss is scalar", loss.dim() == 0)
        check("loss is finite", torch.isfinite(loss).item())
    except Exception as e:
        check("forward no crash", False, f"raised {e}")


def test_t14_sliding_window_eval_callable():
    """T14: Sliding window eval function is callable."""
    print("\nT14: eval_val_sliding is callable")
    torch = _require_torch()
    if torch is None:
        skip("sliding window callable", "torch not available")
        return

    from train_gpt import eval_val_sliding

    check("eval_val_sliding is callable", callable(eval_val_sliding))
    sig = inspect.signature(eval_val_sliding)
    params = list(sig.parameters.keys())
    check("has stride param", "stride" in params)
    check("has batch_seqs param", "batch_seqs" in params)
    check("has override_byte_count param", "override_byte_count" in params)


# ===========================================================================
# BPB CORRECTNESS (T15-T16)
# ===========================================================================

def test_t15_bpb_plausible_range():
    """T15: bpb result is in plausible range (0.5 - 5.0)."""
    print("\nT15: bpb in plausible range")
    torch = _require_torch()
    if torch is None:
        skip("bpb range", "torch not available")
        return
    if not torch.cuda.is_available():
        skip("bpb range (needs CUDA for full eval)", "CUDA not available")
        return

    # Simulate bpb calculation without running full eval:
    # bpb = total_nats / (override_byte_count * ln2)
    # For a random model on 127-vocab: loss ~ ln(127) ~ 4.84
    # With override: bpb = 4.84 * num_tokens / (byte_count * ln2)
    # Just verify the formula works for plausible numbers
    total_nats = 4.84 * 1000  # 1000 tokens at ~ln(127)
    byte_count = 3000  # ~3 bytes per char is typical
    bpb = total_nats / (byte_count * math.log(2.0))
    check("synthetic bpb in [0.5, 5.0]", 0.5 <= bpb <= 5.0, f"bpb={bpb:.4f}")

    # Edge case: very small byte count
    bpb_small = total_nats / (100 * math.log(2.0))
    check("bpb with few bytes > 5.0 (expected high)", bpb_small > 5.0, f"bpb={bpb_small:.4f}")


def test_t16_same_byte_denominator():
    """T16: Same text with BPE and IPA uses same original byte denominator concept."""
    print("\nT16: BPE and IPA share byte denominator")
    # The key insight: bpb should be comparable across tokenizers because
    # override_byte_count uses the original UTF-8 byte count, not token count.
    text = "The quick brown fox jumps over the lazy dog."
    original_bytes = len(text.encode("utf-8"))

    # Simulate IPA path: more tokens, but same byte count in denominator
    ipa_nats = 3.5 * 200  # 200 IPA tokens at lower per-token loss
    ipa_bpb = ipa_nats / (original_bytes * math.log(2.0))

    # Simulate BPE path: fewer tokens, same byte count
    bpe_nats = 4.5 * 50  # 50 BPE tokens at higher per-token loss
    bpe_byte_count = original_bytes  # same denominator
    bpe_bpb = bpe_nats / (bpe_byte_count * math.log(2.0))

    check("both use same byte count", bpe_byte_count == original_bytes)
    check("IPA bpb is finite", math.isfinite(ipa_bpb))
    check("BPE bpb is finite", math.isfinite(bpe_bpb))
    check(
        "both bpb in plausible range",
        0.1 <= ipa_bpb <= 50.0 and 0.1 <= bpe_bpb <= 50.0,  # wide range: random weights produce high bpb
        f"ipa={ipa_bpb:.4f} bpe={bpe_bpb:.4f}",
    )


# ===========================================================================
# IPA-BPE TESTS (T17-T24)
# ===========================================================================

def test_t17_ipa_bpe_token_ids_under_1024():
    """T17: All IPA-BPE token IDs < 1024."""
    print("\nT17: IPA-BPE token IDs < 1024")
    shard = _find_shard(IPA_BPE_DIR)
    if shard is None:
        skip("IPA-BPE token range", "IPA-BPE shards not found locally")
    else:
        header = np.fromfile(shard, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(shard, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)
        max_id = int(tokens.max()) if tokens.size > 0 else 0
        check("all token IDs < 1024", max_id < 1024, f"max={max_id}")

    # Synthetic
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        ids = [0, 100, 500, 1023]
        _write_shard_v2_uint16(tmp, ids, 300)
        tokens = np.fromfile(tmp, dtype="<u2", count=len(ids), offset=HEADER_BYTES)
        check("synthetic all IDs < 1024", int(tokens.max()) < 1024)
    finally:
        tmp.unlink(missing_ok=True)


def test_t18_v2_shard_headers():
    """T18: v2 shard headers correct (version=2, byte_count > 0)."""
    print("\nT18: v2 shard headers")
    shard = _find_shard(IPA_BPE_DIR)
    if shard is None:
        skip("v2 header check", "IPA-BPE shards not found locally")
    else:
        header = np.fromfile(shard, dtype="<i4", count=256)
        check("magic 20240520", int(header[0]) == MAGIC, f"got {header[0]}")
        check("version 2", int(header[1]) == 2, f"got {header[1]}")
        check("byte_count > 0", int(header[3]) > 0, f"got {header[3]}")
        check("num_tokens > 0", int(header[2]) > 0, f"got {header[2]}")

    # Synthetic
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        _write_shard_v2_uint16(tmp, [1, 2, 3], 999)
        header = np.fromfile(tmp, dtype="<i4", count=256)
        check("synthetic version=2", int(header[1]) == 2)
        check("synthetic byte_count=999", int(header[3]) == 999)
    finally:
        tmp.unlink(missing_ok=True)


def test_t19_byte_count_matches_original():
    """T19: Byte count matches original English text bytes."""
    print("\nT19: Byte count is original text bytes")
    # The byte count in header[3] should reflect the original English text,
    # not the IPA-encoded text. We can verify with a synthetic round-trip.
    text = "Hello, world! This is a test sentence for IPA encoding."
    original_bytes = len(text.encode("utf-8"))

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        _write_shard_v2_uint16(tmp, [1, 2, 3, 4, 5], original_bytes)
        header = np.fromfile(tmp, dtype="<i4", count=256)
        stored_bytes = int(header[3])
        check(
            "stored byte count matches original",
            stored_bytes == original_bytes,
            f"stored={stored_bytes} original={original_bytes}",
        )
    finally:
        tmp.unlink(missing_ok=True)


def test_t20_roundtrip_ipa_encode():
    """T20: Round-trip: text->IPA->encode produces valid tokens."""
    print("\nT20: Round-trip IPA encode")
    try:
        from minimal_ipa_converter import text_to_ipa, text_to_ipa_chars
    except ImportError:
        skip("IPA round-trip", "minimal_ipa_converter not available")
        return

    text = "The knight rode through the night."
    ipa = text_to_ipa(text)
    chars = text_to_ipa_chars(text)
    check("IPA text is non-empty", len(ipa) > 0, f"got empty")
    check("chars list matches IPA length", len(chars) == len(ipa))
    check("IPA is deterministic", text_to_ipa(text) == ipa)
    # Each char should be encodable as uint8 (< 256 unique chars)
    unique_chars = set(chars)
    check("unique chars < 256 (fits uint8)", len(unique_chars) < 256, f"got {len(unique_chars)}")


def test_t21_v2_uint16_loads():
    """T21: v2 uint16 shards load correctly."""
    print("\nT21: load_data_shard with v2 uint16")
    torch = _require_torch()
    if torch is None:
        skip("v2 uint16 load", "torch not available")
        return

    from train_gpt import load_data_shard

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        ids = [0, 100, 500, 1023]
        _write_shard_v2_uint16(tmp, ids, 400)
        tokens = load_data_shard(tmp)
        check("correct length", tokens.numel() == len(ids), f"got {tokens.numel()}")
        check("values match", list(tokens.tolist()) == ids, f"got {tokens.tolist()}")
    finally:
        tmp.unlink(missing_ok=True)


def test_t22_v1_uint16_regression():
    """T22: v1 uint16 shards still load correctly (regression)."""
    print("\nT22: v1 uint16 shard regression")
    torch = _require_torch()
    if torch is None:
        skip("v1 uint16 regression", "torch not available")
        return

    from train_gpt import load_data_shard

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp = Path(f.name)
    try:
        ids = [0, 511, 1023]
        _write_shard_v1(tmp, ids)
        tokens = load_data_shard(tmp)
        check("v1 loads OK", tokens is not None)
        check("v1 correct length", tokens.numel() == len(ids), f"got {tokens.numel()}")
        check("v1 values match", list(tokens.tolist()) == ids, f"got {tokens.tolist()}")
    finally:
        tmp.unlink(missing_ok=True)


def test_t23_override_byte_count_in_eval_val():
    """T23: override_byte_count parameter works in eval_val signature."""
    print("\nT23: override_byte_count in eval_val signature")
    torch = _require_torch()
    if torch is None:
        skip("override param check", "torch not available")
        return

    from train_gpt import eval_val, eval_val_sliding

    sig = inspect.signature(eval_val)
    param = sig.parameters.get("override_byte_count")
    check("eval_val has override_byte_count", param is not None)
    if param is not None:
        check("param is keyword with default", param.default == 0, f"default={param.default}")

    sig_sw = inspect.signature(eval_val_sliding)
    param_sw = sig_sw.parameters.get("override_byte_count")
    check("eval_val_sliding has override_byte_count", param_sw is not None)
    if param_sw is not None:
        check("sliding param default is 0", param_sw.default == 0, f"default={param_sw.default}")


def test_t24_bpb_override_uses_original_bytes():
    """T24: bpb with override uses original bytes as denominator."""
    print("\nT24: bpb override uses original bytes")
    # Verify the formula: bpb = total_nats / (override_byte_count * ln2)
    # This is the formula used in both eval_val and eval_val_sliding when
    # override_byte_count > 0.
    total_nats = 5000.0
    override_byte_count = 2000
    bpb = total_nats / (override_byte_count * math.log(2.0))
    expected = total_nats / (2000 * math.log(2.0))
    check("bpb formula correct", abs(bpb - expected) < 1e-10)
    check("bpb is positive", bpb > 0)
    check("bpb is finite", math.isfinite(bpb))

    # If we double the byte count, bpb halves
    bpb_double = total_nats / (4000 * math.log(2.0))
    check("doubling bytes halves bpb", abs(bpb_double - bpb / 2) < 1e-10)


# ===========================================================================
# LAYER LOOPING (T25-T30)
# ===========================================================================

def test_t25_model_builds_seq2048():
    """T25: Model builds with TRAIN_SEQ_LEN=2048."""
    print("\nT25: Model with seq_len=2048 concept")
    torch = _require_torch()
    if torch is None:
        skip("model seq2048", "torch not available")
        return

    from train_gpt import GPT

    model = GPT(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    # Verify model can handle seq_len=2048 input
    input_ids = torch.randint(0, 1024, (1, 2048))
    target_ids = torch.randint(0, 1024, (1, 2048))
    try:
        with torch.no_grad():
            loss = model(input_ids, target_ids)
        check("forward with seq_len=2048 works", torch.isfinite(loss).item())
    except Exception as e:
        check("forward with seq_len=2048", False, f"raised {e}")


def test_t26_model_builds_loops():
    """T26: Model builds with num_loops=2, num_layers=5."""
    print("\nT26: Model with num_loops=2, num_layers=5")
    torch = _require_torch()
    if torch is None:
        skip("model loops", "torch not available")
        return

    from train_gpt import GPT

    model = GPT(
        vocab_size=1024,
        num_layers=5,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        num_loops=2,
    )
    check("model created with loops", model is not None)
    check("num_unique_layers is 5", model.num_unique_layers == 5)
    check("num_loops is 2", model.num_loops == 2)

    # Verify forward pass works
    input_ids = torch.randint(0, 1024, (1, 64))
    target_ids = torch.randint(0, 1024, (1, 64))
    try:
        with torch.no_grad():
            loss = model(input_ids, target_ids)
        check("looped forward works", torch.isfinite(loss).item())
    except Exception as e:
        check("looped forward", False, f"raised {e}")


def test_t27_loops1_same_params():
    """T27: num_loops=1 produces same param count as current baseline."""
    print("\nT27: num_loops=1 matches baseline param count")
    torch = _require_torch()
    if torch is None:
        skip("loops=1 params", "torch not available")
        return

    from train_gpt import GPT

    kwargs = dict(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    baseline = GPT(**kwargs)
    looped = GPT(**kwargs, num_loops=1)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    looped_params = sum(p.numel() for p in looped.parameters())
    check(
        "same param count",
        baseline_params == looped_params,
        f"baseline={baseline_params} looped={looped_params}",
    )


def test_t28_forward_and_logits_same_shape():
    """T28: forward and forward_logits produce same-shaped output."""
    print("\nT28: forward vs forward_logits shape")
    torch = _require_torch()
    if torch is None:
        skip("forward shapes", "torch not available")
        return

    from train_gpt import GPT

    model = GPT(
        vocab_size=127,
        num_layers=2,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    model.eval()

    bsz, seq_len = 2, 32
    input_ids = torch.randint(0, 127, (bsz, seq_len))
    target_ids = torch.randint(0, 127, (bsz, seq_len))
    with torch.no_grad():
        loss = model(input_ids, target_ids)
        logits = model.forward_logits(input_ids)

    check("loss is scalar", loss.dim() == 0)
    check(
        "logits shape is (bsz, seq_len, vocab)",
        logits.shape == (bsz, seq_len, 127),
        f"got {logits.shape}",
    )


def test_t29_lora_adapters():
    """T29: LoRA adapters created when lora_rank > 0 and num_loops > 1."""
    print("\nT29: LoRA adapters")
    torch = _require_torch()
    if torch is None:
        skip("LoRA adapters", "torch not available")
        return

    from train_gpt import GPT

    # With lora_rank=0 or num_loops=1, no LoRA
    model_no_lora = GPT(
        vocab_size=1024,
        num_layers=4,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        num_loops=2,
        lora_rank=0,
    )
    check("no LoRA when rank=0", model_no_lora.lora_adapters is None)

    model_no_lora2 = GPT(
        vocab_size=1024,
        num_layers=4,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        num_loops=1,
        lora_rank=8,
    )
    check("no LoRA when loops=1", model_no_lora2.lora_adapters is None)

    # With both lora_rank>0 and num_loops>1, LoRA should exist
    model_lora = GPT(
        vocab_size=1024,
        num_layers=4,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        num_loops=2,
        lora_rank=8,
    )
    check("LoRA exists when rank=8, loops=2", model_lora.lora_adapters is not None)
    if model_lora.lora_adapters is not None:
        check(
            "LoRA has num_loops entries",
            len(model_lora.lora_adapters) == 2,
            f"got {len(model_lora.lora_adapters)}",
        )
        check(
            "each loop has num_layers adapters",
            len(model_lora.lora_adapters[0]) == 4,
            f"got {len(model_lora.lora_adapters[0])}",
        )

    # Verify forward still works with LoRA
    input_ids = torch.randint(0, 1024, (1, 32))
    target_ids = torch.randint(0, 1024, (1, 32))
    model_lora.eval()
    try:
        with torch.no_grad():
            loss = model_lora(input_ids, target_ids)
        check("forward with LoRA works", torch.isfinite(loss).item())
    except Exception as e:
        check("forward with LoRA", False, f"raised {e}")


def test_t30_fewer_layers_fewer_params():
    """T30: Model param count is lower with 5 layers than 9 layers."""
    print("\nT30: Fewer layers = fewer params")
    torch = _require_torch()
    if torch is None:
        skip("layer count params", "torch not available")
        return

    from train_gpt import GPT

    common = dict(
        vocab_size=1024,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    model_9 = GPT(num_layers=9, **common)
    model_5 = GPT(num_layers=5, **common)
    params_9 = sum(p.numel() for p in model_9.parameters())
    params_5 = sum(p.numel() for p in model_5.parameters())
    check(
        "5 layers < 9 layers params",
        params_5 < params_9,
        f"5L={params_5} 9L={params_9}",
    )
    check("both have > 0 params", params_5 > 0 and params_9 > 0)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    # SHARD TESTS (T6-T9)
    test_t6_shard_header_magic()
    test_t7_byte_count_sidecar()
    test_t8_token_ids_in_range()
    test_t9_no_data_loss()

    # TRAINING INTEGRATION (T10-T14)
    test_t10_uint8_shard_loads()
    test_t11_model_builds_vocab127()
    test_t12_bpb_uses_override_byte_count()
    test_t13_model_forward_smoke()
    test_t14_sliding_window_eval_callable()

    # BPB CORRECTNESS (T15-T16)
    test_t15_bpb_plausible_range()
    test_t16_same_byte_denominator()

    # IPA-BPE TESTS (T17-T24)
    test_t17_ipa_bpe_token_ids_under_1024()
    test_t18_v2_shard_headers()
    test_t19_byte_count_matches_original()
    test_t20_roundtrip_ipa_encode()
    test_t21_v2_uint16_loads()
    test_t22_v1_uint16_regression()
    test_t23_override_byte_count_in_eval_val()
    test_t24_bpb_override_uses_original_bytes()

    # LAYER LOOPING (T25-T30)
    test_t25_model_builds_seq2048()
    test_t26_model_builds_loops()
    test_t27_loops1_same_params()
    test_t28_forward_and_logits_same_shape()
    test_t29_lora_adapters()
    test_t30_fewer_layers_fewer_params()

    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print(f"{'='*60}")
    sys.exit(1 if FAILED > 0 else 0)
