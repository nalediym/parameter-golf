#!/usr/bin/env python3
"""
Fast FineWeb BPE → Morpheme Token Converter

Optimized version of convert_fineweb_to_morph.py.
Targets ~10-50x speedup via:
  1. Bulk BPE decode (100k+ tokens at a time)
  2. Segmentation cache (dict lookup instead of regex per word)
  3. Numpy binary I/O (vectorized read/write)
  4. Multiprocessing for segmentation
  5. Pre-warmed cache from lexicon

Usage:
    python fast_convert.py --max_tokens 1000000  # test run
    python fast_convert.py                        # full 62M tokens
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # 1024 bytes
MAGIC = 20240520
VERSION = 1

DEFAULT_INPUT = "../../data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
DEFAULT_OUTPUT = "data/fineweb_val_morph_v2_000000.bin"
DEFAULT_VOCAB = "data/morph_vocab_v2.json"
DEFAULT_LEXICON = "data/morph_lexicon_v2.json"
DEFAULT_SP_MODEL = "../../data/tokenizers/fineweb_1024_bpe.model"

WORD_RE = re.compile(r"[a-zA-Z']+")

# Affix tables (mirrored from AgglutinativeTokenizer for standalone use)
DERIVATIONAL_PREFIXES = sorted([
    "anti", "auto", "bi", "circum", "co", "con", "com", "contra", "counter",
    "de", "dis", "en", "em", "ex", "extra", "fore", "hetero", "homo",
    "hyper", "hypo", "il", "im", "in", "ir", "inter", "intra", "macro",
    "mal", "micro", "mid", "mis", "mono", "multi", "non", "ob", "oc",
    "of", "op", "out", "over", "para", "peri", "poly", "post", "pre",
    "pro", "pseudo", "re", "retro", "semi", "sub", "super", "supra",
    "sur", "syn", "trans", "tri", "ultra", "un", "under", "uni",
], key=len, reverse=True)

DERIVATIONAL_SUFFIXES = sorted([
    "tion", "sion", "ation", "ition", "ment", "ness", "ity", "ty",
    "er", "or", "ist", "ism", "cy", "ence", "ance", "ure",
    "age", "al", "dom", "ee", "ery", "ess", "ful", "hood", "ing",
    "ship", "th", "y",
    "able", "ible", "ant", "ent", "ar", "ary", "ed", "en",
    "ern", "ese", "ian", "ic", "ical", "ious", "ous", "ish",
    "ive", "less", "ly", "ory", "some", "ward", "wise",
    "ate", "ify", "ise", "ize",
    "wards",
], key=len, reverse=True)

INFLECTIONAL_SUFFIXES = sorted([
    "s", "es", "ed", "ing", "er", "est",
], key=len, reverse=True)

ALL_SUFFIXES = sorted(
    list(set(DERIVATIONAL_SUFFIXES + INFLECTIONAL_SUFFIXES)),
    key=len, reverse=True,
)


# ---------------------------------------------------------------------------
# Standalone segmentation (no class overhead, cacheable)
# ---------------------------------------------------------------------------

def segment_word(word: str) -> List[str]:
    """
    Segment a word into morphemes using the agglutinative strategy.
    Standalone function — no class, no dataclass allocation.
    Returns list of morpheme strings.
    """
    w = word.lower()
    if len(w) <= 3:
        return [w]

    # Try prefix stripping (recursive)
    for prefix in DERIVATIONAL_PREFIXES:
        if w.startswith(prefix) and len(w) > len(prefix) + 2:
            remainder = w[len(prefix):]
            return [prefix] + segment_word(remainder)

    # Try suffix stripping
    for suffix in ALL_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            root = w[:-len(suffix)]
            if len(root) >= 3:
                return [root, suffix]

    return [w]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_bpe_tokens(path: str, max_tokens: int = None) -> np.ndarray:
    """Read BPE tokens from binary via numpy (zero-copy)."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    assert header[0] == MAGIC and header[1] == VERSION, (
        f"Bad header: magic={header[0]}, version={header[1]}"
    )
    num_tokens = int(header[2])

    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def write_morph_binary(path: str, morph_ids: np.ndarray):
    """Write morpheme IDs as uint16 binary (no header, flat)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    morph_ids.astype("<u2").tofile(path)


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_segmentation_cache(
    lexicon: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Pre-build a word→morphemes cache from the lexicon.
    This avoids regex/rule evaluation at runtime for known words.
    """
    cache: Dict[str, List[str]] = {}
    for word, morphemes in lexicon.items():
        cache[word.lower()] = morphemes
    return cache


# ---------------------------------------------------------------------------
# Two-pass converter: decode → collect unique → batch segment → fast emit
# ---------------------------------------------------------------------------

def _segment_batch(words: List[str]) -> Dict[str, List[str]]:
    """Segment a batch of words. Used by multiprocessing."""
    return {w: segment_word(w) for w in words}


def convert_two_pass(
    bpe_tokens: np.ndarray,
    sp_model_path: str,
    token_to_id: Dict[str, int],
    unk_id: int,
    cache: Dict[str, List[str]],
    decode_batch_size: int = 500_000,
    num_workers: int = 0,
) -> Tuple[np.ndarray, Dict[str, List[str]], Counter]:
    """
    Two-pass conversion:
      Pass 1: Decode all BPE → text → words. Collect unique words.
      Pass 2: Segment only unique uncached words (optionally parallel).
      Pass 3: Fast lookup emit — all words are in cache.
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    total = len(bpe_tokens)

    # ---- PASS 1: Decode + collect all words ----
    print("  Pass 1: Decode BPE → text → words...")
    t0 = time.time()
    all_words: List[str] = []  # ordered word stream (for pass 3)

    for start in range(0, total, decode_batch_size):
        end = min(start + decode_batch_size, total)
        chunk = bpe_tokens[start:end].tolist()
        text = sp.decode(chunk)
        words = WORD_RE.findall(text.lower())
        # Filter len>1 inline
        all_words.extend(w for w in words if len(w) > 1)

        pct = 100 * end / total
        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        print(f"    [{pct:5.1f}%] decoded {end:,}/{total:,} tokens | "
              f"{rate:,.0f} tok/s | words so far: {len(all_words):,}")

    t1 = time.time()
    word_counts = Counter(all_words)
    unique_words = set(word_counts.keys())
    uncached = unique_words - set(cache.keys())
    print(f"    Total words: {len(all_words):,} | "
          f"Unique: {len(unique_words):,} | "
          f"Uncached: {len(uncached):,} | "
          f"Time: {t1 - t0:.1f}s")

    # ---- PASS 2: Batch segment uncached words ----
    print(f"  Pass 2: Segmenting {len(uncached):,} uncached words...")
    t2 = time.time()
    uncached_list = list(uncached)

    if num_workers > 1 and len(uncached_list) > 1000:
        # Split into chunks for parallel processing
        chunk_sz = max(500, len(uncached_list) // (num_workers * 4))
        word_chunks = [
            uncached_list[i:i + chunk_sz]
            for i in range(0, len(uncached_list), chunk_sz)
        ]
        print(f"    Dispatching {len(word_chunks)} chunks to {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for result_dict in pool.map(_segment_batch, word_chunks):
                cache.update(result_dict)
    else:
        # Single process — still fast because it's only unique words
        for i, w in enumerate(uncached_list):
            cache[w] = segment_word(w)
            if (i + 1) % 10000 == 0:
                print(f"    Segmented {i + 1:,}/{len(uncached_list):,}")

    t3 = time.time()
    print(f"    Segmentation time: {t3 - t2:.1f}s "
          f"({len(uncached_list) / max(t3 - t2, 0.001):,.0f} words/s)")

    # ---- PASS 3: Fast emit — pure dict lookup ----
    print("  Pass 3: Emitting morpheme IDs...")
    t4 = time.time()

    # Pre-build morpheme→id lookup for speed (avoid repeated dict.get)
    # Most morphemes will be the same few hundred, so a local cache helps
    morph_id_cache: Dict[str, int] = {}

    all_morph_ids: List[int] = []
    for word in all_words:
        morphemes = cache[word]  # guaranteed hit after pass 2
        for m in morphemes:
            mid = morph_id_cache.get(m)
            if mid is None:
                mid = token_to_id.get(m, unk_id)
                morph_id_cache[m] = mid
            all_morph_ids.append(mid)

    t5 = time.time()
    print(f"    Emit time: {t5 - t4:.1f}s | "
          f"morph tokens: {len(all_morph_ids):,} | "
          f"unique morphemes: {len(morph_id_cache):,}")

    return np.array(all_morph_ids, dtype=np.uint16), cache, word_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fast FineWeb BPE → morpheme converter"
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab_path", default=DEFAULT_VOCAB)
    parser.add_argument("--lexicon_path", default=DEFAULT_LEXICON)
    parser.add_argument("--sp_model_path", default=DEFAULT_SP_MODEL)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0,
                        help="0=single-process (default, fastest for <10M), "
                             "N=multiprocess with N workers")
    parser.add_argument("--decode_batch", type=int, default=200_000,
                        help="BPE tokens per decode batch (single-process mode)")
    args = parser.parse_args()

    t_start = time.time()

    # ---- Load vocab ----
    print("Loading vocab...")
    vocab = json.load(open(args.vocab_path))
    token_to_id: Dict[str, int] = vocab["token_to_id"]
    unk_id = token_to_id.get("<UNK>", 1)
    print(f"  Vocab size: {vocab['vocab_size']}")

    # ---- Load lexicon into cache ----
    print("Loading lexicon cache...")
    lexicon_data = json.load(open(args.lexicon_path))
    lexicon = lexicon_data.get("lexicon", {})
    cache = build_segmentation_cache(lexicon)
    print(f"  Pre-cached {len(cache):,} words from lexicon")

    # ---- Read BPE tokens ----
    print("Reading BPE tokens...")
    bpe_tokens = read_bpe_tokens(args.input_path, args.max_tokens)
    print(f"  Loaded {len(bpe_tokens):,} tokens")

    t_load = time.time()
    print(f"  Load time: {t_load - t_start:.1f}s")

    # ---- Convert ----
    print("Converting...")
    sp_model_path = str(Path(args.sp_model_path).resolve())

    morph_ids, cache, word_counts = convert_two_pass(
        bpe_tokens, sp_model_path, token_to_id, unk_id, cache,
        decode_batch_size=args.decode_batch,
        num_workers=args.workers,
    )

    t_convert = time.time()

    # ---- Write output ----
    print("Writing output...")
    write_morph_binary(args.output_path, morph_ids)

    t_end = time.time()

    # ---- Stats ----
    out_size = Path(args.output_path).stat().st_size
    convert_elapsed = t_convert - t_load
    total_elapsed = t_end - t_start
    bpe_count = len(bpe_tokens)
    morph_count = len(morph_ids)
    ratio = morph_count / bpe_count if bpe_count else 0
    tok_per_sec = bpe_count / convert_elapsed if convert_elapsed > 0 else 0

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"  Input:           {args.input_path}")
    print(f"  Output:          {args.output_path}")
    print(f"  Output size:     {out_size:,} bytes ({out_size/1024/1024:.1f} MB)")
    print(f"  BPE tokens:      {bpe_count:,}")
    print(f"  Morph tokens:    {morph_count:,}")
    print(f"  BPE→Morph ratio: {ratio:.3f}x")
    print(f"  Unique words:    {len(word_counts):,}")
    print(f"  Cache entries:   {len(cache):,}")
    print(f"  Convert time:    {convert_elapsed:.1f}s ({tok_per_sec:,.0f} BPE tok/s)")
    print(f"  Total time:      {total_elapsed:.1f}s")
    print()
    print(f"  Top 10 words: {word_counts.most_common(10)}")

    # ---- Save report ----
    report_path = Path(args.output_path).with_suffix(".report.json")
    report = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "bpe_tokens": bpe_count,
        "morph_tokens": morph_count,
        "ratio": round(ratio, 4),
        "unique_words": len(word_counts),
        "cache_entries": len(cache),
        "convert_seconds": round(convert_elapsed, 2),
        "total_seconds": round(total_elapsed, 2),
        "tokens_per_second": round(tok_per_sec, 0),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
