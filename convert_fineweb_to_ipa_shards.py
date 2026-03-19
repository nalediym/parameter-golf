#!/usr/bin/env python3
"""
Convert FineWeb BPE shards to IPA uint8 shards with byte_count sidecars.

Reads BPE uint16 shards, decodes to text, converts to IPA chars,
encodes as uint8 token IDs, and writes binary shards in the same
header format that train_gpt.py expects.

Also writes a .bytes sidecar file per shard recording the original
UTF-8 byte count, needed for correct bpb calculation.

Usage:
    python3 convert_fineweb_to_ipa_shards.py --shards 1     # 1 shard (smoke test)
    python3 convert_fineweb_to_ipa_shards.py --shards 80    # all shards
    python3 convert_fineweb_to_ipa_shards.py --val-only      # just validation
"""
import argparse
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from minimal_ipa_converter import text_to_ipa


def load_vocab(vocab_path):
    """Load IPA vocab mapping from JSON."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['char_to_id'], data['vocab_size']


def build_vocab_from_data(ipa_text, existing_vocab=None):
    """Build or extend vocab from observed IPA characters."""
    if existing_vocab:
        char_to_id = dict(existing_vocab)
        next_id = max(char_to_id.values()) + 1
    else:
        char_to_id = {'<UNK>': 0}
        next_id = 1

    for ch in ipa_text:
        if ch not in char_to_id:
            char_to_id[ch] = next_id
            next_id += 1

    return char_to_id, next_id


def encode_ipa_text(ipa_text, char_to_id):
    """Encode IPA string to uint8 token IDs."""
    unk_id = char_to_id.get('<UNK>', 0)
    return np.array([char_to_id.get(ch, unk_id) for ch in ipa_text], dtype=np.uint8)


def load_bpe_shard(path):
    """Load a FineWeb BPE binary shard (uint16 with 256-int header)."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens


def write_ipa_shard(path, tokens_uint8, original_byte_count):
    """Write IPA shard in same header format as BPE shards, but uint8.

    Header: 256 int32 values
      [0] = magic (20240520)
      [1] = version (2 = IPA uint8 format)
      [2] = num_tokens
      [3] = original_byte_count (for bpb calculation)
    """
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 2  # version 2 = IPA uint8
    header[2] = len(tokens_uint8)
    header[3] = original_byte_count

    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens_uint8.tobytes())

    # Also write byte count as a simple sidecar for easy validation
    bytes_path = str(path).replace('.bin', '.bytes')
    with open(bytes_path, 'w') as f:
        f.write(str(original_byte_count))


def convert_shard(input_path, output_path, sp, char_to_id, chunk_size=5000):
    """Convert one BPE shard to IPA uint8 shard."""
    t0 = time.time()
    bpe_tokens = load_bpe_shard(input_path)

    # Decode BPE tokens to text in chunks, convert to IPA
    all_ipa_chars = []
    total_original_bytes = 0

    for i in range(0, len(bpe_tokens), chunk_size):
        chunk = bpe_tokens[i:i + chunk_size]
        text = sp.Decode(chunk.tolist())
        original_bytes = len(text.encode('utf-8'))
        total_original_bytes += original_bytes

        ipa_text = text_to_ipa(text)
        all_ipa_chars.append(ipa_text)

    full_ipa = ''.join(all_ipa_chars)
    ipa_tokens = encode_ipa_text(full_ipa, char_to_id)

    # Validate: all tokens in range
    vocab_size = max(char_to_id.values()) + 1
    assert ipa_tokens.max() < vocab_size, \
        f"Token ID {ipa_tokens.max()} >= vocab_size {vocab_size}"
    assert ipa_tokens.dtype == np.uint8

    write_ipa_shard(output_path, ipa_tokens, total_original_bytes)

    elapsed = time.time() - t0
    return {
        'bpe_tokens': len(bpe_tokens),
        'ipa_tokens': len(ipa_tokens),
        'original_bytes': total_original_bytes,
        'expansion_ratio': len(ipa_tokens) / len(bpe_tokens),
        'elapsed_sec': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert FineWeb BPE shards to IPA uint8')
    parser.add_argument('--input-dir', type=str,
                        default='data/datasets/fineweb10B_sp1024')
    parser.add_argument('--output-dir', type=str,
                        default='data/datasets/fineweb10B_ipa')
    parser.add_argument('--tokenizer', type=str,
                        default='data/tokenizers/fineweb_1024_bpe.model')
    parser.add_argument('--vocab', type=str,
                        default='data/ipa_vocab.json')
    parser.add_argument('--shards', type=int, default=1,
                        help='Number of train shards to convert (0=skip train)')
    parser.add_argument('--val-only', action='store_true',
                        help='Only convert validation shards')
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    print("=" * 60)
    print("FINEWEB BPE -> IPA UINT8 SHARD CONVERSION")
    print("=" * 60)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"BPE tokenizer loaded: {sp.vocab_size()} vocab")

    # Load or build vocab
    vocab_path = Path(args.vocab)
    if vocab_path.exists():
        char_to_id, vocab_size = load_vocab(vocab_path)
        print(f"IPA vocab loaded: {vocab_size} tokens from {vocab_path}")
    else:
        print("WARNING: No vocab file found, will build from data")
        char_to_id = {'<UNK>': 0}
        vocab_size = 1

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    results = []
    total_original_bytes = 0

    # Convert validation shards
    val_shards = sorted(input_dir.glob("fineweb_val_*.bin"))
    if val_shards:
        print(f"\n--- Converting {len(val_shards)} validation shard(s) ---")
        for val_path in val_shards:
            out_path = output_dir / val_path.name
            print(f"  {val_path.name} ...", end=" ", flush=True)
            stats = convert_shard(val_path, out_path, sp, char_to_id, args.chunk_size)
            total_original_bytes += stats['original_bytes']
            results.append(('val', val_path.name, stats))
            print(f"done ({stats['elapsed_sec']:.1f}s, {stats['ipa_tokens']:,} IPA tokens, "
                  f"{stats['original_bytes']:,} original bytes)")

    # Convert training shards
    if not args.val_only and args.shards > 0:
        print(f"\n--- Converting {args.shards} training shard(s) ---")
        for shard_idx in range(args.shards):
            train_path = input_dir / f"fineweb_train_{shard_idx:06d}.bin"
            if not train_path.exists():
                print(f"  WARNING: {train_path} not found, skipping")
                continue
            out_path = output_dir / train_path.name
            print(f"  {train_path.name} ...", end=" ", flush=True)
            stats = convert_shard(train_path, out_path, sp, char_to_id, args.chunk_size)
            total_original_bytes += stats['original_bytes']
            results.append(('train', train_path.name, stats))
            print(f"done ({stats['elapsed_sec']:.1f}s, {stats['ipa_tokens']:,} IPA tokens, "
                  f"{stats['original_bytes']:,} original bytes)")

    # Summary
    print(f"\n{'=' * 60}")
    print("CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Shards converted: {len(results)}")
    total_bpe = sum(s['bpe_tokens'] for _, _, s in results)
    total_ipa = sum(s['ipa_tokens'] for _, _, s in results)
    print(f"Total BPE tokens:      {total_bpe:,}")
    print(f"Total IPA tokens:      {total_ipa:,}")
    print(f"Total original bytes:  {total_original_bytes:,}")
    print(f"Expansion ratio:       {total_ipa / total_bpe:.2f}x")
    print(f"IPA tokens/orig byte:  {total_ipa / total_original_bytes:.2f}")
    print(f"Vocab size:            {max(char_to_id.values()) + 1}")
    print(f"Output dir:            {output_dir}")

    # Write total byte count for validation
    with open(output_dir / 'total_original_bytes.txt', 'w') as f:
        f.write(str(total_original_bytes))

    # Write conversion log
    log = {
        'shards': [(split, name, stats) for split, name, stats in results],
        'total_bpe_tokens': total_bpe,
        'total_ipa_tokens': total_ipa,
        'total_original_bytes': total_original_bytes,
        'expansion_ratio': total_ipa / total_bpe,
        'vocab_size': max(char_to_id.values()) + 1,
    }
    with open(output_dir / 'conversion_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nConversion log saved to {output_dir / 'conversion_log.json'}")


if __name__ == '__main__':
    main()
