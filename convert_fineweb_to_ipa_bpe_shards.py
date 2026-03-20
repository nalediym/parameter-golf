#!/usr/bin/env python3
"""
Convert FineWeb BPE shards to IPA-BPE shards.

Reads BPE uint16 shards, decodes to text, converts to IPA,
re-tokenizes with the IPA-BPE SentencePiece model, and writes
version 2 uint16 shards with original byte counts in the header.

Usage:
    python3 convert_fineweb_to_ipa_bpe_shards.py --shards 1
    python3 convert_fineweb_to_ipa_bpe_shards.py --shards 80
    python3 convert_fineweb_to_ipa_bpe_shards.py --val-only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from minimal_ipa_converter import text_to_ipa


def load_bpe_shard(path):
    """Load a FineWeb BPE binary shard (uint16 with 256-int header)."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens


def write_ipa_bpe_shard(path, token_ids, original_byte_count):
    """Write IPA-BPE shard as version 2 with uint16 tokens."""
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 2  # version 2 = has original byte count
    header[2] = len(token_ids)
    header[3] = original_byte_count

    tokens_np = np.array(token_ids, dtype="<u2")

    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())

    # Sidecar for easy validation
    bytes_path = str(path).replace('.bin', '.bytes')
    with open(bytes_path, 'w') as f:
        f.write(str(original_byte_count))


def convert_shard(input_path, output_path, sp_bpe, sp_ipa_bpe, chunk_size=5000):
    """Convert one BPE shard to IPA-BPE shard."""
    t0 = time.time()
    bpe_tokens = load_bpe_shard(input_path)

    all_ipa_bpe_ids = []
    total_original_bytes = 0

    for i in range(0, len(bpe_tokens), chunk_size):
        chunk = bpe_tokens[i:i + chunk_size]
        text = sp_bpe.Decode(chunk.tolist())
        original_bytes = len(text.encode('utf-8'))
        total_original_bytes += original_bytes

        ipa_text = text_to_ipa(text)
        ipa_bpe_ids = sp_ipa_bpe.Encode(ipa_text)
        all_ipa_bpe_ids.extend(ipa_bpe_ids)

    # Validate
    max_id = max(all_ipa_bpe_ids) if all_ipa_bpe_ids else 0
    assert max_id < sp_ipa_bpe.vocab_size(), \
        f"Token ID {max_id} >= vocab_size {sp_ipa_bpe.vocab_size()}"
    assert max_id < 65536, f"Token ID {max_id} doesn't fit in uint16"

    write_ipa_bpe_shard(output_path, all_ipa_bpe_ids, total_original_bytes)

    elapsed = time.time() - t0
    return {
        'bpe_tokens': len(bpe_tokens),
        'ipa_bpe_tokens': len(all_ipa_bpe_ids),
        'original_bytes': total_original_bytes,
        'ratio_vs_bpe': len(all_ipa_bpe_ids) / len(bpe_tokens),
        'elapsed_sec': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert FineWeb BPE shards to IPA-BPE')
    parser.add_argument('--input-dir', type=str,
                        default='data/datasets/fineweb10B_sp1024')
    parser.add_argument('--output-dir', type=str,
                        default='data/datasets/fineweb10B_ipa_bpe')
    parser.add_argument('--bpe-tokenizer', type=str,
                        default='data/tokenizers/fineweb_1024_bpe.model')
    parser.add_argument('--ipa-bpe-tokenizer', type=str,
                        default='data/tokenizers/ipa_bpe_1024.model')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--val-only', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    print("=" * 60)
    print("FINEWEB BPE -> IPA-BPE SHARD CONVERSION")
    print("=" * 60)

    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.Load(args.bpe_tokenizer)
    print(f"Source BPE tokenizer: {sp_bpe.vocab_size()} vocab")

    sp_ipa_bpe = spm.SentencePieceProcessor()
    sp_ipa_bpe.Load(args.ipa_bpe_tokenizer)
    print(f"IPA-BPE tokenizer: {sp_ipa_bpe.vocab_size()} vocab")

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
            stats = convert_shard(val_path, out_path, sp_bpe, sp_ipa_bpe, args.chunk_size)
            total_original_bytes += stats['original_bytes']
            results.append(('val', val_path.name, stats))
            print(f"done ({stats['elapsed_sec']:.1f}s, {stats['ipa_bpe_tokens']:,} tokens, "
                  f"ratio={stats['ratio_vs_bpe']:.3f}x)")

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
            stats = convert_shard(train_path, out_path, sp_bpe, sp_ipa_bpe, args.chunk_size)
            total_original_bytes += stats['original_bytes']
            results.append(('train', train_path.name, stats))
            print(f"done ({stats['elapsed_sec']:.1f}s, {stats['ipa_bpe_tokens']:,} tokens, "
                  f"ratio={stats['ratio_vs_bpe']:.3f}x)")

    # Summary
    print(f"\n{'=' * 60}")
    print("CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    total_bpe = sum(s['bpe_tokens'] for _, _, s in results)
    total_ipa_bpe = sum(s['ipa_bpe_tokens'] for _, _, s in results)
    print(f"Shards converted:      {len(results)}")
    print(f"Total BPE tokens:      {total_bpe:,}")
    print(f"Total IPA-BPE tokens:  {total_ipa_bpe:,}")
    print(f"Total original bytes:  {total_original_bytes:,}")
    print(f"Token ratio vs BPE:    {total_ipa_bpe / total_bpe:.3f}x")
    print(f"Vocab size:            {sp_ipa_bpe.vocab_size()}")
    print(f"Output dir:            {output_dir}")

    with open(output_dir / 'total_original_bytes.txt', 'w') as f:
        f.write(str(total_original_bytes))

    log = {
        'shards': [(split, name, stats) for split, name, stats in results],
        'total_bpe_tokens': total_bpe,
        'total_ipa_bpe_tokens': total_ipa_bpe,
        'total_original_bytes': total_original_bytes,
        'token_ratio_vs_bpe': total_ipa_bpe / total_bpe,
        'vocab_size': sp_ipa_bpe.vocab_size(),
    }
    with open(output_dir / 'conversion_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nConversion log: {output_dir / 'conversion_log.json'}")


if __name__ == '__main__':
    main()
