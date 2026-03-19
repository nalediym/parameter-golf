#!/usr/bin/env python3
"""
View internet text from the FineWeb training dataset.

This script decodes the binary token files back into human-readable text.
"""

import numpy as np
import sentencepiece as spm
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='View decoded text from FineWeb training data'
    )
    parser.add_argument(
        '--shard',
        type=int,
        default=0,
        help='Which training shard to view (default: 0)'
    )
    parser.add_argument(
        '--tokens',
        type=int,
        default=200,
        help='How many tokens to decode and show (default: 200)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start position in the shard (default: 0)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/datasets/fineweb10B_sp1024',
        help='Path to dataset folder'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='data/tokenizers/fineweb_1024_bpe.model',
        help='Path to tokenizer model file'
    )

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer not found at {args.tokenizer_path}")
        print("Run: python3 data/cached_challenge_fineweb.py --variant sp1024")
        return

    shard_file = f"{args.dataset_path}/fineweb_train_{args.shard:06d}.bin"
    if not os.path.exists(shard_file):
        print(f"Error: Shard not found at {shard_file}")
        print(f"Available shards: {args.dataset_path}/")
        return

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    print(f"Vocabulary size: {sp.vocab_size()} tokens\n")

    # Load tokens from shard (with header parsing)
    print(f"Loading shard {args.shard} from {shard_file}...")
    
    # Read header (256 int32 values)
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(shard_file, dtype="<i4", count=256)
    
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header format")
    
    num_tokens = int(header[2])
    print(f"Shard contains {num_tokens:,} tokens total\n")
    
    # Read actual tokens (after header)
    tokens = np.fromfile(shard_file, dtype="<u2", count=num_tokens, offset=header_bytes)

    # Validate range
    if args.start >= len(tokens):
        print(f"Error: Start position {args.start} is beyond shard length {len(tokens)}")
        return

    end_pos = min(args.start + args.tokens, len(tokens))
    actual_tokens = end_pos - args.start

    # Decode tokens to text
    print(f"{'='*60}")
    print(f"Showing {actual_tokens} tokens starting at position {args.start}")
    print(f"{'='*60}\n")

    token_slice = tokens[args.start:end_pos]
    text = sp.Decode(token_slice.tolist())

    print(text)

    print(f"\n{'='*60}")
    print(f"End of sample")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
