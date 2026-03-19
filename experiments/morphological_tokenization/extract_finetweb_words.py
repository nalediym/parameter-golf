"""
Extract Word Frequencies from FineWeb Shards

Analyzes actual FineWeb binary files to extract word frequencies,
then builds comprehensive morpheme vocabulary.

Usage:
    python extract_finetweb_words.py \
        --data_path ../../data/datasets/fineweb10B_sp1024/ \
        --sp_model ../../data/tokenizers/fineweb_1024_bpe.model \
        --output data/fineweb_word_freq.json \
        --max_shards 1 \
        --max_tokens_per_shard 10000000
"""

import argparse
import json
import struct
from pathlib import Path
from collections import Counter
import sys
import re
import numpy as np


def load_sentencepiece(model_path: Path):
    """Load SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        return sp
    except ImportError:
        print("Error: sentencepiece not installed")
        sys.exit(1)


def read_shard_tokens(shard_path: Path):
    """
    Read tokens from FineWeb shard with proper header parsing.
    
    Format:
    - 256 int32 header values
    - Header[0] = magic number (20240520)
    - Header[1] = version (1)  
    - Header[2] = number of tokens
    - Tokens are uint16, start after 1024-byte header
    """
    # Read header
    header = np.fromfile(shard_path, dtype="<i4", count=256)
    
    if header.size != 256:
        print(f"Warning: {shard_path.name} - invalid header size {header.size}")
        return []
    
    if int(header[0]) != 20240520:
        print(f"Warning: {shard_path.name} - unexpected magic {header[0]}")
        return []
    
    num_tokens = int(header[2])
    
    # Read tokens after header
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(shard_path, dtype="<u2", count=num_tokens, offset=header_bytes)
    
    return tokens.tolist()


def extract_words_from_tokens(tokens: list, sp_model, batch_size: int = 1000):
    """
    Extract words from BPE tokens by decoding to text.
    
    Args:
        tokens: List of BPE token IDs
        sp_model: SentencePiece processor
        batch_size: Number of tokens to decode at once
    
    Returns:
        Counter of word frequencies
    """
    word_freq = Counter()
    
    # Process in batches to avoid memory issues
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        
        try:
            # Decode batch to text
            text = sp_model.decode(batch)
            
            # Extract words (alphanumeric sequences)
            words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
            
            # Count frequencies
            word_freq.update(words)
            
        except Exception as e:
            print(f"  Warning: decode error at position {i}: {e}")
            continue
    
    return word_freq


def main():
    parser = argparse.ArgumentParser(
        description="Extract word frequencies from FineWeb dataset"
    )
    parser.add_argument("--data_path", type=str,
                       default="../../data/datasets/fineweb10B_sp1024/",
                       help="Path to FineWeb dataset directory")
    parser.add_argument("--sp_model", type=str,
                       default="../../data/tokenizers/fineweb_1024_bpe.model",
                       help="Path to SentencePiece model")
    parser.add_argument("--output", type=str,
                       default="data/fineweb_word_freq.json",
                       help="Output path for word frequency JSON")
    parser.add_argument("--max_shards", type=int, default=1,
                       help="Maximum number of shards to process")
    parser.add_argument("--max_tokens_per_shard", type=int, default=None,
                       help="Maximum tokens per shard (for testing)")
    parser.add_argument("--min_freq", type=int, default=5,
                       help="Minimum word frequency to include in output")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    sp_model = load_sentencepiece(Path(args.sp_model))
    
    # Find shards
    train_shards = sorted(data_path.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_path.glob("fineweb_val_*.bin"))
    
    all_shards = val_shards + train_shards
    
    if args.max_shards:
        all_shards = all_shards[:args.max_shards]
    
    print(f"Found {len(train_shards)} train shards, {len(val_shards)} val shards")
    print(f"Processing {len(all_shards)} shard(s)...")
    print()
    
    # Extract word frequencies
    total_word_freq = Counter()
    total_tokens = 0
    
    for i, shard_path in enumerate(all_shards):
        print(f"[{i+1}/{len(all_shards)}] Processing {shard_path.name}...")
        
        # Read tokens
        tokens = read_shard_tokens(shard_path)
        
        if args.max_tokens_per_shard:
            tokens = tokens[:args.max_tokens_per_shard]
        
        if not tokens:
            print(f"  Skipping empty shard")
            continue
        
        print(f"  Read {len(tokens):,} tokens")
        total_tokens += len(tokens)
        
        # Extract words
        print(f"  Extracting words...")
        shard_words = extract_words_from_tokens(tokens, sp_model)
        
        print(f"  Found {len(shard_words):,} unique words")
        
        # Merge with global counter
        total_word_freq.update(shard_words)
        
        print(f"  Running total: {len(total_word_freq):,} unique words")
        print()
    
    # Filter by minimum frequency
    filtered_freq = {word: count 
                     for word, count in total_word_freq.items() 
                     if count >= args.min_freq}
    
    # Sort by frequency
    sorted_freq = dict(sorted(filtered_freq.items(), 
                              key=lambda x: x[1], 
                              reverse=True))
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metadata': {
            'total_tokens_processed': total_tokens,
            'shards_processed': len(all_shards),
            'unique_words_found': len(total_word_freq),
            'words_above_min_freq': len(sorted_freq),
            'min_frequency': args.min_freq,
        },
        'word_frequencies': sorted_freq
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Unique words found: {len(total_word_freq):,}")
    print(f"Words above min freq ({args.min_freq}): {len(sorted_freq):,}")
    print()
    print("Top 20 words:")
    for word, count in list(sorted_freq.items())[:20]:
        print(f"  {word:20} {count:8,}")
    print()
    print(f"Saved to {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
