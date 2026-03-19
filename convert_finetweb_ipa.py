#!/usr/bin/env python3
"""
Convert FineWeb dataset to IPA using minimal converter.

This script converts the binary FineWeb shards to IPA format using our
minimal rule-based converter instead of the 3MB eng_to_ipa library.
"""

import numpy as np
import sentencepiece as spm
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Import our minimal converter functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from minimal_ipa_converter import text_to_ipa, EXCEPTIONS, G2P_RULES


def load_tokenizer(tokenizer_path):
    """Load the SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    return sp


def load_data_shard(path):
    """Load a FineWeb binary shard."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    
    return tokens


def chunk_tokens(tokens, chunk_size=1000):
    """Split tokens into chunks that likely form complete sentences."""
    # FineWeb uses document boundaries - look for common end-of-doc tokens
    # For now, just chunk by fixed size
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i+chunk_size]


def convert_shard(input_path, output_path, tokenizer, chunk_size=1000):
    """
    Convert a single FineWeb shard to IPA format.
    
    Process:
    1. Load tokens from binary file
    2. Decode to text using tokenizer
    3. Convert text to IPA
    4. Save as UTF-8 text file
    """
    print(f"Loading {input_path}...")
    tokens = load_data_shard(input_path)
    
    print(f"Converting {len(tokens):,} tokens to IPA...")
    
    ipa_chunks = []
    total_original_chars = 0
    total_ipa_chars = 0
    
    # Process in chunks to show progress
    chunk_iter = chunk_tokens(tokens, chunk_size)
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    
    for chunk in tqdm(chunk_iter, total=num_chunks, desc="Converting"):
        # Decode to text
        text = tokenizer.Decode(chunk.tolist())
        total_original_chars += len(text)
        
        # Convert to IPA
        ipa_text = text_to_ipa(text)
        total_ipa_chars += len(ipa_text)
        
        ipa_chunks.append(ipa_text)
    
    # Join all chunks
    full_ipa = '\n'.join(ipa_chunks)
    
    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_ipa)
    
    # Calculate stats
    original_size = os.path.getsize(input_path)
    ipa_size = os.path.getsize(output_path)
    
    return {
        'original_tokens': len(tokens),
        'original_chars': total_original_chars,
        'ipa_chars': total_ipa_chars,
        'original_size': original_size,
        'ipa_size': ipa_size,
        'compression_ratio': original_size / ipa_size,
        'char_expansion': total_ipa_chars / total_original_chars,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert FineWeb to IPA')
    parser.add_argument('--input-dir', type=str, 
                        default='data/datasets/fineweb10B_sp1024',
                        help='Input directory with FineWeb .bin files')
    parser.add_argument('--output-dir', type=str,
                        default='data/ipa_converted_minimal',
                        help='Output directory for IPA files')
    parser.add_argument('--tokenizer', type=str,
                        default='data/tokenizers/fineweb_1024_bpe.model',
                        help='Path to tokenizer model')
    parser.add_argument('--shards', type=int, default=3,
                        help='Number of shards to convert (default: 3)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Tokens per chunk for processing (default: 1000)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINEWEB TO IPA CONVERSION (Minimal Converter)")
    print("=" * 60)
    
    # Show converter info
    print(f"\nConverter size:")
    print(f"  Exception dict: {len(EXCEPTIONS)} entries")
    print(f"  G2P rules: {len(G2P_RULES)} patterns")
    print(f"  Est. total: ~13 KB")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert shards
    results = []
    for shard_idx in range(args.shards):
        input_file = Path(args.input_dir) / f"fineweb_train_{shard_idx:06d}.bin"
        output_file = output_dir / f"fineweb_train_{shard_idx:06d}_ipa.txt"
        
        if not input_file.exists():
            print(f"\n⚠️  Shard {shard_idx} not found: {input_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing shard {shard_idx}")
        print(f"{'='*60}")
        
        stats = convert_shard(input_file, output_file, tokenizer, args.chunk_size)
        results.append(stats)
        
        print(f"\nShard {shard_idx} stats:")
        print(f"  Original: {stats['original_size']:,} bytes")
        print(f"  IPA:      {stats['ipa_size']:,} bytes")
        print(f"  Ratio:    {stats['compression_ratio']:.1f}x")
        print(f"  Expansion: {stats['char_expansion']:.2f}x")
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        total_original = sum(r['original_size'] for r in results)
        total_ipa = sum(r['ipa_size'] for r in results)
        total_tokens = sum(r['original_tokens'] for r in results)
        
        print(f"\nTotal shards converted: {len(results)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total original size: {total_original:,} bytes ({total_original/1024/1024:.1f} MB)")
        print(f"Total IPA size: {total_ipa:,} bytes ({total_ipa/1024:.1f} KB)")
        print(f"Overall compression: {total_original/total_ipa:.1f}x")
        
        # Space calculation
        embedding_save_kb = 942  # from earlier analysis
        converter_cost_kb = 13
        net_save_kb = embedding_save_kb - converter_cost_kb
        
        print(f"\n{'='*60}")
        print("PARAMETER GOLF IMPACT")
        print(f"{'='*60}")
        print(f"\nEmbedding savings: +{embedding_save_kb} KB")
        print(f"Converter cost:    -{converter_cost_kb} KB")
        print(f"Net savings:       +{net_save_kb} KB ✅")
        print(f"\nYou can fit ~{net_save_kb/1024:.1f} MB more model parameters!")
    
    print(f"\nOutput saved to: {output_dir}")


if __name__ == '__main__':
    main()
