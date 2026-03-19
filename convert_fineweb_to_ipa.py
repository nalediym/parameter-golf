#!/usr/bin/env python3
"""
Convert FineWeb training data to IPA notation and analyze file sizes.

Usage:
    python3 convert_fineweb_to_ipa.py [--shards N] [--output-dir DIR]

Default: processes 3 shards
"""

import numpy as np
import sentencepiece as spm
import eng_to_ipa as ipa
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import struct


def load_tokenizer(tokenizer_path):
    """Load the SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    return sp


def read_shard(shard_path):
    """Read a binary shard file and return tokens."""
    # Read header (256 int32 values)
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(shard_path, dtype="<i4", count=256)
    
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header format for {shard_path}")
    
    num_tokens = int(header[2])
    
    # Read actual tokens (after header) - uint16 for sp1024
    tokens = np.fromfile(shard_path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens, num_tokens


def decode_tokens_to_text(tokenizer, tokens, batch_size=10000):
    """Decode tokens to text in batches to manage memory."""
    # Try decoding the entire batch at once for efficiency
    # SentencePiece can handle large sequences
    try:
        text = tokenizer.Decode(tokens.tolist())
        return text
    except Exception as e:
        print(f"  Warning: Batch decode failed, falling back to smaller chunks: {e}")
        # Fallback: decode in smaller chunks
        texts = []
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i+batch_size]
            chunk_text = tokenizer.Decode(batch.tolist())
            texts.append(chunk_text)
        return "".join(texts)


def convert_to_ipa(text, chunk_size=5000):
    """Convert English text to IPA notation using eng_to_ipa.
    
    Processes in chunks to avoid SQL variable limits in eng_to_ipa.
    """
    # Split text into sentences/chunks to avoid SQL limits
    # eng_to_ipa has issues with very long text
    ipa_parts = []
    
    # Simple chunking by approximate word count
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        try:
            ipa_chunk = ipa.convert(chunk)
            ipa_parts.append(ipa_chunk)
        except Exception as e:
            # If a chunk fails, try smaller chunks
            smaller_chunk_size = chunk_size // 2
            for j in range(i, min(i + chunk_size, len(words)), smaller_chunk_size):
                small_chunk = ' '.join(words[j:j+smaller_chunk_size])
                try:
                    ipa_small = ipa.convert(small_chunk)
                    ipa_parts.append(ipa_small)
                except Exception as e2:
                    print(f"    Warning: Failed to convert chunk: {e2}")
                    # Skip this chunk
    
    return ' '.join(ipa_parts)


def save_ipa_text(ipa_text, output_path):
    """Save IPA text to a file."""
    # Use UTF-8 encoding for IPA symbols
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ipa_text)


def save_ipa_binary(ipa_text, output_path):
    """Save IPA text as a compact binary format."""
    # We'll create a simple binary format: [header][symbol_table][char_data]
    # Header: magic number, version, char count, unique symbol count, bytes per char
    # Char data: each character encoded as index into symbol table
    
    # Build symbol table - each unique character gets an index
    symbols = sorted(set(ipa_text))
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    
    # Determine encoding size based on number of unique symbols
    num_symbols = len(symbols)
    if num_symbols <= 256:
        bytes_per_char = 1
        dtype = np.uint8
    elif num_symbols <= 65536:
        bytes_per_char = 2
        dtype = np.uint16
    else:
        bytes_per_char = 4
        dtype = np.uint32
    
    # Encode characters
    encoded = np.array([symbol_to_idx[c] for c in ipa_text], dtype=dtype)
    
    # Create symbol table in UTF-8 with length prefix for each symbol
    # Format: [num_symbols][for each: symbol_len][for each: symbol_utf8_bytes]
    symbols_data = []
    for symbol in symbols:
        symbol_bytes = symbol.encode('utf-8')
        symbols_data.append(struct.pack('<B', len(symbol_bytes)))  # 1-byte length
        symbols_data.append(symbol_bytes)
    symbols_table = b''.join(symbols_data)
    
    # Create header (256 int32 values, similar to original format)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20250101  # Magic number (IPA version)
    header[1] = 1  # Version
    header[2] = len(ipa_text)  # Number of characters
    header[3] = num_symbols  # Number of unique symbols
    header[4] = bytes_per_char  # Bytes per character
    header[5] = len(symbols_table)  # Symbol table size in bytes
    
    # Write to file
    with open(output_path, 'wb') as f:
        # Write header
        header.tofile(f)
        # Write symbol table
        f.write(symbols_table)
        # Write encoded character data
        encoded.tofile(f)


def load_ipa_binary(input_path):
    """Load IPA text from binary format."""
    header = np.fromfile(input_path, dtype=np.int32, count=256)
    
    if header[0] != 20250101 or header[1] != 1:
        raise ValueError(f"Invalid IPA binary format: {input_path}")
    
    num_chars = int(header[2])
    num_symbols = int(header[3])
    bytes_per_char = int(header[4])
    symbol_table_size = int(header[5])
    
    # Read symbol table with length prefixes
    header_bytes = 256 * np.dtype(np.int32).itemsize
    symbols = []
    with open(input_path, 'rb') as f:
        f.seek(header_bytes)
        table_data = f.read(symbol_table_size)
        
        # Parse symbol table: [len][bytes] for each symbol
        offset = 0
        for _ in range(num_symbols):
            sym_len = table_data[offset]
            offset += 1
            sym_bytes = table_data[offset:offset + sym_len]
            offset += sym_len
            symbols.append(sym_bytes.decode('utf-8'))
    
    # Read encoded data
    dtype_map = {1: np.uint8, 2: np.uint16, 4: np.uint32}
    dtype = dtype_map.get(bytes_per_char, np.uint8)
    
    file_offset = header_bytes + symbol_table_size
    encoded = np.fromfile(input_path, dtype=dtype, count=num_chars, offset=file_offset)
    
    # Decode back to string
    ipa_text = ''.join(symbols[idx] for idx in encoded)
    return ipa_text


def analyze_ipa(ipa_text):
    """Analyze the IPA text for statistics and issues."""
    stats = {
        'total_chars': len(ipa_text),
        'unique_symbols': len(set(ipa_text)),
        'has_unknown_words': '*' in ipa_text,  # eng_to_ipa uses * for unknown
        'unknown_count': ipa_text.count('*'),
    }
    
    # Check for homophones (words that become identical in IPA)
    # This is a simplified check - we can't easily recover the original words
    # from IPA without a reverse dictionary
    
    return stats


def process_shard(shard_path, output_dir, tokenizer, save_format='binary', sample_tokens=None):
    """Process a single shard: decode, convert to IPA, save."""
    shard_name = Path(shard_path).stem
    print(f"\nProcessing {shard_name}...")
    
    # Get original file size
    original_size = os.path.getsize(shard_path)
    print(f"  Original size: {original_size:,} bytes")
    
    # Read tokens
    print(f"  Reading tokens...")
    tokens, num_tokens = read_shard(shard_path)
    print(f"  Tokens in shard: {num_tokens:,}")
    
    # Sample if requested
    if sample_tokens and num_tokens > sample_tokens:
        # Take a representative sample from the middle
        start = (num_tokens - sample_tokens) // 2
        tokens = tokens[start:start + sample_tokens]
        print(f"  Sampled {sample_tokens:,} tokens (from middle of shard)")
    
    # Decode to text
    print(f"  Decoding tokens to text...")
    text = decode_tokens_to_text(tokenizer, tokens)
    text_chars = len(text)
    print(f"  Decoded text: {text_chars:,} characters")
    
    # Convert to IPA
    print(f"  Converting to IPA...")
    ipa_text = convert_to_ipa(text)
    ipa_chars = len(ipa_text)
    print(f"  IPA text: {ipa_chars:,} characters")
    
    # Analyze
    stats = analyze_ipa(ipa_text)
    
    # Save IPA output
    if save_format == 'text':
        output_path = os.path.join(output_dir, f"{shard_name}_ipa.txt")
        save_ipa_text(ipa_text, output_path)
    else:
        output_path = os.path.join(output_dir, f"{shard_name}_ipa.bin")
        save_ipa_binary(ipa_text, output_path)
    
    ipa_size = os.path.getsize(output_path)
    print(f"  IPA {save_format} size: {ipa_size:,} bytes")
    
    compression_ratio = original_size / ipa_size if ipa_size > 0 else 0
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return {
        'shard': shard_name,
        'original_size': original_size,
        'original_tokens': num_tokens,
        'original_chars': text_chars,
        'ipa_chars': ipa_chars,
        'ipa_size': ipa_size,
        'compression_ratio': compression_ratio,
        'unique_symbols': stats['unique_symbols'],
        'has_unknown_words': stats['has_unknown_words'],
        'unknown_word_count': stats['unknown_count'],
        'output_path': output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert FineWeb training data to IPA notation'
    )
    parser.add_argument(
        '--shards',
        type=int,
        default=3,
        help='Number of shards to process (default: 3)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/datasets/fineweb10B_sp1024',
        help='Path to FineWeb dataset'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='data/tokenizers/fineweb_1024_bpe.model',
        help='Path to tokenizer model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ipa_converted',
        help='Output directory for IPA files'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'binary'],
        default='binary',
        help='Output format (default: binary)'
    )
    parser.add_argument(
        '--start-shard',
        type=int,
        default=0,
        help='Starting shard index (default: 0)'
    )
    parser.add_argument(
        '--sample-tokens',
        type=int,
        default=500000,
        help='Number of tokens to sample from each shard (default: 500000 for ~10MB of text)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    print("=" * 70)
    print("FINEWEB TO IPA CONVERTER")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify paths
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer not found at {args.tokenizer_path}")
        sys.exit(1)
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset directory not found at {args.dataset_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path}...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"Vocabulary size: {tokenizer.vocab_size()} tokens")
    
    # Find available shards
    dataset_path = Path(args.dataset_path)
    all_shards = sorted(dataset_path.glob("fineweb_train_*.bin"))
    print(f"\nFound {len(all_shards)} total shards in dataset")
    
    # Limit to requested number of shards
    shards_to_process = all_shards[args.start_shard:args.start_shard + args.shards]
    print(f"Will process {len(shards_to_process)} shard(s) starting from index {args.start_shard}")
    
    if not shards_to_process:
        print("Error: No shards to process")
        sys.exit(1)
    
    # Process each shard
    results = []
    for i, shard_path in enumerate(shards_to_process, 1):
        print(f"\n{'=' * 70}")
        print(f"Shard {i}/{len(shards_to_process)}: {shard_path.name}")
        print(f"{'=' * 70}")
        
        result = process_shard(
            str(shard_path),
            args.output_dir,
            tokenizer,
            args.format,
            args.sample_tokens
        )
        results.append(result)
    
    # Summary report
    print(f"\n{'=' * 70}")
    print("SUMMARY REPORT")
    print(f"{'=' * 70}")
    
    total_original = sum(r['original_size'] for r in results)
    total_ipa = sum(r['ipa_size'] for r in results)
    avg_compression = total_original / total_ipa if total_ipa > 0 else 0
    
    print(f"\nShards processed: {len(results)}")
    print(f"Output format: {args.format}")
    print(f"Output directory: {args.output_dir}")
    
    print(f"\n{'Shard':<20} {'Original':>15} {'IPA':>15} {'Ratio':>10} {'Unknowns':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['shard']:<20} {r['original_size']:>15,} {r['ipa_size']:>15,} "
              f"{r['compression_ratio']:>10.2f}x {r['unknown_word_count']:>10,}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_original:>15,} {total_ipa:>15,} "
          f"{avg_compression:>10.2f}x")
    
    # Detailed stats
    print(f"\n{'=' * 70}")
    print("DETAILED STATISTICS")
    print(f"{'=' * 70}")
    
    total_tokens = sum(r['original_tokens'] for r in results)
    total_chars = sum(r['original_chars'] for r in results)
    total_ipa_chars = sum(r['ipa_chars'] for r in results)
    total_unknowns = sum(r['unknown_word_count'] for r in results)
    
    print(f"\nTotal tokens decoded: {total_tokens:,}")
    print(f"Total characters (original): {total_chars:,}")
    print(f"Total characters (IPA): {total_ipa_chars:,}")
    print(f"Character reduction: {(1 - total_ipa_chars/total_chars)*100:.1f}%")
    
    # IPA symbol analysis
    all_unique_symbols = set()
    for r in results:
        # We don't have the actual IPA text here, but we recorded the count
        pass
    
    avg_symbols = sum(r['unique_symbols'] for r in results) / len(results)
    print(f"\nAverage unique IPA symbols per shard: {avg_symbols:.0f}")
    print(f"Unknown word markers (*): {total_unknowns:,}")
    
    # Issues report
    print(f"\n{'=' * 70}")
    print("ISSUES & WARNINGS")
    print(f"{'=' * 70}")
    
    if total_unknowns > 0:
        print(f"⚠️  Found {total_unknowns:,} unknown words (marked with *)")
        print("   These could be:")
        print("   - Proper nouns (names, places)")
        print("   - Technical terms")
        print("   - Non-English words")
        print("   - Misspellings")
    else:
        print("✓ No unknown words detected")
    
    print(f"\n⚠️  Homophone issue: Words like 'knight'/'night', 'to'/'too'/'two'")
    print("   become identical in IPA. This is a fundamental limitation of")
    print("   phonetic representation.")
    
    # Output file details
    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  {r['output_path']}")
    
    # Save JSON report
    report_path = os.path.join(args.output_dir, "conversion_report.json")
    report = {
        'timestamp': datetime.now().isoformat(),
        'shards_processed': len(results),
        'format': args.format,
        'total_original_size': total_original,
        'total_ipa_size': total_ipa,
        'avg_compression_ratio': avg_compression,
        'total_tokens': total_tokens,
        'total_original_chars': total_chars,
        'total_ipa_chars': total_ipa_chars,
        'total_unknown_words': total_unknowns,
        'avg_unique_symbols': avg_symbols,
        'shards': results,
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")
    
    print(f"\n{'=' * 70}")
    print("CONVERSION COMPLETE")
    print(f"{'=' * 70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
