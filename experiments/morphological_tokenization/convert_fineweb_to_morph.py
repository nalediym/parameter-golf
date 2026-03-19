"""
Convert FineWeb Dataset to Morphological Tokens

Converts BPE token binary files to morpheme sequences by:
1. Reading BPE token IDs from binary
2. Decoding to text using SentencePiece
3. Segmenting text into morphemes
4. Writing morpheme IDs to binary

Usage:
    python convert_fineweb_to_morph.py \
        --input_path ../../data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin \
        --output_path data/fineweb_val_morph_000000.bin \
        --vocab_path data/morph_vocab.json \
        --sp_model_path ../../data/tokenizers/fineweb_1024_bpe.model \
        --max_tokens 100000  # For testing, remove for full conversion
"""

import argparse
import json
import struct
from pathlib import Path
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from morphological_tokenizer import AgglutinativeTokenizer


def load_morph_vocab(vocab_path: Path):
    """Load morpheme vocabulary."""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab


def load_sentencepiece(model_path: Path):
    """Load SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        return sp
    except ImportError:
        print("Error: sentencepiece not installed")
        print("Install with: pip install sentencepiece")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading SentencePiece model: {e}")
        sys.exit(1)


def read_bpe_tokens(binary_path: Path, max_tokens: int = None):
    """
    Read BPE token IDs from binary file.
    
    Format:
    - 256 int32 header values
    - Header[0] = magic number (20240520)
    - Header[1] = version (1)
    - Header[2] = number of tokens
    - Tokens are uint16, start after header (offset 1024 bytes)
    """
    import numpy as np
    
    # Read header (256 int32 values)
    header = np.fromfile(binary_path, dtype="<i4", count=256)
    
    if header.size != 256:
        raise ValueError(f"Invalid header size: {header.size}")
    
    if int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header format: magic={header[0]}, version={header[1]}")
    
    num_tokens = int(header[2])
    print(f"  Shard header: {num_tokens:,} tokens declared")
    
    # Read actual tokens (after 1024-byte header)
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(binary_path, dtype="<u2", count=num_tokens, offset=header_bytes)
    
    print(f"  Actually read {len(tokens):,} tokens")
    
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"  Truncated to {max_tokens:,} tokens")
    
    return tokens.tolist()


def tokens_to_text_chunked(bpe_tokens: list, sp_model, chunk_size: int = 1000):
    """
    Convert BPE tokens to text.
    SentencePiece works best with batches.
    """
    text_chunks = []
    
    for i in range(0, len(bpe_tokens), chunk_size):
        chunk = bpe_tokens[i:i+chunk_size]
        # Decode this chunk
        text = sp_model.decode(chunk)
        text_chunks.append(text)
    
    return ' '.join(text_chunks)


def extract_words(text: str):
    """Extract words from text."""
    # Keep alphanumeric words, lowercase
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    return words


def convert_file(input_path: Path, output_path: Path, vocab: dict, 
                 sp_model, morph_tokenizer, max_tokens: int = None,
                 report_interval: int = 10000):
    """
    Convert BPE binary to morpheme binary.
    """
    print(f"Converting {input_path.name}...")
    print(f"  Max tokens: {max_tokens if max_tokens else 'unlimited'}")
    
    token_to_id = vocab.get('token_to_id', {})
    unk_id = token_to_id.get('<UNK>', 1)
    
    # Read BPE tokens
    print("  Reading BPE tokens...")
    bpe_tokens = read_bpe_tokens(input_path, max_tokens)
    print(f"  Loaded {len(bpe_tokens):,} BPE tokens")
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000  # BPE tokens per chunk
    all_morpheme_ids = []
    
    print("  Converting to morphemes...")
    num_chunks = (len(bpe_tokens) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(bpe_tokens), chunk_size):
        if i % report_interval == 0:
            print(f"    Progress: {i:,} / {len(bpe_tokens):,} tokens "
                  f"({100*i/len(bpe_tokens):.1f}%)")
        
        # Get chunk of BPE tokens
        chunk = bpe_tokens[i:i+chunk_size]
        
        # Decode to text
        try:
            text = sp_model.decode(chunk)
        except Exception as e:
            print(f"    Warning: decode error at position {i}: {e}")
            continue
        
        # Extract words
        words = extract_words(text)
        
        # Segment each word into morphemes
        for word in words:
            if len(word) <= 1:
                continue  # Skip single characters
                
            result = morph_tokenizer.segment(word)
            
            # Convert morphemes to IDs
            for morpheme in result.morphemes:
                morpheme_id = token_to_id.get(morpheme, unk_id)
                all_morpheme_ids.append(morpheme_id)
    
    print(f"  Generated {len(all_morpheme_ids):,} morpheme tokens")
    
    # Calculate compression/expansion ratio
    ratio = len(all_morpheme_ids) / len(bpe_tokens) if bpe_tokens else 0
    print(f"  BPE→Morph ratio: {ratio:.2f}x")
    
    # Write to binary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        for morpheme_id in all_morpheme_ids:
            # Ensure within uint16 range
            if morpheme_id > 65535:
                morpheme_id = unk_id
            f.write(struct.pack('<H', morpheme_id))
    
    print(f"  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    
    return {
        'input_tokens': len(bpe_tokens),
        'output_tokens': len(all_morpheme_ids),
        'ratio': ratio,
        'input_file': str(input_path),
        'output_file': str(output_path),
    }


def compare_sample(sp_model, morph_tokenizer, vocab, bpe_path: Path, 
                   max_tokens: int = 1000, num_examples: int = 5):
    """
    Show a comparison between BPE and morphological tokenization.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: BPE vs Morphological Tokenization")
    print("=" * 80)
    
    token_to_id = vocab.get('token_to_id', {})
    id_to_token = vocab.get('id_to_token', {})
    unk_id = token_to_id.get('<UNK>', 1)
    
    # Read sample tokens
    bpe_tokens = read_bpe_tokens(bpe_path, max_tokens)
    
    # Decode to text
    full_text = sp_model.decode(bpe_tokens)
    
    # Extract sample sentences (rough approximation)
    sentences = full_text.split('.')[:num_examples]
    
    print(f"\nSample from first {max_tokens} tokens:\n")
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
            
        print(f"Original: {sent}")
        
        # BPE tokenization
        bpe_ids = sp_model.encode(sent, out_type=int)
        bpe_pieces = [sp_model.id_to_piece(id_) for id_ in bpe_ids]
        print(f"BPE:      {' | '.join(bpe_pieces)}")
        print(f"          ({len(bpe_ids)} tokens)")
        
        # Morphological tokenization
        words = extract_words(sent)
        morphemes = []
        for word in words:
            result = morph_tokenizer.segment(word)
            morphemes.extend(result.morphemes)
        
        morph_ids = [token_to_id.get(m, unk_id) for m in morphemes]
        print(f"MORPH:    {' | '.join(morphemes)}")
        print(f"          ({len(morph_ids)} tokens, IDs: {morph_ids[:10]}{'...' if len(morph_ids) > 10 else ''})")
        
        ratio = len(morph_ids) / len(bpe_ids) if bpe_ids else 0
        print(f"Ratio: {ratio:.2f}x (morph/bpe)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FineWeb BPE to morpheme tokens"
    )
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input FineWeb BPE binary file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to output morpheme binary file")
    parser.add_argument("--vocab_path", type=str,
                       default="data/morph_vocab.json",
                       help="Path to morpheme vocabulary JSON")
    parser.add_argument("--sp_model_path", type=str,
                       default="../../data/tokenizers/fineweb_1024_bpe.model",
                       help="Path to SentencePiece model")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Maximum BPE tokens to process (for testing)")
    parser.add_argument("--compare", action="store_true",
                       help="Show comparison samples before converting")
    parser.add_argument("--compare_tokens", type=int, default=1000,
                       help="Number of tokens for comparison sample")
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    vocab = load_morph_vocab(Path(args.vocab_path))
    print(f"  Loaded morpheme vocab: {vocab['vocab_size']} tokens")
    print(f"    - Roots: {vocab.get('num_roots', 'N/A')}")
    print(f"    - Affixes: {vocab.get('num_affixes', 'N/A')}")
    
    sp_model = load_sentencepiece(Path(args.sp_model_path))
    print(f"  Loaded SentencePiece: {sp_model.vocab_size()} tokens")
    
    morph_tokenizer = AgglutinativeTokenizer()
    print(f"  Initialized morphological tokenizer")
    
    # Show comparison if requested
    if args.compare:
        compare_sample(sp_model, morph_tokenizer, vocab, 
                      Path(args.input_path), 
                      max_tokens=args.compare_tokens)
    
    # Convert file
    print("\n" + "=" * 80)
    result = convert_file(
        Path(args.input_path),
        Path(args.output_path),
        vocab,
        sp_model,
        morph_tokenizer,
        max_tokens=args.max_tokens
    )
    
    # Save conversion report
    report_path = Path(args.output_path).with_suffix('.report.json')
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved conversion report to {report_path}")


if __name__ == "__main__":
    main()
