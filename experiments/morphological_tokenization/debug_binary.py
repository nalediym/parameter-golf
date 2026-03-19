"""
Debug FineWeb binary format
"""

import struct
from pathlib import Path

# Read first few tokens from validation file
binary_path = Path("/Users/naledi/Projects/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")

with open(binary_path, 'rb') as f:
    # Read first 50 tokens
    tokens = []
    for _ in range(50):
        data = f.read(2)
        if not data or len(data) < 2:
            break
        token_id = struct.unpack('<H', data)[0]
        tokens.append(token_id)

print(f"First 50 token IDs: {tokens}")
print(f"Max token ID: {max(tokens)}")
print(f"Min token ID: {min(tokens)}")
print(f"Unique tokens: {len(set(tokens))}")

# Try loading with sentencepiece
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("/Users/naledi/Projects/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")

print(f"\nSentencePiece vocab size: {sp.vocab_size()}")

# Check if tokens are in range
invalid = [t for t in tokens if t >= sp.vocab_size()]
print(f"Invalid tokens (>= vocab_size): {invalid}")

# Try decoding valid tokens only
valid_tokens = [t for t in tokens if t < sp.vocab_size()]
print(f"\nValid tokens: {valid_tokens[:20]}")

if valid_tokens:
    try:
        text = sp.decode(valid_tokens)
        print(f"\nDecoded text: {text[:200]}...")
    except Exception as e:
        print(f"Decode error: {e}")
        # Try piece by piece
        print("\nPiece by piece:")
        for i, tid in enumerate(valid_tokens[:20]):
            try:
                piece = sp.id_to_piece(tid)
                print(f"  {i}: ID={tid} -> '{piece}'")
            except Exception as e2:
                print(f"  {i}: ID={tid} -> ERROR: {e2}")
