#!/usr/bin/env python3
"""
Quick test of minimal IPA converter on FineWeb sample.
"""

import numpy as np
import sentencepiece as spm
from pathlib import Path
from minimal_ipa_converter import text_to_ipa

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.Load('data/tokenizers/fineweb_1024_bpe.model')

# Load a small sample from shard 0
shard_file = 'data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin'
header_bytes = 256 * np.dtype("<i4").itemsize
tokens = np.fromfile(shard_file, dtype="<u2", count=10000, offset=header_bytes)

# Decode
print("Loading first 10,000 tokens...")
text = sp.Decode(tokens.tolist())
print(f"Sample text ({len(text)} chars):")
print(text[:500])
print("\n" + "="*60 + "\n")

# Convert to IPA
print("Converting to IPA...")
ipa = text_to_ipa(text)
print(f"IPA version ({len(ipa)} chars):")
print(ipa[:500])

# Stats
print(f"\n{'='*60}")
print("STATS")
print(f"{'='*60}")
print(f"Original: {len(text)} chars")
print(f"IPA:      {len(ipa)} chars")
print(f"Expansion: {len(ipa)/len(text):.2f}x")

# Size comparison
original_bytes = len(text.encode('utf-8'))
ipa_bytes = len(ipa.encode('utf-8'))
print(f"\nOriginal bytes: {original_bytes}")
print(f"IPA bytes: {ipa_bytes}")
print(f"Ratio: {original_bytes/ipa_bytes:.2f}x")
