#!/usr/bin/env python3
"""
Quick IPA model training - simplified version.
"""

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from pathlib import Path
import sys
from minimal_ipa_converter import text_to_ipa

print("="*60)
print("QUICK IPA MODEL TRAINING")
print("="*60)

# Device
device = torch.device("cpu")  # Keep it simple
print(f"Using device: {device}")

# Build IPA vocabulary from a sample
def build_ipa_vocab():
    """Build vocabulary from common IPA symbols."""
    # Common English phonemes in IPA
    symbols = set('abcdefghijklmnopqrstuvwxyz ')
    symbols.update(['æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɪ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʤ', 'ʧ', 'θ', 'ˌ', 'ˈ'])
    return sorted(list(symbols))

IPA_VOCAB = build_ipa_vocab()
VOCAB_SIZE = len(IPA_VOCAB)
print(f"Vocabulary size: {VOCAB_SIZE}")

char_to_id = {ch: i for i, ch in enumerate(IPA_VOCAB)}
id_to_char = {i: ch for i, ch in enumerate(IPA_VOCAB)}

# Load tokenizer
print("\nLoading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.Load('data/tokenizers/fineweb_1024_bpe.model')

# Load sample data
print("Loading sample data (first 10K tokens)...")
shard_file = 'data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin'
header_bytes = 256 * np.dtype("<i4").itemsize
tokens = np.fromfile(shard_file, dtype="<u2", count=10000, offset=header_bytes)
text = sp.Decode(tokens.tolist())

print(f"Text length: {len(text)} chars")

# Convert to IPA
print("Converting to IPA...")
ipa_text = text_to_ipa(text)
print(f"IPA length: {len(ipa_text)} chars")

# Simple dataset
def encode(text):
    return [char_to_id.get(c, 0) for c in text]

class SimpleDataset:
    def __init__(self, data, seq_len=64):
        self.data = encode(data)
        self.seq_len = seq_len
    
    def get_batch(self, batch_size=8):
        import random
        xs, ys = [], []
        for _ in range(batch_size):
            idx = random.randint(0, len(self.data) - self.seq_len - 1)
            xs.append(self.data[idx:idx+self.seq_len])
            ys.append(self.data[idx+1:idx+self.seq_len+1])
        return torch.tensor(xs), torch.tensor(ys)

# Tiny model
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(128, d_model)
        
        encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos(pos)
        x = self.transformer(x)
        return self.fc(x)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# Create model
print("\n" + "="*60)
print("MODEL")
print("="*60)
model = TinyTransformer(VOCAB_SIZE)
print(f"Parameters: {model.count_params():,}")
print(f"Vocab: {VOCAB_SIZE}")

# Train
print("\n" + "="*60)
print("TRAINING")
print("="*60)

dataset = SimpleDataset(ipa_text)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for step in range(200):
    x, y = dataset.get_batch(4)
    
    logits = model(x)
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 50 == 0:
        print(f"Step {step+1}/200, Loss: {loss.item():.4f}")

# Test generation
print("\n" + "="*60)
print("GENERATION TEST")
print("="*60)

model.eval()
with torch.no_grad():
    # Start with "ðə " (the)
    input_ids = encode("ðə ")
    input_tensor = torch.tensor([input_ids[-64:]])  # Last 64 chars
    
    generated = input_ids.copy()
    for _ in range(50):
        logits = model(input_tensor)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        input_tensor = torch.tensor([generated[-64:]])
    
    result = ''.join([id_to_char.get(i, '?') for i in generated])
    print(f"Seed: 'ðə '")
    print(f"Generated: '{result}'")

# Save
print("\n" + "="*60)
print("SAVING")
print("="*60)
torch.save({
    'model': model.state_dict(),
    'vocab': IPA_VOCAB,
    'char_to_id': char_to_id,
    'id_to_char': id_to_char,
}, 'quick_ipa_model.pt')

print("Saved: quick_ipa_model.pt")
print("\nTraining complete!")
