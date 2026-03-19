#!/bin/bash
# Train Small IPA Model for Parameter Golf
# Usage: ./train_ipa_model.sh [OPTIONS]
#
# Options:
#   --iterations N      Number of training iterations (default: 1000)
#   --batch-size N      Batch size in tokens (default: 4096)
#   --learning-rate F   Learning rate (default: 0.001)
#   --model-size SIZE   tiny|small|medium (default: tiny)

set -e

# Default parameters
ITERATIONS=1000
BATCH_SIZE=4096
LEARNING_RATE=0.001
MODEL_SIZE="tiny"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "IPA MODEL TRAINING SCRIPT"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Batch size: $BATCH_SIZE tokens"
echo "  Learning rate: $LEARNING_RATE"
echo "  Model size: $MODEL_SIZE"
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Run: uv venv"
    exit 1
fi

# Check for required files
echo ""
echo "Checking prerequisites..."

if [ ! -f "minimal_ipa_converter.py" ]; then
    echo "✗ minimal_ipa_converter.py not found"
    exit 1
fi
echo "✓ Minimal IPA converter found"

if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "✗ FineWeb dataset not found. Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    exit 1
fi
echo "✓ FineWeb dataset found"

if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "✗ Tokenizer not found"
    exit 1
fi
echo "✓ Tokenizer found"

# Create training script
echo ""
echo "Creating training script..."

cat > train_ipa_lm.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Train a small language model on IPA-converted text.
"""

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import json
from tqdm import tqdm
from minimal_ipa_converter import text_to_ipa

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# IPA Vocabulary (from minimal converter)
IPA_SYMBOLS = set()
# Add all IPA chars from converter
for ipa_str in ['ðə', 'kɑffi', 'lɛgɛnd', 'əv', 'naɪt', 'θru', 'raɪt', 'ɛks', 'bri']:
    IPA_SYMBOLS.update(ipa_str)
# Add common symbols
IPA_SYMBOLS.update([
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'z',
    'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɪ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʤ', 'ʧ', 'θ', 'ˌ', 'ˈ'
])
IPA_VOCAB = sorted(list(IPA_SYMBOLS))
IPA_VOCAB_SIZE = len(IPA_VOCAB)
print(f"IPA vocabulary size: {IPA_VOCAB_SIZE}")

# Create char-to-id and id-to-char mappings
char_to_id = {ch: i for i, ch in enumerate(IPA_VOCAB)}
id_to_char = {i: ch for i, ch in enumerate(IPA_VOCAB)}

def encode_ipa(text):
    """Convert IPA text to token IDs."""
    return [char_to_id.get(c, char_to_id.get('ə', 0)) for c in text]

def decode_ipa(ids):
    """Convert token IDs to IPA text."""
    return ''.join([id_to_char.get(i, ' ') for i in ids])

class IPADataset(Dataset):
    """Dataset for IPA text."""
    def __init__(self, ipa_text, seq_length=128):
        self.data = encode_ipa(ipa_text)
        self.seq_length = seq_length
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y

class IPATransformer(nn.Module):
    """Small transformer for IPA language modeling."""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(512, d_model)  # Max 512 positions
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Add positional encoding
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        logits = self.fc(x)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

def train_model(model, train_loader, val_loader, iterations=1000, lr=0.001, device='cpu'):
    """Train the IPA model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModel has {model.count_parameters():,} parameters")
    print(f"Training for {iterations} iterations...\n")
    
    model.train()
    losses = []
    best_val_loss = float('inf')
    
    for iteration in range(iterations):
        # Training step
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for cross-entropy
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            losses.append(loss.item())
            
            if batch_idx >= len(train_loader) - 1:
                break
        
        # Logging
        if (iteration + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            print(f"Iteration {iteration+1}/{iterations}, Loss: {avg_loss:.4f}")
            
            # Validation
            if val_loader:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                        val_losses.append(loss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"  Validation Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_ipa_model.pt')
                
                model.train()
    
    return losses

def generate_text(model, seed_text="ðə ", length=100, temperature=1.0, device='cpu'):
    """Generate IPA text from the model."""
    model.eval()
    
    # Encode seed
    input_ids = encode_ipa(seed_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(length):
            # Get model prediction
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            input_tensor = torch.tensor([generated[-128:]], dtype=torch.long).to(device)
    
    return decode_ipa(generated)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--model-size', type=str, default='tiny', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--seq-length', type=int, default=128)
    parser.add_argument('--sample-shards', type=int, default=1, help='Number of shards to sample for training')
    args = parser.parse_args()
    
    # Model size configurations
    configs = {
        'tiny': {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128},
        'small': {'d_model': 128, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 256},
        'medium': {'d_model': 256, 'nhead': 8, 'num_layers': 4, 'dim_feedforward': 512},
    }
    
    config = configs[args.model_size]
    
    print("="*60)
    print("IPA MODEL TRAINING")
    print("="*60)
    print(f"\nModel config: {args.model_size}")
    print(f"  d_model: {config['d_model']}")
    print(f"  nhead: {config['nhead']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  dim_feedforward: {config['dim_feedforward']}")
    
    # Load and convert training data
    print(f"\nLoading training data from {args.sample_shards} shard(s)...")
    
    sp = spm.SentencePieceProcessor()
    sp.Load('data/tokenizers/fineweb_1024_bpe.model')
    
    all_ipa_text = []
    for shard_idx in range(args.sample_shards):
        shard_file = f'data/datasets/fineweb10B_sp1024/fineweb_train_{shard_idx:06d}.bin'
        if not Path(shard_file).exists():
            print(f"Warning: {shard_file} not found, skipping")
            continue
        
        print(f"  Processing shard {shard_idx}...")
        header_bytes = 256 * np.dtype("<i4").itemsize
        tokens = np.fromfile(shard_file, dtype="<u2", count=50000, offset=header_bytes)  # Sample 50K tokens
        text = sp.Decode(tokens.tolist())
        ipa_text = text_to_ipa(text)
        all_ipa_text.append(ipa_text)
    
    full_ipa_text = '\n'.join(all_ipa_text)
    print(f"\nTotal IPA text: {len(full_ipa_text):,} characters")
    
    # Split train/val
    split_point = int(len(full_ipa_text) * 0.9)
    train_text = full_ipa_text[:split_point]
    val_text = full_ipa_text[split_point:]
    
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")
    
    # Create datasets
    train_dataset = IPADataset(train_text, seq_length=args.seq_length)
    val_dataset = IPADataset(val_text, seq_length=args.seq_length)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    model = IPATransformer(
        vocab_size=IPA_VOCAB_SIZE,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward']
    )
    
    # Train
    losses = train_model(model, train_loader, val_loader, 
                        iterations=args.iterations, 
                        lr=args.learning_rate,
                        device=device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab': IPA_VOCAB,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'losses': losses,
    }, 'ipa_model_final.pt')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: ipa_model_final.pt")
    print(f"Final train loss: {sum(losses[-100:])/min(100,len(losses)):.4f}")
    
    # Generate samples
    print("\n" + "="*60)
    print("GENERATED SAMPLES")
    print("="*60)
    
    seeds = ["ðə ", "aɪ ", "hi ", "ðɛr "]
    for seed in seeds:
        generated = generate_text(model, seed_text=seed, length=50, temperature=0.8, device=device)
        print(f"\nSeed: '{seed}'")
        print(f"Generated: '{generated}'")
    
    # Save results summary
    results = {
        'model_size': args.model_size,
        'parameters': model.count_parameters(),
        'vocab_size': IPA_VOCAB_SIZE,
        'final_train_loss': sum(losses[-100:])/min(100,len(losses)),
        'training_iterations': args.iterations,
        'device': str(device),
    }
    
    with open('ipa_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Results saved: ipa_training_results.json")
    print("="*60)

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

chmod +x train_ipa_lm.py

echo "✓ Training script created"

# Run training
echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="
echo ""

python3 train_ipa_lm.py \
    --iterations "$ITERATIONS" \
    --batch-size 32 \
    --learning-rate "$LEARNING_RATE" \
    --model-size "$MODEL_SIZE" \
    --sample-shards 1

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
echo ""

# Show results
if [ -f "ipa_training_results.json" ]; then
    echo "Results:"
    cat ipa_training_results.json
    echo ""
fi

if [ -f "ipa_model_final.pt" ]; then
    echo "Model saved: ipa_model_final.pt"
    ls -lh ipa_model_final.pt
    echo ""
fi

echo "To test the model:"
echo "  python3 -c \"import torch; print('Model loaded:', torch.load('ipa_model_final.pt').keys())\""
