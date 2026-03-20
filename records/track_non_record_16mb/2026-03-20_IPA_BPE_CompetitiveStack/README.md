# IPA-BPE Phonetic Tokenizer + Competitive Stack

**Non-record submission | val_bpb: 1.2055 | 4xH100 SXM**

## Key Idea: Phonetic Tokenization

Instead of training on raw English text with standard BPE, we convert text to IPA (International Phonetic Alphabet) pronunciation symbols, then train a BPE tokenizer on that phonetic representation.

This makes homophones like "knight" and "night" share the same token (`naɪt`), and morphological patterns become more regular. BPE on IPA produces 1.07x the tokens of standard BPE-1024 while preserving phonetic structure.

### IPA-BPE Pipeline

```
English text → G2P converter (4795 CMUdict exceptions + rules) → IPA text
IPA text → SentencePiece BPE (vocab=1024) → IPA-BPE tokens
IPA-BPE tokens → train_gpt.py (with byte-count correction for bpb)
```

### Competitive Stack (on top of IPA-BPE)

Techniques borrowed from top leaderboard entries:
- **Int6 STE QAT**: Straight-through estimator fake-quantization during training
- **BigramHash**: 4096-bucket hash embedding for bigram context (~590K params)
- **SmearGate**: Learned gate blending current/previous token embeddings
- **OrthoInit**: Orthogonal weight initialization with muP output scaling
- **MLP 3x**: Wider feedforward (1536 hidden dim)
- **Tuned hyperparams**: lr=0.02, momentum=0.99, warmdown=3000, grad_clip=0.3, batch=786K

## Results

| Metric | Value |
|--------|-------|
| val_loss (standard) | 2.0704 |
| val_bpb (standard) | 1.2262 |
| val_loss (sliding, stride=64) | 2.0354 |
| **val_bpb (sliding, stride=64)** | **1.2055** |
| Training steps | 3509 |
| Step avg | 171ms |
| Peak memory | 17,053 MiB |
| Model size (int8+zlib) | 19.5MB |
| Hardware | 4xH100 SXM |

## Known Issues

1. **Model size over 16MB**: MLP 3x + BigramHash pushes the int8+zlib compressed model to 19.5MB. Implementing int6 export quantization (vs int8) would recover ~25% and bring it under budget. QAT is already training with int6 fake-quantization, so the model is ready for int6 export.

2. **4xH100, not 8xH100**: This run used half the competition GPUs. On 8xH100, approximately 7000+ steps would be achievable, likely pushing bpb below 1.19.

3. **IPA shard conversion**: The evaluators would need to convert BPE shards to IPA-BPE format before training. This could be integrated as a setup step or the IPA conversion could happen on-the-fly.

## IPA Research Findings

We tested three IPA variants on 1xH100:

| Approach | Tokens vs BPE | val_bpb (1xH100) |
|----------|--------------|-------------------|
| IPA char-level (127 vocab) | 2.28x more | 1.369 |
| IPA-BPE-1024 (seq=1024) | 1.07x more | 1.329 |
| IPA-BPE-1024 (seq=2048) | 1.07x more | 1.357 |
| BPE baseline | 1.00x | 1.335 |

Key finding: IPA-BPE beats BPE baseline on the same hardware (1.329 vs 1.335). The phonetic structure provides a small but measurable advantage.

## Configuration

```bash
VOCAB_SIZE=1024 TRAIN_SEQ_LEN=2048 MLP_MULT=3 QAT=1 ORTHO_INIT=1 \
USE_BIGRAM_HASH=1 USE_SMEAR_GATE=1 MATRIX_LR=0.02 SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
TRAIN_BATCH_TOKENS=786432 EVAL_STRIDE=64 EVAL_BATCH_SEQS=1024 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Files

- `train_gpt.py` — Training script with competitive stack features
- `train.log` — Full training log from 4xH100 run
- `submission.json` — Metadata
