#!/bin/bash
# Quick test of IPA model training pipeline
# This demonstrates the setup without full training

echo "=========================================="
echo "IPA MODEL TRAINING - QUICK SETUP TEST"
echo "=========================================="
echo ""

# Check Python environment
echo "1. Checking Python environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "   ✓ Virtual environment activated"
    python3 --version
else
    echo "   ✗ Virtual environment not found"
    echo "   Run: uv venv && source .venv/bin/activate"
    exit 1
fi

# Check required files
echo ""
echo "2. Checking required files..."
REQUIRED_FILES=(
    "minimal_ipa_converter.py"
    "data/tokenizers/fineweb_1024_bpe.model"
    "data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (missing)"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo ""
    echo "Missing files. Please run:"
    echo "  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    exit 1
fi

# Test IPA converter
echo ""
echo "3. Testing IPA converter..."
python3 -c "
from minimal_ipa_converter import text_to_ipa
test = 'The knight rode through the night'
result = text_to_ipa(test)
print(f'   Input:  {test}')
print(f'   Output: /{result}/')
"

# Show training command
echo ""
echo "=========================================="
echo "READY TO TRAIN!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  ./train_ipa_model.sh [OPTIONS]"
echo ""
echo "Options:"
echo "  --iterations N      Training iterations (default: 1000)"
echo "  --batch-size N      Tokens per batch (default: 32)"
echo "  --learning-rate F   Learning rate (default: 0.001)"
echo "  --model-size SIZE   tiny|small|medium (default: tiny)"
echo ""
echo "Examples:"
echo ""
echo "  # Quick test (1-2 minutes):"
echo "  ./train_ipa_model.sh --iterations 100 --model-size tiny"
echo ""
echo "  # Small model (5-10 minutes):"
echo "  ./train_ipa_model.sh --iterations 1000 --model-size small"
echo ""
echo "  # Full training (30+ minutes):"
echo "  ./train_ipa_model.sh --iterations 5000 --model-size medium"
echo ""
echo "Output files:"
echo "  ipa_model_final.pt         - Trained model"
echo "  best_ipa_model.pt          - Best validation checkpoint"
echo "  ipa_training_results.json  - Training metrics"
echo ""
