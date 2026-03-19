"""
Build Comprehensive Morpheme Vocabulary from FineWeb

Uses extracted word frequencies to build a large morpheme vocabulary
that covers the most common morphologically complex words in FineWeb.

Usage:
    python build_comprehensive_lexicon.py \
        --word_freq data/fineweb_word_freq.json \
        --output data/morph_lexicon_v2.json \
        --vocab_path data/morph_vocab_v2.json \
        --max_words 20000 \
        --min_word_freq 10
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from morphological_tokenizer import AgglutinativeTokenizer


def load_word_frequencies(freq_path: Path):
    """Load word frequency data."""
    with open(freq_path, 'r') as f:
        data = json.load(f)
    
    return data.get('word_frequencies', {})


def build_morpheme_vocabulary(word_freq: dict, max_words: int, min_freq: int):
    """
    Build comprehensive morpheme vocabulary.
    
    Args:
        word_freq: Dict of word -> frequency
        max_words: Maximum words to process (by frequency)
        min_freq: Minimum word frequency to include
    
    Returns:
        Dictionary with morpheme vocabulary and lexicon
    """
    print(f"Building morpheme vocabulary...")
    print(f"  Input: {len(word_freq):,} words")
    print(f"  Max words to process: {max_words:,}")
    print(f"  Min word frequency: {min_freq}")
    print()
    
    # Filter and sort words
    filtered_words = {w: f for w, f in word_freq.items() if f >= min_freq}
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    
    if max_words:
        sorted_words = sorted_words[:max_words]
    
    print(f"Processing {len(sorted_words):,} words above frequency threshold...")
    
    # Initialize tokenizer
    tokenizer = AgglutinativeTokenizer()
    
    # Build lexicon and morpheme frequency
    lexicon = {}
    morpheme_freq = Counter()
    segmentation_stats = Counter()
    
    for word, freq in sorted_words:
        result = tokenizer.segment(word)
        
        # Skip character-level segmentations (not useful)
        if result.segmentation_type == "character":
            continue
        
        # Add to lexicon
        lexicon[word] = result.morphemes
        
        # Update morpheme frequencies
        for morpheme in result.morphemes:
            morpheme_freq[morpheme] += freq
        
        # Track segmentation types
        segmentation_stats[result.segmentation_type] += 1
    
    print(f"Segmentation statistics:")
    for seg_type, count in segmentation_stats.most_common():
        print(f"  {seg_type:20}: {count:,} words")
    print()
    
    # Categorize morphemes
    roots = set()
    prefixes = set()
    suffixes = set()
    
    for morpheme, freq in morpheme_freq.items():
        # Categorize based on patterns
        if morpheme in tokenizer.DERIVATIONAL_PREFIXES:
            prefixes.add(morpheme)
        elif morpheme in tokenizer.DERIVATIONAL_SUFFIXES or morpheme in tokenizer.INFLECTIONAL_SUFFIXES:
            suffixes.add(morpheme)
        else:
            # Check if it's likely a root (appears in multiple words, not just as affix)
            roots.add(morpheme)
    
    # Create vocabulary
    vocab = {
        'roots': sorted(list(roots)),
        'prefixes': sorted(list(prefixes)),
        'suffixes': sorted(list(suffixes)),
        'lexicon': lexicon,
        'morpheme_freq': dict(morpheme_freq),
        'stats': {
            'num_roots': len(roots),
            'num_prefixes': len(prefixes),
            'num_suffixes': len(suffixes),
            'total_morphemes': len(roots) + len(prefixes) + len(suffixes),
            'num_lexicon_entries': len(lexicon),
            'words_processed': len(sorted_words),
            'words_with_segmentation': len(lexicon),
        }
    }
    
    return vocab


def create_token_to_id_mapping(vocab: dict, max_vocab_size: int = 5000):
    """
    Create token-to-ID mapping with vocabulary size limit.
    
    Strategy:
    1. Keep all special tokens (<PAD>, <UNK>, etc.)
    2. Include most frequent morphemes up to max_vocab_size
    3. Exclude rare morphemes (treated as <UNK> during encoding)
    """
    # Start with special tokens
    token_to_id = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
        '<ROOT>': 4,  # Marks start of root morpheme
        '<AFFIX>': 5,  # Marks affix morpheme
    }
    
    current_id = 6
    
    # Get morpheme frequencies
    morpheme_freq = vocab.get('morpheme_freq', {})
    
    # Sort by frequency
    sorted_morphemes = sorted(morpheme_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Take top N morphemes
    max_morphemes = max_vocab_size - len(token_to_id)
    top_morphemes = [m for m, _ in sorted_morphemes[:max_morphemes]]
    
    print(f"Selected top {len(top_morphemes):,} morphemes by frequency")
    
    # Add to vocabulary
    for morpheme in top_morphemes:
        token_to_id[morpheme] = current_id
        current_id += 1
    
    return token_to_id


def create_coverage_report(vocab: dict, word_freq: dict, top_n: int = 1000):
    """
    Create report on how well the morpheme vocabulary covers the corpus.
    """
    token_to_id = {}
    for morpheme in vocab['roots'] + vocab['prefixes'] + vocab['suffixes']:
        token_to_id[morpheme] = 1  # Just mark as valid
    
    covered_words = 0
    uncovered_words = 0
    covered_tokens = 0
    uncovered_tokens = 0
    
    # Check coverage for top N most frequent words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    for word, freq in sorted_words:
        if word in vocab['lexicon']:
            covered_words += 1
            covered_tokens += freq
        else:
            uncovered_words += 1
            uncovered_tokens += freq
    
    coverage_pct = 100 * covered_tokens / (covered_tokens + uncovered_tokens)
    
    return {
        'top_n_words': top_n,
        'covered_words': covered_words,
        'uncovered_words': uncovered_words,
        'covered_tokens': covered_tokens,
        'uncovered_tokens': uncovered_tokens,
        'token_coverage_pct': coverage_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build comprehensive morpheme vocabulary from FineWeb"
    )
    parser.add_argument("--word_freq", type=str,
                       default="data/fineweb_word_freq.json",
                       help="Path to word frequency JSON")
    parser.add_argument("--output", type=str,
                       default="data/morph_lexicon_v2.json",
                       help="Output path for morpheme lexicon")
    parser.add_argument("--vocab_path", type=str,
                       default="data/morph_vocab_v2.json",
                       help="Output path for token-to-ID vocabulary")
    parser.add_argument("--max_words", type=int, default=20000,
                       help="Maximum words to process (by frequency)")
    parser.add_argument("--min_word_freq", type=int, default=10,
                       help="Minimum word frequency to include")
    parser.add_argument("--max_vocab_size", type=int, default=3000,
                       help="Maximum morpheme vocabulary size")
    
    args = parser.parse_args()
    
    # Load word frequencies
    print(f"Loading word frequencies from {args.word_freq}...")
    word_freq = load_word_frequencies(Path(args.word_freq))
    print(f"Loaded {len(word_freq):,} words\n")
    
    # Build morpheme vocabulary
    vocab = build_morpheme_vocabulary(
        word_freq,
        max_words=args.max_words,
        min_freq=args.min_word_freq
    )
    
    print("=" * 60)
    print("VOCABULARY STATISTICS")
    print("=" * 60)
    print(f"Total morphemes: {vocab['stats']['total_morphemes']:,}")
    print(f"  - Roots: {vocab['stats']['num_roots']:,}")
    print(f"  - Prefixes: {vocab['stats']['num_prefixes']:,}")
    print(f"  - Suffixes: {vocab['stats']['num_suffixes']:,}")
    print(f"Lexicon entries: {vocab['stats']['num_lexicon_entries']:,}")
    print()
    
    # Create token-to-ID mapping
    print(f"Creating token-to-ID mapping (max vocab size: {args.max_vocab_size})...")
    token_to_id = create_token_to_id_mapping(vocab, max_vocab_size=args.max_vocab_size)
    
    print(f"Final vocabulary size: {len(token_to_id):,} tokens")
    print()
    
    # Create coverage report
    coverage = create_coverage_report(vocab, word_freq, top_n=1000)
    print("=" * 60)
    print("COVERAGE REPORT (Top 1000 words)")
    print("=" * 60)
    print(f"Words with segmentation: {coverage['covered_words']:,}/{coverage['top_n_words']:,}")
    print(f"Token coverage: {coverage['token_coverage_pct']:.1f}%")
    print()
    
    # Save vocabulary
    vocab_path = Path(args.vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    
    vocab_output = {
        'token_to_id': token_to_id,
        'id_to_token': {str(v): k for k, v in token_to_id.items()},
        'special_tokens': ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<ROOT>', '<AFFIX>'],
        'vocab_size': len(token_to_id),
        'num_roots': vocab['stats']['num_roots'],
        'num_prefixes': vocab['stats']['num_prefixes'],
        'num_suffixes': vocab['stats']['num_suffixes'],
        'coverage': coverage,
    }
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_output, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save lexicon
    lexicon_output = {
        'metadata': {
            'num_roots': vocab['stats']['num_roots'],
            'num_prefixes': vocab['stats']['num_prefixes'],
            'num_suffixes': vocab['stats']['num_suffixes'],
            'num_lexicon_entries': vocab['stats']['num_lexicon_entries'],
            'coverage': coverage,
        },
        'lexicon': vocab['lexicon'],
        'morpheme_frequencies': vocab['morpheme_freq'],
    }
    
    lexicon_path = Path(args.output)
    with open(lexicon_path, 'w') as f:
        json.dump(lexicon_output, f, indent=2)
    print(f"Saved lexicon to {lexicon_path}")
    
    # Print top morphemes
    print("\n" + "=" * 60)
    print("TOP 20 MORPHEMES BY FREQUENCY")
    print("=" * 60)
    sorted_morphemes = sorted(vocab['morpheme_freq'].items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:20]
    for morpheme, freq in sorted_morphemes:
        category = 'prefix' if morpheme in vocab['prefixes'] else \
                   'suffix' if morpheme in vocab['suffixes'] else 'root'
        print(f"  {morpheme:20} {freq:8,} ({category})")
    
    # Print sample tokenizations
    print("\n" + "=" * 60)
    print("SAMPLE TOKENIZATIONS")
    print("=" * 60)
    sample_words = [
        "running", "happiness", "impossibility", "representation",
        "international", "multidimensional", "unpredictability",
        "antidisestablishmentarianism"
    ]
    
    for word in sample_words:
        if word in vocab['lexicon']:
            morphemes = vocab['lexicon'][word]
            ids = [token_to_id.get(m, token_to_id['<UNK>']) for m in morphemes]
            print(f"{word:30} → {' + '.join(morphemes):40} → {ids}")
        else:
            print(f"{word:30} → (not in lexicon)")
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
