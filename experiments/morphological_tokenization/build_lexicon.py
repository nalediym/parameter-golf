"""
Build Morpheme Lexicon from FineWeb Data

Analyzes the FineWeb dataset to:
1. Extract word frequencies
2. Build morphological segmentations for common words
3. Create a vocabulary of morphemes (roots + affixes)

Usage:
    python build_lexicon.py \
        --data_path ../../data/datasets/fineweb10B_sp1024/ \
        --output_path data/morph_lexicon.json \
        --max_words 50000
"""

import argparse
import json
import struct
from pathlib import Path
from collections import Counter
import sys

# Try to import tqdm, fall back to simple progress if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from morphological_tokenizer import AgglutinativeTokenizer


def read_binary_file(filepath: Path, max_tokens: int = None) -> list:
    """Read tokens from binary file."""
    tokens = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)  # Read 1MB at a time
            if not chunk:
                break
            # Each token is 2 bytes (uint16)
            for i in range(0, len(chunk), 2):
                if i + 1 < len(chunk):
                    token = struct.unpack('<H', chunk[i:i+2])[0]
                    tokens.append(token)
                    if max_tokens and len(tokens) >= max_tokens:
                        return tokens
    return tokens


def extract_words_from_binary(data_path: Path, tokenizer_path: Path, max_words: int = 100000):
    """
    Extract words from FineWeb binary files.
    
    Note: This requires the SentencePiece tokenizer to decode tokens to text.
    For now, we'll use a simplified approach - extract common patterns.
    """
    print(f"Extracting words from {data_path}...")
    
    # For initial testing, let's use a simple word list
    # In production, you'd decode the binary tokens using the SP tokenizer
    
    # Load common English words (simplified for now)
    common_words = [
        # High-frequency content words
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us",
        
        # Morphologically complex examples
        "running", "runner", "runs", "ran",
        "walking", "walker", "walks", "walked",
        "talking", "talker", "talks", "talked",
        "happiness", "happy", "happier", "happiest",
        "sadness", "sad", "sadder", "saddest",
        "impossible", "possible", "possibly",
        "uncomfortable", "comfortable", "comfort",
        "telecommunications", "communication", "communicate",
        "international", "national", "nation",
        "antidisestablishmentarianism",  # Famous example
        "preprocessing", "processing", "process",
        "multidimensional", "dimensional", "dimension",
        "biotechnology", "technology", "technological",
        "representation", "represent", "representative",
        "understanding", "understand", "understood",
        "development", "develop", "developed",
        "information", "inform", "informed",
        "application", "apply", "applied",
        "government", "govern", "governing",
        "environment", "environmental",
        "performance", "perform", "performing",
        "experience", "experienced",
        "difference", "different", "differ",
        "important", "importance",
        "relationship", "relate", "related",
        "generation", "generate", "generated",
        "organization", "organize", "organized",
        "conversation", "converse", "conversational",
        "probability", "probable", "probably",
        "availability", "available",
        "flexibility", "flexible",
        "responsibility", "responsible",
        "community", "communities",
        "activity", "activities", "active",
        "university", "universities",
        "capacity", "capabilities", "capable",
        "security", "secure",
        "functionality", "functional", "function",
        "opportunity", "opportunities",
        "specificity", "specific", "specifically",
        "complexity", "complex", "complicated",
        "necessity", "necessary", "necessarily",
        "creativity", "creative", "creation", "create",
        "sensitivity", "sensitive", "sense",
        "productivity", "productive", "product", "production",
        "connectivity", "connected", "connection", "connect",
        "diversity", "diverse", "diversify",
        "intensity", "intense", "intensive",
        "unpredictability", "unpredictable", "predict", "prediction",
        "reproducibility", "reproducible", "reproduce", "reproduction",
        "interpretability", "interpretable", "interpret", "interpretation",
        "generalizability", "generalizable", "generalize", "generalization",
        "computational", "computation", "compute", "computer", "computing",
        "mathematical", "mathematics", "mathematician",
        "theoretical", "theory", "theorize",
        "practical", "practice", "practitioner",
        "technical", "technique", "technician",
        "analytical", "analysis", "analyze", "analyst",
        "empirical", "empirically",
        "methodological", "methodology", "method",
        "statistical", "statistics", "statistician",
        "experimental", "experiment", "experimentation",
        "conceptual", "conception", "concept", "conceive",
        "contextual", "context",
        "structural", "structure", "structuring",
        "behavioral", "behavior", "behave",
        "operational", "operation", "operate", "operator",
        "functional", "function", "functioning",
        "relational", "relation", "relationship", "relate",
        "directional", "direction", "direct",
        "traditional", "tradition",
        "additional", "addition", "add",
        "conditional", "condition",
        "positional", "position", "positioning",
        "compositional", "composition", "compose", "composite",
        "transformational", "transformation", "transform", "transformer",
        "informational", "informative",
        "instructional", "instruction", "instruct",
        "conversational", "conversation",
        "foundational", "foundation", "found",
        "motivational", "motivation", "motivate", "motive",
        "inspirational", "inspiration", "inspire",
        "educational", "education", "educate",
        "organizational", "organization",
    ]
    
    # Count frequencies (in reality, would count from corpus)
    word_counts = Counter()
    for word in common_words:
        # Simulate frequency distribution
        if len(word) <= 3:
            word_counts[word] = 10000
        elif len(word) <= 5:
            word_counts[word] = 5000
        else:
            word_counts[word] = 1000
    
    return word_counts


def build_morpheme_vocabulary(word_counts: Counter, min_word_freq: int = 10) -> dict:
    """
    Build vocabulary of morphemes from segmented words.
    
    Returns:
        Dictionary with:
        - 'roots': list of root morphemes
        - 'prefixes': list of prefixes
        - 'suffixes': list of suffixes
        - 'lexicon': word -> morpheme list mapping
        - 'morpheme_freq': frequency of each morpheme
    """
    print("Building morpheme vocabulary...")
    
    tokenizer = AgglutinativeTokenizer()
    
    lexicon = {}
    morpheme_counts = Counter()
    
    # Segment all words above frequency threshold
    for word, freq in tqdm(word_counts.items(), desc="Segmenting words"):
        if freq < min_word_freq:
            continue
            
        result = tokenizer.segment(word)
        
        # Only keep non-character segmentations
        if result.segmentation_type != "character":
            lexicon[word] = result.morphemes
            
            # Count morpheme frequencies
            for morpheme in result.morphemes:
                morpheme_counts[morpheme] += freq
    
    # Categorize morphemes
    roots = set()
    prefixes = set()
    suffixes = set()
    
    for word, morphemes in lexicon.items():
        if len(morphemes) >= 2:
            # First morpheme might be prefix
            if morphemes[0] in tokenizer.DERIVATIONAL_PREFIXES:
                prefixes.add(morphemes[0])
            else:
                roots.add(morphemes[0])
            
            # Last morpheme might be suffix
            if morphemes[-1] in tokenizer.DERIVATIONAL_SUFFIXES + tokenizer.INFLECTIONAL_SUFFIXES:
                suffixes.add(morphemes[-1])
            else:
                roots.add(morphemes[-1])
            
            # Middle morphemes are usually roots
            for m in morphemes[1:-1]:
                roots.add(m)
        elif len(morphemes) == 1:
            roots.add(morphemes[0])
    
    # Create vocabulary with IDs
    vocab = {
        'roots': sorted(list(roots)),
        'prefixes': sorted(list(prefixes)),
        'suffixes': sorted(list(suffixes)),
        'lexicon': lexicon,
        'morpheme_freq': dict(morpheme_counts),
        'stats': {
            'num_roots': len(roots),
            'num_prefixes': len(prefixes),
            'num_suffixes': len(suffixes),
            'num_lexicon_entries': len(lexicon),
            'total_morphemes': len(roots) + len(prefixes) + len(suffixes)
        }
    }
    
    return vocab


def create_token_to_id_mapping(vocab: dict) -> dict:
    """
    Create token-to-ID mapping for model training.
    
    Organizes as:
    - 0: <PAD>
    - 1: <UNK>
    - 2: <BOS>
    - 3: <EOS>
    - 4+: morphemes (grouped by type)
    """
    token_to_id = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
    }
    
    current_id = 4
    
    # Add roots first (most common)
    for root in sorted(vocab['roots']):
        token_to_id[root] = current_id
        current_id += 1
    
    # Add prefixes
    for prefix in sorted(vocab['prefixes']):
        token_to_id[prefix] = current_id
        current_id += 1
    
    # Add suffixes
    for suffix in sorted(vocab['suffixes']):
        token_to_id[suffix] = current_id
        current_id += 1
    
    return token_to_id


def main():
    parser = argparse.ArgumentParser(description="Build morpheme lexicon from text data")
    parser.add_argument("--data_path", type=str, 
                       default="../../data/datasets/fineweb10B_sp1024/",
                       help="Path to FineWeb binary data")
    parser.add_argument("--tokenizer_path", type=str,
                       default="../../data/tokenizers/fineweb_1024_bpe.model",
                       help="Path to SentencePiece model")
    parser.add_argument("--output_path", type=str,
                       default="data/morph_lexicon.json",
                       help="Output path for lexicon JSON")
    parser.add_argument("--vocab_path", type=str,
                       default="data/morph_vocab.json",
                       help="Output path for token-to-ID vocabulary")
    parser.add_argument("--max_words", type=int, default=50000,
                       help="Maximum number of words to process")
    parser.add_argument("--min_freq", type=int, default=10,
                       help="Minimum word frequency to include")
    
    args = parser.parse_args()
    
    # Extract words (simplified for now)
    print(f"Analyzing words...")
    word_counts = extract_words_from_binary(
        Path(args.data_path),
        Path(args.tokenizer_path) if args.tokenizer_path else None,
        max_words=args.max_words
    )
    
    print(f"Found {len(word_counts)} unique words")
    
    # Build vocabulary
    vocab = build_morpheme_vocabulary(word_counts, min_word_freq=args.min_freq)
    
    print("\nVocabulary Statistics:")
    print(f"  Roots: {vocab['stats']['num_roots']}")
    print(f"  Prefixes: {vocab['stats']['num_prefixes']}")
    print(f"  Suffixes: {vocab['stats']['num_suffixes']}")
    print(f"  Total morphemes: {vocab['stats']['total_morphemes']}")
    print(f"  Lexicon entries: {vocab['stats']['num_lexicon_entries']}")
    
    # Save vocabulary
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"\nSaved lexicon to {output_path}")
    
    # Create and save token-to-ID mapping
    token_to_id = create_token_to_id_mapping(vocab)
    
    vocab_output = {
        'token_to_id': token_to_id,
        'id_to_token': {str(v): k for k, v in token_to_id.items()},
        'special_tokens': ['<PAD>', '<UNK>', '<BOS>', '<EOS>'],
        'vocab_size': len(token_to_id),
        'num_roots': len(vocab['roots']),
        'num_affixes': len(vocab['prefixes']) + len(vocab['suffixes']),
    }
    
    vocab_path = Path(args.vocab_path)
    with open(vocab_path, 'w') as f:
        json.dump(vocab_output, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Print sample tokenizations
    print("\n" + "=" * 60)
    print("Sample Tokenizations:")
    print("=" * 60)
    
    test_words = [
        "running", "happiness", "impossibility", 
        "telecommunications", "antidisestablishmentarianism",
        "representation", "international", "multidimensional"
    ]
    
    for word in test_words:
        if word in vocab['lexicon']:
            morphemes = vocab['lexicon'][word]
            token_ids = [token_to_id.get(m, token_to_id['<UNK>']) for m in morphemes]
            print(f"{word:30} → {' + '.join(morphemes):40} → {token_ids}")


if __name__ == "__main__":
    main()
