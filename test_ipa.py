#!/usr/bin/env python3
"""
Test script for IPA tokenization idea.

Converts English text to IPA, then attempts to convert back to show
the lossy nature of the conversion (homophones problem).
"""

import eng_to_ipa as ipa
import sys

def convert_to_ipa(text):
    """Convert English text to IPA notation."""
    return ipa.convert(text)

def demonstrate_homophones():
    """Show words that sound the same but are spelled differently."""
    homophones = [
        ("knight", "night"),
        ("right", "write", "rite"),
        ("to", "too", "two"),
        ("sea", "see"),
        ("meet", "meat"),
        ("break", "brake"),
        ("their", "there", "they're"),
    ]
    
    print("=" * 60)
    print("HOMOPHONE PROBLEM")
    print("=" * 60)
    for group in homophones:
        print(f"\nWords: {', '.join(group)}")
        for word in group:
            phonetic = ipa.convert(word)
            print(f"  {word} → /{phonetic}/")
    print()

def test_conversion_roundtrip():
    """Test converting English → IPA → (attempted) English."""
    test_sentences = [
        "The knight rode through the night.",
        "I need to write the right answer.",
        "The coffee legend of Kaldi is interesting.",
        "Break the brake pedal.",
    ]
    
    print("=" * 60)
    print("ROUNDTRIP TEST (English → IPA)")
    print("=" * 60)
    
    for sentence in test_sentences:
        ipa_version = convert_to_ipa(sentence)
        print(f"\nOriginal: {sentence}")
        print(f"IPA:      /{ipa_version}/")
        print("         ^^^ Note: IPA is lowercase, no punctuation, one representation")
    print()

def analyze_finetext_sample():
    """Analyze a sample from FineWeb data."""
    # Sample text similar to what we saw earlier
    sample = "The coffee legends of Kaldi and Omar the Dervish are interesting stories."
    
    print("=" * 60)
    print("FINEWEB SAMPLE ANALYSIS")
    print("=" * 60)
    
    # Convert to IPA
    ipa_text = convert_to_ipa(sample)
    
    # Count unique characters
    unique_chars = set(ipa_text.replace('/', '').replace(' ', ''))
    
    print(f"\nOriginal text ({len(sample)} chars):")
    print(sample)
    
    print(f"\nIPA representation ({len(ipa_text)} chars):")
    print(f"/{ipa_text}/")
    
    print(f"\nUnique IPA symbols: {len(unique_chars)}")
    print(f"Symbols: {sorted(unique_chars)}")
    
    # Character reduction
    reduction = (1 - len(unique_chars) / 128) * 128  # ASCII is 128 chars
    print(f"\nvs ASCII: ~{len(unique_chars)} vs 128 characters = {reduction:.0f}x smaller alphabet")
    print()

def show_ipa_vocabulary():
    """Show the full IPA symbol set for English."""
    print("=" * 60)
    print("IPA VOCABULARY SIZE FOR ENGLISH")
    print("=" * 60)
    
    # Common IPA symbols used in English
    consonants = ['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 
                  'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 'tʃ', 'dʒ']
    vowels = ['i', 'ɪ', 'e', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ʊ', 'u', 'ə', 'ɜ', 'ʌ', 'ɑɪ', 
              'aʊ', 'ɔɪ', 'eɪ', 'oʊ', 'ɛ', 'ɝ', 'ɚ', 'iə', 'eə', 'ʊə']
    
    print(f"\nEnglish consonants in IPA: ~{len(consonants)} symbols")
    print(f"English vowels in IPA: ~{len(vowels)} symbols")
    print(f"Total unique symbols: ~{len(consonants) + len(vowels)}")
    print(f"\nCompare to challenge tokenizers:")
    print(f"  - byte260: 260 symbols")
    print(f"  - sp1024: 1024 symbols")
    print(f"  - YOUR IPA: ~{len(consonants) + len(vowels)} symbols ✓ smaller!")
    print()

def calculate_compression_win():
    """Calculate the theoretical compression win from IPA."""
    print("=" * 60)
    print("COMPRESSION ANALYSIS")
    print("=" * 60)
    
    # Typical embedding size is vocab_size * embedding_dim * 4 bytes (float32)
    # For a small model, let's say embedding_dim = 256
    
    EMBED_DIM = 256
    BYTES_PER_PARAM = 4
    
    sp1024_embed_size = 1024 * EMBED_DIM * BYTES_PER_PARAM
    ipa_embed_size = 80 * EMBED_DIM * BYTES_PER_PARAM
    
    print(f"\nAssuming embedding dimension = {EMBED_DIM}")
    print(f"\nEmbedding layer size comparison:")
    print(f"  sp1024:  {sp1024_embed_size:,} bytes ({sp1024_embed_size/1024/1024:.2f} MB)")
    print(f"  IPA:     {ipa_embed_size:,} bytes ({ipa_embed_size/1024/1024:.2f} MB)")
    print(f"  Savings: {(1 - ipa_embed_size/sp1024_embed_size)*100:.1f}% smaller embeddings")
    
    # But we need to account for the conversion tables
    print(f"\n⚠️  BUT: Need to store conversion tables in your 16MB budget!")
    print(f"   - eng-to-ipa library data: ~3MB")
    print(f"   - IPA-to-eng dictionary: potentially very large")
    print()

def main():
    print("\n" + "=" * 60)
    print("IPA TOKENIZATION PROTOTYPE")
    print("=" * 60)
    print("\nTesting if IPA can help compress models for Parameter Golf\n")
    
    demonstrate_homophones()
    test_conversion_roundtrip()
    analyze_finetext_sample()
    show_ipa_vocabulary()
    calculate_compression_win()
    
    print("=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("""
✅ WINS:
  - IPA has ~80 symbols vs 1024 tokens = smaller embeddings
  - Could potentially fit a bigger model in 16MB

❌ CHALLENGES:
  - IPA is LOSSY: knight/night/to/too/two all become same symbols
  - Converting back to English for evaluation is ambiguous
  - Conversion tables eat up your 16MB budget
  - FineWeb evaluation is on English text, not IPA

💡 POSSIBLE SOLUTIONS:
  1. Accept the loss - maybe context helps disambiguate?
  2. Store a small "context-free" IPA→English mapping
  3. Train on both IPA and English (bilingual model)
  4. Challenge: Is the compression win worth the complexity?

Next step: Try converting actual FineWeb data to IPA and see how
much you can compress it + how much dictionary space you need.
""")

if __name__ == '__main__':
    main()
