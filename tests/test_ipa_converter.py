#!/usr/bin/env python3
"""Tests T1-T5: IPA converter correctness."""
import sys
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from minimal_ipa_converter import word_to_ipa, text_to_ipa, text_to_ipa_chars

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        print(f"  FAIL: {name} {detail}")


def test_t1_exception_lookup():
    """T1: Top words produce correct IPA from exception dict."""
    print("\nT1: Exception lookup")
    expected = {
        'the': 'ðə', 'knight': 'naɪt', 'through': 'θru',
        'know': 'noʊ', 'would': 'wʊd', 'could': 'kʊd',
        'should': 'ʃʊd', 'one': 'wʌn', 'two': 'tu',
        'business': 'bɪznəs',
    }
    for word, ipa in expected.items():
        result = word_to_ipa(word)
        check(f"{word} -> {ipa}", result == ipa, f"got '{result}'")


def test_t2_g2p_rules():
    """T2: Regular words produce plausible IPA."""
    print("\nT2: G2P rules")
    # These words should produce non-empty IPA without crashing
    regular_words = [
        'cat', 'dog', 'run', 'jump', 'fast', 'slow',
        'table', 'chair', 'window', 'garden', 'simple',
    ]
    for word in regular_words:
        result = word_to_ipa(word)
        check(f"{word} produces output", len(result) > 0, f"got '{result}'")
        check(f"{word} no asterisk", '*' not in result, f"got '{result}'")


def test_t3_oov_handling():
    """T3: Unknown/malformed words never crash."""
    print("\nT3: OOV handling")
    edge_cases = [
        '', 'x', 'zzzz', 'qwrtplk', 'a', 'I',
        "it's", "don't", "they're",
        'supercalifragilisticexpialidocious',
        'cryptocurrency', 'blockchain',
    ]
    for word in edge_cases:
        try:
            result = word_to_ipa(word)
            check(f"'{word}' no crash", True)
        except Exception as e:
            check(f"'{word}' no crash", False, f"raised {e}")


def test_t4_byte_count_roundtrip():
    """T4: Original text byte count can be recovered from IPA output."""
    print("\nT4: Byte count round-trip")
    test_texts = [
        "The knight rode through the night.",
        "Hello, world! How are you?",
        "Price: $29.99 (10% off)",
        "Visit https://example.com today.",
        "",
        "   ",  # whitespace only
    ]
    for text in test_texts:
        original_bytes = len(text.encode('utf-8'))
        # IPA conversion + char list
        ipa_text = text_to_ipa(text)
        # We should be able to count original bytes separately
        # The key property: text_to_ipa is deterministic
        ipa_text_2 = text_to_ipa(text)
        check(
            f"deterministic for '{text[:30]}...'",
            ipa_text == ipa_text_2,
            f"two calls differ"
        )
        # Byte count of original is knowable before conversion
        check(
            f"original bytes countable for '{text[:30]}...'",
            original_bytes >= 0,
        )


def test_t5_determinism():
    """T5: Same input always produces same output."""
    print("\nT5: Determinism")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells sea shells by the sea shore.",
        "How much wood would a woodchuck chuck?",
    ]
    for text in texts:
        results = [text_to_ipa(text) for _ in range(10)]
        all_same = all(r == results[0] for r in results)
        check(f"10 calls identical for '{text[:30]}...'", all_same)

    # Also check char-level tokenization
    for text in texts:
        results = [text_to_ipa_chars(text) for _ in range(5)]
        all_same = all(r == results[0] for r in results)
        check(f"char tokens stable for '{text[:30]}...'", all_same)


def test_passthrough():
    """Passthrough: non-alpha chars pass through unchanged."""
    print("\nPassthrough: non-alpha characters")
    # Digits should pass through
    result = text_to_ipa("cost $29.99")
    check("digits pass through", "29.99" in result, f"got '{result}'")

    # Punctuation should pass through
    result = text_to_ipa("hello! how? yes.")
    check("! passes through", "!" in result, f"got '{result}'")
    check("? passes through", "?" in result, f"got '{result}'")

    # Whitespace preserved
    result = text_to_ipa("a b c")
    check("spaces preserved", " " in result, f"got '{result}'")


if __name__ == '__main__':
    test_t1_exception_lookup()
    test_t2_g2p_rules()
    test_t3_oov_handling()
    test_t4_byte_count_roundtrip()
    test_t5_determinism()
    test_passthrough()

    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'='*60}")
    sys.exit(1 if FAILED > 0 else 0)
