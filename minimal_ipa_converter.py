#!/usr/bin/env python3
"""
Minimal IPA Converter for Parameter Golf

Uses rule-based grapheme-to-phoneme conversion with a small exception
dictionary for irregular words. Target size: <500KB total.

Approach:
1. Simple letter-to-sound rules for most words
2. Small lookup table for common irregular words
3. No database dependencies
"""

import re
import json
import sys
from pathlib import Path

# =============================================================================
# EXPANDED EXCEPTION DICTIONARY
# Load ~4800 entries from CMUdict-derived JSON if available,
# fall back to inline dict for standalone use.
# =============================================================================

_EXCEPTIONS_JSON = Path(__file__).parent / "data" / "ipa_exceptions_2k.json"
_EXPANDED_EXCEPTIONS = None

def _load_expanded_exceptions():
    global _EXPANDED_EXCEPTIONS
    if _EXPANDED_EXCEPTIONS is not None:
        return _EXPANDED_EXCEPTIONS
    if _EXCEPTIONS_JSON.exists():
        with open(_EXCEPTIONS_JSON, 'r', encoding='utf-8') as f:
            _EXPANDED_EXCEPTIONS = json.load(f)
        return _EXPANDED_EXCEPTIONS
    _EXPANDED_EXCEPTIONS = {}
    return _EXPANDED_EXCEPTIONS

# =============================================================================
# EXCEPTION DICTIONARY - Top irregular words only
# These are words that don't follow simple rules
# =============================================================================

EXCEPTIONS = {
    # Very common irregular words
    'the': 'ðə', 'a': 'ə', 'an': 'æn', 'to': 'tu', 'of': 'əv',
    'and': 'ænd', 'for': 'fɔr', 'from': 'frʌm', 'have': 'hæv',
    'been': 'bɪn', 'was': 'wʌz', 'were': 'wɜr', 'said': 'sɛd',
    'they': 'ðeɪ', 'their': 'ðɛr', 'there': 'ðɛr', 'theyre': 'ðɛr',
    'where': 'wɛr', 'were': 'wɜr', 'one': 'wʌn', 'two': 'tu',
    'once': 'wʌns', 'enough': 'ɪˈnʌf', 'through': 'θru',
    'though': 'ðoʊ', 'thought': 'θɔt', 'bought': 'bɔt', 'brought': 'brɔt',
    'daughter': 'ˈdɔtər', 'laugh': 'læf', 'cough': 'kɔf',
    'tough': 'tʌf', 'dough': 'doʊ', 'plough': 'plaʊ',
    'is': 'ɪz', 'are': 'ɑr', 'as': 'æz', 'has': 'hæz',
    'does': 'dʌz', 'done': 'dʌn', 'come': 'kʌm', 'some': 'sʌm',
    'love': 'lʌv', 'move': 'muv', 'prove': 'pruv', 'improve': 'ɪmˈpruv',
    'eye': 'aɪ', 'eyes': 'aɪz', 'buy': 'baɪ', 'guy': 'gaɪ',
    'key': 'ki', 'quay': 'ki',
    'know': 'noʊ', 'knew': 'nu', 'knows': 'noʊz', 'knowledge': 'ˈnɑlɪdʒ',
    'knife': 'naɪf', 'knight': 'naɪt', 'knit': 'nɪt', 'knock': 'nɑk',
    'knot': 'nɑt', 'knowing': 'ˈnoʊɪŋ',
    'write': 'raɪt', 'wrote': 'roʊt', 'written': 'ˈrɪtən',
    'wrong': 'rɔŋ', 'wrist': 'rɪst', 'wrap': 'ræp',
    'who': 'hu', 'whom': 'hum', 'whose': 'huz', 'whole': 'hoʊl',
    'what': 'wʌt', 'when': 'wɛn', 'where': 'wɛr', 'why': 'waɪ',
    'hour': 'aʊər', 'honest': 'ˈɑnɪst', 'honor': 'ˈɑnər',
    'ghost': 'goʊst', 'ghoul': 'gul', 'ghetto': 'ˈgɛtoʊ',
    'could': 'kʊd', 'would': 'wʊd', 'should': 'ʃʊd',
    'half': 'hæf', 'calf': 'kæf', 'calm': 'kɑm', 'palm': 'pɑm',
    'talk': 'tɔk', 'walk': 'wɔk', 'chalk': 'tʃɔk', 'folk': 'foʊk',
    'would': 'wʊd', 'should': 'ʃʊd', 'could': 'kʊd',
    'friend': 'frɛnd', 'fiend': 'find',
    'great': 'greɪt', 'break': 'breɪk', 'steak': 'steɪk',
    'read': 'rid', 'lead': 'lid', 'live': 'lɪv', 'give': 'gɪv',
    'put': 'pʊt', 'cut': 'kʌt', 'shut': 'ʃʌt', 'but': 'bʌt',
    'busy': 'ˈbɪzi', 'business': 'ˈbɪznəs', 'bury': 'ˈbɛri',
    'many': 'ˈmɛni', 'any': 'ˈɛni',
    'said': 'sɛd', 'again': 'əˈgɛn', 'against': 'əˈgɛnst',
    'heart': 'hɑrt', 'hearth': 'hɑrθ', 'heard': 'hɜrd',
    'bear': 'bɛr', 'tear': 'tɛr', 'wear': 'wɛr', 'swear': 'swɛr',
    'tear': 'tɪr', 'pear': 'pɛr',
    'wind': 'wɪnd', 'wild': 'waɪld', 'mild': 'maɪld', 'child': 'tʃaɪld',
    'sword': 'sɔrd', 'word': 'wɜrd', 'work': 'wɜrk', 'world': 'wɜrld',
    'answer': 'ˈænsər', 'sword': 'sɔrd',
    'iron': 'ˈaɪərn', 'ion': 'ˈaɪən',
}

# =============================================================================
# G2P RULES - Letter patterns to phonemes
# Ordered from most specific to most general
# =============================================================================

G2P_RULES = [
    # Digraphs (two-letter combinations)
    ('ch', 'tʃ'),   # church, chair
    ('sh', 'ʃ'),    # ship, shop  
    ('th', 'θ'),    # think, bath (voiceless)
    ('ph', 'f'),    # phone, photo
    ('ng', 'ŋ'),    # sing, ring
    ('gh', ''),     # silent in most cases: night, light
    ('wh', 'w'),    # what, when (simplified)
    ('ck', 'k'),    # back, check
    ('qu', 'kw'),   # queen, quick
    ('wr', 'r'),    # write, wrong
    ('kn', 'n'),    # knight, know
    ('gn', 'n'),    # gnat, gnaw
    ('ph', 'f'),    # phone
    ('rh', 'r'),    # rhyme
    ('sc', 's'),    # scene (before e,i)
    ('tch', 'tʃ'),  # catch, match
    ('dge', 'dʒ'),  # judge, edge
    ('ai', 'eɪ'),   # rain, train
    ('au', 'ɔ'),    # auto, pause
    ('aw', 'ɔ'),    # law, saw
    ('ay', 'eɪ'),   # day, say
    ('ea', 'i'),    # read, lead (simplified - often "ee")
    ('ee', 'i'),    # see, tree
    ('ei', 'eɪ'),   # eight, weight
    ('eu', 'u'),    # neutral (simplified)
    ('ew', 'u'),    # new, few
    ('ey', 'eɪ'),   # they, grey
    ('ie', 'aɪ'),   # pie, tie
    ('oa', 'oʊ'),   # boat, road
    ('oe', 'oʊ'),   # toe, foe
    ('oi', 'ɔɪ'),   # oil, boil
    ('oo', 'u'),    # food, moon (long)
    ('ou', 'aʊ'),   # out, house
    ('ow', 'aʊ'),   # now, cow (sometimes 'oʊ')
    ('oy', 'ɔɪ'),   # boy, toy
    ('ua', 'wɑ'),   # quarrel (simplified)
    ('ue', 'u'),    # true, blue
    ('ui', 'u'),    # fruit, suit
    ('ar', 'ɑr'),   # car, far
    ('er', 'ər'),   # her, teacher
    ('ir', 'ɜr'),   # bird, stir
    ('or', 'ɔr'),   # for, more
    ('ur', 'ɜr'),   # turn, burn
    
    # Single letters (context-dependent handled in code)
]

# Vowel sounds by letter position
VOWEL_SHORT = {
    'a': 'æ',  # cat
    'e': 'ɛ',  # bed
    'i': 'ɪ',  # sit
    'o': 'ɑ',  # hot
    'u': 'ʌ',  # cut
}

VOWEL_LONG = {
    'a': 'eɪ',  # cake
    'e': 'i',   # see
    'i': 'aɪ',  # kite
    'o': 'oʊ',  # boat
    'u': 'u',   # cute
}

CONSONANTS = {
    'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
    'h': 'h', 'j': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm',
    'n': 'n', 'p': 'p', 'q': 'k', 'r': 'r', 's': 's',
    't': 't', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j',
    'z': 'z',
}


def apply_g2p_rules(word):
    """
    Convert word to IPA using grapheme-to-phoneme rules.
    Returns IPA string or None if word not handled.
    """
    word = word.lower()
    result = []
    i = 0
    
    while i < len(word):
        matched = False
        
        # Try digraphs first (3-letter, then 2-letter)
        for pattern, phoneme in G2P_RULES:
            if word[i:].startswith(pattern):
                result.append(phoneme)
                i += len(pattern)
                matched = True
                break
        
        if matched:
            continue
        
        # Single letter handling
        char = word[i]
        
        if char in CONSONANTS:
            result.append(CONSONANTS[char])
        elif char in VOWEL_SHORT:
            # Simple vowel - use short sound as default
            result.append(VOWEL_SHORT[char])
        elif char == 'y':
            # Y as vowel (simplified)
            result.append('i')
        elif char == ' ':
            pass  # skip spaces
        else:
            # Unknown character, keep as is but mark
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def word_to_ipa(word):
    """
    Convert a single word to IPA.
    Checks expanded CMUdict exceptions, then inline exceptions, then G2P rules.
    """
    word_lower = word.lower()

    # Check expanded exceptions (CMUdict-derived, ~4800 entries)
    expanded = _load_expanded_exceptions()
    if word_lower in expanded:
        return expanded[word_lower]

    # Check inline exception dictionary
    if word_lower in EXCEPTIONS:
        return EXCEPTIONS[word_lower]

    # Apply G2P rules
    ipa = apply_g2p_rules(word_lower)

    # Fallback: return rule-based result (never fail)
    if not ipa or len(ipa) == 0:
        return word_lower

    return ipa


def text_to_ipa(text):
    """
    Convert English text to IPA notation.
    Alphabetic words go through G2P. Everything else passes through as-is.
    """
    text = text.lower()

    # Split into alphabetic words vs everything else
    parts = re.findall(r"[a-z']+|[^a-z']+", text)

    result = []
    for part in parts:
        if re.match(r"[a-z']+", part):
            # Alphabetic word -> G2P conversion
            result.append(word_to_ipa(part))
        else:
            # Non-alpha (punctuation, digits, whitespace, URLs, HTML) -> passthrough
            result.append(part)

    return ''.join(result)


def text_to_ipa_chars(text):
    """
    Convert English text to a list of IPA characters (token-level).
    Each character becomes one token. Used for building training shards.
    """
    ipa_text = text_to_ipa(text)
    return list(ipa_text)


def analyze_converter_size():
    """Calculate the size of this converter module."""
    # Exception dict
    exceptions_json = json.dumps(EXCEPTIONS)
    exceptions_size = len(exceptions_json.encode('utf-8'))
    
    # Rules (as code, so just estimate)
    rules_size = len(G2P_RULES) * 10  # rough estimate
    
    # Code size
    code_lines = 200  # approximate
    code_size = code_lines * 50  # rough estimate
    
    total_estimate = exceptions_size + rules_size + code_size
    
    print("=" * 60)
    print("MINIMAL IPA CONVERTER SIZE ANALYSIS")
    print("=" * 60)
    print(f"\nException dictionary:")
    print(f"  Entries: {len(EXCEPTIONS)}")
    print(f"  JSON size: {exceptions_size:,} bytes ({exceptions_size/1024:.2f} KB)")
    
    print(f"\nG2P Rules:")
    print(f"  Rules: {len(G2P_RULES)}")
    print(f"  Estimated size: {rules_size:,} bytes ({rules_size/1024:.2f} KB)")
    
    print(f"\nCode overhead:")
    print(f"  Lines: ~{code_lines}")
    print(f"  Estimated: {code_size:,} bytes ({code_size/1024:.2f} KB)")
    
    print(f"\nTOTAL ESTIMATED SIZE: {total_estimate:,} bytes ({total_estimate/1024:.2f} KB)")
    
    if total_estimate < 500 * 1024:
        print(f"\n✅ UNDER TARGET! (<500KB)")
        space_left = 500 * 1024 - total_estimate
        print(f"   Space remaining: {space_left/1024:.1f} KB")
    else:
        print(f"\n⚠️ OVER TARGET (>500KB)")
        overage = total_estimate - 500 * 1024
        print(f"   Need to reduce by: {overage/1024:.1f} KB")
    
    return total_estimate


def test_converter():
    """Test the converter on sample text."""
    test_cases = [
        "The knight rode through the night",
        "I need to write the right answer",
        "The coffee legend of Kaldi is interesting",
        "Break the brake pedal",
        "To be or not to be",
        "She sells sea shells by the sea shore",
        "How much wood would a woodchuck chuck",
    ]
    
    print("\n" + "=" * 60)
    print("CONVERTER TEST RESULTS")
    print("=" * 60)
    
    total_chars = 0
    total_ipa_chars = 0
    
    for text in test_cases:
        ipa = text_to_ipa(text)
        total_chars += len(text)
        total_ipa_chars += len(ipa)
        
        print(f"\nOriginal: {text}")
        print(f"IPA:      /{ipa}/")
    
    print("\n" + "=" * 60)
    print(f"Average expansion: {total_ipa_chars/total_chars:.2f}x")


def main():
    print("=" * 60)
    print("MINIMAL IPA CONVERTER FOR PARAMETER GOLF")
    print("=" * 60)
    
    # Size analysis
    size = analyze_converter_size()
    
    # Test conversion
    test_converter()
    
    # Compare to eng_to_ipa
    print("\n" + "=" * 60)
    print("COMPARISON TO ENG_TO_IPA")
    print("=" * 60)
    print(f"\nThis converter: ~{size/1024:.1f} KB")
    print(f"eng_to_ipa library: ~3,000 KB")
    print(f"Space saved: ~{3000 - size/1024:.0f} KB")
    
    # Net calculation
    embedding_savings = 0.92 * 1024  # KB
    converter_cost = size / 1024  # KB
    net_savings = embedding_savings - converter_cost
    
    print(f"\n" + "=" * 60)
    print("PARAMETER GOLF IMPACT")
    print("=" * 60)
    print(f"\nEmbedding savings: +{embedding_savings:.1f} KB")
    print(f"Converter cost:    -{converter_cost:.1f} KB")
    print(f"Net result:        {'+' if net_savings > 0 else ''}{net_savings:.1f} KB")
    
    if net_savings > 0:
        print(f"\n✅ NET POSITIVE! You save {net_savings:.1f} KB")
    else:
        print(f"\n⚠️ NET NEGATIVE - still need to optimize more")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
