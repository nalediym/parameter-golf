"""
Morphological Tokenizer for Parameter Golf

Segments English words into morphemes (roots + affixes) to enable
compositional generalization with smaller models.

Usage:
    from morphological_tokenizer import MorphologicalTokenizer
    tokenizer = MorphologicalTokenizer()
    tokens = tokenizer.segment("impossibility")
    # Returns: ["im", "poss", "ible", "ity"]
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SegmentationResult:
    """Result of word segmentation."""
    word: str
    morphemes: List[str]
    segmentation_type: str  # "dictionary", "prefix", "suffix", "compound", "character"
    confidence: float  # 0.0-1.0


class MorphologicalTokenizer:
    """
    English morphological tokenizer.
    
    Strategy:
    1. Dictionary lookup for known words
    2. Prefix stripping (un-, re-, dis-, pre-, etc.)
    3. Suffix stripping (-ing, -tion, -ness, etc.)
    4. Compound splitting (treehouse → tree + house)
    5. Character fallback for unknowns
    """
    
    # Common English prefixes (derivational)
    DERIVATIONAL_PREFIXES = [
        "anti", "auto", "bi", "circum", "co", "con", "com", "contra", "counter",
        "de", "dis", "en", "em", "ex", "extra", "fore", "hetero", "homo",
        "hyper", "hypo", "il", "im", "in", "ir", "inter", "intra", "macro",
        "mal", "micro", "mid", "mis", "mono", "multi", "non", "ob", "oc",
        "of", "op", "out", "over", "para", "peri", "poly", "post", "pre",
        "pro", "pseudo", "re", "retro", "semi", "sub", "super", "supra",
        "sur", "syn", "trans", "tri", "ultra", "un", "under", "uni"
    ]
    
    # Common English suffixes by category
    INFLECTIONAL_SUFFIXES = [
        "s", "es",  # plural
        "ed", "ing",  # verb
        "er", "est",  # comparative/superlative
    ]
    
    DERIVATIONAL_SUFFIXES = [
        # Nominalization
        "tion", "sion", "ation", "ition", "ment", "ness", "ity", "ty",
        "er", "or", "ist", "ism", "cy", "ence", "ance", "ure", "ure",
        "age", "al", "dom", "ee", "ery", "ess", "ful", "hood", "ing",
        "ship", "th", "ure", "y",
        # Adjectival
        "able", "ible", "al", "ant", "ent", "ar", "ary", "ed", "en",
        "ern", "ese", "ful", "ian", "ic", "ical", "ious", "ous", "ish",
        "ive", "less", "ly", "ory", "ous", "some", "ward", "wise", "y",
        # Verbal
        "ate", "en", "ify", "ise", "ize",
        # Adverbial
        "ly", "ward", "wards", "wise",
    ]
    
    # Greek/Latin combining forms (bound roots)
    COMBINING_FORMS = [
        "bio", "cardio", "chemo", "chrono", "cosmo", "cryo", "crypto",
        "cyber", "demo", "eco", "electro", "geo", "graph", "hemo",
        "hydro", "hypno", "iso", "macro", "mega", "meta", "micro",
        "mono", "multi", "neo", "neuro", "photo", "physio", "poly",
        "pseudo", "psycho", "pyro", "radio", "retro", "socio", "techno",
        "tele", "thermo", "toxico", "xeno",
    ]
    
    def __init__(self, lexicon_path: Optional[Path] = None):
        """
        Initialize tokenizer.
        
        Args:
            lexicon_path: Path to JSON lexicon file. If None, uses built-in rules.
        """
        self.lexicon: Dict[str, List[str]] = {}
        self.word_freq: Dict[str, int] = {}
        
        if lexicon_path and lexicon_path.exists():
            with open(lexicon_path, 'r') as f:
                data = json.load(f)
                self.lexicon = data.get('lexicon', {})
                self.word_freq = data.get('word_freq', {})
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for common morphological operations."""
        # Sort prefixes/suffixes by length (longest first) for greedy matching
        sorted_prefixes = sorted(self.DERIVATIONAL_PREFIXES, key=len, reverse=True)
        sorted_suffixes = sorted(
            self.DERIVATIONAL_SUFFIXES + self.INFLECTIONAL_SUFFIXES,
            key=len, reverse=True
        )
        
        self.prefix_pattern = re.compile(
            f"^({'|'.join(re.escape(p) for p in sorted_prefixes)})(.+)$",
            re.IGNORECASE
        )
        self.suffix_pattern = re.compile(
            f"^(.+)({'|'.join(re.escape(s) for s in sorted_suffixes)})$",
            re.IGNORECASE
        )
        
    def segment(self, word: str) -> SegmentationResult:
        """
        Segment a word into morphemes.
        
        Args:
            word: The word to segment
            
        Returns:
            SegmentationResult with morphemes and metadata
        """
        word_lower = word.lower()
        
        # 1. Dictionary lookup (most reliable)
        if word_lower in self.lexicon:
            return SegmentationResult(
                word=word,
                morphemes=self.lexicon[word_lower],
                segmentation_type="dictionary",
                confidence=1.0
            )
        
        # 2. Try prefix stripping
        prefix_result = self._try_prefixes(word_lower)
        if prefix_result:
            return SegmentationResult(
                word=word,
                morphemes=prefix_result,
                segmentation_type="prefix",
                confidence=0.8
            )
        
        # 3. Try suffix stripping
        suffix_result = self._try_suffixes(word_lower)
        if suffix_result:
            return SegmentationResult(
                word=word,
                morphemes=suffix_result,
                segmentation_type="suffix",
                confidence=0.7
            )
        
        # 4. Try compound splitting
        compound_result = self._try_compound_split(word_lower)
        if compound_result:
            return SegmentationResult(
                word=word,
                morphemes=compound_result,
                segmentation_type="compound",
                confidence=0.6
            )
        
        # 5. Character fallback (last resort)
        return SegmentationResult(
            word=word,
            morphemes=list(word_lower),  # Character-level
            segmentation_type="character",
            confidence=0.3
        )
    
    def _try_prefixes(self, word: str) -> Optional[List[str]]:
        """Try to segment by stripping prefixes."""
        for prefix in sorted(self.DERIVATIONAL_PREFIXES, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                root = word[len(prefix):]
                # Check if root is valid (dictionary or can stand alone)
                if self._is_valid_root(root):
                    return [prefix, root]
        return None
    
    def _try_suffixes(self, word: str) -> Optional[List[str]]:
        """Try to segment by stripping suffixes."""
        # Try derivational first (more meaningful)
        for suffix in sorted(self.DERIVATIONAL_SUFFIXES, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                if self._is_valid_root(root):
                    return [root, suffix]
        
        # Then try inflectional
        for suffix in sorted(self.INFLECTIONAL_SUFFIXES, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                if self._is_valid_root(root):
                    return [root, suffix]
        
        return None
    
    def _try_compound_split(self, word: str) -> Optional[List[str]]:
        """Try to split compound words."""
        # Simple heuristic: look for valid words concatenated
        for i in range(3, len(word) - 3):
            first = word[:i]
            second = word[i:]
            if self._is_valid_word(first) and self._is_valid_word(second):
                return [first, second]
        return None
    
    def _is_valid_root(self, root: str) -> bool:
        """Check if a root is valid (in dictionary or common)."""
        if root in self.lexicon:
            return True
        if root in self.word_freq and self.word_freq[root] > 10:
            return True
        # Accept roots that are 3+ characters
        return len(root) >= 3
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if standalone word exists."""
        return word in self.lexicon or word in self.word_freq
    
    def batch_segment(self, words: List[str]) -> List[SegmentationResult]:
        """Segment multiple words."""
        return [self.segment(w) for w in words]
    
    def encode_text(self, text: str) -> List[str]:
        """
        Encode text into morphological tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of morpheme tokens
        """
        # Simple word tokenization (can be improved)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        all_tokens = []
        for word in words:
            result = self.segment(word)
            all_tokens.extend(result.morphemes)
        
        return all_tokens
    
    def save_lexicon(self, path: Path):
        """Save current lexicon to JSON."""
        data = {
            'lexicon': self.lexicon,
            'word_freq': self.word_freq
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_word_list(cls, words: List[str], output_path: Optional[Path] = None) -> "MorphologicalTokenizer":
        """
        Build tokenizer from word list using rule-based segmentation.
        
        Args:
            words: List of words to analyze
            output_path: Optional path to save lexicon
            
        Returns:
            Initialized tokenizer
        """
        tokenizer = cls()
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        tokenizer.word_freq = word_counts
        
        # Build lexicon using rule-based segmentation
        for word in word_counts.keys():
            result = tokenizer.segment(word)
            if result.segmentation_type != "character":
                tokenizer.lexicon[word] = result.morphemes
        
        if output_path:
            tokenizer.save_lexicon(output_path)
        
        return tokenizer


class AgglutinativeTokenizer(MorphologicalTokenizer):
    """
    Aggressive agglutinative segmentation.
    Maximizes morpheme boundaries for compositional learning.
    """
    
    def segment(self, word: str) -> SegmentationResult:
        """Aggressive multi-level segmentation."""
        word_lower = word.lower()
        
        # Dictionary lookup first
        if word_lower in self.lexicon:
            return SegmentationResult(
                word=word,
                morphemes=self.lexicon[word_lower],
                segmentation_type="dictionary",
                confidence=1.0
            )
        
        # Recursive prefix + suffix stripping
        morphemes = self._recursive_segment(word_lower)
        
        if len(morphemes) > 1:
            return SegmentationResult(
                word=word,
                morphemes=morphemes,
                segmentation_type="agglutinative",
                confidence=0.6
            )
        
        # Fallback
        return super().segment(word)
    
    def _recursive_segment(self, word: str) -> List[str]:
        """Recursively segment word into morphemes."""
        if len(word) <= 3:
            return [word]
        
        # Try prefix
        for prefix in sorted(self.DERIVATIONAL_PREFIXES, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                remainder = word[len(prefix):]
                sub_result = self._recursive_segment(remainder)
                return [prefix] + sub_result
        
        # Try suffix
        all_suffixes = self.DERIVATIONAL_SUFFIXES + self.INFLECTIONAL_SUFFIXES
        for suffix in sorted(all_suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                if len(root) >= 3:
                    return [root, suffix]
        
        return [word]


# Example usage and testing
if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = MorphologicalTokenizer()
    
    test_words = [
        "running", "runner", "impossible", "impossibility",
        "uncomfortable", "telecommunications", "treehouse",
        "antidisestablishmentarianism", "hello", "world"
    ]
    
    print("Morphological Segmentation Tests:")
    print("=" * 60)
    
    for word in test_words:
        result = tokenizer.segment(word)
        print(f"{word:30} → {' + '.join(result.morphemes):40} ({result.segmentation_type}, conf={result.confidence:.2f})")
    
    print("\n" + "=" * 60)
    print("Agglutinative Segmentation Tests:")
    print("=" * 60)
    
    agg_tokenizer = AgglutinativeTokenizer()
    for word in test_words:
        result = agg_tokenizer.segment(word)
        print(f"{word:30} → {' + '.join(result.morphemes):40} ({result.segmentation_type}, conf={result.confidence:.2f})")
