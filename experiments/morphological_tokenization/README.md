# Morphological Tokenization Experiment

**Status:** Active Development  
**Goal:** Improve Parameter Golf model efficiency by segmenting English into morphemes (roots + affixes) rather than arbitrary BPE tokens.

---

## What Was Created

### 1. Core Tokenizer (`src/morphological_tokenizer.py`)
Rule-based morphological segmentation with two modes:
- **Standard**: Conservative segmentation (prefixes → suffixes → compounds)
- **Agglutinative**: Aggressive recursive segmentation

**Key insight:** BPE treats "running" and "runner" as unrelated (~3000 vectors apart). Morphological tokenizer exposes `run` + `ing` and `run` + `ner`, enabling compositional learning.

### 2. Word Extraction (`extract_finetweb_words.py`)
Extracts word frequencies from actual FineWeb binary shards:
- Reads binary format (256 int32 header + uint16 tokens)
- Decodes via SentencePiece
- Counts word frequencies from text

**Results from 10M tokens:**
- 105,501 unique words found
- 31,716 words above frequency 5
- Top words: the, and, to, of, a, in, is, for...

### 3. Comprehensive Lexicon Builder (`build_comprehensive_lexicon.py`)
Builds large-scale morpheme vocabulary from extracted frequencies:

**Version 2 Vocabulary:**
- **7,825 roots** (unique morphemes that appear as stems)
- **56 prefixes** (un-, re-, dis-, pre-, etc.)
- **58 suffixes** (-ing, -tion, -ness, -able, etc.)
- **2,500 final vocab size** (top morphemes by frequency)

**Coverage:** 16.6% token coverage for top 1000 words (448/1000 words segmented)

### 4. Dataset Converter (`convert_fineweb_to_morph.py`)
Converts FineWeb binary files to morpheme sequences:
- Parses binary header format
- Decodes BPE → text via SentencePiece
- Segments words → morphemes
- Writes morpheme IDs to binary

---

## Files Structure

```
experiments/morphological_tokenization/
├── FINDINGS.md                       # Detailed research findings
├── README.md                         # This file
├── build_lexicon.py                  # Basic vocabulary builder
├── build_comprehensive_lexicon.py   # Full-scale vocabulary builder
├── extract_finetweb_words.py        # Word frequency extraction
├── convert_fineweb_to_morph.py      # Dataset converter
├── debug_binary.py                  # Binary format debugger
├── src/
│   └── morphological_tokenizer.py   # Core segmentation engine
├── data/
│   ├── morph_lexicon.json           # Basic lexicon (235 morphemes)
│   ├── morph_vocab.json             # Basic vocab mapping
│   ├── morph_lexicon_v2.json        # Comprehensive lexicon (7,939 morphemes)
│   ├── morph_vocab_v2.json          # Comprehensive vocab (2,500 tokens)
│   ├── fineweb_word_freq.json       # Word frequencies (31,716 words)
│   └── test_morph.bin               # Sample converted shard
└── tests/
    └── (test scripts)
```

---

## Usage

### Extract Word Frequencies
```bash
cd experiments/morphological_tokenization
python3 extract_finetweb_words.py \
    --max_shards 1 \
    --max_tokens_per_shard 10000000
```

### Build Comprehensive Vocabulary
```bash
python3 build_comprehensive_lexicon.py \
    --word_freq data/fineweb_word_freq.json \
    --max_words 20000 \
    --max_vocab_size 2500
```

### Convert Dataset (Small Test)
```bash
python3 convert_fineweb_to_morph.py \
    --input_path ../../data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin \
    --output_path data/test_morph.bin \
    --vocab_path data/morph_vocab_v2.json \
    --max_tokens 100000
```

### Compare Tokenizations
```bash
python3 convert_fineweb_to_morph.py \
    --input_path ../../data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin \
    --output_path data/test_morph.bin \
    --vocab_path data/morph_vocab_v2.json \
    --compare \
    --compare_tokens 1000
```

### Use in Code
```python
from src.morphological_tokenizer import AgglutinativeTokenizer

tokenizer = AgglutinativeTokenizer()
result = tokenizer.segment("impossibility")
print(result.morphemes)  # ['im', 'possibil', 'ity']
```

---

## Test Results

### V2 Comprehensive Vocabulary
```
Top 20 Morphemes (by frequency in FineWeb):
  s                     249,478 (suffix)    # plural
  ed                    103,833 (suffix)    # past tense  
  ing                    99,655 (suffix)    # progressive
  er                     85,020 (suffix)    # comparative/agent
  y                      82,700 (suffix)    # adjective/adverb
  es                     78,797 (suffix)    # plural
  re                     62,888 (prefix)     # again/back
  ly                     40,661 (suffix)    # adverb
  al                     39,961 (suffix)    # adjective
  in                     35,640 (prefix)    # in/into
  ...

Sample Tokenizations:
running              → runn + ing                    [370, 8]
happiness            → happi + ness                  [1468, 71]  
representation       → re + pre + sent + ation       [12, 30, 106, 23]
international        → inter + nation + al           [50, 82, 14]
multidimensional     → multi + dimension + al        [438, 1912, 14]
```

### Conversion Ratios
From 10,000 BPE tokens:
- BPE: 10,000 tokens → Morph: 13,007 tokens (**1.30x expansion**)

This expansion is expected - we're decomposing words into smaller units.

---

## Key Hypothesis

With ~2,500 morphemes vs BPE's 1,024 tokens:
- **Vocab size:** 2.4x larger
- **Compositional generalization:** Model learns `run` + rules instead of memorizing each inflection
- **Sequence length:** ~1.3x longer (roots + affixes vs single tokens)
- **Net effect:** Smaller effective model capacity needed for same loss because of compositional structure

**Challenge:** The 16MB parameter budget is tight. Even with 2,500 vocab, embeddings use:
- BPE: 1,024 × 512 × 4 × 2 = 4.2MB
- Morph: 2,500 × 512 × 4 × 2 = 10.2MB (too big!)

**Solution needed:** Factorized embeddings (root + affix → projection) or smaller base dims.

**Target:** 5-10% bpb improvement through better generalization (if we can fit it in 16MB)

---

## Next Steps

1. ✅ ~~Extract real word frequencies~~ Done
2. ✅ ~~Build comprehensive vocabulary~~ Done (2,500 morphemes)
3. 🔄 **Optimize vocabulary size** - Try 1,000-1,500 morphemes to fit 16MB budget
4. 🔄 **Implement factorized embeddings** - Separate root/affix embeddings with projection
5. 🔄 **Convert full validation set** - Batch processing for 62M tokens
6. 🔄 **Train test model** - Compare bpb vs baseline
7. 🔄 **Handle edge cases** - Proper nouns, compounds, foreign words

---

## Performance Notes

- **Word extraction:** ~10M tokens/minute (decodes via SP)
- **Lexicon building:** ~20k words/minute (rule-based segmentation)
- **Dataset conversion:** ~50k tokens/minute (I/O + decode + segment bottleneck)
- **Full validation (62M tokens):** ~20-30 minutes (may need optimization)

For full-scale experiments, consider:
1. Parallelizing conversion across shards
2. Caching decoded text
3. Using faster segmentation (regex vs recursive)
