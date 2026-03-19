# Morphological Segmentation Experiment

**Date:** March 19, 2026  
**Hypothesis:** English has hidden agglutinative and fusional morphology that, if exposed through better tokenization, could improve compression (bpb) by allowing compositional generalization with fewer parameters.

---

## Executive Summary

**Verdict: 🔄 PROMISING - Requires Implementation**

Inspired by Zulu's Bantu agglutination, Cherokee's polysynthesis, and Amharic's root-and-pattern morphology, English reveals itself as a linguistic chimera with:
- Germanic compounding (compound words)
- Romance inflection (plurals, past tense)  
- Greek/Latin bound roots (tele-phone, bi-cycle)

**Core Insight:** BPE treats "running" and "runner" as unrelated tokens (~3000 embeddings apart in vector space). A morphological tokenizer would encode them as `run` + `ing` and `run` + `ner`, forcing the model to learn compositional rules rather than memorizing every inflection.

---

## The Grammar Lessons

### Zulu (Bantu Agglutination)

**Structure:** Extensive noun classes (15), subject/object agreement on verbs, tense/aspect/mood baked into morphology. One verb encodes: subject + object + tense + aspect + negation + applicative + causative + reciprocal.

**Lesson:** Agglutination = **compositional structure**. If boundaries were clean, a model learns rules, not memorization.

**English Application:** We have vestigial agglutination:
```
un-comfort-able-ness → un + comfort + able + ness
anti-dis-establish-ment-arian-ism → anti + dis + establish + ment + arian + ism
```

### Cherokee (Polysynthesis)

**Structure:** Words = entire sentences. "I saw it" = one word with incorporated object. Pronouns bound inside verbs. **Evidentials** mark knowledge source (I saw it vs I heard it vs I infer it).

**Lesson:** Information packing varies radically. Cherokee packs more per word than English spreads across tokens.

**English Application:** Our "light verbs" (make a decision, take a walk, do research) are polysynthetic remnants. Tokenizing by **semantic unit** could reduce sequence length.

### Amharic (Root-and-Pattern)

**Structure:** Triconsonantal roots (k-t-b = "write") + vowel patterns = verb forms, participles, nouns. **Non-concatenative** - you can't just string morphemes left-to-right.

**Lesson:** Information is **distributed non-locally**. Root consonants = meaning, vowels = grammar.

**English Application:** We have non-concatenative traces:
```
sing / sang / sung / song - vowel alternation marks tense/voice
man / men - vowel change marks plural  
drive / drove / driven - ablaut pattern
```

BPE treats these as unrelated. A root-and-pattern tokenizer would link them.

---

## English Morphological Analysis

### The Hybrid Nature of English

English pretends to be an **isolating language** (like Mandarin) but is actually a **weird hybrid**:

| Layer | Examples | Morphological Type |
|-------|----------|-------------------|
| Germanic core | compounds, strong verbs | Agglutinative + Ablaut |
| Romance overlay | -tion, -sion, -ment | Concatenative suffixes |
| Greek/Latin | tele-, bio-, micro- | Bound roots (cranberry morphemes) |
| Modern coinages | -gate, -holic, -zilla | Analogical compounding |

### Current BPE Problems

**Problem 1: Related words scattered**
```
"running" → token 2847
"runner" → token 4921  
"ran" → token 1834
"runs" → token 1567
```

The 16MB model learns 4 separate embeddings for one concept. With morphology:
```
"running" → [run] [ing]
"runner" → [run] [ner]  
"ran" → [run] [past_ablaut]
"runs" → [run] [3sg_present]
```

Now the model learns 1 root embedding + morphological rules.

**Problem 2: Productivity untapped**
If the model sees "treehugger" (compound), BPE might tokenize as [tree] [hug] [ger] or [treehug] [ger] depending on frequency. With morphology:
```
"treehugger" → [tree] [hug] [agent_noun]
```

The model now understands it's [noun] + [verb] + [person_who_does] = tree enthusiast, even if it never saw the word before.

---

## Proposed Morphological Tokenizer Design

### Option A: Agglutinative Splitting

**Strategy:** Maximize morpheme boundaries

```
impossibility → im + poss + ible + ity
telecommunications → tele + com + mun + ic + ate + ion + s
uncomfortableness → un + comfort + able + ness
```

**Vocabulary Size:** ~5000 roots + ~200 affixes = 5200 tokens vs BPE's 1024-32000

**Trade-offs:**
- ✅ Longer sequences but semantically regular
- ✅ Model learns compositionality  
- ❌ Suffix variants (ation, ition, ion) need multiple tokens

### Option B: Root-and-Pattern Hybrid

**Strategy:** Encode ablaut/mutation patterns explicitly

```
sing/sang/sung → [sing] [present] vs [sing] [past_ablaut] vs [sing] [participle]
drive/drove/driven → [drive] [present] vs [drive] [past_ablaut_o] vs [drive] [participle_en]
man/men → [man] [singular] vs [man] [plural_umlaut]
```

**Vocabulary Size:** ~8000 roots + ~50 pattern markers

**Trade-offs:**
- ✅ Captures non-concatenative morphology  
- ✅ Regularizes strong verbs
- ❌ More complex token vocabulary
- ❌ Pattern markers are rare (sparse embeddings)

### Option C: Semantic Constituency

**Strategy:** Inspired by Cherokee - pack semantic units

```
make a decision → [make_decision]
take a walk → [take_walk]  
do research → [do_research]
```

These "light verb constructions" are semantically equivalent to simple verbs (decide, walk, research).

**Implementation:** Train tokenizer to merge light verbs with their objects when they form semantic units.

---

## Implementation Plan

### Phase 1: Morpheme Dictionary

**Source:** Use existing resources
- CELEX (Dutch/English lexicon with morphological analysis)
- UniMorph (universal morphological features)
- MorphyNet (English morphological database)

**Build mapping:**
```python
MORPHOLOGICAL_LEXICON = {
    "running": ["run", "V", "PROG"],      # verb, progressive aspect
    "runner": ["run", "N", "AGENT"],       # noun, agentive
    "impossibility": ["im", "poss", "ible", "ity"],  # negation, root, adjective, noun
    "telecommunications": ["tele", "communicate", "ion", "s"],  # distant, verb, nominalization, plural
}
```

### Phase 2: Rule-Based Segmenter

**For OOV (out-of-vocabulary) words:**
```python
# Unsupervised morphology induction
# Use Harris's successor variety or Goldsmith's Linguistica

def segment_unknown(word):
    # Try prefix stripping
    for prefix in ["un", "re", "dis", "pre", "anti", "de"]:
        if word.startswith(prefix):
            return [prefix, word[len(prefix):]]
    
    # Try suffix stripping (Porter stemmer style)
    for suffix in ["ing", "tion", "ness", "ment", "able", "ible"]:
        if word.endswith(suffix):
            return [word[:-len(suffix)], suffix]
    
    # Try compound splitting (use frequency data)
    # "treehouse" → ["tree", "house"] if both exist in vocab
    
    return [word]  # atomic if no pattern matches
```

### Phase 3: Neural Morphological Tokenizer

**Train a small BERT-style model** to predict morpheme boundaries:

```python
# Input: "impossibility"
# Output: [im|poss|ible|ity]

# Architecture:
# - 2-layer transformer
# - 128 dim
# - 8K vocabulary
# - Trained on CELEX + MorphyNet
# - Predicts BIO tags: B-ROOT, I-ROOT, B-PREFIX, I-SUFFIX, etc.
```

**This becomes your tokenizer** - replaces SentencePiece BPE.

### Phase 4: Integration with Parameter Golf

**Modified train_gpt.py flow:**
```python
# 1. Text → Morphemes (via neural tokenizer)
tokens = morphological_tokenizer.encode(text)  # "running" → ["run", "_ing"]

# 2. Tokens → IDs
ids = [vocab[t] for t in tokens]

# 3. Train as usual
# Model learns that "run" + "_ing" predicts ...

# 4. Evaluation
# Model predicts morpheme sequence
# Decode back to text for bpb calculation
```

---

## Expected Benefits for 16MB Constraint

### Compression Analysis

**BPE 1024 approach:**
- Vocab: 1024 tokens
- Sequence: ~1.2 tokens per word (average)
- Embeddings: 1024 × 512 × 4 bytes = ~2MB

**Morphological approach (estimated):**
- Vocab: ~5000 morphemes (roots) + ~200 affixes = 5200
- Sequence: ~2 tokens per word (roots + affixes)
- Embeddings: 5200 × 512 × 4 bytes = ~10.6MB

Wait - that's **worse**! More parameters in embeddings.

### The Real Win: Smaller Model Capacity Needed

**Key insight:** With compositional tokenization, you need **fewer layers/smaller hidden dims** to achieve same loss.

**Why:**
- BPE model must memorize: "running", "runner", "runs", "ran" as 4 unrelated concepts
- Morph model learns: "run" + rules for [ing], [ner], [s], [past]

The morph model delegates inflection to **positional patterns** rather than distinct embeddings.

**Hypothesis:** A 9-layer 512-dim BPE model ≈ 6-layer 384-dim morph model in loss, but morph model is **smaller overall** despite larger vocab.

**Math:**
```
BPE model:
- Embeddings: 1024 × 512 × 4 × 2 (in + out) = 4.2MB
- Transformer: ~11MB (9 layers, 512 dim)
- Total: ~15.2MB

Morph model (projected):
- Embeddings: 5200 × 384 × 4 × 2 = 16MB ❌ (too big!)

# Need smaller embeddings - try shared + projection
Morph model with factorized embeddings:
- Root embeddings: 5000 × 128 × 4 = 2.5MB  
- Affix embeddings: 200 × 128 × 4 = 0.1MB
- Projection: 128 → 384 = small
- Transformer: ~7MB (6 layers, 384 dim)
- Total: ~9.6MB ✅
```

### Factorized Embeddings

**Idea from low-rank compression:**
```python
# Instead of full embeddings
root_emb = embedding_table[root_id]  # 128-dim
affix_emb = affix_table[affix_id]    # 128-dim
combined = torch.cat([root_emb, affix_emb])  # 256-dim
projected = projection_layer(combined)     # 384-dim for transformer
```

**Benefits:**
- Smaller per-token storage
- Roots and affixes share semantic space (all "-ing" verbs cluster)
- Model learns morphological relationships for free

---

## Risks and Open Questions

### Risk 1: Ambiguity

**Problem:** Many segmentations possible:
```
"unlockable" = un + lock + able (can be unlocked)
              OR un + lockable (not able to be locked)
```

**Mitigation:** 
- Use most frequent segmentation (from training corpus)
- Or: let model learn both and use context to disambiguate

### Risk 2: OOV Handling

**Problem:** New words ("webinar", "podcast") not in morpheme lexicon.

**Mitigation:**
- Keep a fallback to character-level or subword for unknowns
- Or use unsupervised segmentation as backup

### Risk 3: Evaluation Complexity

**Problem:** Need to convert morph tokens → text → calculate bpb.

**Mitigation:**
- Build deterministic decoder
- Cache validation set conversion (one-time cost)

### Risk 4: Challenging Assumptions

**Question:** Does English actually have enough regular morphology for this to help?

**Data point:** ~40% of English words are morphologically complex (have 2+ morphemes). But they're the **high-frequency** words in text. So impact is outsized.

---

## Recommended Next Steps

1. **Build morpheme lexicon** from CELEX/UniMorph (~2 hours)
2. **Create rule-based segmenter** for OOV words (~4 hours)  
3. **Convert validation set** to morphological tokens (~1 hour)
4. **Train small test model** (6-layer, 384-dim, factorized embeddings) (~30 min on H100)
5. **Compare bpb** against baseline BPE model
6. **Iterate** on segmentation quality based on results

---

## Related Work

- **Byte-level BPE** (GPT-2): Handles any language but loses morphology
- **SentencePiece with IPA** (your current experiment): Phonetic level, loses orthography
- **Morfessor**: Unsupervised morphological segmentation - could provide segmentation hints
- **CANINE**: Character-level transformer with learned morphological awareness

---

## Files to Create

1. `morphological_tokenizer.py` - Segmentation engine
2. `morph_vocab.json` - Root + affix vocabulary  
3. `convert_to_morph.py` - Dataset conversion script
4. `train_gpt_morph.py` - Modified training with factorized embeddings

---

**Experiment Status:** Planning  
**Confidence:** Medium-High (strong linguistic motivation, implementation complexity unknown)  
**Estimated Impact:** 5-10% bpb improvement through better generalization  
**Time to MVP:** 2-3 days

