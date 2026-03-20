# Parameter Golf Submission Checklist (IPA Tokenizer Approach)

Research date: 2026-03-19
Based on: openai/parameter-golf repo analysis of 6 accepted submissions

---

## 1. Folder Structure

The PR must add exactly ONE new folder under `records/`:

```
records/
  track_10min_16mb/          # if training completes in <10min on 8xH100
    YYYY-MM-DD_ShortName/
      README.md
      submission.json
      train_gpt.py            # self-contained training + eval script
      train.log               # at least 1 run log
      train_seed42.log        # additional seed logs for p<0.01 significance
      train_seed7.log         # (3 seeds typical for record submissions)
```

Or for non-record:

```
records/
  track_non_record_16mb/
    YYYY-MM-DD_ShortName/
      (same files)
```

Naming convention: `2026-03-20_IPA_BPE_SlidingWindow` (date + concise technique name).

The PR must ONLY add files in this new folder. No changes to root files.

---

## 2. Required Files

### 2a. `submission.json`

Two formats observed in accepted submissions:

**Minimal format (baseline style):**
```json
{
  "author": "Your Name",
  "github_id": "your_github_username",
  "name": "Short descriptive name",
  "blurb": "1-2 sentence description of the approach and key result.",
  "date": "2026-03-20T00:00:00Z",
  "val_loss": 1.98000000,
  "val_bpb": 1.17200000,
  "bytes_total": 15900000,
  "bytes_code": 65000
}
```

**Extended format (record style with multiple seeds):**
```json
{
  "track": "10min_16mb",
  "date": "2026-03-20",
  "name": "IPA BPE + Sliding Window Eval",
  "author": "Your Name",
  "seed_results": {
    "1337": {"val_loss": 1.980, "val_bpb": 1.172, "steps": 10400, "ms_per_step": 57.0},
    "42":   {"val_loss": 1.982, "val_bpb": 1.174, "steps": 10700, "ms_per_step": 56.0},
    "7":    {"val_loss": 1.983, "val_bpb": 1.174, "steps": 10500, "ms_per_step": 57.0}
  },
  "mean_val_loss": 1.98200000,
  "mean_val_bpb": 1.17300000,
  "p_value": 0.0001,
  "artifact_bytes": 15374243,
  "code_bytes": 50651
}
```

**Key fields:**
- `val_bpb` is the post-quantization, post-sliding-window (if used) score
- `bytes_total` = compressed model bytes + code bytes
- `bytes_code` = byte length of `train_gpt.py` (calculated as `len(Path(__file__).read_text().encode("utf-8"))`)
- For record submissions, include `seed_results` with 3 seeds and `p_value`

### 2b. `README.md`

Based on accepted submissions, include these sections:

```markdown
# <Technique Name>

**Mean val_bpb: X.XXXX** (N seeds, p<0.01)

## Key Idea

1-2 paragraphs explaining the core approach.

## Results

| Metric | Baseline | This Submission |
|---|---|---|
| Post-quant val_bpb | 1.2244 | X.XXXX |
| Improvement | -- | -0.0XXX |
| Training steps | ... | ... |
| Eval time (8xH100) | ... | ... |
| Artifact size | ... | ... bytes |

(If multiple seeds, show a seed table like the SOTA submission does.)

## Configuration

List all env vars / hyperparameters.

## Command

Exact `torchrun` command to reproduce the run.

## Key Metrics (from train.log)

- Training stop step
- Pre-quant and post-quant val_loss / val_bpb
- Training time
- Peak memory
- Model size breakdown
- Eval time

## Included Files

Bullet list of every file in the submission folder.
```

### 2c. `train_gpt.py`

- Must be the EXACT script used for the run
- Must be self-contained: compile and run from within the records folder
- The script measures its own size via `Path(__file__).read_text()`

### 2d. Train logs

- At least 1 log file (named `train.log` or `train_seed<N>.log`)
- For record submissions: 3 runs with different seeds to prove p<0.01 significance
- Must show the `final_int8_zlib_roundtrip_exact val_bpb:X.XXXXXXXX` line

---

## 3. Artifact Size Constraint

Total must be under **16,000,000 bytes** (decimal, not MiB).

```
artifact = bytes_model_compressed + bytes_code
```

- `bytes_code` = size of `train_gpt.py` in UTF-8 bytes
- `bytes_model_compressed` = size of `final_model.int8.ptz` (int8 quantized + zlib compressed)

**Current code_bytes for our train_gpt.py: ~65,088 bytes** -- this is already large compared to the baseline (47,642) and SOTA (50,651). If we embed IPA data, it will grow further.

---

## 4. IPA-Specific Risks and Mitigations

### RISK 1: "Submissions that edit the tokenizer will be examined much more carefully"

**Status: HIGH RISK.** We use a completely custom IPA-based tokenizer (`ipa_bpe_1024.model`). This is the most scrutinized category of submissions.

**Mitigation:**
- The README must extensively explain the IPA tokenization approach
- Must prove the val_bpb calculation is correct (see Risk 3)
- Include a clear explanation of why IPA improves compression
- Be prepared for reviewers to deeply audit the byte-counting logic

### RISK 2: Self-contained artifact requirement

The competition says: "No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible."

**Current state of our dependencies:**
- `ipa_bpe_1024.model` (14,165 bytes) -- SentencePiece model for IPA tokens
- `data/ipa_exceptions_2k.json` (118,692 bytes) -- exception dictionary
- `minimal_ipa_converter.py` (12,832 bytes) -- text-to-IPA converter
- `convert_fineweb_to_ipa_bpe_shards.py` (7,300 bytes) -- shard converter

**Critical question:** The evaluation runs `train_gpt.py` which loads pre-tokenized data from `DATA_PATH`. The tokenizer file is only needed for building the byte lookup tables (LUTs). If the IPA shards store original byte counts in their headers (version 2 format), and the LUT is built differently in IPA mode, then:

- The tokenizer `.model` file may NOT be needed at eval time if the LUTs are hardcoded or derived from the shard data
- BUT if `train_gpt.py` loads the tokenizer at startup, it must be present
- The IPA exception dict and converter are only needed for the DATA CONVERSION step (creating shards), not during training/eval

**Mitigation:**
- Verify exactly which files `train_gpt.py` needs at runtime vs. at data-prep time
- If the tokenizer is needed at runtime, embed it or ship it in the submission folder
- The data conversion pipeline (`minimal_ipa_converter.py`, `ipa_exceptions_2k.json`) runs BEFORE training. Reviewers need to be able to reproduce the tokenized data, so either:
  - (a) Include the conversion scripts and data in the submission folder, or
  - (b) Upload the pre-tokenized IPA shards to HuggingFace and reference them (like the baseline does with `cached_challenge_fineweb.py`)
- Option (b) is strongly preferred -- the baseline already uses cached HF data

### RISK 3: Correct val_bpb with override_byte_count

Our IPA approach changes the tokenization, which means the standard per-token byte count LUT does not apply. Instead, we use `override_byte_count` which stores the original (pre-IPA) byte count in the shard headers.

**The formula:**
```
val_bpb = total_nats / (original_byte_count * ln(2))
```

This is correct IF AND ONLY IF:
- `original_byte_count` is the exact number of UTF-8 bytes in the original FineWeb validation text
- The same validation documents are used as in the standard SP-1024 pipeline
- No bytes are lost or gained during the IPA conversion

**Mitigation:**
- Cross-validate: run the SAME validation text through both the standard SP-1024 pipeline and our IPA pipeline, confirm the original byte counts match
- Include the cross-validation result in the README
- Show the exact byte count and how it was computed
- Reviewers will want to see that `override_byte_count` matches the known FineWeb val byte count

### RISK 4: Significance threshold

Record submissions must beat SOTA by >= 0.005 nats with p<0.01.

**Current SOTA:** 1.1748 val_bpb (notapplica, Muon WD + 10 layer)

**Target:** Must achieve <= 1.1698 val_bpb

**Mitigation:**
- Run 3 seeds minimum
- Compute mean and standard deviation
- Report the t-test p-value explicitly in submission.json

### RISK 5: code_bytes inflation

Every extra file embedded in `train_gpt.py` inflates code_bytes, eating into the 16MB budget.

**Current sizes:**
- `train_gpt.py`: 65,088 bytes
- `ipa_exceptions_2k.json`: 118,692 bytes (if embedded, this DOUBLES the code size)
- `ipa_bpe_1024.model`: 14,165 bytes (binary, would need base64 encoding = ~19KB)

**Mitigation:**
- If `ipa_exceptions_2k.json` is only needed for data conversion (not training), do NOT embed it
- If the tokenizer model is needed at training time, embed it as a base64 constant (~19KB overhead)
- Aggressively minimize any embedded data

### RISK 6: Eval time limit

Eval must complete in under 10 minutes on 8xH100 (separate from training time).

**Mitigation:**
- Sliding window eval with stride=64 takes ~70-160s in existing submissions
- Our IPA eval should be similar or faster (same architecture, different tokenization)
- Monitor and report eval time in the submission

---

## 5. PR Description Template

```markdown
## Record: IPA BPE Tokenization + [Other Techniques] (val_bpb=X.XXXX)

### Summary
- IPA (International Phonetic Alphabet) tokenization reduces the vocabulary's phonetic ambiguity, improving per-byte compression
- Custom BPE vocabulary of 1024 IPA tokens trained on FineWeb
- [Other techniques: sliding window eval, FP16 embed, etc.]
- Mean val_bpb: X.XXXX over 3 seeds (p < 0.01)

### Tokenizer Change Justification
This submission uses a custom IPA-based tokenizer. Per the competition rules, we provide:
1. Full source code for the IPA conversion pipeline
2. Cross-validation showing original byte counts are preserved exactly
3. Detailed explanation of the val_bpb calculation correctness

### Artifact Size
- Model (int8+zlib): XX,XXX,XXX bytes
- Code: XX,XXX bytes
- Total: XX,XXX,XXX bytes (under 16,000,000 cap)

### Reproducibility
[Exact commands to reproduce from scratch]
```

---

## 6. Pre-Submission Checklist

- [ ] Folder named `YYYY-MM-DD_ShortName` in the correct track subfolder
- [ ] `submission.json` with all required fields and exact numeric values from logs
- [ ] `README.md` with technique explanation, results table, command, and metrics
- [ ] `train_gpt.py` is the exact script used (byte-for-byte)
- [ ] At least 1 training log (3 logs for record submissions)
- [ ] Logs contain `final_int8_zlib_roundtrip_exact val_bpb:` line
- [ ] `bytes_total` in submission.json < 16,000,000
- [ ] val_bpb beats SOTA by >= 0.005 (for record track)
- [ ] 3 seeds with p < 0.01 statistical significance (for record track)
- [ ] `train_gpt.py` runs successfully from within the records subfolder
- [ ] No external downloads or network calls during training/eval
- [ ] IPA byte count cross-validated against standard SP-1024 byte count
- [ ] README explicitly addresses tokenizer change and bpb correctness
- [ ] PR only adds files in the new submission folder (no root changes)
- [ ] PR description includes tokenizer justification section

### IPA-Specific Checks

- [ ] Determine if `ipa_bpe_1024.model` is needed at training runtime (check tokenizer_path usage)
- [ ] Determine if `ipa_exceptions_2k.json` is needed at training runtime (likely NO, only for data prep)
- [ ] If tokenizer is needed at runtime, embed as base64 in train_gpt.py or include in submission folder
- [ ] Verify shard version 2 header stores correct original byte count
- [ ] Cross-validate: `sum(original_bytes)` across val shards == known FineWeb val UTF-8 byte count
- [ ] IPA conversion is lossless at the byte-counting level (no bytes dropped)
- [ ] Consider uploading pre-tokenized IPA shards to HuggingFace for reproducibility
