# Sign-Language-Inspired Tokenization TODO

Session IDs
- ASL linguistics/history research: `ses_2f8c0e34bffeuxLJUVl3WAUvVs`
- BSL linguistics/history research: `ses_2f8c0e2c8ffeJ2eIcf8Eo1DJCF`

## Objective
- [ ] Build and test a compositional tokenizer that keeps a tiny base vocabulary while restoring structure through parallel feature channels.

## 1) Token format and schema
- [ ] Define a token record with `base_id` and optional feature IDs.
- [ ] Support two base modes: `ipa` and `morph`.
- [ ] Add `ORTH_HINT` feature channel (small spelling disambiguation hints).
- [ ] Add `MORPH_ROLE` feature channel (`ROOT`, `PFX`, `SFX`, `INFLECT`, `DERIVE`, `COMPOUND`).
- [ ] Add `CLAUSE_FUNC` feature channel (`DECL`, `Q`, `NEG`, `FOCUS`, `COND`).
- [ ] Add `DISCOURSE_ROLE` feature channel (`NARRATION`, `QUOTE`, `SHIFT`).
- [ ] Add `VARIANT_TAG` feature channel (`std`, `regional_x`, `oov_backoff`).
- [ ] Define `NULL` feature value for every channel.

## 2) Preprocessing pipeline
- [ ] Implement minimal normalization policy and document it.
- [ ] IPA path: run minimal G2P converter to generate base units.
- [ ] Morph path: run rule-based segmenter + lexicon lookup for base units.
- [ ] Attach feature IDs with deterministic taggers.
- [ ] Serialize aligned streams for training and keep reversible metadata.
- [ ] Add explicit OOV/backoff behavior so encoding never fails.

## 3) Model integration
- [ ] Implement channel embeddings for all enabled feature channels.
- [ ] Start with additive fusion (`e_total = e_base + e_features`).
- [ ] Add optional concat+projection fusion as an ablation.
- [ ] Keep feature-channel embedding dims small to protect budget.
- [ ] Tie weights where possible to reduce parameter footprint.

## 4) Core experiments to try
- [ ] `A0` baseline: current BPE-1024 setup.
- [ ] `A1` IPA-only base channel.
- [ ] `A2` IPA + `ORTH_HINT`.
- [ ] `A3` morph-only base channel.
- [ ] `A4` morph + `MORPH_ROLE`.
- [ ] `A5` hybrid: IPA base + `MORPH_ROLE` + `ORTH_HINT`.
- [ ] `A6` full feature bundle.
- [ ] `A7` leave-one-channel-out from full bundle.

## 5) Evaluation and probes
- [ ] Track primary metric: validation `bpb`.
- [ ] Track embedding memory and total compressed artifact size.
- [ ] Track training throughput/step time.
- [ ] Build homophone disambiguation probe set (`night/knight`, `to/too/two`).
- [ ] Build morphological-family generalization probes (`run/running/runner/ran`).
- [ ] Build clause-function probe set (questions, negation, focus).
- [ ] Report performance by `VARIANT_TAG` bucket.

## 6) Success criteria
- [ ] Beat BPE baseline `bpb` at equal total parameter budget.
- [ ] Improve homophone probe accuracy over IPA-only model.
- [ ] Improve morphological-family generalization over BPE baseline.
- [ ] Keep feature-channel memory overhead small (target: <= 10-15% vs base embedding setup).

## 7) Recommended implementation order
- [ ] First pass: implement and run `A1`, `A3`, and `A5` only.
- [ ] Add probe-eval scripts once first pass trains end-to-end.
- [ ] Expand to `A6` and `A7` only if `A5` is promising.
- [ ] Freeze schema and tune architecture only after ablation signal is clear.
