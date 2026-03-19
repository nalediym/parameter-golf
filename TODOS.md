# TODOS

## Timer output clutter
**What:** The `rpod` countdown timer prints on new lines instead of overwriting in-place because training output on stdout interferes with `\r` carriage returns.
**Why:** Makes the terminal hard to read during training runs.
**Fix:** Move timer to terminal title bar (`\033]0;...\007`) or only print every 60s instead of every second.
**Priority:** Low — cosmetic, doesn't affect results.

## Phase 4: Full 8xH100 run
**What:** Run `./rpod train` on an 8xH100 pod for competition-grade bpb.
**Why:** 1xH100 gets ~1500 steps in 10 min. 8xH100 gets ~12,000 steps — needed for convergence.
**Depends on:** 1xH100 results showing IPA bpb trending in right direction.
**Cost:** ~$3.50 (10 min at $20/hr).

## Tests T6-T16
**What:** Build remaining test suite (shard integrity, training integration, bpb correctness).
**Why:** Competition scrutinizes tokenizer changes. Tests are proof of correctness.
**Status:** T1-T5 done (66/66 passing). T6-T16 not yet built.
**Priority:** Medium — needed before submission, not before experiments.
