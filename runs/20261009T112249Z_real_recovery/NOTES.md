# Real-crop recovery, recomputed to satisfy rule 8 — 27.8%

Replaces `20261008T152006Z_real_recovery` (34.2%), which was computed
on six lines from `group1_1.128` and `group1_1.198` — both in
`demo_pages/`, so CLAUDE.md rule 8 forbids reporting it.

| | withdrawn | this run |
|---|---|---|
| lines | 6, demo pages | **13, train pages** |
| reader | `gemini-3.5-flash-lite` (0.428 CER) | **`gemini-3.5-flash`** (0.370) |
| consensus-wrong characters | 73 | **126** |
| recovered in some sample | 25 (34.2%) | **35 (27.8%)** |
| mean sample disagreement | 0.379 | **0.163** |
| mean fraction contested | 0.528 | **0.246** |

Readings come from `20261009T105809Z_demo_build_real`, 65 calls on
`gemini-3.5-flash`, 5 samples per crop at temperature 0.8, U-Net
binarized input.

## The number went down, and that is the finding

DEMO_PLAN.md §6a expected the better reader to give a cleaner figure,
and noted that if it stayed near 34% the correlated-error explanation
was confirmed. It fell to 27.8%, which confirms it more strongly:
**disagreement collapsed by more than half (0.379 → 0.163) while
recovery fell only a fifth**, so the better reader is not finding the
right character more often in its spread — it is producing less spread
to look in. Confidently-wrong is precisely the failure an ensemble
cannot bracket.

This is an argument for the project's own design: a CTC posterior is
per-frame and keeps mass on rivals even where the top-1 is wrong (80.6%
of wrong frames hold the correct symbol in the top-5), which an LLM
ensemble does not.

## Why 13 and not 16

Quota. 16 reportable lines exist; an earlier run crashed on an uncaught
`ConnectionResetError` and spent ~17 of the day's 100 `3.5-flash` calls
for nothing (`20261009T104824Z_demo_build_real/NOTES.md`), leaving room
for 13 × 5. The three remaining lines — `1.140/line_11`, `line_15`,
`line_19` — can be added on any later day's quota, and the 13 already
read are cached under `data/demo_cache/readings/`, so topping up costs
15 calls, not 80.

## Caveats carried forward

- An **ensemble over a borrowed vision model**, not the CTC soft bridge.
  Same claim, different mechanism.
- Scored against a single blind hand transcription by one reader, which
  carries its own error rate (the ಶ/ಕ correction demonstrated it).
- The comparison with the withdrawn figure varies both the reader and
  the line set, because the old line set may not be reported. It is
  evidence, not a controlled A/B.

## Provenance

`git_commit.txt` records `16e0e9e`, which **predates the scripts that
produced this run** -- `setu.eval.make_recovery_set`, the reading cache
and the `OSError` fix in `sample_readings` were all uncommitted when it
ran. Same situation as DEMO_PLAN.md 9 records for 2026-10-08. They are
committed in the commit that adds this note, unchanged.
