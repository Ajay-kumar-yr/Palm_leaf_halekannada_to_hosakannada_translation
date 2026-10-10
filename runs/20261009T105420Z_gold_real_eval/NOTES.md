# Does binarized input help the *fine-tuned* recogniser? No.

Two runs, identical but for the image the recogniser is fed:

| run | input | true CER vs hand transcription |
|---|---|---|
| `20261009T105420Z_gold_real_eval` (this one) | U-Net binarized (`gold_unet/`) | **0.6961** |
| `20261009T105426Z_gold_real_eval` | photo crop, grayscale (`train/`) | **0.6925** |

Same checkpoint (`20261008T104839Z_crnn_finetune_real/best_model.pt`),
same 16 lines, same gold. The gap is **0.0036 on 16 lines** — noise.

This was the open question DEMO_PLAN.md 6a listed for midday: binarized
input was known to help the *vision model* (0.428 vs 0.447,
`ocr_input_check`) and to lift the *original* checkpoint's confidence
(0.6453 vs 0.6265 mean top-1, `20261008T141042Z_real_vs_synthetic_confidence`),
but had never been tried on the fine-tuned model.

**It changes nothing.** Note that the fine-tune trained on grayscale
photo crops, so binarized input is mildly out-of-distribution for it and
still scores the same — the error is not in the preprocessing. That is
the same conclusion RESULTS.md 3.1 reached from the other direction:
94% of the domain gap is letter shape.

## Why 16 lines and not 32

CLAUDE.md rule 8. The 32 hand-transcribed gold lines are half
demo-page lines (`group1_1.128`, `.198`, `.26`, `.28`) and half
train-page lines (`1.108`, `1.12`, `1.140`). Only the 16 train-page
lines may appear in a reported number; `setu.eval.make_recovery_set`
builds that subset and documents the split. The 0.687 figure in
RESULTS.md row 9 was computed over all 32 and needs the same treatment;
on the 16 reportable lines alone it is **0.696**.

## A measurement bug this run caught

The first pass of `make_recovery_set` wrote the *hand transcription*
into the labels file's `text` field. Everything downstream reads `text`
as the **machine** label, so `gold_real` scored the gold against itself
and reported the vision model's labels as 97% accurate — mean CER
**0.0283** where the true figure is **0.3723**. With the machine label
restored, the 16 reportable lines give 0.3723, against 0.370 over all
32: a clean cross-check that the subset is representative.

Added to RESULTS.md 6.

## Provenance

`git_commit.txt` records `16e0e9e`, which **predates the scripts that
produced this run** -- `setu.eval.make_recovery_set`, the reading cache
and the `OSError` fix in `sample_readings` were all uncommitted when it
ran. Same situation as DEMO_PLAN.md 9 records for 2026-10-08. They are
committed in the commit that adds this note, unchanged.
