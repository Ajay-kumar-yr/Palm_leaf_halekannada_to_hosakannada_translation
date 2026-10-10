# Grayscale arm of the binarization comparison

Identical to `20261009T105420Z_gold_real_eval` except that the
recogniser is fed the grayscale photo crop instead of the U-Net
binarized one. True CER **0.6925** against **0.6961** binarized --
a 0.0036 gap on 16 lines, i.e. noise.

The write-up, the rule 8 reasoning and the measurement bug this pair
caught are all in the twin run's NOTES.md. Read that one.

## Provenance

`git_commit.txt` records `16e0e9e`, which **predates the scripts that
produced this run** -- `setu.eval.make_recovery_set`, the reading cache
and the `OSError` fix in `sample_readings` were all uncommitted when it
ran. Same situation as DEMO_PLAN.md 9 records for 2026-10-08. They are
committed in the commit that adds this note, unchanged.
