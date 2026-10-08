# Provenance caveat for `baseline_reproduction_check`

Added after the run; `results.json` is unaltered.

The synthetic side here is **not** measured on the same image files as the
committed baseline `20261006T152242Z_measure_confidence_distribution`
(mean top-1 0.9606, 11.69% of frames below 0.9). That baseline was
measured on the **RTX 3060's** `data/s1` render. This run is on the
**laptop's own** `data/s1`, which was rendered before the `textures.py`
sort fix (commit `1c71fe7`) and is therefore a different set of pictures
of the same 23,346 texts — see `TRANSFER_TO_LAPTOP.md`, which makes the
same point about S2 and which is why `data/s1` was deliberately not
copied (Tier 3).

Evidence the two renders differ: this run counts **585,505** frames over
the val split against the baseline's **586,747**. Frame count is
`sum(width // 8)` over the split and is independent of batching, so a
difference can only come from the images themselves. Verified directly:
scanning this machine's val-split image headers gives
`sum(w // 8) = 585,505`, exactly matching what the run counted. The
measurement is internally exact; the data differs.

So `baseline_reproduction_check` should be read as **agreement across two
independent renders** (delta +0.0006 mean top-1, −0.0020 fraction below
0.9), not as a bit-exact reproduction. That is the stronger claim of the
two: the synthetic confidence level is a property of the pipeline, not of
one particular render.

Either way it is ~500x smaller than the real-vs-synthetic gap this run
exists to measure (0.9613 → 0.6265).
