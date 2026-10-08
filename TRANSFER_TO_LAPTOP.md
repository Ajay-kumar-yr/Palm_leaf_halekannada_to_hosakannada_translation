# Moving SETU from the 3060 desktop back to the laptop

Written 2026-10-08 on the RTX 3060 machine. Like `HANDOFF.md` and
`STATUS.md` this is session state, not a rules document — delete it once
the move is done and verified.

**Why move at all:** the only remaining blocker in `STATUS.md` §4.3 is
Palmira, and Palmira exists *only* on the laptop (a `palmira` conda env
with Detectron2 + a custom-compiled DefGrid CUDA extension, no recorded
from-scratch setup). Three of the remaining tasks need it. Rebuilding
that env on the 3060 is hours of GPU-arch-specific CUDA pinning that may
not succeed; carrying ~865 MB to the laptop takes minutes.

**The headline:** you do **not** need to move the 17 GB of data. The
recogniser's cached per-frame output over S2 (614 MB) is a complete
stand-in for the rendered images for every remaining task. `data/s1`
(12 GB) and `data/real` (3.1 GB) stay where they are — the laptop already
has its own `data/real`, and nothing left to do needs S1.

---

## Part 0 — Check these on the laptop first, before copying anything

Do this before building the package. If any of it has rotted, you want to
know now.

```bash
# --- on the LAPTOP ---

# 1. Does the Palmira env still work? This is the entire reason for the move.
conda activate palmira
python -c "import detectron2, torch; print(detectron2.__version__, torch.__version__, torch.cuda.is_available())"
python -c "from defgrid.config import add_defgrid_maskhead_config; print('defgrid OK')"
#    ^ the DefGrid CUDA extension is the fragile part. If this import fails,
#      STOP and reconsider — the move buys you nothing without it.

# 2. Are Palmira's weights still there?
ls -la pretrained/Palmira_indiscapes.pth   # inside the Palmira checkout

# 3. Is the repo's real-image dataset still in place?
ls /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/data/real/dataset | head
#    ^ note the laptop path is major_proj, NOT major_project/updated_maj_proj.
#      run_palmira_survey.py has this path hardcoded and is correct FOR THE
#      LAPTOP, so leave it alone there.

# 4. Does the laptop repo have anything the 3060 never saw?
cd /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation
git status --short
git log --oneline origin/main..HEAD     # any unpushed laptop commits?
#    ^ if either is non-empty, deal with it BEFORE pulling.

# 5. Does it already have the S3 pretrain checkpoint and the vocab?
ls -la runs/20260930T052116Z_modernizer_pretrain_s3/best_model.pt
ls -la data/modernizer_vocab.json

# 6. Free space — ~1 GB for the minimum, ~3.5 GB if you take S2 too.
df -h /mnt/d
```

---

## Part 1 — Code: nothing to copy

The working tree on the 3060 is clean and fully pushed (verified:
`git status` empty, `origin/main..HEAD` empty). All source, `CLAUDE.md`,
`STATUS.md`, this file, `data/splits/*` (including the **frozen test
split** and its audit manifest) and `data/raw_corpus/*.py` travel by git.

```powershell
# --- on the 3060, to send this document itself ---
git add TRANSFER_TO_LAPTOP.md make_transfer_package.ps1 .gitignore
git commit -m "Add laptop transfer plan and packaging script"
git push
```

```bash
# --- on the LAPTOP ---
git pull
```

---

## Part 2 — Build the binary package (on the 3060)

Everything git cannot carry is gitignored. The script assembles it:

```powershell
# --- on the 3060, Windows PowerShell, repo root ---
.\make_transfer_package.ps1 -Tier 1               # ~865 MB, the minimum
# .\make_transfer_package.ps1 -Tier 2             # ~3.1 GB, adds S2 images + corpora
# .\make_transfer_package.ps1 -Tier 3             # ~15 GB, adds S1 (only if retraining)
# .\make_transfer_package.ps1 -Tier 1 -SkipHash   # skip checksums if in a hurry
```

It writes `transfer_to_laptop/` (gitignored) plus `MANIFEST.sha256` in
`sha256sum -c` format.

**A Tier 1 package was already built and verified on 2026-10-08** —
`transfer_to_laptop/` is sitting in the repo root right now: **76 files,
0.82 GB**, built in 3 s, all 76 checksums confirmed with `sha256sum -c`,
and the packaged CRNN checkpoint hash cross-checked against the source.
If you want Tier 1, skip the script and copy that folder. Re-run the
script with `-Force` only if you want a different tier.

Also verified: the eval path hardcodes exactly three file paths —
`data/splits/s2_inband_verse_ids.txt` (committed, arrives by git),
`data/modernizer_vocab.json` and
`runs/20260930T052116Z_modernizer_pretrain_s3/best_model.pt` (both in the
package). So Tier 1 + `git pull` is self-sufficient for the Part 5 smoke
test and for the §4.4 B4 re-fine-tune.

### What each tier contains, and why

**Tier 1 — essential, ~865 MB.** Without these the laptop cannot continue.

| Item | Size | Why irreplaceable |
|---|---|---|
| `runs/**` minus all `*.pt` | ~30 MB | Every config/results/git-commit record. Per `CLAUDE.md` rule 4 a result without a run folder does not exist — this *is* the audit trail for the whole report |
| `runs/20261007T022632Z_cache_s2_recogniser/` | 401 MB | Recogniser output over all 3,500 S2 lines: top-1 strings **and** full per-frame distributions. Replaces the S2 images for every remaining task |
| `runs/20261007T143817Z_cache_s2_recogniser/` | 213 MB | Same for the 2,075-line in-band top-up |
| `runs/20261006T120718Z_crnn_train_s1/best_model.pt` | 13 MB | **The recogniser.** Epoch 10, 0.39% val CER, 1.53% frozen-test CER. 12 epochs × ~45 min to reproduce — and not reproducible at all on the laptop's older renders |
| `runs/20261007T144601Z_modernizer_finetune_s2_b3/best_model.pt` | 69 MB | Best B3 (val_loss 1.5276) |
| `runs/20261007T131859Z_modernizer_finetune_s2_b4_T1.5/best_model.pt` | 69 MB | Best B4 |
| `runs/20260930T052116Z_modernizer_pretrain_s3/best_model.pt` | 69 MB | The shared S3 warm start, `finetune_s2.py`'s default `--resume-from`. The laptop made this one, so it should already have it — included anyway, it is cheap insurance |
| `data/modernizer_vocab.json` | 4 KB | Tiny and gitignored. Both eval and fine-tune hard-code this path and will not run without it |

**Tier 2 — cheap insurance, +2.2 GB.** Not needed for anything in
`STATUS.md` §4, but these are the two things that cannot be regenerated
*identically*:

- `data/s2/` (1.4 GB, 3,502 files) — includes the **frozen test split
  images**. The laptop's own S2 render predates the `textures.py` sort fix
  (commit `1c71fe7`), so its images are *different pictures* from the ones
  the reported 1.53% CER was measured on. If you ever need to look at a
  test image, you want these, not the laptop's.
- `data/s2_extra/` (707 MB, 2,076 files) — the in-band top-up render.
- `data/raw_corpus/*.txt` + `*_meta.jsonl` (126 MB) — deterministic output
  of the committed build scripts from the checksum-verified
  KannadaLit4NLP zip, so technically regenerable, but copying beats
  re-running the builds.

**Tier 3 — only if you plan to retrain the CRNN, +12 GB.** `data/s1/`
(23,348 files). You almost certainly do not: `STATUS.md` §3.4 found
perfect OCR would move the end-to-end score by 0.30 chrF++, and the
laptop's 4 GB card trains at ~31 min/epoch using only 75% of the corpus
against the 3060's ~45 min using 95.6%. Retraining on the laptop would be
a step backwards. Leave S1 here.

### Deliberately NOT copied

- `data/real/` (3.1 GB) — the laptop already has it, same Mendeley download.
- `data/s1/` at Tiers 1–2 — see above.
- The 5 superseded modernizer checkpoints and 3 superseded CRNN
  checkpoints (~345 MB + 41 MB). Their `results.json` files still travel,
  so the numbers in `STATUS.md` stay auditable; only the weights are
  dropped.
- `runs/20261007T134952Z_joint_unlocked_train/*.pt` (82 MB) — the joint run
  showed no end-to-end gain (§3.6). Its results record travels; weights
  do not.
- `.venv/` — never copy a virtualenv between machines.
- The three incomplete run folders listed in `STATUS.md` §6. The script
  skips them; delete them here too if you like.

### How long the copy takes

Source disk read speed measured on this machine: **1,632 MB/s** for one
large file, **441 MB/s / ~1,010 files/s** for small image files. So the
source is never the bottleneck — your transfer medium is:

| Tier | Size | Files | USB 3.0 (~100 MB/s) | USB 2.0 (~30 MB/s) | Gigabit LAN (~110 MB/s) |
|---|---|---|---|---|---|
| 1 | 865 MB (measured) | 76 (measured) | ~9 min | ~29 min | ~8 min |
| 2 | ~3.1 GB | ~5,700 | ~31 min | ~1 h 45 | ~28 min |
| 3 | ~15 GB | ~29,000 | ~2 h 30 | ~8 h+ | ~2 h 20 |

Small-file counts hurt disproportionately on USB 2.0 and on any
network/FAT32 path — Tiers 2 and 3 can run well over these estimates.
Add ~10 s/GB for the SHA-256 pass at each end. **Tier 1 has almost no
small files, which is another reason to prefer it.**

---

## Part 3 — Carry it across

Pick one. In all cases the destination is the laptop repo root.

**USB / external drive (simplest).** Copy `transfer_to_laptop/` to the
drive, then on the laptop copy its *contents* over the repo root so
`runs/` merges with `runs/` and `data/` with `data/`.

```powershell
# on the LAPTOP, Windows side — merge, do not replace
robocopy "E:\transfer_to_laptop" "D:\major_proj\Palm_leaf_halekannada_to_hosakannada_translation" /E
```

Avoid FAT32 if you take Tier 2 or 3 — use exFAT or NTFS.

**Direct over LAN** (both machines on, faster and no middleman):

```powershell
# from the 3060, pushing to a share on the laptop
robocopy "transfer_to_laptop" "\\LAPTOP-NAME\d$\major_proj\Palm_leaf_halekannada_to_hosakannada_translation" /E /Z
#   /Z = restartable mode, worth it over a flaky link
```

**Do not use cloud sync** for the caches. The 401 MB `logprobs.npy` is
exactly the kind of file the WSL-networking corruption in `HANDOFF.md`
mangled — and if you do, the checksum step below is non-optional.

---

## Part 4 — Verify on the laptop

```bash
# --- on the LAPTOP, WSL, repo root ---
sha256sum -c MANIFEST.sha256
#   every line must say OK. One FAILED = recopy that file; do not proceed.
```

If `sha256sum` is unavailable, in PowerShell:

```powershell
Get-Content MANIFEST.sha256 | ForEach-Object {
  $parts = $_ -split '\s+', 2
  $expected = $parts[0]
  $path = $parts[1].Trim()
  $actual = (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
  if ($actual -ne $expected) { Write-Host "FAILED $path" -ForegroundColor Red }
}
```

---

## Part 5 — Smoke test: prove the transfer actually works

Run this before doing any new work. It exercises the caches, the vocab,
the frozen split, the checkpoint and the metrics module in one go.

```bash
# --- on the LAPTOP, WSL ---
cd /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation
source ~/setu-venv/bin/activate      # or whichever env has torch
export PYTHONPATH=src

python -m setu.eval.run_comparison \
  --cache-dir runs/20261007T022632Z_cache_s2_recogniser \
  --b3 runs/20261007T144601Z_modernizer_finetune_s2_b3/best_model.pt
```

Expected, from `runs/20261007T151916Z_eval_b0_b3_b4/results.json`:

| Quantity | Expected | Tolerance |
|---|---|---|
| `n_frozen_test` | 1358 | **exact** |
| `n_inband_test_scored` | 465 | **exact** |
| `recogniser_cer_all_frozen_test` | 0.015272 | **exact** |
| `B0_recognised` chrF++ | 21.0146 | **exact** |
| `B0_oracle` chrF++ | 21.3101 | **exact** |
| `B3_argmax` chrF++ | 13.6669 | ±0.1 |

The first five come straight out of the cache and the metrics module with
no model forward pass, so they must match **exactly** — any difference
means a corrupted cache, the wrong `s2_inband_verse_ids.txt`, or a bad
vocab file. The B3 number needs greedy generation through the
modernizer, so tiny cross-GPU/cuDNN floating-point drift is acceptable
there; a difference of more than ~0.1 is not.

This runs on CPU if CUDA is unavailable, so it works even if the laptop's
torch install is unhappy.

---

## Part 6 — Laptop-specific settings you must change

These are the traps. The committed defaults are now tuned for the 3060's
12 GB card and **will not work on the laptop's 4 GB RTX 3050**.

1. **CRNN training flags.** `src/setu/recogniser/train.py` now defaults to
   `--max-pixels-per-batch 3000000` and `--max-single-image-pixels
   5000000` (commit `cc3d4db`, re-profiled for the 3060). The 3050's
   empirically tuned values were **2200000** and **1500000**. If you run
   any CRNN training on the laptop, pass the lower values explicitly. Per
   `CLAUDE.md` rule 7, pass them on the command line — do not quietly
   edit the defaults back.

2. **Always `--time-steps N` first.** Every training/eval script takes it;
   it measures real per-step cost and peak memory, projects the run, and
   writes nothing. `STATUS.md` §3.8: arithmetic estimates were wrong in
   *both* directions (predicted 11.7 GB for a run that used 3.78 GB;
   predicted ~10 min for a fine-tune that took ~47). This matters more on
   4 GB than it did on 12 GB.

3. **Expect the `/mnt/d` header-scan cost back.** Scanning 23,346 S1
   image headers took **~4 min** on the laptop's WSL↔Windows bridge vs
   **43 s** here. Only bites S1-touching runs, which you are probably not
   doing.

4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is already set
   unconditionally at the top of the five entry points (§3.8). Do not
   remove it; on 4 GB it matters even more.

5. **B4 re-fine-tune cost.** `STATUS.md` §4.4 budgets ~26 min for it on
   the 3060. Assume meaningfully longer on the 3050, and possibly memory
   pressure at `--max-tokens-per-batch 8192`. Measure with `--time-steps
   15` before committing.

---

## Part 7 — Where to pick up, in order

From `STATUS.md` §4, reordered so the Palmira-dependent work (the reason
you moved) comes first while the env is known-good:

1. **Add crop-saving to `data/raw_corpus/run_palmira_survey.py`.** It
   already holds `predictions["instances"]` and reads only
   `pred_classes` off it — the `pred_masks` for all 9,631 detected line
   segments were computed and discarded. This is a ~20-line change to a
   script that already ran successfully on this exact env. `STATUS.md`
   §4.3 calls the committed survey output unreusable, which is true of
   the *file*, not of the script.
   - You only need ~50 pages for §6.4 plus 10–15 for the demo, not all
     738 — minutes of GPU time.
   - Keep `CLAUDE.md`'s rule: original colour photo in, never the U-Net
     black-and-white output.
2. **§6.4 measurement #2** — recogniser top-1 confidence, real vs
   synthetic. The synthetic half already exists (mean 0.9606, 11.69% of
   frames below 0.9, `runs/20261006T152242Z_measure_confidence_distribution`).
   This is the valuable one: it quantifies the synthetic→real domain gap
   with zero ground truth.
3. **§6.4 measurement #3** — flagging rate real vs synthetic, reusing
   `src/setu/bridge/flagging.py`.
4. **Demo pages.** `demo_pages/` is empty on both machines (just a
   `.gitkeep`). Needs the crops from step 1. `CLAUDE.md` rule 8 still
   holds: never appears in a reported number, curation disclosed.
5. **§4.1, the forward rule-based orthographic modernizer** — pure text
   work, no GPU, no Palmira, and `STATUS.md` ranks it highest-value. It
   needs your judgement on the Kannada rules, so it is the one task that
   cannot be front-loaded by a machine.
6. **§4.4 B4 re-fine-tune** — only if you still want the B3-vs-B4 claim.
   `STATUS.md` is honest that this is probably not where the remaining
   time should go.

**Free, no Palmira, do it any time:** re-rank demo-page candidates using
the per-page `Character Line Segment` counts in the committed
`data/raw_corpus/palmira_survey_results.jsonl` (738 pages). That is a far
better page-quality signal than the brightness/contrast heuristic behind
`data/splits/hkhpl_page_survey.csv`, where only 25 of 738 rows were
human-reviewed.

---

## Part 8 — What stays on the 3060, and do not delete it

Treat the 3060 as the archive, not as decommissioned:

- It holds the **only** copy of `data/s1` (12 GB) and of the S2/S2-extra
  renders the reported numbers were actually measured on.
- It is the only machine that can retrain the CRNN at full corpus
  coverage in reasonable time (95.6% of S1 vs the laptop's 75%).
- If a reviewer asks for a retrain, an ablation, or more synthetic data,
  you come back here. Do not wipe it until the report is submitted and
  graded.

The move is one-directional for *work in progress* only. Anything new the
laptop produces still has to come back by git (code, results.json, small
files) or by hand (checkpoints) — same asymmetry, same gitignore.

---

## Appendix — irreplaceable vs regenerable

| Asset | Status |
|---|---|
| `runs/20261006T120718Z_crnn_train_s1/best_model.pt` | **Irreplaceable.** Trained on this machine's S1 render; the laptop's render differs (pre-`1c71fe7` texture ordering) |
| The two `*_cache_s2_recogniser/` folders | **Irreplaceable** without both the checkpoint and this machine's S2 images |
| `data/s2/`, `data/s2_extra/` | **Irreplaceable as rendered.** Regenerable as *different images* |
| `data/s1/` | Same — regenerable as different images, ~hours |
| S3 pretrain + S2 fine-tune checkpoints | Regenerable: 30–60 min + 10–47 min, but only if the data above exists |
| `data/raw_corpus/*.txt` | Regenerable, deterministic, from committed scripts + the verified zip |
| `data/modernizer_vocab.json` | Regenerable but gitignored — **copy it**, it is 4 KB and everything depends on it |
| `data/splits/*` | Committed. **Never regenerate** — `s2_test_verse_ids.txt` is the frozen split (`CLAUDE.md` rule 1) |
| `data/real/` | Re-downloadable from Mendeley; laptop already has it |
| All source, docs, build scripts, survey JSONL | Committed and pushed |
