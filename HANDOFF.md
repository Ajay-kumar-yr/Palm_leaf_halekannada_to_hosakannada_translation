# Handoff: full setup from a bare clone + HKHPL copy

Written at commit `ba004af`, for the RTX 3060 (12GB) desktop, which as of
this writing has **only** `git clone` done and the HKHPL dataset copied
into `data/real/` — nothing else. This replaces the previous version of
this file, which wrongly assumed more would carry over automatically.

Read alongside `CLAUDE.md` (project rules) — this file is session state,
not a rules document, and should be deleted once it's stale rather than
kept updated indefinitely.

## Standing preference

**Ask before starting any new process** (rendering, training, installing
things) — confirm with the user first rather than just proceeding. This
was an explicit, repeated instruction across the session that produced
this handoff; it's not written down anywhere else, so carry it forward.

## What `git clone` actually gives you, and what it doesn't

Git gets you: all source code, `CLAUDE.md`/this file, the Navilu font
(`data/external_models/fonts/navilu/Navilu.ttf` — small enough to commit
directly, SIL OFL licensed), the Sajjan U-Net binarizer's `LICENSE` and
`PROVENANCE.md` (not its weights, see below), `data/splits/*` (the
**frozen S2 test-verse-ID split and its append-only audit trail —
already correct, do not regenerate**, see the warning below), a handful
of small committed result files (`data/raw_corpus/palmira_survey_results.jsonl`,
`src/setu/pos/tagged_sample*.{jsonl,tsv}`), and the `*.py` build scripts
under `data/raw_corpus/`.

Git does **not** get you (all gitignored, by design — see `.gitignore`):
every `runs/<timestamp>_*/` folder (meaning **every checkpoint and every
numeric result from this whole project only exists on the old laptop**
unless someone copies them over by hand), `data/s1/images/` and
`data/s2/images/` (rendered line images), `data/raw_corpus/*.txt` and
`extracted/` (the built corpora — only the scripts that produce them are
committed), `data/external_models/sajjan_unet/*.pth` (93MB weights),
`data/modernizer_vocab.json`, and the `.venv/`.

**If you want the modernizer's S3-pretrained checkpoint
(`best_model.pt`, val_char_acc 95.9%) without retraining it, it has to be
copied from the old machine by hand — it is not retrievable any other
way.** Same for any CRNN checkpoint, once one exists.

## Step 1 — Python environment (Windows, for rendering)

Rendering needs a Windows-accessible font path for the Nirmala UI
fallback (Navilu itself is just a `.ttf` file and works anywhere, but
`fonts.py`'s fallback is hardcoded to `C:\Windows\Fonts\Nirmala.ttc` —
fine on Windows, a no-op elsewhere since Navilu alone covers 97%+ of S1
already).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

This installs `torch` CPU-only by default on Windows — that's fine,
rendering doesn't need a GPU. `sacrebleu` and `segmentation-models-pytorch`
are both already in `requirements.txt` from this session; the latter was
genuinely added mid-session (CLAUDE.md "ask before adding a dependency"
was satisfied for it already, nothing further to ask).

## Step 2 — Download KannadaLit4NLP and rebuild the corpus

The real KannadaLit4NLP dataset (Mendeley Data, CC BY 4.0, DOI
`10.17632/nvjydxpxjr.3`) is not committed (34MB zip). Download it fresh:

```
mkdir -p data/raw_corpus/extracted
curl -s "https://data.mendeley.com/api/datasets/nvjydxpxjr/files" -o /tmp/mendeley_files.json
# parse the "download_url" field out of that JSON (file id may differ run
# to run; don't hardcode it) and:
curl -sL -o data/raw_corpus/KannadaLit4NLP.zip "<that download_url>"
```

Verify before trusting it: `sha256sum data/raw_corpus/KannadaLit4NLP.zip`
should be `013245506e09a9f51c9102faa8f0db42410db888500605dd5c233df0dbeae1b1`.
Then:

```
cd data/raw_corpus && unzip -q KannadaLit4NLP.zip -d extracted && cd ../..
.venv\Scripts\python data/raw_corpus/build_corpora.py
.venv\Scripts\python data/raw_corpus/build_s3.py
.venv\Scripts\python data/raw_corpus/build_s2_render_set.py
```

**Before doing anything else, run `git diff data/splits/` and confirm it
is empty.** `build_corpora.py` unconditionally recomputes the S2
test-split freeze every time it runs — it *should* reproduce the exact
same split deterministically (same corpus version + same seeded hash of
verse_id), since `data/splits/s2_test_verse_ids.txt` is already correctly
frozen and committed. But CLAUDE.md rule 1 is "never rewrite a frozen
test split," so treat any actual diff here as a stop-everything problem
(likely a different KannadaLit4NLP version got downloaded), not something
to shrug off and continue past.

## Step 3 — Render S1 (and S2, if needed yet)

```
PYTHONPATH=src .venv\Scripts\python -m setu.render.generate --corpus data/raw_corpus/s1_corpus.txt --out-dir data/s1 --manifest data/s1/manifest.jsonl --seed 0
```

~2 hours, CPU-bound (font rasterization + damage simulation, not GPU),
should take about as long here as it did on the laptop. Expect 0 skipped
(the danda/double-danda glyph fallback to Nirmala handles the ~15% of
lines Navilu alone can't render). S2 isn't needed for CRNN training —
defer it (`--corpus data/raw_corpus/s2_corpus_render.txt --paired`) until
Week 3 work actually starts.

## Step 4 — WSL2 + CUDA PyTorch (for training)

```
wsl --install -d Ubuntu   # if not already present
wsl -d Ubuntu -- sudo apt update && sudo apt install -y python3-venv python3-pip
wsl -d Ubuntu -- python3 -m venv ~/setu-venv
wsl -d Ubuntu -- bash -c "source ~/setu-venv/bin/activate && pip install torch torchvision"
wsl -d Ubuntu -- bash -c "source ~/setu-venv/bin/activate && pip install numpy Pillow tqdm pyyaml sacrebleu uharfbuzz freetype-py segmentation-models-pytorch"
```

`sudo apt install` needs an interactive password — that part can't be
done from an automated session, do it in a real terminal first. Verify
CUDA actually works before anything else:

```
wsl -d Ubuntu -- bash -c "source ~/setu-venv/bin/activate && python -c \"import torch; print(torch.cuda.get_device_name(0))\""
```

should print the RTX 3060.

## Step 5 — Re-profile GPU memory settings for the 12GB card

**Do not reuse the numbers already in `train.py`'s defaults
(`--max-single-image-pixels 1500000`, `--max-pixels-per-batch 2200000`)
— those were empirically tuned against the 4GB RTX 3050 and are almost
certainly far too conservative for 12GB.** At that 1.5M-px cutoff, only
75% of S1 (17,505/23,346 lines) actually gets used for training; on 12GB
this ceiling should move way up, likely close to the full corpus.

Profiling method used last time (repeat this, don't guess): time a single
forward+backward+optimizer step on images of increasing pixel area, with
`torch.cuda.synchronize()` around the timed block and a warmup call
first (the first call includes one-time cuDNN algorithm-selection
overhead that will mislead you if counted). Watch for a *non-linear*
jump, not just OOM — on the 3050, per-step time was well-behaved
(~0.2-0.3s) up to ~1.9M px, then jumped to 3.5s at 3.8M px (13x slower
for 2x the pixels), well before actually hitting OOM. The CRNN's later
conv blocks (4-5) don't reduce width, only height, so very wide+tall
images get disproportionately expensive, not just memory-heavy — this is
architectural, not GPU-specific, so expect the same qualitative jump on
the 3060, just at a higher pixel count. A single 700-char outlier line
was measured at 924×14,561px and used 3.3GB for a forward pass ALONE on
the 4GB card — even on 12GB, some extreme outliers may still need
excluding rather than batched around.

Measured on the 4GB card at its tuned settings: **~31 min/epoch**
training, plus a one-time ~4min startup cost scanning 23,346 image-file
headers (slow specifically because of the WSL↔Windows `/mnt/d`
filesystem bridge on that machine — worth checking whether this is still
the bottleneck here, or whether local disk changes that number).

Also: don't run two GPU processes at once even on 12GB without checking
`nvidia-smi` first — a concurrent-job OOM already happened once on the
4GB card from exactly this.

## Step 6 — Confirm before starting the actual CRNN training run

Per the standing preference above. Recommended epoch count from the
earlier discussion: **8-15 epochs**, given the stated bar is "pass
without major objections," not full convergence (CTC/CRNN models
typically want 20-30+ epochs to fully converge) — but confirm this with
the user before committing to a number, especially once real per-epoch
timing on the 3060 is known and the total wall-clock budget can be
reconsidered.

## Other gotchas already hit and fixed (don't re-discover these)

- **Python 3.14 on Linux defaults multiprocessing to `forkserver`**,
  which requires DataLoader `collate_fn` to be picklable — a local
  closure function fails silently with a pickling error. Both
  `train.py` and `pretrain.py` use plain functions/picklable classes for
  this reason; don't refactor back to a closure.
- **Manifest image paths must use `.as_posix()`**, not `str(Path)` —
  rendering runs on Windows (needs a Windows-accessible font) but
  training runs on WSL/Linux, and a Windows backslash is a literal
  filename character on Linux, not a path separator. Already fixed in
  `generate.py` and in every reader that loads a manifest written on
  Windows (`.replace("\\", "/")` before joining); don't revert either
  side.
- **PIL's decompression-bomb guard** rejects some real HKHPL scans
  (300+ megapixels) — disabled in `textures.py`
  (`Image.MAX_IMAGE_PIXELS = None`) since these are trusted local files,
  not untrusted uploads.
- **`textures.py` caches the real-image directory listing** at module
  level rather than re-scanning on every call. If this regresses,
  rendering gets very slow again (it's called multiple times per line:
  once for background, once per hole).
- **sacrebleu's BLEU reads as 0 on short sentences** (a line under 4
  words has no 4-grams to match, independent of quality) — documented,
  expected behavior, not a bug to fix with a different smoothing method;
  it's part of why CLAUDE.md treats chrF++ as the primary metric.

## Where things actually stand (project status, not machine status)

**Week 1: complete**, including the real (not heuristic-proxy) §6.4
segmentation number — Palmira run over all 738 unique real HKHPL pages,
**96.9% segmentation success rate** (`runs/20261003T081302Z_palmira_full_page_survey`
on the laptop; the small results JSONL is committed at
`data/raw_corpus/palmira_survey_results.jsonl`). A 25-photo stratified
human-review sample also corrected the page-survey's "broken" bucket,
which the heuristic had mostly wrong (dark background dominating
whole-image stats, not actual illegibility) — the committed
`data/splits/hkhpl_page_survey.csv` reflects this for those 25 rows; the
other ~713 are still heuristic-proxy.

**S1 (23,346 lines) and S2 (3,500 lines: all 1,358 frozen test +
2,142 sampled train) are both built and rendered** on the laptop with
every fix in place (Navilu font, multiply-blend compositing, stroke
distortion, danda fallback) — but the rendered images themselves are
gitignored, so this machine needs to redo Step 3 above (or receive a
manual file copy) regardless.

**S3 modernizer pretraining: done, 4 epochs**, val_char_acc 95.9% — but
the checkpoint is gitignored (`runs/.../best_model.pt`), so either copy
it from the laptop or retrain here if needed.

**CRNN training: still not started anywhere.** This machine (3060, 12GB)
is specifically for that.

**Soft bridge, confidence flagging, chrF++/BLEU metrics, POS tagging
prototype: all built and tested** against synthetic data (no real CRNN
checkpoint exists yet to test against real recognizer output) —
`src/setu/bridge/soft_bridge.py`, `src/setu/bridge/flagging.py`,
`src/setu/eval/metrics.py`, `src/setu/pos/`. Nothing further to build
here until a trained CRNN exists.

**Real-photo binarizer wired in** (`src/setu/demo/binarize.py`, wraps
S.P. Sajjan's U-Net — the same model that produced HKHPL's own ground
truth) but its weights are gitignored; re-download per
`data/external_models/sajjan_unet/PROVENANCE.md` if the demo pipeline is
needed here. Not required for CRNN training itself.

**Palmira: not set up on this machine, and not needed here.** The §6.4
Palmira work is already done and recorded on the laptop (see above);
redoing it here is only necessary if the live demo needs to run on this
machine too. Its environment (a `palmira` conda env with Detectron2 + a
custom-compiled DefGrid CUDA extension) predates this session's own
work — no clean from-scratch setup script was ever recorded, so expect
real friction if it's ever needed here; ask before attempting it, since
it previously involved GPU-arch-specific CUDA toolkit pinning and
nontrivial compilation troubleshooting.

## Suggested order of operations on this machine

1. Steps 1-3 above (venv, corpus, render S1) — no GPU needed for any of
   this, can run while Step 4 (WSL setup) also happens.
2. Step 4 (WSL2 + CUDA torch) — confirm `torch.cuda.get_device_name(0)`
   actually prints the 3060 before trusting anything built on top of it.
3. Step 5 (re-profile GPU memory settings) — do not skip this and reuse
   the laptop's numbers.
4. Step 6 — confirm epoch count and get explicit go-ahead, then train.
5. Everything after that (soft bridge integration with a real
   checkpoint, S2 fine-tuning, joint training, Week 3/4 work) follows
   the roadmap's own sequencing in `CLAUDE.md`.
