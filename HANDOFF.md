# Handoff notes — moving from RTX 3050 (4GB) laptop to RTX 3060 (12GB) desktop

Written at commit `f330c5e`. Read this alongside `CLAUDE.md` (project
rules) — this file is session state, not a rules document, and should be
deleted once it's stale rather than kept updated indefinitely.

## Standing preference from the previous session

**Ask before starting any new process** (rendering, training, installing
things) — confirm with the user first rather than just proceeding. This
was an explicit, repeated instruction in the session that produced this
handoff; it's not written down anywhere else, so carry it forward.

## Where things actually stand

**Week 1: complete.** Page survey verdict filled (heuristic proxy, spot
checked), S2 test-verse-ID split frozen (`data/splits/`), CRNN 8-example
memorize check passes, damage simulator built, Palmira original-vs-B&W
check done.

**Corpus:** Real KannadaLit4NLP dataset downloaded and used (Mendeley
Data, CC BY 4.0) — not a placeholder. Built via
`data/raw_corpus/build_corpora.py` (S1/S2) and `build_s3.py` (S3). The
raw downloaded/generated corpus files are gitignored (reproducible from
those two scripts + the public dataset); only the scripts are committed.

**S1 (23,346 rendered lines):** Fully re-rendered at current HEAD with
every fix below already applied. Font is Navilu (SIL OFL, handwriting-
style — replaced Nirmala UI, which read as "typed text pasted on a
photo"; Nirmala kept only as a per-line fallback for danda/double-danda,
which Navilu lacks). Ink/background compositing uses a multiply blend
(not flat-replace) so leaf grain shows through strokes. Stroke distortion
(a smooth per-glyph elastic warp, `apply_stroke_distortion` in
`damage.py`) breaks up vector-font geometric perfection. **Fake-vs-real
classifier result: 74.7%** on a 2,000-line sample — inside the roadmap's
70-80% target band (`runs/20260930T101006Z_fake_vs_real_classifier`).

**S3 modernizer pretraining: done, 4 epochs.** Final checkpoint:
`runs/20260930T052116Z_modernizer_pretrain_s3/best_model.pt`. val_char_acc
95.9%, val_loss 0.137, still improving when stopped (diminishing returns
judged not worth chasing further — S3 is synthetic rule-generated data,
only a warm-start before Week 3's real S2 fine-tuning). Vocab at
`data/modernizer_vocab.json` (rebuild deterministically from the same
corpus if lost). **Bug fixed along the way:** PyTorch's default post-LN
transformer would not converge at all from scratch (loss stuck ~3.3 for
350+ epochs on the 8-example memorize check) — fixed by setting
`norm_first=True` in `setu/modernizer/model.py`. If anyone "simplifies"
that back to default, it will silently break training again.

**CRNN training: NOT started.** `setu/recogniser/train.py` works
end-to-end on this GPU after real debugging (see below) but no actual
training run has happened yet — only timing/smoke tests. Recommended
epoch count discussed with the user: **8-15 epochs**, given the stated
bar is "pass without major objections," not full convergence (CTC/CRNN
models typically want 20-30+ epochs to fully converge).

**Real-photo binarizer wired in:** `setu/demo/binarize.py` wraps S.P.
Sajjan's U-Net (same model that produced HKHPL's own Ground_Truth_images;
MIT licensed). Weights are gitignored
(`data/external_models/sajjan_unet/unet_best_weights.pth`, 93MB) —
re-download from the source repo noted in `PROVENANCE.md` in that folder
if missing. Only used for the real-photo inference path (Palmira → cut
lines → this → CRNN), never for synthetic S1/S2.

## GPU-memory tuning that MUST be redone on the 3060

Everything below was empirically profiled against the 4GB RTX 3050 and
is almost certainly wrong (too conservative) for a 12GB card. Do not just
reuse these numbers — re-profile, the same way this session did:

- `setu/recogniser/train.py --max-single-image-pixels` (currently
  1,500,000) and `--max-pixels-per-batch` (currently 2,200,000): these
  bound actual scanned pixel height×width, NOT text length — height
  varies independently of character count because `damage.py`'s skew
  rotation uses a random angle per line (`expand=True` grows the canvas
  unpredictably). A single 700-char outlier line was measured at
  924×14,561px and used 3.3GB for a forward pass ALONE on the 4GB card.
  At the current 1.5M-px cutoff, only 75% of S1 (17,505/23,346 lines) is
  actually used for training — the rest is excluded for speed, not
  dropped from the dataset itself. On 12GB this ceiling should move way
  up, likely close to the full corpus.
- Measured on the 4GB card at the current settings: **~31 min/epoch**
  training (plus a one-time ~4min startup cost scanning image dimensions
  — this scan reads 23,346 file headers and was slow specifically because
  of the WSL↔Windows `/mnt/d` filesystem bridge; consider whether that's
  still true on the new machine's setup).
- Profiling method used (repeat this, don't guess): time a single
  forward+backward+optimizer step on images of increasing pixel area
  (`torch.cuda.synchronize()` around the timed block, with a warmup call
  first — the first call includes one-time cuDNN algorithm-selection
  overhead and will mislead you if included in the timing). Watch for a
  *non-linear* jump, not just OOM — on the 3050, time was well-behaved
  (~0.2-0.3s/step) up to ~1.9M px then jumped to 3.5s/step at 3.8M px
  (13x slower for 2x the pixels) well before actually hitting OOM. The
  CRNN's later conv blocks (4-5) don't reduce width, only height, so
  very wide+tall images get disproportionately expensive, not just
  memory-heavy — this is architectural, not GPU-specific, so the same
  qualitative jump should appear on the 3060, just at a higher pixel
  count.
- `setu/modernizer/pretrain.py --batch-size`: was knocked down from 64 to
  16 on the 4GB card after an OOM (concurrent GPU jobs also caused a
  spurious OOM once — don't run two GPU processes at the same time on a
  memory-constrained card; less of a concern with 12GB but still worth
  checking `nvidia-smi` before launching a second job).

## Other gotchas already hit and fixed (don't re-discover these)

- **Python 3.14 on Linux defaults multiprocessing to `forkserver`**,
  which requires DataLoader `collate_fn` to be picklable — a local
  closure function fails silently with a pickling error. Both
  `train.py` and `pretrain.py` use plain functions/picklable classes for
  this reason; don't refactor back to a closure.
- **Manifest image paths must use `.as_posix()`**, not `str(Path)` —
  rendering runs on Windows (needs a Windows-accessible font), training
  runs on WSL/Linux, and a Windows backslash is a literal filename
  character on Linux, not a path separator. Already fixed in
  `generate.py`; if S1/S2 ever gets re-rendered, this stays fixed
  automatically, just don't revert it.
- **PIL's decompression-bomb guard** rejects some real HKHPL scans
  (300+ megapixels) — disabled in `textures.py` (`Image.MAX_IMAGE_PIXELS
  = None`) since these are trusted local files, not untrusted uploads.
- **`textures.py` re-scanned the real-image directory on every call** —
  now cached at module level. If this regresses, rendering gets very
  slow again (it's called multiple times per line: once for background,
  once per hole).

## Suggested next steps on the new machine

1. `git pull`, copy `data/s1/images/` + `data/s1/manifest.jsonl` over (or
   re-render — ~2h, CPU-bound, should be similar wall-clock on either
   machine since it's not GPU-bound).
2. Set up WSL2 Ubuntu + venv + CUDA torch (same process as this session:
   `pip install torch torchvision` from the default PyPI index pulled in
   a CUDA build fine on Linux; `segmentation-models-pytorch` additionally
   needed for the binarizer).
3. Re-profile the GPU-memory numbers above before launching any real
   training run.
4. **Confirm epoch count and get explicit go-ahead before starting CRNN
   training** — per the standing preference noted at the top.
5. After CRNN training: fake-vs-real check was only run on a 2,000-line
   sample so far; Week 3 work (soft bridge, S2 rendering, modernizer
   fine-tuning on S2, joint training) hasn't started.
