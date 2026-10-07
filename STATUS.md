# SETU — status, findings and next steps

Written 2026-10-07 on the RTX 3060 machine. Read alongside `CLAUDE.md`
(project rules) and `HANDOFF.md` (machine setup, now complete). Like
`HANDOFF.md` this is session state, not a rules document — delete it when
it goes stale rather than maintaining it forever.

---

## The one-paragraph version

**The recogniser works. The modernizer does not.** The CRNN reads
synthetic old-Kannada lines at **1.53% CER** on the frozen test split. The
modernizer, in both the conventional (B3) and soft-bridge (B4) variants,
scores **6–7.5 chrF++ BELOW simply copying its input unchanged**. Three
separate hypotheses for why were tested and refuted (data starvation,
degenerate decoding, missing copy mechanism). The root cause appears to be
a **task mismatch that predates this session**: the project title promises
*halekannada → hosakannada*, i.e. orthographic modernization, but the
training targets are KannadaLit4NLP's **ಭಾವಾರ್ಥ (scholarly commentary)**,
which shares only **6.8%** of its words with the source. The recommended
next step is a **forward rule-based orthographic modernizer**, which would
match the title, cannot hallucinate, and demos well.

---

## 1. What works (reportable results)

| Result | Number | Run folder / source |
|---|---|---|
| Recogniser CER, frozen test split (all 1,358 lines) | **1.53%** | `20261007T151916Z_eval_b0_b3_b4` |
| Recogniser CER, S1 validation split | **0.39%** | `20261006T120718Z_crnn_train_s1` |
| Recogniser CER over all of S2 (3,500 lines) | 1.57% (median **0.00%**) | `20261007T022632Z_cache_s2_recogniser` |
| Palmira line-segmentation success rate | **96.9%** over all 738 real HKHPL pages | `data/raw_corpus/palmira_survey_results.jsonl` |
| Fake-vs-real classifier | **68.8%** test accuracy | `20261006T153251Z_fake_vs_real_classifier` |
| CTC confidence distribution (real checkpoint) | mean top-1 **0.9606**; **11.69%** of frames below 0.9 | `20261006T152242Z_measure_confidence_distribution` |
| Joint unlocked training | stable, recogniser never needed locking | `20261007T134952Z_joint_unlocked_train` |
| POS tagging | 104 hand-applied tags | `src/setu/pos/` (earlier session) |

### CRNN training, 12 epochs total

Run in three batches via `--resume-from` / `--start-epoch`:

| Epoch | train_loss | val_CER | | Epoch | train_loss | val_CER |
|---|---|---|---|---|---|---|
| 1 | 0.7476 | 4.60% | | 7 | 0.0963 | 0.40% |
| 2 | 0.1605 | 3.61% | | 8 | 0.1008 | 1.24% |
| 3 | 0.1360 | 0.72% | | 9 | 0.0970 | 2.55% |
| 4 | 0.1166 | 0.71% | | **10** | **0.0949** | **0.39% ← best** |
| 5 | 0.1105 | 2.39% | | 11 | 0.1201 | 0.62% |
| 6 | 0.1054 | 0.45% | | 12 | 0.0962 | 0.81% |

**Best checkpoint: `runs/20261006T120718Z_crnn_train_s1/best_model.pt`**
(epoch 10). ~45 min/epoch, peak ~5–6 GB. Zero batches skipped throughout.

### Data built on this machine

| Set | Size | Location |
|---|---|---|
| S1 (recogniser training) | 23,346 lines, 0 skipped | `data/s1/` |
| S2 (original render) | 3,500 lines (1,358 frozen test + 2,142 train) | `data/s2/` |
| S2 extra (in-band top-up) | 2,075 lines, **all train-side** | `data/s2_extra/` |
| S3 (modernizer pretrain corpus) | 135,724 pairs | `data/raw_corpus/s3_corpus_for_generate.txt` |
| Recogniser output caches | 0.41 GB + 0.22 GB, full per-frame distributions | the two `*_cache_s2_recogniser` runs |

All rendered images and caches are **gitignored** — they exist only on
this machine. `data/raw_corpus/*.py` and `data/splits/*` are committed, so
everything is reproducible from a clone plus the Mendeley download.

---

## 2. What does not work

### The reported comparison (465 in-band frozen-test lines)

| System | chrF++ | BLEU | mean chars |
|---|---|---|---|
| **B0_recognised** (copy recogniser output, unmodernized) | **21.01** | 0.39 | 171 |
| **B0_oracle** (copy ground-truth old text) | **21.31** | 0.39 | 171 |
| B3_argmax | 14.97 | 0.05 | 192 |
| B4_soft_bridge (T=1.5) | 14.68 | 0.04 | 205 |
| B3 after 3.9× more data | 13.67 | 0.04 | 208 |
| B3 after 3.9× data + no-repeat-3gram | 13.44 | 0.05 | 148 |

Reference mean length 194 chars. `20261007T141237Z_eval_b0_b3_b4` (first),
`20261007T151916Z` (more data), `20261007T152849Z` (+ repetition control).

**B4 − B3 = −0.28 chrF++.** No soft-bridge advantage at any point measured.

BLEU is ≈0 for everything. `CLAUDE.md` already predicted this for short
lines; **read chrF++ only**.

### Modernizer fine-tunes

| Run | Train lines | best val_loss | val_char_acc |
|---|---|---|---|
| `20261007T130517Z_..._b3` | 651 | 1.7187 | 0.4914 |
| `20261007T131103Z_..._b4_T1` | 651 | 1.7257 | 0.4908 |
| `20261007T131859Z_..._b4_T1.5` | 651 | **1.7239** (best B4) | 0.4919 |
| `20261007T132658Z_..._b4_T2` | 651 | 1.7279 | 0.4911 |
| `20261007T144601Z_..._b3` | **2,516** | **1.5276** | **0.5463** |

---

## 3. Findings — the important part

### 3.1 The title and the training targets are different tasks

This is the central finding. Measured over the 465 in-band frozen-test
lines:

- reference words appearing **verbatim** in the old text: **6.8%**
- reference words sharing a 4-char stem with an old word: **24.8%**
- reference 4-char n-grams present in the old text: **13.1%**

So `CLAUDE.md`'s premise that *"old and modern Kannada share most
vocabulary"* — the stated basis for B0 being a meaningful floor — **is not
supported by this corpus**. KannadaLit4NLP's `interpretationN` fields are
explanatory commentary (they open with ಅರ್ಥ / ಭಾವಾರ್ಥ / ಸರಳಾನುವಾದ, discuss
neighbouring verses, address the reader as ವಾಚಕರೆ), not spelling
modernizations. A 20M-parameter from-scratch transformer cannot learn
semantic exegesis of medieval Kannada from a few thousand examples.

B0 wins not because the task is near-copy, but because at the character
n-gram level chrF++ measures, the old text stays closer to the modern
reference than the model's invented text does.

### 3.2 `val_loss` is anti-correlated with chrF++ here — and it chose B4's temperature

With 3.9× more data: `val_loss` **improved** 1.7187 → 1.5276 and
`val_char_acc` **improved** 0.4914 → 0.5463, while **chrF++ got worse**
14.97 → 13.67.

Every early-stopping decision this session used `val_loss`, **and so did
the B4 temperature selection that picked T=1.5**. That choice therefore
rests on a metric now shown to move the wrong way. Any future B3-vs-B4
claim needs selection redone on chrF++ (which costs generation at each
eval — that is why it was not done this way originally).

### 3.3 Three refuted hypotheses

| Hypothesis | Test | Outcome |
|---|---|---|
| Modernizer is data-starved | 721 → 2,796 train lines | **Refuted** — chrF++ fell |
| Degenerate greedy decoding | no-repeat 3-gram blocking | **Refuted** — length 208→148, chrF++ 13.67→13.44 |
| Copy-with-edits task, no copy mechanism | measured source/target overlap | **Refuted** — only 6.8% verbatim overlap |

Intra-line word repetition did rise with more data (0.158 → 0.253 vs
references' 0.038), so the repetition is real — but suppressing it did not
help, meaning repetition was a *symptom* of having nothing correct to say.

### 3.4 The recogniser is not the bottleneck

`B0_oracle` (21.31) beats `B0_recognised` (21.01) by only **0.30 chrF++**.
Perfect OCR would barely move the end-to-end number. Do not spend further
effort on recognition for the sake of the end-to-end score.

### 3.5 `pool_size >= 2` is mandatory for the soft bridge, not optional

`CLAUDE.md` treats pooling as optional. It is not: one in-band
frozen-test line is **2,151 frames** at `pool_size=1`, over the
modernizer's `MAX_LEN=2048`. `pool_size=2` caps it at 1,076. A bonus is
that at pool=2 B4's encoder length (~232) nearly matches B3's (~196), so
the branches stay well matched.

`pos_embed` is a learned `nn.Embedding(2048, 256)`, so raising `MAX_LEN`
would add untrained rows and invalidate the S3 checkpoint — not a clean
escape hatch.

### 3.6 Joint unlocked training: stable, no gain

300 steps, 104 s. No NaN, no instability, so the recogniser never needed
locking (which `CLAUDE.md` says weakens the claim). But `val_loss`
1.6831 → 1.7105 and `val_char_acc` 0.5000 → 0.4920, i.e. **no end-to-end
gain**.

Recogniser CER read 0.0135 → 0.0104, which looks like a 23% improvement.
**Do not report it as one.** Its trajectory across the six evaluations was
0.0122 / 0.0147 / 0.0138 / 0.0154 / 0.0133 / 0.0104 — noisier than the
start-to-end difference, on 68 validation lines. The defensible claim is
"CER stayed stable under unlocking", which is still worth stating because
nothing in the loss asks the CRNN to keep reading correctly.

Also note this run resumed a modernizer that had **already early-stopped**,
so some of the degradation is plain resumed-optimization overfitting, not
attributable to joint training.

### 3.7 The in-band evaluation subset

Roadmap §2.1 step 2 specifies keeping only interpretations running 0.7–1.5×
the original's length. **That filter was never implemented** in
`build_corpora.py`. Measured over the 3,500 rendered S2 lines the
modern/old ratio has median **1.77×**, p90 **11.07×**, max **32.6×**, with
targets up to 5,284 characters.

Decision taken (yours, 2026-10-07): **leave the frozen split untouched and
report end-to-end metrics on the in-band subset only, disclosed.**
`data/splits/s2_test_verse_ids.txt` is unmodified (verified: empty
`git diff`); `data/splits/s2_inband_verse_ids.txt` is an additional list,
with a provenance record appended to `data/splits/manifest.jsonl`.

Of the rendered lines, **465/1,358 frozen test** and **721/2,142 train**
are in band. The filter also removed a hard blocker: in-band targets top
out at 796 chars, so the 186 lines exceeding `MAX_LEN=2048` drop to zero.

### 3.8 Infrastructure gotchas found this session

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required.** S1's
  wildly variable image sizes fragment PyTorch's default allocator so badly
  that `memory.used` climbed 5 GB → 12 GB/12 GB *during* an epoch with no
  single batch growing. Now set unconditionally at the top of
  `recogniser/train.py`, `bridge/cache_s2.py`, `bridge/joint_train.py`,
  `modernizer/finetune_s2.py`, `eval/run_comparison.py`.
- **Estimate from `--time-steps`, never arithmetic.** Arithmetic was wrong
  in both directions: it predicted ~11.7 GB peak for joint training that
  actually used 3.78 GB, and under-called a fine-tune at ~10 min that took
  ~47. Every training/eval script now has `--time-steps N`, which measures
  per-step cost and peak memory, projects the run, and writes nothing.
- **`textures.py` now sorts its real-image listing.** It used `rglob` order,
  i.e. filesystem order, so the same seed picked different real photos on
  different machines — silently defeating rule 6 for S1/S2 rendering and the
  fake-vs-real check.
- **Three `data/raw_corpus/build_*.py` scripts had the old laptop's absolute
  path hardcoded.** Fixed to derive from `__file__`.
- **WSL2 networking on this machine corrupts large downloads.** See
  `HANDOFF.md` for the full write-up; the working fix is to retry the
  install in a loop (torch took 8 attempts).

---

## 4. Next steps — pick up here

### 4.1 RECOMMENDED: forward rule-based orthographic modernizer

This is the highest-value remaining work. Rationale:

- It matches what the **title** claims (*halekannada → hosakannada* =
  orthographic modernization), unlike the commentary targets
- `src/setu/modernizer/reverse_spelling.py` already implements three
  **documented** orthographic rules in the modern→old direction
- The **forward** direction is strictly better-posed: the historical
  mergers were many-to-one (ಱ and ರ both collapsed into ರ), so old→modern
  is **deterministic**, whereas modern→old needs guesswork about which
  words historically carried ಱ
- Output is copy-plus-targeted-edits, so it **cannot hallucinate** — no
  word salad, content always faithful
- It demos extremely well: side-by-side old/modern with changed characters
  highlighted
- It should score at or above B0, since it *is* B0 plus edits toward modern
  orthography

The three rules to invert:

1. homorganic nasal cluster → anusvara (ಙ್+ಕ → ಂ+ಕ, ಞ್+ಚ → ಂಚ, ಣ್+ಟ → ಂಟ,
   ನ್+ತ → ಂತ, ಮ್+ಪ → ಂಪ) — deterministic
2. **ಱ → ರ** — deterministic
3. **ೞ → ಳ** — deterministic

**What is needed from you:** validate these three, and decide which
*additional* old→modern rules are worth adding (archaic inflections such as
ಇರ್ದ → ಇದ್ದ, ಕಾಣಾ, locative -ಅಲ್ಲಿ forms, etc.). Claude deliberately did
not invent Kannada linguistic rules beyond inverting the documented three —
that needs a Kannada reader's judgement.

### 4.2 Demo strategy

**Demoing on training data will not work.** Measured on 300 training
examples: chrF++ mean **14.70** (vs 13.44 held-out — only 1.3 better), and
only **8 of 300** beat the B0 floor. The best single training example
(24.76) is still visibly incoherent Kannada. The model barely memorised its
own training set, so cherry-picking does not produce demo-quality output
and any Kannada-reading examiner will see it.

What *will* demo well:

1. **Recognition** — real page → Palmira lines → CRNN reading old Kannada
   at 1.5% CER. This genuinely works and is the project's strong result.
2. **Rule-based orthographic modernization** (§4.1) — visibly correct,
   faithful, explainable.
3. **Confidence flagging** — which lines the system would send to a human.

Note `CLAUDE.md` rule 8 already sanctions a curated demo:
`demo_pages/` never appears in any reported number, and "curation is for
the demo only and is disclosed explicitly in the report." Keep that
disclosure, and have the answer to *"is that a held-out example?"* ready —
being unable to answer it is what damages you, not the curation itself.

### 4.3 §6.4 real-imagery measurements — BLOCKED

Three label-free measurements (no transcriptions of real HKHPL lines exist,
so CER on real images is impossible):

1. line-segmentation success rate — **done**, 96.9% over 738 pages
2. recogniser top-1 confidence, **real vs synthetic** — not run
3. flagging rate, **real vs synthetic** — not run

#2 is the valuable one: the synthetic half already exists (mean 0.9606,
11.69% of frames below 0.9), so the real-page comparison would quantify the
synthetic→real domain gap with zero ground truth.

**Blocker:** both need real **line crops**, which need Palmira.

- the committed survey saved only per-page **counts** (`n_instances`,
  `class_counts`) — **no polygons or boxes**, so it cannot be reused
- HKHPL ships **images only** — no annotation files anywhere in `data/real/`
- Palmira is **not installed here**; `HANDOFF.md` warns it needs conda +
  Detectron2 + a custom-compiled DefGrid CUDA extension, with
  GPU-arch-specific toolkit pinning

Options: (a) set up Palmira here, hours and may fail; (b) **check the old
laptop for saved line crops or Palmira outputs — cheapest path by far**;
(c) report §6.4 partially, noting the segmentation number already exceeds
what the roadmap asked for (738 pages vs "~50").

Do **not** substitute whole pages or arbitrary strips for line crops — the
CRNN was trained only on single lines, so the confidence numbers would
measure nothing while looking like a result.

### 4.4 Not done, if the B3-vs-B4 comparison is to be revived

- **B4 was never re-fine-tuned on the enlarged data.** Only B3 was
  (`20261007T144601Z`). For a matched comparison B4 needs the same
  treatment (~26 min).
- **Temperature needs re-selecting on chrF++**, not `val_loss` (§3.2).
- Honestly: given B4−B3 has been ≤0.3 chrF++ at every point measured, and
  both arms sit 6–7 points below the copy floor, this is probably not where
  the remaining time should go.

---

## 5. Commands to resume

All training/eval runs from WSL (CUDA); rendering from the Windows venv
(needs the Windows font path).

```bash
# --- inside WSL ---
cd /mnt/d/major_project/updated_maj_proj/Palm_leaf_halekannada_to_hosakannada_translation
source ~/setu-venv/bin/activate
export PYTHONPATH=src

CACHE1=runs/20261007T022632Z_cache_s2_recogniser
CACHE2=runs/20261007T143817Z_cache_s2_recogniser
CRNN=runs/20261006T120718Z_crnn_train_s1/best_model.pt

# re-score the current best B3 (no retraining)
python -m setu.eval.run_comparison --cache-dir $CACHE1 \
  --b3 runs/20261007T144601Z_modernizer_finetune_s2_b3/best_model.pt

# B4 on the enlarged data, to match B3 (~26 min; measure first)
python -m setu.modernizer.finetune_s2 --cache-dir $CACHE1 $CACHE2 \
  --branch b4 --temperature 1.5 --time-steps 15      # measure
python -m setu.modernizer.finetune_s2 --cache-dir $CACHE1 $CACHE2 \
  --branch b4 --temperature 1.5                      # run

# full comparison once both branches match
python -m setu.eval.run_comparison --cache-dir $CACHE1 \
  --b3 <b3_ckpt> --b4 <b4_ckpt> --temperature 1.5
```

```powershell
# --- Windows venv, for rendering only ---
$env:PYTHONPATH = "src"
.venv\Scripts\python -m setu.render.generate --corpus <corpus> `
  --out-dir <dir> --manifest <dir>/manifest.jsonl --paired --seed 0
```

**Every script takes `--time-steps N`** — use it before committing to any
long run.

---

## 6. Housekeeping

Three run folders are **incomplete** (no `results.json`, so per rule 4 they
are not results) — leftovers from interrupted runs. Safe to delete:

- `runs/20261004T111145Z_crnn_train_s1/` (killed: allocator thrashing at 6M batch budget)
- `runs/20261004T134542Z_crnn_train_s1/` (killed: same, before the expandable_segments fix)
- `runs/20261007T031640Z_modernizer_finetune_s2_b3/` (killed mid-run; **note there are two `_b3` folders and only `20261007T130517Z` is the real 651-line one**)

Everything else is committed and pushed. Rendered images, caches and
checkpoints are gitignored and exist **only on this machine** — if it is
rebuilt, re-render from the committed `data/raw_corpus/*.py` scripts.
