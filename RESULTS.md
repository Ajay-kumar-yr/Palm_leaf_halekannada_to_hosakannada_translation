# SETU — measured results

Every number this project can defend, with the run folder it came from.
Written 2026-10-08. This is the spine of the report's results chapter;
`GOLD_FINDINGS.md` holds the long-form discussion of the
hand-transcription findings, `DEMO_PLAN.md` the plan and the decisions.

Nothing here is quoted from memory — each row names a run folder, and
`CLAUDE.md` rule 4 applies: a result without one does not exist.

---

## 1. Summary of every measurement

| # | Measurement | Value | Run folder |
|---|---|---|---|
| 1 | Recogniser CER, frozen synthetic test (1,358 lines) | **1.53%** | `20261007T151916Z_eval_b0_b3_b4` |
| 2 | Recogniser CER, S1 validation | 0.39% | `20261006T120718Z_crnn_train_s1` |
| 3 | **Soft bridge, frame level** — wrong top-1 frames where the correct symbol survived in the top-5 | **80.6%** (2,907 / 3,607) | `20261008T033215Z_recovery_examples` |
| 4 | Weight the bridge puts on that correct symbol / argmax puts | 0.200 / **0.000** | same |
| 5 | Palmira line segmentation | 96.9% of 738 photos; **97.2%** of 568 distinct pages | `data/raw_corpus/palmira_survey_results.jsonl` |
| 6 | Domain gap, mean top-1 confidence: synthetic → real | **0.9613 → 0.6265** | `20261008T061052Z_real_vs_synthetic_confidence` |
| 7 | Lines flagged by confidence: synthetic → real | 1.8% → **100%** | same |
| 8 | Domain gap decomposition: texture / letter shape | **6% / 94%** | `20261008T141042Z_real_vs_synthetic_confidence` |
| 9 | CRNN on real lines, true CER vs human transcription | **0.687** | `20261008T130753Z_gold_real_eval` |
| 10 | Vision-model labels vs human transcription | **0.370** (0.389 before the ಶ/ಕ correction) | `20261008T150323Z_gold_fix_sha_ka` |
| 11 | Archaic ಱ in 32 real lines: human / machine | **6 (in 4 lines) / 0** | `20261008T130753Z_gold_real_eval` |
| 12 | Data scaling: 13 / 52 / 105 labelled lines | 0.756 / 0.719 / **0.700** CER | `20261008T1026–1048Z_crnn_finetune_real` |
| 13 | **Real-crop recovery** — consensus-wrong characters where some sample held the right one | **34.2%** (25 / 73) | `20261008T152006Z_real_recovery` |
| 14 | Build B branch divergence | 5 of 6 lines | `20261008T151237Z_demo_build_real` |
| 15 | Fake-vs-real classifier | 68.8% | `20261006T153251Z_fake_vs_real_classifier` |

---

## 2. The soft bridge — the project's contribution

### 2.1 On synthetic lines, where the recogniser works (rows 3–4)

The frozen test split, 383,427 aligned non-blank frames, true ground
truth from the corpus, CTC forced alignment verified by checking that
the collapse of every alignment reproduces its target.

- The recogniser's top-1 was **wrong at 3,607 frames**.
- At **2,907 of them (80.6%)** the correct symbol was still inside the
  bridge's top-5.
- The bridge carries that symbol with mean weight **0.200**; argmax
  carries it with **0.000** — by construction, not by measurement,
  because argmax keeps one symbol and discards the rest.

**The claim this supports, exactly:** *the information survives the
interface.* It does **not** show the bridge produced a correct final
answer, and must not be allowed to drift into that under questioning.
It declined to throw the answer away; that is the whole claim.

Six worked examples are published for the demo, chosen for being
*readable* — near-perfect lines (0.4% CER) with exactly one wrong
character where the bridge held the right one. The clearest:

> argmax committed to **ಮುಗ್ಹೆ** at 0.38 — not a word.
> The bridge also held **ಧ** at 0.35, giving **ಮುಗ್ಧೆ** — the correct
> word, and the ground truth.

295 frozen-test lines qualify as recovery cases, so these are instances
of a pattern, not anecdotes.

### 2.2 On real manuscript crops (row 13)

Our CRNN cannot read real crops at all (row 9), so the same *principle*
was tested with a different uncertainty source: read each crop five
times at temperature 0.8 and treat disagreement between samples as the
uncertainty. B3 takes the consensus string alone; B4 takes it plus the
alternatives and their frequencies.

Scored against the human transcriptions:

| | |
|---|---|
| characters the consensus read wrong | 73 |
| …where at least one sample held the right one | **25 (34.2%)** |
| synthetic CTC top-5 equivalent | 80.6% |

**The principle transfers, but weakly.** A third of the consensus's
errors are recoverable from what B4 carries and B3 discards — against
zero for argmax by construction — but less than half the rate the CTC
bridge achieves where the recogniser is competent.

**Why, and this is worth reporting:** a confident vision model tends to
be **wrong the same way five times**. Its sampling errors are
correlated. A CTC posterior is per-frame and genuinely brackets the
truth; an LLM ensemble mostly does not. That is an argument *for* the
project's own design choice, arrived at by measurement rather than
assertion.

**What this is not:** it is an ensemble over a borrowed recogniser, not
the CTC soft bridge. Same claim, different mechanism, and the report
must say so.

---

## 3. Why the recogniser fails on real manuscripts

### 3.1 The gap is letter shape, not preprocessing (row 8)

Binarizing with the Sajjan U-Net removes the leaf texture and leaves
only the difference in letterforms. Over all 712 sample crops, original
checkpoint, no retraining:

| input | mean top-1 | frames below 0.9 |
|---|---|---|
| synthetic (S1 val) | 0.9613 | 11.5% |
| real, binarized | 0.6453 | 76.0% |
| real, grayscale | 0.6265 | 79.8% |

Of the 0.335 total gap: **leaf texture 0.019 (6%)**, **letter shape
0.316 (94%)**.

Binarization is free, reliably helps, and should be used — and closes
one sixteenth of the gap. The rest is that the recogniser learned
letterforms from a rendered font and real manuscripts are handwritten.
**This one line explains every negative result below.**

### 3.2 More labelled data will not fix it (row 12)

Same held-out 32 lines, same recipe, only the training-set size varying:

| labelled training lines | held-out CER |
|---|---|
| 13 | 0.7556 |
| 52 | 0.7189 |
| 105 | 0.7002 |

Log-linear fit: `CER = 0.824 − 0.0184 × log2(lines)`, i.e. **~0.018 per
doubling**. Extrapolated, CER 0.50 needs ~200,000 lines and CER 0.30
needs ~380 million. The most that could be labelled in the remaining
time is ~450 lines, predicting 0.66.

Two honest caveats. The curve was measured with labels later shown to
be ~37% wrong, so it understates what clean labels might achieve. And
three points cannot prove nothing lies further along. But the model's
error (0.687) sits far above the label-noise floor (0.370), so the model
is the binding constraint, and clean labels at that scale would need
human transcription of thousands of lines — the very bottleneck that
made machine labelling necessary.

### 3.3 Same-page training does not help either

Training on 52 lines from the very pages held out from gave **0.711**,
identical to training on different pages. Training loss reached 0.03 —
the model memorises and does not generalise. S1 replay held synthetic
CER at 0.0033 throughout, so nothing was broken.

### 3.4 Synthetic handwriting cannot be stitched from real glyphs

A recogniser overfit on the 32 transcribed lines force-aligns all of
them, yielding 1,787 glyphs over 49 of 66 symbols. The glyphs carry
real symbol identity (1-NN 14.5% against 3.5% chance). Binarizing first
removes the brightness seams at the joins. But the stitched lines still
show inconsistent baselines, irregular spacing and neighbour fragments:
in connected cursive the ink genuinely overlaps, and no vertical cut
separates it. Four cutting strategies hit the same ceiling.

Next steps for anyone resuming it: baseline alignment by ink centroid,
and seam blending. Details in `GOLD_FINDINGS.md` §4–4a.

---

## 4. Label quality (rows 10–11)

A Kannada reader transcribed the 32 held-out lines **blind** — the page
never showed the machine's reading.

- **The machine labels are ~37% wrong** (0.370 CER). Only 13 of 32 fall
  inside the 0.35 threshold `build_label_set` uses to accept a label.
- **The real leaves carry archaic orthography and the vision model
  erases it.** The human read ಱ **6 times across 4 of 32 lines**; the
  machine labels for those same lines contain **zero**. This answers the
  question `STATUS.md` §3.9 left open: that finding is a property of the
  *corpus* (2001 critical editions), not of the collection.
  `CLAUDE.md`'s original instinct about ಱ/ೞ was right. A recogniser
  cannot learn ಱ from a teacher that never emits it.
- The transcriber then identified a systematic ಶ→ಕ confusion in their
  own typing (ಶ appeared 49 times against ಕ's 23, backwards for
  Kannada). Correcting it moves the machine-label CER from 0.389 to
  **0.370**. Three gold variants are kept; the quoted figure is the
  machine-independent one.

**Limits of this gold set:** one reader, one pass, no double annotation,
32 lines. It is ground truth for this project's purposes and carries its
own error rate — as the ಶ/ಕ correction itself demonstrates.

---

## 5. What is claimed, and what is not

| Claimed | Not claimed |
|---|---|
| The recogniser reads *synthetic* old-Kannada lines at 1.53% CER | That it reads real manuscripts (it does not: 0.687) |
| Information argmax discards survives the bridge — 80.6% of wrong frames | That the bridge produces a correct final answer |
| The principle transfers to real crops at 34.2% | That this is the CTC soft bridge (it is an ensemble over a borrowed reader) |
| 94% of the domain gap is letter shape | That preprocessing can close it |
| Real manuscripts contain ಱ | That ೞ does (absent from these 32 lines) |
| Machine labels are ~37% wrong | That the gold is error-free |

The modernizer shown in any demo is an **LLM**, not this project's own
model, which scores *below copying its input* and is documented as a
measured negative result (`STATUS.md` §3.1). Demo pages are curated and
appear in no reported number (`CLAUDE.md` rule 8); the §6.4 figures come
from a 50-page sample disjoint from them.

---

## 6. Measurement bugs found and fixed

Recorded because each would have produced a confident, wrong number, and
because the same traps are easy to fall into again.

| Bug | Effect if unnoticed |
|---|---|
| Recovery scored against the display-capped list (10 contested chars/line) | Reported **9.6%** where the true figure is **34.2%** — under-reporting our own result 3.5× |
| Demo-line selection compared characters position-by-position | One insertion shifts everything after it: a line with 3 real errors showed 98 "errors" and ranked as a spectacular recovery |
| Ranking recovery lines by raw count | Selected the *most broken* lines (0.58 CER), illegible either way |
| `is not` used to filter identical sample strings | Python interns them, so a unanimously-read line reported as **maximally uncertain** |
| Vocabulary flag kept in a module global | DataLoader workers re-import, so labels were encoded with the wrong vocabulary — silent until a numeral raised |
| Quota detection read only the first 200 bytes of a 429 | The quota id sits further down, so daily exhaustion was misread as a rate limit |
| Retry loop did not rotate keys on 503 | Hammered one unlucky key five times while others were free |
| `nan < best` never true with no val set | An overfit run saved no checkpoint at all |

---

## 7. Operational notes

- **Free-tier quota is 20 requests per key per day, _per model_**
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, ~9h retry). Five
  keys give 100 calls/day on one model. Because the cap is per model,
  reading and modernizing should use different models so they do not
  compete.
- Reading quality by model, against the human transcriptions:
  `gemini-3.5-flash` **0.370**, `gemini-3.5-flash-lite` 0.428.
  `3.7`/`3.8-flash` returned 503 under real transcription load.
- Binarized input reads slightly better than grayscale for the vision
  model too (0.428 vs 0.447 on flash-lite, 10 lines).
