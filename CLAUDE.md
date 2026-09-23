# SETU — project rules and plan (Roadmap v4)

Reading old Kannada palm-leaf manuscripts and rewriting them in modern Kannada.

This file supersedes any earlier CLAUDE.md written from Roadmap v2. The
biggest change from v2: **Track A (UniLipi fine-tuning) is dropped.**
UniLipi weights are not arriving in time, so Track B (our own small CRNN)
is the only recogniser path — there is no fallback branching to reason
about anymore. Evaluation is also now synthetic-first by design, not by
necessity: the reported comparison runs on a frozen synthetic test set,
with real HKHPL imagery used for a separate, label-free measurement (Part
6 below), not for the headline numbers.

**Scope:** 4 weeks, fixed deadline, team of 4, ~5–8 person-hours/week total
(~80–130 person-hours for the whole project). Roles: **A** recognition,
**B** data, **C** modernization, **D** infrastructure/demo.

Do not start building anything from this file alone — it is context for
future requests in this project, not a standing instruction to act.

## The one novel contribution

Every existing system for this problem recognises text, commits to a single
character string (argmax), and only then modernizes it. Once the recogniser
picks a character it was only 55% sure about, the uncertainty is gone and
the modernizer never sees it.

**Our system never collapses to a single string before modernization.** The
recogniser's full confidence distribution is carried forward — blended per
CTC frame, temperature-adjusted, top-5 symbols (blank included) kept and
renormalised — and the modernizer consumes that blended signal directly.
This is the **soft bridge**, the only genuinely new part of the project.
Everything else (line finding, the recognition architecture, the
manuscript images) is existing work, adapted, and the report says so
plainly.

## What's cut and staying cut

Visual cross-attention, era/dynasty conditioning, fine-tuning Palmira, the
four-baseline comparison grid, ablation experiments, calibration statistics
as a chapter (ECE, reliability diagrams, risk-coverage curves), double
annotation / inter-annotator agreement, the full patent-differentiation
chapter (one paragraph instead, for the viva).

## What's locked in (from the submitted synopsis)

POS tagging and modernization. Both handled cheaply: POS tagging is an
existing tagger or one fixed LLM prompt over the modernizer's output,
hand-checked on ~100 tags, done in an afternoon — not a trained model.

## Non-negotiable engineering rules

1. NEVER write to a frozen test split (synthetic test IDs, and the gold
   test split if one exists). Freezing synthetic test data means committing
   the held-out verse ID list on day one, before rendering — trivial
   because the data is generated, not collected.
2. If gold lines ever arrive, split assignment is by hash of the line ID at
   creation time, recorded in an append-only manifest — never a fixed
   checksum taken once while the set is still growing.
3. Before any training run longer than 10 minutes, prove the model can
   memorise 8 examples to near-zero error first. If it can't, something is
   broken; do not start the real run.
4. Every run writes `runs/<timestamp>/` with its config, results, and git
   commit hash. A result without a run folder does not exist.
5. Metrics are computed ONLY by `src/setu/eval/metrics.py`. Never inline,
   never invented.
6. Set random seeds in every entry point.
7. Never quietly change a setting to make a run work. Fail loudly.
8. `demo_pages/` is for the live demo only. It never appears in any
   reported number.
9. The modern-Kannada side of the S2 pairs (used for the headline
   end-to-end numbers) must come from real scholarly interpretations
   (KannadaLit4NLP), never from our own rule engine — scoring against your
   own rules' output measures nothing.
10. Both B3's and B4's modernizers must be fine-tuned on the recogniser's
    actual noisy output (cached top-1 strings / frame confidences from one
    run over S2), not on clean text — otherwise the comparison is unfair to
    whichever one meets noise for the first time at test time.

## Facts about this project

- **Track B only.** No UniLipi branching; build the CRNN ourselves.
- Recogniser output is romanised (WX), ~50–60 symbols, not 600–900 Kannada
  syllables — with only 25,000 training lines, syllable-level output starves
  rare/archaic characters (notably ಱ and ೞ, the very letters that mark a
  text as old Kannada) of training examples while training metrics still
  look healthy. WX also converts back to Kannada script exactly.
- CRNN target size: **3–5 million parameters** — 5 conv blocks (32→256
  channels, pool in height only after block 3), one BiLSTM (256 hidden
  units), output over ~60 symbols + blank. Input height 64, variable width.
- Modernizer: a **small transformer trained from scratch** (~6 encoder + 6
  decoder layers, hidden size 256, 4 heads, ~20M params) — NOT ByT5-small.
  ByT5's Kannada-script pretraining doesn't transfer to WX-romanised input,
  and it costs ~10x the training time (4–8h vs 30–60min) for that reason.
  ByT5-small remains only a stretch option if there's a spare overnight
  slot and the from-scratch model's chrF++ disappoints.
- CTC gives one confidence distribution per image **frame** (a vertical
  strip), not per letter — most frames are blank/repeats. There is no
  clean "top-5 per letter position" without collapsing frames first, which
  is itself the hard decision the bridge exists to avoid. **The bridge
  operates per frame**, not per letter.
- Soft bridge per-frame procedure: apply temperature dial → softmax → keep
  top 5 symbols including blank → renormalise those 5 to sum to 1 → blend
  embeddings by confidence → optionally pool every 2–4 frames by averaging
  to shorten the sequence → feed to the modernizer's encoder. Frames where
  blank confidence exceeds ~0.95 may be dropped (defensible as removing
  padding, not choosing between competing letters — distinct from
  collapsing, and worth being able to explain that distinction under
  questioning).
- Temperature is tuned on validation data, not fixed at 1 — CTC models are
  typically overconfident (98–99% on the top choice), which would make the
  blend numerically identical to argmax and erase the B3-vs-B4 difference.
  Measure the fraction of frames with top-1 confidence below 0.9 in
  **Week 3**, not Week 4; if that fraction is tiny, raise the temperature.
- Joint training default: recogniser unlocked, ~1/10th normal learning
  rate, a few hundred steps on S2. Only lock the recogniser (weakening the
  claim) if training is genuinely unstable (NaN loss / falling accuracy),
  and only after first trying a lower learning rate and a shorter phase.
- Palmira is used exactly as downloaded, no fine-tuning, only for real
  HKHPL pages (synthetic images are already single lines). It receives the
  **original photo**, never the black-and-white U-Net output — Palmira was
  trained on colour photos and black-and-white is out-of-distribution for
  it. Order: photo → Palmira → cut lines → black-and-white conversion only
  if the recogniser needs it.
- Survey every HKHPL page once (Week 1): mark good/acceptable/broken. This
  produces both the demo page selection (10–15 clean pages, kept in
  `demo_pages/`, separate from `test_split/`) and the §6.4 segmentation
  success-rate number. Curation is for the demo only and is disclosed
  explicitly in the report; reported accuracy always comes from the frozen
  synthetic test set.

## Data plan

| Set | Contents | Size | Used for |
|---|---|---|---|
| S1 | rendered old-Kannada line image + old text | 20–30k | Training the recogniser |
| S2 | rendered image + old text + **modern text** (from KannadaLit4NLP) | 2–4k | Joint training + all end-to-end reported numbers |
| S3 | old/modern text pairs, no images, rule-generated | 100–300k | Pre-training the modernizer |
| R | real HKHPL photos, no transcriptions | ~50–500 | Demo + label-free real-imagery measurements (Part 6) |
| G | real image + old text, best effort | 0–60 | One real CER number, if it arrives — never blocking |

S1 damage effects (5, down from 9 — dropped fibre grain, fungal staining,
stylus groove shading, elastic letter distortion): background texture and
holes/damage patches cut from real HKHPL leaves, ink bleed, page warp,
baseline skew. Expensive effects (warp, texture, holes) are baked in at
generation time; cheap ones (rotation, brightness, blur, stretch) are
applied fresh on the GPU each time an image is used, in batches — this is
the single most important dataloader change for training speed on this
machine (Ryzen 5 5500, 6 cores).

Fake-vs-real classifier check (S1 vs real HKHPL crops): target 70–80%
accuracy — not near-100%, which would mean the synthetic images are too
clean.

## Evaluation

Three systems: **B0** (copy input unchanged — the floor, since old and
modern Kannada share most vocabulary), **B3** (recogniser → argmax →
modernizer, the conventional approach), **B4** (recogniser → soft bridge →
modernizer, ours). Optional B1 (Tesseract) only if under an hour. The
comparison that matters is B3 vs B4: same recogniser, same data, same
modernizer training procedure — only the interface differs.

Report: CER on the frozen synthetic test split (+ gold CER if any gold
lines exist); chrF++ (primary) and BLEU (secondary) both shown against B0
(BLEU alone is misleading here — plain copying already scores well given
vocabulary overlap, and the patent's reported 0.81 should be read with that
in mind); a flagged-vs-unflagged accuracy chart; 3–4 worked examples where
the bridge recovered a wrong top-1 guess; the §6.4 real-imagery
measurements; POS accuracy on ~100 hand-checked tags.

## Part 6 — what HKHPL (the real dataset) is actually used for

Since training/eval is synthetic-first, real data still matters in four
ways: (1) appearance source for the damage simulator's textures/holes —
synthetic images are conditioned on HKHPL, not invented; (2) real crops for
the fake-vs-real classifier; (3) the qualitative live demo; (4) **label-free
real-imagery measurements** — run the full pipeline over ~50 real pages and
report line-segmentation success rate (free, from the Week 1 survey), the
recogniser's top-1 confidence distribution on real vs synthetic lines (one
script — a sharp drop quantifies the domain gap with zero transcriptions
needed), and the flagging rate real vs synthetic (reuses bridge machinery).
This is what answers "did you actually use the dataset your project is
about?" without needing ground truth. Budget half a day, Week 4.

## Training time budget (RTX 3060 12GB)

The GPU is not the constraint — one clean pipeline pass is ~4–6 GPU-hours;
budget 12–20 GPU-hours across the month for retraining/tuning. The
constraint is the ~80–130 person-hours. Rough figures: CRNN on S1, 2–4h;
modernizer pre-train on S3, 30–60min; per-branch modernizer fine-tune on
S2, 10–20min each; joint unlocked training, 20–40min. Use WSL2/Linux, not
native Windows (dataloader workers are much slower on Windows); keep S1 in
RAM as a single memory-mapped uint8 array rather than individual files; do
augmentation on the GPU in batches where possible; check `nvidia-smi` early
in any run — under 70% utilization means CPU-bound, fix the dataloader
before touching the model; don't generate data and train at the same time
on this 6-core CPU.

## Week-by-week (high level)

- **Week 1:** repo/CLAUDE.md/environment; damage simulator; build S1 and
  S2 (freeze S2 test verse IDs on day one); CRNN built and passing the
  8-example memorisation test; Palmira original-vs-black-and-white check
  and full-page survey (→ demo set + segmentation number); gold-line
  enquiry emails sent, then move on.
- **Week 2:** train CRNN on S1; fake-vs-real check; build reverse-spelling
  rule engine and S3; pre-train modernizer on S3; measure synthetic CER.
- **Week 3:** measure fraction of frames with top-1 confidence < 0.9 first;
  build the soft bridge (frame-level, blank included, renormalised,
  temperature, pooling); cache recogniser outputs over S2; fine-tune both
  modernizers on their respective noisy inputs; joint unlocked training;
  confidence flagging; POS tagging. **End of Week 3: feature freeze.**
- **Week 4:** run B0/B3/B4 on frozen test split; 3–4 worked recovery
  examples (one hour, don't skip); §6.4 real-imagery measurements (half a
  day); confidence chart; demo on curated pages; write report; rehearse
  demo twice; leave real buffer time.

## Ask before

Changing the loss function, changing the vocabulary, touching frozen test
splits, or adding a dependency.

## Viva notes

Prior-work paragraph and fallback answers for "why is accuracy below the
patent's 93.7%" and "why is the evaluation synthetic" are written out in
full in Roadmap v4 Part 10 — reuse them verbatim rather than improvising,
and volunteer the synthetic-evaluation limitation before being asked.
