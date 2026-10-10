# SETU — 4-day demo plan (decided 2026-10-08)

**Read this first.** It supersedes `STATUS.md` §4.2 wherever they
disagree. Like `STATUS.md` it is session state, not a rules document;
`CLAUDE.md`'s rules still apply to everything below.

**Deadline: 2026-10-12. Feature freeze: end of 2026-10-11.**

**Start at §6c** — the record of 2026-10-10 and what is left. §6b is
2026-10-09. §6a was
the plan for that day; §1-5c are the record of how the approach got
here, and §6 (Day 1-4) is superseded. Every number lives
in `RESULTS.md`; how to run the demo is in `DEMO_RUNBOOK.md`.

---

## 1. What changed today, and why it matters

1. **The recogniser cannot read real manuscripts.** Measured on 712 real
   line crops from 50 pages: mean top-1 confidence **0.63** vs **0.96**
   on synthetic; **100%** of real lines flagged vs **1.8%** synthetic.
   The decoded text is garbage (strings of repeated ಅ). Run
   `20261008T061052Z_real_vs_synthetic_confidence`.
2. **Why:** S1 trained the CRNN on *font-rendered* text pasted onto leaf
   texture. The texture is cut from real HKHPL photos and contains real
   handwriting — but always as *background*. The model learned to read
   the font and ignore handwriting. Not a bug to patch; it needs real
   handwriting with labels.
3. **The trained modernizer cannot be rescued in 4 days.** It scores
   below copying its input (STATUS.md §2–3); its training targets are
   scholarly commentary, not modernizations.
4. **Palmira works on the laptop**, and real line crops now exist
   (`data/raw_corpus/run_palmira_crops.py`; 263 demo + 712 sample).
5. **"738 pages" is 568 distinct pages** + 170 hd re-photos (STATUS.md
   §3.10). Segmentation success over distinct pages is 97.2%. Fix the
   wording wherever "738 pages" appears.

## 2. The goal

A demo that, **on real palm-leaf crops**, does recognition and
modernization, and shows **argmax (B3) vs soft bridge (B4)** side by side
— and looks right at face value.

## 3. The approach

**Teacher–student.** A free vision LLM reads real crops and produces
labels. We fine-tune **our own** CRNN on them. The CRNN then gives real
per-frame probability distributions on real crops, so argmax and the
soft bridge both run natively. The novel contribution stays in our model.

Why not just use another OCR as the recogniser: the bridge needs a
probability distribution over symbols. Tesseract has no handwriting
model for Kannada and fails on palm leaf. A vision LLM reads well but
returns text only — nothing for argmax to throw away, so no comparison.

**Modernizer for the demo: an LLM**, fed two ways —

- **B3:** the argmax string.
- **B4:** the same line with uncertain characters marked, alternatives
  and probabilities attached, e.g. `ದ/ಧ (0.52/0.48)`.

That shows the claim directly: one path commits early, the other passes
the uncertainty forward. The from-scratch modernizer stays in the report
as a measured negative result. **Disclose both substitutions.**

## 4. No human verification — how quality is checked automatically

We have no Kannada reader to grade labels, so every check is automatic:

| Check | How | Used for |
|---|---|---|
| **Cross-model consensus** | Two *independent* model families transcribe each crop. Keep only lines where they agree within ~10–15% character error | Training labels; disagreement rate is reported |
| **Self-consistency** | Same model twice, different sampling | Cheap extra filter |
| **Corpus plausibility** | Fraction of transcribed words found in the KannadaLit4NLP vocabulary (27M chars) | Relative ranking only — some manuscripts are not literary texts |
| **Held-out consensus set** | Highest-agreement lines, kept out of training | Real-line CER, **reported as against machine consensus, not human gold** |
| **Demo referee** | A full-page LLM reading decides which of B3/B4 got a line right | Picking the 3–4 worked examples, disclosed |

**Known blind spot:** LLMs silently normalise ಱ/ೞ to ರ/ಳ (STATUS.md
§3.9). Consensus labels will never contain them. Say so in the report.

## 5. APIs (free or near-free)

Free-tier limits change without notice — Google made large cuts in
December 2025. **Check live limits in AI Studio / the provider console
before relying on any number.**

| Role | Recommended | Why |
|---|---|---|
| **Labeller A** | **Google Gemini API, free tier** (AI Studio key, no card). Flash-Lite for volume, Flash for quality | Strongest Indic-script reading we have seen; the team already used it on a real page |
| **Labeller B** (independent) | **Qwen-VL via OpenRouter `:free` models**, or **Groq**'s free vision endpoint | A *different model family* is the whole point of consensus |
| **Modernizer** | Gemini free tier (text only, cheap) | Same key |
| **Paid fallback** | Gemini Flash-Lite paid | ~600 short line crops is cents, not dollars |

Volume needed: ~600 crops × 2 labellers + ~300 modernizer calls ≈ 1,500
requests over two days — inside free tiers if spread out; switch to
paid Flash-Lite if quotas bite.

**Dependencies:** adding an API client is a dependency change — approved
by the user 2026-10-08. Keep keys in environment variables, never in git.

## 5b. Decision 2026-10-08 (evening): writer-dependent demo

**Measured first.** A fine-tune on 43 labelled real lines
(`20261008T082646Z_crnn_finetune_real`) memorised them — train loss
0.08 — but did not generalise: held-out real CER stayed at **0.71** and
the output was still garbage. Against that, the rule-3 check drove 8
real lines to **CER 0.0000 in 22 s**. The architecture can read this
handwriting; it is short of labelled examples, not capability. S1
replay held synthetic CER at **0.0029** throughout, so adaptation costs
nothing on the synthetic side.

**Decision (user, 2026-10-08): train on the demo pages themselves**,
holding out a quarter of each page's lines, and demo the held-out
lines. Same hand, same ink, same leaf, so far fewer labels are needed —
the only route to a working real-crop demo inside the deadline.

**This makes the demo writer-dependent and page-dependent, and that
must be said plainly**, in the report and unprompted in the viva:

- the recogniser was adapted on lines from *these same pages*;
- the lines shown were held out from training, but the model has seen
  other lines in the same hand, from the same leaf;
- so it demonstrates **adaptation to a known hand**, not generalisation
  to unseen manuscripts — and the honest generalisation number is the
  one above: **0.71 CER on held-out pages**, which belongs in the
  report next to it;
- `CLAUDE.md` rule 8 still holds: demo pages appear in **no reported
  number**. The §6.4 domain-gap figures stay on the 50-page sample set,
  which is disjoint from the demo pages and from training.

**Keys:** 5 distinct keys are in `.env`, from different accounts except
1 and 5 (same account, separate projects). Free-tier quota is metered
per project, so extra keys in one project add nothing. Two lines in
`.env` were both named `GEMINI_API_KEY_4`, so one key was silently
overwritten — if a key seems to be missing, check for a duplicated
name first.

## 5c. The data-scaling measurement — option A is closed (2026-10-08)

Before spending the remaining days labelling, we measured what labelling
would buy. Same held-out 32 lines, same recipe, only the amount of
training data changed:

| labelled training lines | held-out CER | S1 CER |
|---|---|---|
| 13 | 0.7556 | 0.0030 |
| 52 | 0.7189 | 0.0038 |
| 105 | 0.7002 | 0.0031 |

Eight times the data bought **0.055 CER**. Fitting the obvious
log-linear form gives

    CER = 0.824 - 0.0184 x log2(lines)

i.e. **~0.018 CER per doubling**. Extrapolated, a usable recogniser
needs absurd amounts of data:

| target CER | lines needed |
|---|---|
| 0.50 | ~200,000 |
| 0.30 | ~380,000,000 |

Against that, the most we could label in three days on every free key is
~450 lines, predicting **CER 0.66**; even 5,000 lines predicts 0.60.
The S1 column shows nothing is broken — synthetic ability is intact
throughout.

**So option A fails for a reason money cannot fix.** It is not key
quota, not API budget, not the four days. A CRNN pre-trained on
*rendered fonts* is simply the wrong starting point for handwriting, and
closing that gap needs orders of magnitude more labelled handwriting
than this project can obtain. Paid API credit would have bought ~2,000
lines and changed the number by ~0.03.

Extrapolating a three-point curve is not proof that no breakthrough
exists further along — but it is strong evidence of no *imminent* one,
and it is the honest basis for the decision. The curve itself belongs in
the report: it converts "our recogniser cannot read real manuscripts"
from an apology into a quantified finding about transfer from rendered
text to handwriting.

**Decision: stop labelling for recogniser training.** The remaining
routes are:

1. **Human transcriptions of the 32 held-out lines** (the transcription
   page) — not training data, but they turn every real-image CER from
   *agreement with a vision model* into a true error rate, and they
   measure how good the machine labels were.
2. **Build B — uncertainty by resampling** (§7a below): the only route
   to real-corpus output that still demonstrates the project's claim.
3. **Keep the synthetic bridge demo** as the rigorous half: true ground
   truth, frozen test split, genuine recoveries.

## 7a. Build B — carrying uncertainty on real crops

Our soft bridge needs per-frame CTC distributions from our own
recogniser, which on real crops produces nothing readable. So on real
manuscripts the *same principle* is demonstrated with a different
uncertainty source, and the report says exactly that.

Read each real crop **N times at non-zero temperature** with the vision
model. Where the readings agree, it is confident; where they disagree,
the disagreement *is* the uncertainty, with sample frequencies as
probabilities. Then:

- **B3** takes the majority reading — one string, uncertainty discarded.
- **B4** takes the reading plus the alternatives and their frequencies.

Same modernizer, same prompt, same temperature; only the input differs —
exactly the B3/B4 contrast the project is about. ~5 reads + 2
modernizations per line, so ~70 calls for 10 demo lines.

**What must be disclosed:** this is an *ensemble* uncertainty estimate
from a vision LLM, not the CTC soft bridge. The mechanism differs. The
claim — that collapsing to one string early destroys information the
modernizer could have used — is the same one, and the frame-level
measurement on synthetic data (80.6%) remains the quantified result.

## 6. Day by day

> **The Day 1-4 schedule below is superseded.** It was written before
> the measurements of 2026-10-08 closed option A, before build B ran,
> and before the web UI existed. Kept for the record; §6a is the plan.

## 6a. Plan from 2026-10-09 (written 2026-10-08 evening)

### What is already done

| | |
|---|---|
| Synthetic bridge demo | published, true ground truth, 6 worked recoveries |
| Build B on real crops | ~~34.2%~~ — withdrawn under rule 8; **26.0%** on all 16 reportable lines (§6b) |
| Web UI, both pipelines | built, tested end to end, `DEMO_RUNBOOK.md` |
| All numbers | `RESULTS.md`, each with its run folder |
| 32 hand transcriptions + ಶ/ಕ correction | machine labels measured at 0.370 |

### The constraint that shapes the day

Free tier is **20 calls per key per day, per model** → 100/day on
`gemini-3.5-flash`. Budget:

| task | reads (3.5-flash) | modernizer (second model) |
|---|---|---|
| pre-cache 2 real pages x 4 lines x 5 reads | 40 | 16 |
| re-measure recovery at better quality | *reuses those same calls* | — |
| synthetic demo lines (CRNN is local) | 0 | ~6 |
| **total** | **40 of 100** | ~22 |

**Do not run build B separately.** Pre-cache the demo pages through the
UI, then compute the recovery number from those cached samples: one set
of calls, both the demo cache and the measurement.

### Morning — while quota is fresh

1. Check quota; confirm `3.5-flash` has reset (~9 h after exhaustion).
2. **Pre-cache the demo set** through the UI at 3.5-flash quality — two
   real pages plus the synthetic recovery lines. This *is* the demo
   preparation; afterwards it runs offline.
3. **Recompute recovery** from those samples. Today's 34.2% was
   flash-lite reading at 0.428 CER; at 0.370 it should be cleaner. If
   it stays near 34%, the correlated-error explanation is confirmed,
   which is itself a result worth reporting.
4. Switch the demo to `--no-live` once cached, so it *cannot* call out.

### Midday — the 15 extra transcriptions

Transcribe them as a **clean test set**, independent of the 32 (which
are now entangled with the glyph experiment). Two things only they can
buy: an honest real-line CER, and whether binarized input helps the
*fine-tuned* model, which is still untested.

### Afternoon — the decision

**Does Act 2 earn its place?** Judge on the morning's numbers. If the
readings are visibly a third wrong, it is probably stronger to lead
with the findings — the 94% shape decomposition, the scaling curve, the
ಱ discovery — and keep the real route as "here is the pipeline, and
here is exactly how it fails and why".

Then report sections, with `RESULTS.md` as the spine.

### Friday and Saturday

Rehearse twice. Keep real buffer.

### Cut first if time runs short

- The 3.5-flash re-run — the flash-lite cache from 2026-10-08 already
  demonstrates the pipeline.
- Anything further on glyph synthesis (`GOLD_FINDINGS.md` §4a).
- The B3-vs-B4 re-fine-tune of the old trained modernizer; it has been
  below the copy floor at every point measured.

### Standing rules for the demo

- Never call an API live in front of an examiner — pre-cache, then
  `--no-live`.
- Start the Palmira worker first and check `worker_ready`.
- Do not describe the real route as "our system reads manuscripts".
  It does not: 0.696 CER, and `RESULTS.md` §3 says why.

## 6b. What actually happened on 2026-10-09

**Quota reset confirmed** — 5/5 keys live on `gemini-3.5-flash`
(`setu.label.quota_check`, new). ~27 of the day's 100 went on probes and
a crashed run; 65 on the measurement. Flash-lite is **untouched**: the
demo pre-cache moved to 2026-10-10 at the user's call.

**A rule 8 violation, found and fixed.** The 32 hand-transcribed gold
lines are 16 train-page and 16 **demo-page** lines, and nothing said so.
The 34.2% recovery figure was computed on six demo-page lines, so rule 8
forbade reporting it. `setu.eval.make_recovery_set` now builds the
reportable 16; see `RESULTS.md` §6a.

**Recovery recomputed: 27.8%** (35/126 characters, 13 lines,
`20261009T112249Z_real_recovery`). §6a asked what a better reader would
do. It made the figure **worse**, and that is the result worth having:
disagreement collapsed 0.379 → 0.163 while recovery fell only 34.2% →
27.8%, so the better reader is not finding the right character more
often in its spread — it is producing less spread to look in.
Confidently-wrong is exactly what an ensemble cannot bracket, and a CTC
posterior can. Argument for our own design, by measurement.

**The binarization question is closed: it changes nothing.** The
fine-tuned recogniser scores 0.696 on binarized input against 0.693 on
grayscale (16 lines, `20261009T105420Z` / `...5426Z_gold_real_eval`).
Binarizing helps the *vision model* and lifts the *original*
checkpoint's confidence, but does not move the fine-tuned model's
accuracy.

**Transcription Set 2 is live** — 15 lines drawn seeded-random from the
50-page sample set, no quality filter, max 2 per page, all four
manuscript groups (`setu.eval.make_test_set`). Independent of the 32,
which are entangled with the glyph bank and the scaling curve. Not yet
transcribed.

**Infrastructure, from today's two bugs:** readings are now cached per
crop under `data/demo_cache/readings/`, and `sample_readings` catches
bare `OSError`. A crash or a rerun no longer costs quota — topping the
measurement up from 13 to 16 lines costs 15 calls, not 80.

### The Act 2 decision — SETTLED (user, 2026-10-09)

§6a said to judge whether the real route earns its place. **Decision:
Act 2 is the ensemble on real crops, and only that.** Build B as
described in §7a — read each crop N times at temperature, consensus for
B3, consensus plus alternatives and frequencies for B4.

An alternative was offered and **declined**: making Act 2 the
label-free confidence contrast instead (our own CRNN's mean top-1
falling 0.96 → 0.63 from synthetic to real, 1.8% of lines flagged
against 100%, RESULTS.md rows 6–7). That stays available as a
*finding* for the report and the viva, not as an act.

What Act 2 therefore needs, and nothing else counts as ready:

- the demo pages pre-cached through the UI, then `--no-live`;
- the modernizer run on both branches for each shown line, so B3 and
  B4 both display a modern reading rather than `—`;
- the disclosure said out loud, unprompted: the reading is a vision
  model, not our recogniser (0.696 CER on real crops); the uncertainty
  is ensemble disagreement, not the CTC bridge; recovery is 26.0%
  against the bridge's 80.6%.

The supporting findings are the answer to "why is the real half
weaker", asked or not: 94% of the gap is letter shape, the scaling
curve puts a usable recogniser at ~200,000 lines, the vision model
erases ಱ, and a better reader *lowers* ensemble recovery. Each is a
measurement rather than an apology.

## 6c. 2026-10-10 — Act 2 is cached and the demo passes under `--no-live`

**Done.**

- **Both demo pages pre-cached** at the settings the demo runs at (5
  reads, 4 lines, modernizer on): `group1_1.128` and `group2_2.15`, 8
  lines, every line read, both branches modernized, **branches differ
  on all 8**. 41 reader calls, 16 modernizer calls.
- **All 8 synthetic lines cached with the modernizer**, so Act 1 shows
  modern Kannada rather than `—`. `line_000859` is the line to lead
  with: B3's modern text keeps argmax's non-word **ಬೀಗಿಯ**, B4's gives
  **ಬಾಗಿಲ** ("door"). Say it precisely — carrying the uncertainty
  stopped a non-word propagating; it did not reconstruct the gold word
  (ಬೀದಿಯ).
- **Recovery finished on all 16 reportable lines: 26.0%** (38/146),
  against 27.8% on the first 13, so it is stable (run
  `20261010T075245Z_real_recovery`).
- **`--no-live` acceptance test passes**: 10 staged files replay in
  0.1–0.8 s with no empty lines and both branches modernized; the 3
  uncached pages are refused; nothing broken.

**Four bugs fixed, two of which would have ended the demo** (RESULTS.md
§6): the `--no-live` gate computed a cache key that could never match,
so it refused everything; and on a cache hit the UI changed the key by
dropping `read_fn`, replaying a readings-free entry with nothing on
screen. Also: a cache-only test stores empty entries that later replay
(three deleted), and the CRNN forward is not bit-reproducible here, so
flagged-character counts move between runs and must be read off the
cache the demo will actually replay.

### Still open

1. The 15 transcriptions (Transcription Set 2 is published, store
   empty) — needs a Kannada reader, then an independent real-line CER.
2. Report sections, `RESULTS.md` as the spine.
3. Rehearse twice. Start the worker and **wait for `ready (pid …)`**
   before the app, or the banner lies for the rest of the session.
4. Commit: today's and yesterday's work is still uncommitted.

### Day 1 (Oct 9) — labels + go/no-go
- Cut ~600 training crops from pages **outside** the 14 demo pages and
  the 50 measurement pages (extend `run_palmira_crops.py` with a third
  set). Spread across the four manuscript groups.
- **Go/no-go (morning):** both labellers on 30 crops. If they agree on
  most lines → go. If they mostly disagree → narrow to the one
  manuscript group with highest agreement (`group1`–`group4` look like
  four different hands; one hand needs far fewer labels).
- Label all ~600 with both models; build consensus set; hold out ~100.
- In parallel: write the two modernizer prompts; demo skeleton (Gradio
  is already in `requirements.txt`).

### Day 2 (Oct 10) — fine-tune
- **Rule 3 first:** memorise 8 real lines to near-zero error.
- Fine-tune from `runs/20261006T120718Z_crnn_train_s1/best_model.pt` on
  consensus real lines **mixed with S1 replay** (keeps synthetic
  ability). Laptop 3050: pass `--max-pixels-per-batch 2200000
  --max-single-image-pixels 1500000`; `--time-steps` first.
- Measure: CER on held-out consensus lines; real-vs-synthetic confidence
  again (now 0.63 → ?); synthetic CER must not collapse.
- Retune bridge temperature on real held-out lines.
- **Go/no-go (evening):** real CER under ~30% → plan on. Some error is
  *good* — uncertainty is where the bridge has something to show.

### Day 3 (Oct 11) — end to end, then freeze
- Photo → Palmira → crops → fine-tuned CRNN → B3 / B4 → LLM modernizer,
  side by side, on the 14 demo pages.
- **Precompute and cache every result.** The live demo must not depend
  on the network or an API quota.
- Pick 3–4 lines where B4's modern reading is right and B3's is wrong,
  using the full-page referee.
- **Feature freeze tonight.**

### Day 4 (Oct 12) — rehearse twice, slides, report sections, buffer.

## 7. Fallbacks

| If | Then |
|---|---|
| Labellers disagree on most lines | One manuscript group only |
| Fine-tuned CRNN still poor (Day 2) | Show the LLM's reading on real pages; run B3 vs B4 on synthetic lines + the frame-level evidence (correct symbol in top-5 for **80.6%** of wrong frames) |
| API quota exhausted | Paid Flash-Lite; cache everything |
| Network down during demo | Cached results only — never call an API live |

## 8. Honesty checklist for the report and viva

- Real-line CER is against **machine consensus**, not human gold.
- The demo modernizer is an **LLM**, not our trained model; ours is a
  documented negative result.
- B4's LLM input passes uncertainty as **text alternatives** derived from
  the bridge — a demo-time adaptation of the frame-level bridge. The
  quantified claim remains the frame-level one.
- Demo pages are curated and never appear in any reported number (rule 8).
- ಱ/ೞ are lost by LLM labellers.
- The domain gap (0.96 → 0.63) is reported **before** fine-tuning, as the
  §6.4 result; the after-number is reported separately.

## 9. Provenance note for today's runs

`20261008T055116Z_palmira_line_crops` and
`20261008T061052Z_real_vs_synthetic_confidence` record git commit
`2b2dc90`, but were produced from scripts that were **uncommitted at the
time**. Both scripts are committed in the commit that adds this file,
unchanged except for the result-key rename noted in the second run's
`NOTES.md`.
