# SETU — status, findings and next steps

> **2026-10-08: read `RESULTS.md` first** — every defensible number in
> one table, each with its run folder. Then `GOLD_FINDINGS.md` and
> `DEMO_PLAN.md`.
>
> **Also 2026-10-08:**
> `GOLD_FINDINGS.md` reports 32 hand-transcribed real lines, which
> **answer §3.9's open question** (the real leaves DO carry ಱ, in 1 line
> in 8, and the vision labeller erases every one) and show the machine
> labels are ~39% wrong. `DEMO_PLAN.md` holds the current plan and
> supersedes §4.2 below wherever they disagree.

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
which shares only **6.8%** of its words with the source.

**Update 2026-10-08 (laptop).** Two things resolved since the above.
**§6.4 is done** — Palmira works on the laptop, 975 real line crops were
cut, and the domain gap is now a measured number: mean top-1 confidence
**0.9613 synthetic → 0.6265 real**, with **100% of real lines flagged**
against 1.8% synthetic (§4.3). **And the recogniser cannot read real
manuscripts at all** — its output on real crops is garbage, because S1
trained it on font-rendered text over leaf texture in which real
handwriting only ever appears as *background* (§4.3b). The demo
therefore runs recognition on synthetic lines, with the real-page
attempt shown beside it as the domain-gap result.

**Update 2026-10-08:** the fallback recommended here — a forward
rule-based orthographic modernizer — is **also dead**. The corpus carries
no archaic orthography to normalise (§3.9): the three forward rules fire
**3 times in 3,500 lines**. KannadaLit4NLP is built from 2001 critical
editions (plus a 1943 text), so its "old Kannada" is old *language* in
modern *orthography*. Both modernization routes are therefore closed, and
the current priority is a **recognition-centred demo** (§4.2), which is
where the project's genuine result lives.

---

## 1. What works (reportable results)

| Result | Number | Run folder / source |
|---|---|---|
| Recogniser CER, frozen test split (all 1,358 lines) | **1.53%** | `20261007T151916Z_eval_b0_b3_b4` |
| Recogniser CER, S1 validation split | **0.39%** | `20261006T120718Z_crnn_train_s1` |
| Recogniser CER over all of S2 (3,500 lines) | 1.57% (median **0.00%**) | `20261007T022632Z_cache_s2_recogniser` |
| Palmira line-segmentation success rate | **96.9%** over all 738 surveyed photos (**97.2%** over the 568 *distinct pages* — see §3.10) | `data/raw_corpus/palmira_survey_results.jsonl` |
| Fake-vs-real classifier | **68.8%** test accuracy | `20261006T153251Z_fake_vs_real_classifier` |
| CTC confidence distribution (real checkpoint) | mean top-1 **0.9606**; **11.69%** of frames below 0.9 | `20261006T152242Z_measure_confidence_distribution` |
| Joint unlocked training | stable, recogniser never needed locking | `20261007T134952Z_joint_unlocked_train` |
| POS tagging | 104 hand-applied tags | `src/setu/pos/` (earlier session) |
| **Soft bridge, measured at frame level** | on frozen-test frames where top-1 was wrong, the correct symbol was still in the top-5 **80.6%** of the time; bridge carries it with mean weight **0.200**, argmax with **0.000** | `20261008T033215Z_recovery_examples` |
| Palmira line crops from real pages | 263 (14 demo pages) + 712 (50 sample pages) | `20261008T055116Z_palmira_line_crops` |
| **§6.4 domain gap, real vs synthetic** | mean top-1 confidence **0.9613 → 0.6265**; flagged lines **1.8% → 100%** | `20261008T061052Z_real_vs_synthetic_confidence` |

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

### 3.10 The "738 unique photos" are 568 distinct pages (2026-10-08)

Found while drawing the §6.4 sample. The page survey deduped by photo
signature (width, height, sharpness, contrast, brightness), which
correctly collapses the train/val/test mirrors but **cannot see
`dataset/hd_images/`** — 170 higher-resolution re-photographs of pages
that are already in `Dataset/`. Their resolution differs, so their
signature differs.

So the 738 are **568 distinct pages + 170 hd duplicates**. The headline
number barely moves (552/568 = **97.2%** over distinct pages vs 96.9%
over all 738), so nothing reported is wrong — but the *wording* is:
"738 pages" should read "738 photos of 568 pages", or quote 97.2%.

`run_palmira_crops.py` dedupes by page stem and prefers the standard
`Dataset/` copy, so the 50-page §6.4 sample is 50 genuinely distinct
pages. The first attempt sampled three pages twice; that run was deleted
and redone rather than patched.

### 3.9 The corpus contains no archaic orthography at all (2026-10-08)

Found while checking whether a Gemini transcription of a real palm-leaf
photo had silently normalised ಱ/ೞ. It had — but so has everything else,
including our own training data.

| Source | ಱ (U+0CB1) | ೞ (U+0CDE) |
|---|---|---|
| Raw `KannadaLit4NLP_master.jsonl`, 27.1M chars, untouched | **11** | **0** |
| S1 — recogniser training, 23,347 lines / 5.2M chars | **6** | **0** |
| S2 — backs every reported end-to-end number, 3,500 lines | **0** | **0** |
| Gemini's reading of one real HKHPL page, 1,282 chars | **0** | **0** |

For scale, the raw corpus holds 1,156,698 ರ and 297,118 ಳ. Five lines out
of 23,347 contain either archaic character.

**This is not our pipeline.** The raw source has the same property, and
`build_corpora.py`'s cleaning only strips punctuation and ASCII — ಱ/ೞ
round-trip through WX fine (`rY`, `zY`) and were never filtered.

**Cause**, from the dataset's own `SOURCE.md`: the corpus is vachanas
(Basavanna, Allamaprabhu, Akkamahadevi, Siddharameshwara…), Sarvajna's
tripadis, and D.V.G.'s *Mankutimmana Kagga* — **a 1943 text, 946 verses**
— all taken from **2001 critical editions** (Kannada Pusthaka Pradhikara).
Critical editions normalise orthography. Vachana literature was also
plain-register by design, unlike the ornate halegannada of Pampa and Ranna.

**So our "old Kannada" is old *language* in modern *orthography*.**

Consequences:

- **§4.1 as originally written is refuted** — see there.
- **`CLAUDE.md`'s stated justification for WX output is unsupported.** It
  argues syllable-level output would starve "ಱ and ೞ, the very letters
  that mark a text as old Kannada." They were never present to starve. WX
  remains the right choice on CTC class-count grounds (57 vs 600–900), but
  the reason in `CLAUDE.md` is not the real one, and should not be
  defended as written in the viva.
- **The recogniser cannot read ಱ or ೞ.** Six training instances and zero
  respectively; the WX symbols `rY`/`zY` carry essentially no signal. If
  real manuscripts contain them, it will fail on precisely the characters
  that define the task.
- It partly explains the 1.53% CER: modern-orthography text in a modern
  font is an easier read than true halekannada.

**ANSWERED 2026-10-08 — yes, they do.** A Kannada reader transcribed 32
held-out real lines blind: **ಱ appears 6 times across 4 of the 32
lines**, and the vision model's labels for those same lines contain
**zero**. So the finding below is a property of the *corpus*, not of the
collection: our training text is old language in modern orthography, but
the leaves themselves carry the archaism. `CLAUDE.md`'s original
instinct about ಱ/ೞ was right, and the machine-labelling route erases
exactly the letters the project is about. See `GOLD_FINDINGS.md` §1.
ೞ did not appear in these 32 lines; a larger sample may yet show it.

**The original question, now settled:** do the real HKHPL
manuscripts actually contain ಱ/ೞ? An hour of a Kannada reader's time over
five pages decides it. If **yes**, we have a sharp, honest domain-gap
finding for §6.4. If **no**, the title's task does not exist for this
collection and the framing changes. Not answerable from this machine — the
one page examined was a late calendrical text, not early kavya, and one
page proves nothing.

The soft-bridge claim survives either answer: it is about the interface
between recognition and modernization, not about halekannada specifically.

---

## 4. Next steps — pick up here

### 4.1 ~~RECOMMENDED: forward rule-based orthographic modernizer~~ — REFUTED (2026-10-08)

**Do not build this.** The earlier recommendation assumed the old text
carried archaic orthography for the rules to normalise. It does not — see
§3.9. Measured by applying all three forward rules to every S2 line:

```
rule 1 (nasal+virama → anusvara) fired :  3
rule 2 (ಱ → ರ)                   fired :  0
rule 3 (ೞ → ಳ)                   fired :  0
lines changed : 3 / 3,500  (0.09%)
```

The engine would be a no-op: byte-identical to B0 on 3,497 of 3,500 lines.
Every argument in the original recommendation still *reads* correctly —
forward really is the deterministic direction, output really cannot
hallucinate — but all of it is moot when nothing fires. The error was
reasoning from what halekannada is in general rather than measuring this
corpus, which takes about a minute:

```bash
python -c "t=open('data/raw_corpus/s2_corpus_render.txt',encoding='utf-8').read(); print(t.count(chr(0x0CB1)), t.count(chr(0x0CDE)))"
```

**Both modernization routes are therefore closed:** the learned one
because the targets are ಭಾವಾರ್ಥ commentary (§3.1), the rule-based one
because there is no archaic orthography to normalise (§3.9). There is at
present no working modernization story, and the remaining effort should go
to the demo (§4.2) rather than to a third attempt at this.

### 4.2 Demo strategy — THE PRIORITY (restated 2026-10-08)

A good working demo is now the stated goal, and with §4.1 refuted the demo
has to be built on **recognition**, not modernization.

**Demoing the modernizer's output will not work.** Measured on 300
training examples: chrF++ mean **14.70** (vs 13.44 held-out — only 1.3
better), and only **8 of 300** beat the B0 floor. The best single training
example (24.76) is still visibly incoherent Kannada. The model barely
memorised its own training set, so cherry-picking produces nothing
presentable and any Kannada-reading examiner will see it immediately.

#### The three acts that do work

1. **Recognition on a real page.** Real HKHPL photo → Palmira finds lines
   → crop → CRNN reads each line. The project's genuine result (1.53% CER
   on synthetic). Needs Palmira, hence the laptop.
2. **Confidence flagging as triage.** `src/setu/bridge/flagging.py` is
   built and tested. Show which lines the system would route to a human.
   This turns the synthetic→real domain gap from an embarrassment into a
   feature: *the system knows what it does not know.* Strongest honest
   card in the deck.
3. **The soft bridge, shown at frame level — BUILT, 2026-10-08.**
   `src/setu/bridge/recovery_examples.py`, run
   `20261008T033215Z_recovery_examples`. The novel contribution,
   demonstrated **without** depending on the modernizer being any good.

   Measured over all 1,358 frozen-test lines (383,427 aligned non-blank
   frames):

   | | |
   |---|---|
   | frames where the recogniser's top-1 was wrong | **3,607** |
   | …of those, correct symbol still inside the top-5 | **2,907 (80.6%)** |
   | mean weight the **bridge** puts on that correct symbol | **0.200** |
   | mean weight **argmax** puts on it | **0.000** |

   That aggregate is the slide. It beats three anecdotes because it is a
   quantified property of the whole frozen test split, and it pre-empts
   "aren't those cherry-picked?" — the 12 worked examples in
   `examples.md` are the clearest instances of a pattern that holds
   across 2,907 frames.

   **Best examples for the demo are #2 and #3**: `x` vs `X` (ದ vs ಧ) at a
   0.500/0.500 split. A genuinely confusable minimal pair, the recogniser
   exactly undecided, argmax forced to bet — and wrong. Example #1 is a
   blank-vs-letter split, less visually compelling to a Kannada reader.

   Per-frame truth comes from **CTC forced alignment** (Viterbi against
   the ground-truth old text), verified by checking that the CTC-collapse
   of every alignment reproduces its target exactly — 400/400 on a test
   sample, 0 mismatches.

   Note this is **weaker than roadmap §4.5's "worked recovery examples"**,
   which wanted the bridge to produce the right *final* answer. That needs
   a working modernizer and cannot be shown. The defensible claim is
   "**the information survives the interface**" — argmax assigns the
   correct symbol weight 0 by construction; the bridge does not. Do not
   let it drift to "the bridge fixed the error" under questioning. It did
   not; it declined to throw the answer away.

**Modernization in the demo:** show it, framed as a measured negative
result with the §3.1/§3.9 root-cause analysis attached. Hiding it invites
"so where is the modernization your title promises?" with no answer ready.
A documented, well-analysed failure is respectable; an unexplained gap is
not.

#### Order of operations

1. **Palmira crops on the laptop** (`TRANSFER_TO_LAPTOP.md` §7 step 1).
   Everything else is blocked on this, and it also unblocks §4.3.
2. **Look at CRNN output on ~10 real pages before building anything
   around it.** This decides whether act 1 is viable. The roadmap's own
   Part 3.3 forecast is **40–70% CER on real palm leaf** against 8–15% on
   synthetic — and we beat the synthetic forecast by a wide margin, which
   says nothing about the real side. Assume nothing here; look first.
3. **Then pick the demo shape** (below).
4. **Curate `demo_pages/`** — still empty, just a `.gitkeep`. Rank
   candidates by `Character Line Segment` count in the committed
   `data/raw_corpus/palmira_survey_results.jsonl` (738 pages); that is a
   far better quality signal than the brightness heuristic in
   `data/splits/hkhpl_page_survey.csv`, where only 25 of 738 rows were
   human-reviewed. Free, needs no Palmira, can be done before the move.

#### Contingency, decided by step 2

- **If the CRNN reads real lines respectably** — demo acts 1 + 2 + 3 on
  real pages. Best case, and the strongest version of the project.
- **If it reads them badly** — demo act 1 on *synthetic* lines (where the
  1.53% is real and reportable), then show the real-page attempt
  side by side with the measured confidence drop as the §6.4 domain-gap
  result, then acts 2 + 3. Less flashy, equally honest, and it still
  answers "did you use the dataset your project is about?"

Either way the demo has a spine that does not depend on the modernizer.

`CLAUDE.md` rule 8 already sanctions curation: `demo_pages/` never appears
in any reported number, and "curation is for the demo only and is disclosed
explicitly in the report." Keep that disclosure, and have the answer to
*"is that a held-out example?"* ready — being unable to answer it is what
damages you, not the curation itself.

### 4.3 §6.4 real-imagery measurements — DONE (2026-10-08, on the laptop)

**Unblocked and completed.** Palmira was never rebuilt on the 3060; the
work moved to the laptop, where the `palmira` conda env still imports
(detectron2 0.4, CUDA, the DefGrid extension) and the weights are intact.

`data/raw_corpus/run_palmira_crops.py` (new) cuts line crops from the
original colour photos, writing a mask-filled crop and a raw
bounding-box crop per detected `Character Line Segment`, plus a manifest
with box/score/area/centroid. Run `20261008T055116Z_palmira_line_crops`:

| Set | Pages | Line crops |
|---|---|---|
| `demo` (the 14 files in `demo_pages/`) | 14 | 263 |
| `sample` (seeded, disjoint from demo) | 50 | 712 |

Crops are gitignored (`data/real_lines/`). The sample excludes any page
sharing a filename stem or photo signature with a demo page, asserted in
code — rule 8.

**All three measurements now exist:**

1. line-segmentation success rate — 96.9% over 738 pages (earlier)
2. **recogniser top-1 confidence, real vs synthetic — done**
3. **flagging rate, real vs synthetic — done**

Run `20261008T061052Z_real_vs_synthetic_confidence`, 712 real crops
against the 1,055-line S1 validation split:

| | mean top-1 | frames < 0.9 | lines flagged (t=0.85) |
|---|---|---|---|
| synthetic (S1 val) | **0.9613** | 11.5% | **1.8%** |
| real (`crop:as_is`) | **0.6265** | 79.8% | **100%** |

**Every single real line is flagged at every threshold tested**
(0.75/0.80/0.85/0.90); real per-line confidence spans 0.57–0.71 and
never once reaches the synthetic median of 0.966. That is the §6.4
domain-gap number, obtained with zero transcriptions.

Measured in **four** conditions so it cannot be waved away as a
preprocessing artifact — mask-filled vs raw bounding box, native
resolution vs resized to the renderer's nominal 64px line height. All
four land in 0.6254–0.6352. Inverted polarity was also checked by hand
and is no better. 12 of 712 crops are under the architecture's 32px
minimum (five height-pooling stages) and are padded, not dropped, and
counted in `n_padded_to_min_height`.

The synthetic half agrees with the committed 3060 baseline to +0.0006
mean top-1 — but on **this laptop's own S1 render**, not the same image
files (585,505 frames vs 586,747). See the run folder's `NOTES.md`; read
it as agreement across two independent renders, which is the stronger
claim, not as bit-exact reproduction.

### 4.3b What the recogniser actually says on real lines

Worth stating plainly because it decides the demo (§4.2 step 2). The
decoded output on real crops is **not degraded text, it is garbage** —
strings like `ಅಅಅನನ ಅಲ್ಲ್ನಅರಅಅವಾ`, dominated by repeated ಅ, on lines
Palmira segmented with score 0.99.

This should not be surprising, and the report should say so rather than
present it as a shock: **S1 trains the CRNN on crisp font-rendered text
composited onto real leaf texture.** Those texture patches are cut from
real HKHPL photos and visibly contain real handwriting — which is always
*background*, never the target. The model was therefore trained, for
23,346 lines, to read rendered glyphs and ignore handwriting. Real crops
are nothing but handwriting. The 1.53% synthetic CER and this result are
consistent: the recogniser reads the font it was shown.

Consequence: **the §4.2 contingency fires — demo act 1 runs on synthetic
lines**, with the real-page attempt shown beside it as the measured
domain gap. Do not promise real-page recognition.

### 4.3c (superseded) §6.4 real-imagery measurements — was BLOCKED

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
