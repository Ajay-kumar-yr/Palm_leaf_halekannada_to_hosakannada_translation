# What 32 hand-transcribed lines established (2026-10-08)

A Kannada reader transcribed the 32 held-out real lines **blind** — the
page never showed the machine's reading, so it could not anchor theirs.
Run `20261008T130753Z_gold_real_eval`; per-line comparisons in that
folder's `per_line.jsonl`.

Thirty minutes of typing settled three things, one of which changes how
the project should be described.

---

## 1. The real manuscripts DO contain archaic orthography — and the vision model erases it

**This answers the open question in `STATUS.md` §3.9.**

| | ಱ (U+0CB1) | ೞ (U+0CDE) |
|---|---|---|
| human reading of 32 real lines | **6, across 4 lines** | 0 |
| the vision model's labels for the same lines | **0** | 0 |

§3.9 established that the *corpus* (KannadaLit4NLP, from 2001 critical
editions) carries essentially no archaic orthography, and left open
whether the real leaves do. They do: **1 line in 8 contains ಱ**, and the
machine labeller silently normalised every one of them away.

What the reader saw, against what the machine reported:

```
group1_1.26/line_03
  human  : ॥ನಡದರೆ ಸ್ಥಿರತ್ರೆತೀನು।ಶುಱನಾಯ್ಯರದಿಂದಕ್ಯೆಡೆವದು।ತತ್ವಕ್ರಮ।೩
  machine: ॥ ನಡದರೆ ಸ್ಥಿರತೆ ತೀಸುಳು ಮಿನಾಯ್ಕ ಸ್ವರದಿಂದ ಕೇಳಿಡುವದು । ತತ್ವಕ್ರಮ ೧೦

group1_1.26/line_07
  human  : ತಿದ್ದರೆ।ರಾಜ್ಯಕಾಯ್ಯರಿಂ ಕೆಡವದೂ ॥ ಱ್ ಱ್ ಯದ್ದಕ್ರಮಾವಾವರತಿ। ವರ
  machine: ॥ ತಿದರೆ ರಾಜಕಾರ್ಯ ದಿಂ ಕೆಡವದೂ ॥ ೨ ॥ ಆಯುದ್ಧಕ್ರಮ ಪಾಪರತಿ ವರ
```

Consequences, and they are not small:

- **The project's subject matter is real.** §3.9 concluded "our old
  Kannada is old *language* in modern *orthography*", which put the
  title's task in doubt. That holds for the corpus, not for the
  manuscripts. The leaves carry the orthography; our *training data*
  does not.
- **`CLAUDE.md`'s original instinct was right.** It singled out ಱ and ೞ
  as "the very letters that mark a text as old". §3.9 appeared to
  refute that. It did not — it located the problem in the corpus.
- **The teacher-student route destroys exactly what the project is
  about.** A vision model cannot teach a recogniser to read ಱ when it
  never outputs ಱ. Any recogniser trained on these labels is trained to
  normalise away the archaism, whatever its CER.
- **The report should state this plainly**, and it is the strongest
  argument for human transcription at scale being the real unlock for
  this collection.

---

## 2. The machine labels are ~39% wrong

| | |
|---|---|
| mean CER, machine label vs human | **0.3886** |
| median | 0.3730 |
| best / worst line | 0.193 / 0.818 |
| within the 0.35 filter threshold | **13 of 32** |

Every real-image number reported before today was *agreement with these
labels*. They are wrong in well over a third of their characters, and
the `build_label_set` disagreement filter (≤0.35) would have rejected
**19 of the 32** had a second opinion existed.

The best and worst cases show the spread is real, not an artifact of
one bad page:

```
best  (CER 0.19)  human  : ॥ವನರಂದೆನಾಮ।ಬವಕೊಳಗೆಪ್ಯಾಂಗವಬ್ಬನತ್ತುವಬ್ಬ ಬದುಕಿದಾ ಪಾಂಗೆ ಚಂ
                  machine: ॥ ವನರವಂದೇನಾಮ। ಯುವಕೊಳಗೆ ಹ್ಯಾಂಗವಬ್ಬನತ್ತವಬ್ಬಬದುಕಿದಾ ಹಾಂಗೆ ಚಂ
worst (CER 0.82)  human  : ಶ್ರೀ ಗುರುವ್ರಾಗ್ನೆ ಗ್ರನನಗಳು ಏದಶಂದಶ್ರೆ ಬಂದಏಲ॥೫
                  machine: ೨೦ ಋಣಗ್ರುಹಗಳುಪ್ರದಕ್ಷಿರಾಕ್ಕೆಬಂದವಲು೨೧ ಹೊ
```

---

## 3. True CER for the fine-tuned recogniser: 0.687

| measured against | CER |
|---|---|
| machine labels (reported until today) | 0.700 |
| **hand transcriptions (true)** | **0.687** |

The two agree closely, and that matters: the recogniser is **not**
secretly better than it looked. Label noise was not hiding a good model.

**But it does add a caveat to the data-scaling curve** (`DEMO_PLAN.md`
§5c). That curve was measured with labels now known to be ~39% wrong, so
it understates what clean labels might achieve, and the fitted slope
should not be read as a property of the architecture alone. The
conclusion stands anyway, for two reasons: the model's error (0.687) is
far above the label-noise floor (0.389), so the model — not the labels —
is the binding constraint today; and clean labels at the needed scale
would require human transcription of thousands of lines, which is the
very bottleneck that made the machine-labelling route necessary.

---

## Honest limits of this gold set

- **One reader, one pass, no double annotation.** It is ground truth for
  this project's purposes and carries its own error rate. Two readers on
  the same lines would let us quote an inter-annotator figure; we do not
  have one.
- **32 lines.** Enough for a mean CER with a wide interval, not enough to
  break down by page or hand.
- These are **held-out** lines, so none of this leaked into training.

---

## 4. Glyph extraction from the gold lines: works, but not well enough to stitch

Tested on 2026-10-08, time-boxed, because it was the only idea that
attacks the data bottleneck without more human transcription: cut each
character out of the 32 transcribed lines, build a bank of real glyphs
in the scribe's hand, and stitch unlimited training lines with perfect
labels.

**The alignment half works.** A recogniser overfit on the gold lines
(loss 0.026) force-aligns all 32 of them, zero rejections, yielding
**1,787 glyphs covering 49 of 66 symbols (74%)**, 34 of them with 5+
exemplars. Overfitting is legitimate here: the model only has to
reproduce text it has memorised in order to say *where* each character
sits.

**The glyphs carry real symbol identity, weakly.** Nearest-neighbour on
raw pixels: **14.5%** same-symbol against 3.5% chance, top-5 35% against
~15%. Four times chance is well clear of noise — these are genuinely
characters, not slices of background.

**But they cannot be cut cleanly.** CTC marks where a character *peaks*,
not where it begins and ends, and three cutting strategies all hit the
same ceiling:

| cut | median width | 1-NN | top-5 |
|---|---|---|---|
| no margin | 8px | 0.105 | 0.273 |
| 1 frame margin | 24px | **0.145** | 0.351 |
| 2 frame margin | 40px | 0.145 | 0.355 |
| midpoint between peaks | 16px | 0.101 | 0.294 |

Too narrow clips the character; too wide drags in its neighbours;
splitting the gap between peaks does neither well. This is not a
parameter to tune — in connected handwriting the ink of adjacent
characters genuinely overlaps horizontally, and no vertical cut
separates them.

**The stitched lines settle it.** Rebuilding a line from the bank
produces visible brightness seams at every join, clipped characters, and
none of the flow of the real line beside it. A recogniser trained on
that would learn that characters are separated by seams — which real
lines are not.

### 4a. Binarizing first removes the seams — and exposes the next problem

The conclusion above blamed the seams on a brightness mismatch, so the
obvious follow-up was to binarize before cutting. Done, and it works:
**the seams are gone.** Glyphs cut from binarized lines stitch onto a
uniform white ground with no visible join.

Two binarizers were compared on the same line. **HKHPL's own
`Ground_Truth_images` are not usable here** — they cover only 3 of the 7
gold pages, and on `1.108` a large black blob swallows a third of the
line (ink fraction 0.411 against the U-Net's 0.134). The **Sajjan U-Net**
already in the repo (`data/external_models/sajjan_unet`, trained on this
dataset) produces clean, crisp ink on white across every crop, and is
clearly the better choice. All 32 gold lines were binarized with it; a
recogniser overfit on them reaches loss 0.0122 and force-aligns all 32,
giving the same 1,787 glyphs over 49 symbols.

The pixel probe barely moves (1-NN 0.126 against grayscale's 0.145,
top-5 0.344 against 0.351) — binarization strips the leaf texture the
probe was partly keying on, so this is not evidence either way.

**What the stitched line now shows** is the real remaining obstacle:
characters sit at inconsistent heights, spacing is irregular, and
fragments of neighbouring ink still ride along at the cuts. The real
line flows along a baseline; the stitched one bounces.

**Verdict: stop here, but the direction is sound.** Each fix exposes the
next problem — seams gone, baseline jitter and neighbour fragments
remain — and the last of those is structural: in connected cursive the
ink genuinely overlaps, and no vertical cut separates it. Baseline
alignment (match each glyph's ink centroid) and seam blending are the
next steps for anyone picking this up, and they may well be enough. With
the deadline where it is, the time belongs on the demo.

**A finding worth keeping regardless of the glyph idea:** the Sajjan
U-Net turns these crops into clean ink on white. Whether that *helps the
recogniser read real lines* is untested and cheap to test, and needs
held-out binarized lines — one more reason the next batch of hand
transcriptions should be a clean test set.

`src/setu/render/glyphs.py` (with `--midpoint` and margin controls) and
`data/raw_corpus/binarize_lines.py` are kept, so all of this is
reproducible rather than merely described.

---

## 5. The domain gap is letter shape, not leaf texture

Prompted by a good question: the pipeline runs real photo → Palmira →
CRNN with **no binarization**, so was the recogniser being fed something
unlike its training data? Checked: no. S1 is grayscale textured
(148–256 grey levels, ink fraction 0.11–1.00) and the real crops are
grayscale too (143 levels, 0.19). The pipeline was consistent.

But the question points at a real decomposition. The gap between
synthetic and real has two parts — **texture** (rendered leaf background
vs a real photographed leaf) and **shape** (a font's letterforms vs a
scribe's). Binarizing with the Sajjan U-Net removes the first and leaves
the second. Measured on all 712 sample crops with the original S1
checkpoint, no retraining (`20261008T141042Z_real_vs_synthetic_confidence`):

| input | mean top-1 | frames below 0.9 |
|---|---|---|
| synthetic (S1 val) | 0.9613 | 11.5% |
| real, binarized | 0.6453 | 76.0% |
| real, grayscale | 0.6265 | 79.8% |

Decomposing the 0.335 total gap:

| component | size | share |
|---|---|---|
| leaf texture | 0.019 | **6%** |
| letter shape | 0.316 | **94%** |

**Binarization is worth having and nowhere near enough.** It is free,
it reliably improves confidence (and should be used), but it closes
one-sixteenth of the gap. Ninety-four per cent of the problem is that
the recogniser learned letterforms from a rendered font and real
manuscripts are handwritten.

This explains every negative result above in one line, and it is the
cleanest statement of the project's central obstacle: **not noise, not
contrast, not preprocessing — shape.** It is also why no amount of
preprocessing rescues the recogniser, and why the data-scaling curve
(§5c in `DEMO_PLAN.md`) is the right place to look for the real cost.
