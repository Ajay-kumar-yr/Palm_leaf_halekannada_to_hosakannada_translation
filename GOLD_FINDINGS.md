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
