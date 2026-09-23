# SETU — Roadmap v4
### Reading old Kannada palm-leaf manuscripts and rewriting them in modern Kannada

**Scope:** 4 weeks, fixed deadline. Team of 4, roughly 5–8 hours per week *in total* (not each) — about 80–130 person-hours.

**Changes in v4:**
- **Track B is the only path.** UniLipi weights are not coming in time. All Track A branching removed.
- **New Part 6:** how HKHPL is actually used, and a cheap way to get real numbers from it without transcriptions.
- **New Part 11:** how long each training run takes on the RTX 3060.
- **Modernizer changed** from ByT5-small to a small transformer trained from scratch. Reasoning in §5.2 — it's roughly ten times cheaper and the pretrained model wasn't buying what it looked like it was buying.

---

## Part 0 — What we are building

### The one thing that is genuinely ours

Every existing system works in two separate steps. A recogniser looks at the image and decides which character it sees at each spot. Then a second program takes that finished string and rewrites it in modern Kannada.

The trouble is the word *decides*. When the recogniser is only 55% sure whether a mark is one letter or another, it still picks one. The moment it picks, the fact that it was unsure is gone. The modernization step gets a confident-looking string with no idea which parts were guesses.

**Our system doesn't make that decision until the end.** The recogniser passes all its candidates forward with confidence attached, and the modernization step works with that uncertainty directly.

This is our one novel feature. Everything else is existing work, adapted, and we say so.

### Synthetic-first, and why that's acceptable

We generate training and test images by rendering real old-Kannada text in Kannada fonts and artificially damaging it to look like palm leaf. All reported accuracy comes from these images.

The obvious objection is that synthetic damage isn't real damage. True. But our claim isn't "this reads real manuscripts well" — it's **"carrying uncertainty forward beats committing early."** That's a controlled comparison: same recogniser, same images, same modernizer, same training budget, with only the interface differing. A synthetic test set is a valid place to run it.

We still run the full pipeline on real HKHPL pages, and Part 6 shows how to get genuine measurements from them without needing transcriptions.

### What stays cut

Not coming back: visual cross-attention, era conditioning, Palmira fine-tuning, the four-baseline grid, ablation experiments, calibration statistics as a chapter, double annotation, the full patent-differentiation chapter.

### What's locked in

POS tagging and modernization are in the submitted synopsis. Both stay, both handled cheaply.

---

## Part 1 — How it fits together

```
  ┌────────────── TRAINING & SCORING (synthetic) ──────────────┐
  │  old Kannada text ─► render in fonts ─► damage ─► image    │
  │         └── paired modern Kannada text ──────────┐         │
  └──────────────────────────────────────────────────┼─────────┘
                                                     │
  ┌────────── DEMO & MEASUREMENT (real HKHPL pages) ─┼─────────┐
  │  photo ─► Palmira line finder ─► line images ────┤         │
  │  (ORIGINAL photo, not black-and-white)           │         │
  └──────────────────────────────────────────────────┼─────────┘
                                                     ▼
                    ┌────────────────────────────────────────────┐
                    │ Recogniser (our small CRNN)                │
                    │ For every image frame: a confidence number │
                    │ for each possible symbol. Not a string.    │
                    └────────────────────┬───────────────────────┘
                                         ▼
                    ┌────────────────────────────────────────────┐
                    │ ★ SOFT BRIDGE — our contribution ★         │
                    │ per frame: top 5 symbols (blank included), │
                    │ renormalised, temperature-adjusted,        │
                    │ blended into one embedding.                │
                    └────────────────────┬───────────────────────┘
                                         ▼
                    ┌────────────────────────────────────────────┐
                    │ Modernizer (small transformer)             │
                    │ → modern Kannada                           │
                    └────────────────────┬───────────────────────┘
                                         │
                         ┌───────────────┴──────────────┐
                         ▼                              ▼
              Confidence below threshold      POS tagging: existing tagger
              → flag line for review          or LLM prompt on the output
```

### Plain-English glossary

- **OCR / recogniser** — turns a picture of writing into text.
- **Confidence** — a number between 0 and 1 for how sure the model is about one option. All options at one spot sum to 1.
- **Softmax** — the standard maths turning raw scores into those confidence numbers.
- **Temperature** — a dial on the softmax. Higher temperature makes confidences less extreme, so second and third choices keep meaningful weight.
- **Argmax** — "take the highest-scoring option, discard the rest." What we avoid doing early.
- **Embedding** — a list of numbers representing a symbol, so the model can do arithmetic on it. Blending two symbols means blending their number lists.
- **CTC** — a training method for reading a line when you know *what* it says but not *where* each letter sits. Avoids cutting the line into separate characters, which works badly on damaged palm leaf.
- **Frame** — CTC slices the line image into a strip-by-strip sequence. Each strip is a frame. A 30-character line might produce 150 frames.
- **Blank** — a special CTC symbol meaning "nothing here / still the same letter." Most frames are blank.
- **Gradients / backpropagation** — how models learn. An error at the output is traced backwards to work out what to adjust. It only travels through smooth maths; argmax isn't smooth, which is why it blocks learning.
- **CER** — Character Error Rate. Percentage of characters wrong. Lower is better.
- **chrF++ / BLEU** — how close generated text is to a correct reference. Higher is better.
- **Epoch** — one pass through the whole training set.
- **WX** — writing Indic scripts in ordinary Roman letters, designed so conversion back is exact.
- **bf16 / mixed precision** — using 16-bit numbers instead of 32-bit for most of the maths. Roughly twice as fast, half the memory, no meaningful accuracy loss on this kind of model.

---

## Part 2 — The data

| Set | Contents | Size | Used for |
|---|---|---|---|
| **S1 — rendered lines** | image + old text | 20–30k | Training the recogniser |
| **S2 — rendered pairs** | image + old text + **modern text** | 2–4k | Joint training, and all reported end-to-end numbers |
| **S3 — text pairs** | old text + modern text, no images | 100–300k | Pre-training the modernizer |
| **R — real pages** | HKHPL photos, no text | ~50–500 | Demo and the no-label measurements in Part 6 |
| **G — gold lines** | real image + old text | 0–60, best effort | One real CER number, if we get any |

### 2.1 S2 — rendered pairs (the backbone)

This is what makes joint training and end-to-end scoring possible.

1. Take KannadaLit4NLP verses that come with scholarly interpretations.
2. Keep entries where the interpretation runs 0.7×–1.5× the length of the original — those tend to be direct rewrites rather than loose commentary.
3. Render the **old text** through the damage simulator to get an image.
4. You now have image + old text + modern text.

Expect a few thousand. Small, but sufficient, because the modernizer is pre-trained on S3 and only adapted here.

**Freeze the test split before rendering.** Pick a set of verse IDs, set them aside, render separately, commit the ID list on day one. Because this data is generated rather than collected, freezing it is trivial.

**Never generate the modern-text side with your own rule engine.** You'd be scoring the system against output from the same rules it learned, which measures nothing.

### 2.2 S1 — rendered lines for the recogniser

**20,000–30,000 line images.** Text from Kannada Wikisource (Pampa, Ranna, vachana literature) and any other old-Kannada source. No modern text needed — the recogniser only learns to read.

**Five damage effects, applied when the images are generated:**

1. Background texture — real patches cut from blank areas of HKHPL leaves
2. Ink bleed — letters slightly smudged and blurred
3. Page warp — gentle bending, like a curled leaf
4. Baseline skew — lines tilted a few degrees
5. Holes and damage — real damaged patches from HKHPL composited in

Dropped from the original nine: fibre grain, fungal staining, stylus groove shading, elastic letter distortion.

**Split the work between generation time and training time.** This matters for speed (Part 11):

- **Generate once, bake in:** page warp, background texture, holes. These are slow to compute.
- **Apply fresh each time an image is used:** small rotation, brightness shift, mild blur, slight horizontal stretch. These are cheap.

The cheap ones give you effectively unlimited variety at no cost. The expensive ones would starve the GPU if recomputed every batch.

**Cheap quality check:** train a small classifier to tell your fake images from real HKHPL crops. If it succeeds instantly, your fakes are too clean. You want it struggling, around 70–80%. Ten minutes, and the number reads well.

### 2.3 S3 — text pairs for pre-training the modernizer

Take modern Kannada text and run old-Kannada spelling rules backwards — ಹ becomes ಪ, restore the obsolete letters ಱ and ೞ, restore archaic endings. A YAML rules file plus a script.

**100,000–300,000 pairs is the right size.** More than that costs training time you don't need to spend.

**State plainly in the report** that these have believable spelling but not authentic old grammar, which is why S2's real pairs carry the reported numbers.

### 2.4 G — gold lines (best effort, never blocking)

If anyone — the team, a Kannada department, an epigraphy contact — can read old palm-leaf script reliably, collect what you can. **30–60 lines is useful. Zero is survivable.**

Worth one email each to Dr. Bannigidad, a Kannada department, and the Oriental Research Institute. Then move on; don't wait.

If lines do arrive, assign each to test or adaptation by a hash of its ID at creation, and keep an append-only list. Don't checksum a set that's still growing.

### 2.5 POS tagging (required by the synopsis)

Run an existing Kannada part-of-speech tagger, or one fixed LLM prompt, over the **modern Kannada output**. Hand-check about 100 tags, report accuracy on those. A script, not a model. One afternoon.

---

## Part 3 — The recogniser

**Before building from scratch, spend half a day trying to fine-tune any pretrained handwriting recogniser you can find** — PLATTER, or any general HTR checkpoint. Fine-tuning something mediocre beats training a good design from nothing on 25,000 lines. Build from zero only if nothing is available.

### 3.1 Romanised output, roughly 50–60 symbols

Not 600–900 Kannada syllables. This matters more than it looks.

25,000 lines gives around 750,000 symbol instances. Spread over 900 options that's about 800 each on average — but real text is very uneven. A few common syllables dominate; rare ones appear a handful of times. A model can't learn something it has seen eight times.

And **the archaic letters ಱ and ೞ, which are exactly what marks a text as old Kannada, sit deep in that rare tail.** The model would quietly fail on precisely the characters the project is about, while training metrics looked healthy.

With 50–60 romanised symbols, the same 750,000 instances spread evenly and every symbol gets enough examples. Convert back to Kannada script at the very end; the conversion is exact.

### 3.2 Model size

**3–5 million settings.** One BiLSTM layer, fewer convolutional channels. A large model on a small dataset memorises instead of learning.

Rough shape: 5 convolutional blocks (32 → 256 channels, pooling in height only after block 3), one bidirectional LSTM with 256 hidden units, output layer over ~60 symbols plus blank. Input height 64, variable width.

### 3.3 Expected accuracy — set this expectation now

| Measured on | Expected CER |
|---|---|
| Synthetic test images | 8–15% |
| Real palm leaf | 40–70% |

The synthetic figure is what we report. The real figure explains why the demo on real pages looks rough, and it goes in the limitations section alongside the Part 6 measurements.

---

## Part 4 — The soft bridge

**Claude Code writes this.** Budget one hour to read it line by line before the viva — it's the one part you will definitely be asked to explain.

### 4.1 Why "top 5 per letter" doesn't exist

CTC doesn't give you letters. It gives one confidence distribution **per frame** — per vertical strip of the image. A 30-character line might produce 150 frames, most of them blank or repeats.

There are no clean letter positions until you collapse the frames. And collapsing is itself a hard decision — exactly the thing the bridge exists to avoid.

**So the bridge works frame by frame.** The modernizer reads a sequence of blended frames, not blended letters.

### 4.2 What it does, per frame

1. Take the recogniser's scores for that frame.
2. Apply the **temperature dial** (§4.3), then softmax into confidences.
3. Keep the **top 5 symbols, including blank** — blank gets its own embedding like any other symbol. It's real information: it says "nothing is here."
4. **Renormalise those 5 so they sum to 1.** The top 5 of a full distribution don't on their own.
5. Blend: multiply each symbol's embedding by its confidence and add them up.

```
blended frame = (0.55 × embedding of "k")
              + (0.30 × embedding of "K")
              + (0.10 × embedding of blank)
              + ...
```

6. **Pool every 2–4 frames** by averaging, to shorten the sequence. Worth doing — 150 frames per line is long and slow.
7. Feed the sequence into the modernizer's encoder.

**One optional step, and know how to defend it.** Most frames are overwhelmingly blank, which dilutes everything. You may drop frames where blank confidence exceeds about 0.95. This is defensible where collapsing isn't: you're removing padding, not choosing between competing letters. Have that distinction ready — it's what a sharp panelist would probe.

### 4.3 The temperature dial

**The single item most likely to decide whether the project produces a result at all.**

CTC recognisers are usually extremely overconfident. It's normal for the top choice to hold 98% or 99% of the confidence at most frames. If that happens, the blend is arithmetically almost identical to just taking the top choice — and B4 matches B3. Not because the idea is wrong, but because there was no uncertainty left to carry.

**Temperature spreads confidence out before blending.** One line of code. Tune it on validation like any other setting.

This is not the calibration chapter we cut. We report no calibration statistics. This is one hyperparameter that makes the mechanism work.

**Measure in Week 3, not Week 4:** what fraction of frames have top-1 confidence below 0.9? If it's tiny, raise the temperature. Finding this out in Week 3 is a tuning problem. In Week 4 it's a dead project.

### 4.4 Training with the recogniser unlocked

Our claim has two halves. The first — we blend rather than pick — is true however we train.

The second is stronger: **because the blend is smooth maths rather than a hard choice, learning signal from the modernizer can travel backwards into the recogniser and improve it.** That half is only true if they're trained together with the recogniser unlocked.

**Default:** a short joint phase with the recogniser unlocked, at about one-tenth the normal learning rate so it adjusts gently. A few hundred steps on S2.

**Emergency only:** if training becomes unstable — loss goes to NaN, or accuracy clearly drops — lower the learning rate, then shorten the phase, and only then lock the recogniser. Locking weakens the claim to "we pass a blended signal," which is easier to argue with.

### 4.5 Worked recovery examples

**One hour in Week 4. The highest-value hour in the project.**

Find 3–4 cases where the recogniser's top choice was **wrong**, the correct symbol was in its top 5, and the bridge produced the **right** answer.

Show each as a figure: the line image, the top-5 candidates with confidences, what the old approach produced, what ours produced.

A table asserts your idea works. These *show the mechanism working*, which for a panel that isn't grading research rigour is far more persuasive — and it pre-empts "is this really different from argmax?" before it's asked.

---

## Part 5 — The modernizer

### 5.1 What it is

A small encoder-decoder transformer: roughly 6 encoder layers and 6 decoder layers, hidden size 256, 4 attention heads — about 20 million settings. Input vocabulary is your ~60 WX symbols; output vocabulary is the same.

Pre-trained on S3 (the rule-generated pairs), then fine-tuned on S2 (the real pairs, with recogniser noise — see Part 7).

### 5.2 Why not ByT5-small

Earlier drafts specified ByT5-small, a 300-million-setting pretrained byte-level model. Two reasons it isn't the right call here:

**Its pretraining doesn't transfer.** ByT5 has seen Kannada *script*. You're working in WX romanisation. The Kannada knowledge it carries is attached to Kannada Unicode bytes, not Roman ones, so what you actually inherit is general sequence-modelling ability — worth something, but not the Kannada competence it appears to offer.

**It costs roughly ten times the training time** (Part 11: 4–8 hours versus 30–60 minutes) and pushes memory usage to the edge of the 3060.

At your budget, a small transformer trained from scratch on 100–300k pairs is the better trade. It also makes the bridge simpler — you own the embedding table, so blending is a direct lookup rather than plumbing into someone else's model.

**Keep ByT5-small as a stretch option.** If the small transformer's chrF++ is disappointing and you have a spare overnight slot in Week 3, try it. Don't start there.

---

## Part 6 — What HKHPL is actually used for

Reasonable question to ask, since the project is nominally about this dataset and the training path is synthetic. Four uses, and the fourth is worth real effort.

### 6.1 Appearance source for the damage simulator

Background textures and damaged-region patches in S1 and S2 are cut from real HKHPL leaves. Your synthetic images are **conditioned on HKHPL**, not invented. Say this explicitly — it's the difference between "we made up some noise" and "we modelled the appearance of the target collection."

### 6.2 The fake-versus-real check

The classifier that tries to tell synthetic lines from real ones needs real crops, which are HKHPL. The 70–80% target is a measurement of how close your synthetic set comes to **HKHPL specifically**.

### 6.3 The live demo

Real HKHPL pages run through Palmira, the recogniser, the bridge and the modernizer, end to end. Qualitative.

### 6.4 Real measurements without any transcriptions — do this

If a panelist asks "did you use the dataset your project is about?", "we took background patches from it" is thin. Here's how to get a genuine results section on real imagery with no ground truth at all.

Run the pipeline over ~50 real HKHPL pages and report three things:

| Measurement | How | Cost |
|---|---|---|
| **Line segmentation success rate** | Classify each page good / acceptable / broken by eye | Free — you're doing this survey anyway to curate demo pages |
| **Confidence distribution, real vs synthetic** | Mean and spread of the recogniser's top-1 confidence on real lines versus synthetic lines | One script, one afternoon |
| **Flagging rate, real vs synthetic** | What proportion of lines fall below your confidence threshold in each case | Reuses the Part 4 machinery directly |

The middle one is the good one. **If mean confidence drops sharply on real leaves, you have quantified the domain gap using no labels whatsoever.** That's a real measurement on real data, it's genuinely interesting, and it makes your synthetic-first choice look considered rather than evasive.

It also lets you write a "Results on real manuscript imagery" section that claims nothing you can't verify.

**Half a day in Week 4.** Slots in next to the confidence chart.

---

## Part 7 — Line finding with Palmira

Used exactly as downloaded. No fine-tuning, no benchmark table. Needed only for real pages — synthetic images are generated as single lines already.

### 7.1 Give it the original photo

**Do not feed Palmira the black-and-white U-Net output.** Palmira was trained on ordinary colour photographs. Black-and-white is input it has never seen, and it can degrade badly for no reason.

**Correct order:** original photo → Palmira finds lines → cut out each line → *then* convert to black-and-white if the recogniser wants it.

Check this in your first hour: run both ways on ten pages, compare visually.

### 7.2 Survey every page, then curate the demo set

A live demo doesn't fail on the average page. It fails on the one page the panel picks, where two lines merged or a line at the leaf edge was missed.

Once, in Week 1: run Palmira over every page, look through the outputs, mark them good / acceptable / broken (this also produces the §6.4 segmentation number), and set aside **10–15 clearly good pages** as demo inputs.

**Keep it honest.** Separate folders (`demo_pages/`, `test_split/`), and one sentence in the report:

> *The live demonstration uses hand-selected pages where line segmentation succeeded cleanly. All reported accuracy figures are computed on a synthetic test set frozen before development began. Real-page measurements are reported separately in §6.4 and do not depend on selection.*

---

## Part 8 — Evaluation

| ID | What it is | Why |
|---|---|---|
| **B0** | Copy the input unchanged | The floor. Old and modern Kannada share most words, so copying scores surprisingly well. Without this the other numbers mean nothing. |
| **B3** | Recogniser → take top symbol → modernizer | The conventional approach. |
| **B4** | Recogniser → soft bridge → modernizer | Ours. |

Optional **B1** (Tesseract) only if it takes under an hour.

### The fairness rule

**Both modernizers must be trained on realistic input.**

Run the recogniser once over the S2 images and save two things: top-1 text strings, and frame-level confidences. Then:

- **B3's modernizer** is fine-tuned on (noisy top-1 text → modern text)
- **B4's modernizer** is fine-tuned on (soft blended frames → modern text)

Same recogniser, same images, same training budget. Only the interface differs.

Without this, B3's modernizer trains on clean text and meets noisy text at test time while B4's gets what it expects — and a panelist can correctly say B4 only won because it was trained properly. One extra training run, and it's what makes the comparison mean anything.

### What to report

- Recognition: CER on the frozen synthetic test split (plus gold lines if any exist)
- Modernization: chrF++ primary, BLEU secondary, **both against B0**
- One chart: accuracy on flagged versus unflagged lines
- The 3–4 worked recovery examples
- The §6.4 real-imagery measurements
- POS accuracy on the ~100 hand-checked tags

**On BLEU:** the patent reports 0.81. Because old and modern Kannada overlap heavily, plain copying already scores high, so that number means much less than it appears to. That's why B0 is in the table. One sentence, then move on.

---

## Part 9 — Week by week

Roles: **A** recognition, **B** data, **C** modernization, **D** infrastructure and demo.

### Week 1 — foundations

| Task | Who |
|---|---|
| Repository, `CLAUDE.md`, environment | D |
| Build the 5-effect damage simulator | B |
| **Build S2: filter KannadaLit4NLP pairs, freeze test verse IDs, render** | B |
| Generate S1 (20–30k lines) | B |
| Half a day trying pretrained HTR checkpoints | A |
| Build the CRNN; prove it can memorise 8 examples | A |
| Palmira on original photos; compare against black-and-white on 10 pages | A |
| Palmira survey across all pages → demo set + segmentation success rate | A |
| Send gold-line enquiry emails, then carry on | D |

**Checkpoint:** S1 and S2 exist, test IDs frozen, CRNN passes the memorisation test.

### Week 2 — recogniser and modernizer

| Task | Who |
|---|---|
| Train the CRNN on S1 (overnight) | A |
| Fake-vs-real classifier check | B |
| Build the reverse-spelling rule engine, generate S3 | C |
| Pre-train the small transformer on S3 (overnight) | C |
| Measure CER on the synthetic test split | A |

**Checkpoint:** a recogniser that reads synthetic lines, and a modernizer that handles clean text.

### Week 3 — the bridge

| Task | Who |
|---|---|
| **Measure what fraction of frames have top-1 confidence below 0.9** — do this first | A |
| **Build the soft bridge** — frame-level, blank included, renormalised, temperature dial, pooling | A+C |
| Run the recogniser over S2; save top-1 strings and frame confidences | A |
| **Fine-tune both modernizers on their respective noisy inputs** | C |
| **Short joint training with the recogniser unlocked** | A+C |
| Confidence threshold and flagging | D |
| POS tagging on the output | B |

**End of Week 3: feature freeze.**

### Week 4 — results and writing

| Task | Who |
|---|---|
| Run B0, B3, B4 on the frozen test split | D |
| **Collect 3–4 worked recovery examples** — one hour, don't skip | A |
| **§6.4 real-imagery measurements** — half a day | A |
| Confidence chart | D |
| Build the demo on the curated real pages | D |
| Write the report | All |
| Rehearse the demo end to end, twice | All |
| Buffer | — |

---

## Part 10 — The viva

### Prior work

> *We reviewed the Bannigidad et al. patent, which is filed but not yet granted, along with their HDOCNET recogniser. Their system combines two existing OCR tools, Tesseract and Lipi Gnani, and runs recognition and modernization as separate sequential steps — the recogniser commits to a final character string before modernization begins. Our system differs at exactly that point. Rather than choosing a single character early, we carry the recogniser's confidence across all its candidates forward into the modernization step, and settle on a final answer only at the end. We also cite UniLipi, a multi-script recognition model published at ICDAR 2026, whose weights were not publicly available during our development window. Our contribution is not better raw recognition — it is what happens between recognition and modernization, which existing work treats as two independent stages with a hard commitment in between.*

### If asked why accuracy is below the patent's 93.7%

> *Those figures come from a patent specification, which is not peer-reviewed and states no dataset split or test protocol, so there's no comparable basis. We evaluate on our own frozen test set and report against a copy baseline so the reader can see the actual headroom.*

### If asked why the evaluation is synthetic

> *Our claim is about the interface between recognition and modernization, not about absolute reading accuracy. That's a controlled comparison — same recogniser, same images, same modernizer, same training budget, with only the interface differing. We do report measurements on real HKHPL imagery in section 6.4, including the recogniser's confidence gap between real and synthetic lines, which quantifies the domain gap without requiring transcriptions we were unable to obtain.*

Volunteer the limitation before you're asked. Conceding it under questioning reads differently.

---

## Part 11 — Training times on the RTX 3060 12GB

All estimates assume mixed precision (bf16), `channels_last` for the convolutional model, `torch.compile` where it works, and Linux or WSL2 rather than native Windows.

### The headline

| Job | Wall-clock | VRAM | When |
|---|---|---|---|
| Generate S1 (25k images) | 45–90 min (CPU only, 10 processes) | — | Week 1 |
| Generate S2 (~3k images) | under 10 min | — | Week 1 |
| Fake-vs-real classifier | 10–15 min | ~2 GB | Week 2 |
| **Train the CRNN on S1** | **2–4 hours** | ~3–4 GB | Week 2, overnight |
| **Pre-train the modernizer on S3** | **30–60 min** | ~3 GB | Week 2 |
| Run recogniser over S2, cache outputs | under 10 min | ~2 GB | Week 3 |
| Fine-tune B3 modernizer | 10–20 min | ~3 GB | Week 3 |
| Fine-tune B4 modernizer | 10–20 min | ~4 GB | Week 3 |
| **Joint unlocked training** | **20–40 min** | ~5–6 GB | Week 3 |
| Palmira inference, 500 pages | 5–10 min | ~4 GB | Week 1 |

**Total GPU time for one clean pass: roughly 4–6 hours.**

Realistically you'll retrain the CRNN two or three times and the bridge four or five times while tuning temperature and pooling. **Budget 12–20 GPU-hours across the month.**

**The GPU is not your constraint.** Two or three overnight sessions cover the entire project. Your constraint is the 80–130 person-hours. Plan the calendar around people, not the card.

### Where the CRNN estimate comes from

25,000 images per epoch. At roughly 150–250 images per second on a 3060 with mixed precision, that's about 2 minutes per epoch. CTC typically needs 60–100 epochs to converge on synthetic data, giving 2–3.5 hours. Add restarts and call it 2–4.

**The CPU is the tighter half of this machine.** A Ryzen 5 5500 gives you 6 cores and 12 threads. That's enough, but only if the dataloader isn't doing avoidable work. Three things, in order of how much they matter:

**1. Keep the whole dataset in RAM.** 25,000 grayscale lines at height 64 and average width ~600 is about 960 MB uncompressed. That fits in memory. Store S1 as a single memory-mapped `uint8` numpy array rather than 25,000 individual PNG files.

This eliminates disk reads and image decoding entirely, which together are usually the largest cost in a dataloader. Requires at least 16 GB system RAM.

**2. Do augmentation on the GPU, in batches.** Rotation, brightness, blur and stretch all work fine as batched tensor operations (`torch.nn.functional.affine_grid` and `grid_sample`, or `kornia`). Your model is small and the 3060 has ample spare capacity, so moving augmentation there costs almost nothing and removes the six-core constraint completely.

This is the single most useful change for this CPU. If you only do one thing from this list, do this one.

**3. If you keep augmentation on the CPU, use OpenCV rather than PIL** — typically two to five times faster for these operations. Set `num_workers=5`, `persistent_workers=True`, `pin_memory=True`, `prefetch_factor=4`.

Even naive CPU augmentation across five workers should manage 400–800 images per second, comfortably above the 150–250 the GPU needs. The thing that would break it is expensive per-batch warping — which is why **page warp, texture and holes are baked in at generation time** (§2.2). Thin-plate-spline warping on every batch would halve your throughput on this CPU.

**Check `nvidia-smi` in the first ten minutes of any run.** Below 70% utilisation means you're CPU-bound. Fix the dataloader before touching the model.

**Don't generate data and train at the same time.** With six cores, a generation job will starve the training dataloader. Run them in sequence.

**PCIe 3.0 is not a problem here.** The Ryzen 5 5500 runs the 3060 at PCIe 3.0 rather than 4.0, but a batch of 64 grayscale line images is around 8 MB in half precision — about a millisecond of transfer. Irrelevant at this scale.

### Why the modernizer is fast

A 20-million-setting transformer on 100–300k short sequences is a small job. At batch 64 and sequence length 128 you'll see roughly 20–40 steps per second, so 5 epochs over 200k pairs is around 15,000 steps — 10 to 20 minutes of compute. Budget 30–60 minutes with warm-up and checkpointing.

**ByT5-small, for comparison:** 300 million settings, byte-level sequences that run longer, batch 8 with gradient checkpointing and 8-bit Adam, roughly 8–10 GB of VRAM, and **4–8 hours** for the same job. That's the ten-fold difference cited in §5.2.

### Memory notes

Nothing here comes close to filling 12 GB except ByT5. If you hit an out-of-memory error:

- Joint training is the tightest stage, holding the recogniser, bridge and modernizer at once (~5–6 GB). If it does OOM, cache the recogniser's outputs to disk and train the bridge and modernizer alone — that drops to about 3 GB. Only the short unlocked phase needs everything resident.
- Reduce batch size before reducing model size. A smaller batch with gradient accumulation gives the same result more slowly.

### Practical

- **Use WSL2 or Linux.** Windows dataloader workers are dramatically slower, and you are likely to be CPU-bound.
- **Log GPU temperature on long runs.** A 3060 in a small case over three hours will thermal-throttle, and you'll blame the model.
- **One GPU, four people.** Keep a run queue — a text file of commands and a `runner.sh` loop in `tmux` — so the card isn't idle overnight. Person A owns the schedule.
- Develop against 500 images; run the full set only when the code is proven.

---

## Part 12 — Risks

| Risk | Warning sign | Response |
|---|---|---|
| **Confidences too extreme, bridge shows no effect** | Nearly all top-1 confidences above 0.9 | Raise the temperature. Measured in Week 3 for this reason. |
| **Recogniser CER too high even on synthetic** | Above 25% at Week 2 | More epochs first; then simplify the damage simulator (drop warp); then shorten line length. Don't enlarge the model. |
| **CPU-bound training** | GPU utilisation under 70% | More dataloader workers; move expensive augmentation to generation time. |
| **S2 too small after filtering** | Under ~1,000 usable pairs | Loosen the length filter to 0.5–2.0; supplement with rule-generated pairs for *training only*, never the test split. |
| **Joint unlocked training unstable** | NaN loss or falling accuracy | Lower learning rate, then shorten the phase, then lock as a last resort. |
| **Palmira breaks on many pages** | Widespread merged or missed lines | Try different anchor aspect ratios at inference — palm-leaf lines are far longer and thinner than its defaults assume. No retraining needed. |
| **No gold lines at all** | No reader found by Week 2 | Proceed. Report synthetic results plus the §6.4 real-imagery measurements. Planned for, not a failure. |
| **A team member unavailable** | Any week | Protect roles A and D. Drop optional Tesseract first, POS verification depth second. |

---

## Part 13 — Day one checklist

- [ ] Create the repository, add all four members
- [ ] Write `CLAUDE.md`
- [ ] Set up the environment on the 3060 machine — WSL2 or Linux, verify `torch.cuda.is_available()`
- [ ] Download KannadaLit4NLP; count how many pairs survive the length filter
- [ ] **Choose and freeze the test verse IDs, commit the list**
- [ ] Run Palmira on ten pages, original photo versus black-and-white, compare
- [ ] Send the gold-line enquiry emails — then carry on without waiting

Building S2 is the critical path. Start there.
