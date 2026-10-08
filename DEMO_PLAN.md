# SETU — 4-day demo plan (decided 2026-10-08)

**Read this first.** It supersedes `STATUS.md` §4.2 wherever they
disagree. Like `STATUS.md` it is session state, not a rules document;
`CLAUDE.md`'s rules still apply to everything below.

**Deadline: 2026-10-12. Feature freeze: end of Day 3 (2026-10-11).**

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

## 6. Day by day

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
