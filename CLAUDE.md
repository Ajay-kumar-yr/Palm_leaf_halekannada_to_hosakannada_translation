# SETU — project rules

## Non-negotiable
1. NEVER write to data/splits/test.* — the test set is frozen.
2. Split assignment is by hash, not by date. Each gold line's split (test vs
   adaptation) is decided once, at creation time, by hashing its line ID
   (e.g. `sha256(line_id) % 100 < 20` → test). This makes the split stable
   as new lines are collected continuously — do NOT freeze a fixed checksum
   on day one and then add more lines to the same file afterward. Record
   the assignment in an append-only manifest (`data/splits/manifest.jsonl`)
   so it's auditable, and record a checksum of that manifest per run.
3. Before any training run longer than 10 minutes, prove the model can
   memorise 8 examples down to near-zero error. If it can't, something is
   broken. Do not start the real run.
4. Every run writes runs/<timestamp>/ with its config, its results, and the
   git commit hash. A result without a run folder does not exist.
5. Metrics are computed ONLY by src/setu/eval/metrics.py. Never inline,
   never invented.
6. Set random seeds in every entry point.
7. Never quietly change a setting to make a run work. Fail loudly.
8. demo_pages/ is for the live demo only. It never appears in any
   reported number.

## Facts about this project
- The recogniser outputs romanised (WX) symbols, roughly 50-60 of them.
  Not Kannada syllables.
- Palmira receives the ORIGINAL photo. Black-and-white conversion happens
  after line segmentation, not before.
- The soft bridge has a temperature parameter. It is tuned, not fixed at 1.
- Line images are height 64, variable width.
- The recogniser is CTC-based. CTC produces one distribution per image
  FRAME, not one per letter position — most frames are blank or repeats.
  There is no clean "top-5 candidates per letter" until frames are
  collapsed, and collapsing is itself a hard decision. The soft bridge
  blends PER FRAME (blank gets its own embedding in the candidate set),
  with optional pooling of a few frames to shorten the sequence before it
  reaches the modernizer. Do not design the bridge assuming per-letter
  positions exist for free.
- There is no dataset that pairs a manuscript-line image with a MODERN
  Kannada reference. Real (old-text, modern-text) pairs come from
  KannadaLit4NLP with no images; gold lines have images with OLD-text
  transcriptions only. To do end-to-end (image -> modern text) joint
  training or evaluation, render the old-text side of the real pairs
  through the damage simulator to synthesize the missing images, and say
  so explicitly wherever those results are reported. Gold-line results
  are CER only unless a modern-Kannada reference is separately collected
  for those lines.
- The modernizer must be trained on the recogniser's actual (noisy) output
  saved to disk, not on clean one-hot text pairs, for BOTH B3 and B4 — one
  training procedure so the two conditions differ only in the bridge.

## Ask before
Changing the loss function, changing the vocabulary, touching data/splits/,
or adding a dependency.
