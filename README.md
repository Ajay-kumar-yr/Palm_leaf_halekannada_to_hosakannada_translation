# SETU

Reading old Kannada palm-leaf manuscripts and rewriting them in modern Kannada.

See [SETU_Roadmap_v4.md](SETU_Roadmap_v4.md) for the plan and [CLAUDE.md](CLAUDE.md)
for the condensed project rules that Claude Code (and everyone else) must
follow when working in this repository. Track A (UniLipi) is dropped as of
v4 — Track B, our own small CRNN, is the only recogniser.

## Layout

- `src/setu/render/` — font rendering + the 5-effect damage simulator that produces S1 and S2 images
- `src/setu/recogniser/` — the CRNN (CTC, romanised WX output, ~50–60 symbols + blank)
- `src/setu/bridge/` — the soft bridge (hand-written, not AI-generated — see CLAUDE.md)
- `src/setu/modernizer/` — the small from-scratch transformer, old WX text → modern WX text
- `src/setu/data/` — dataset loading/ingestion (KannadaLit4NLP, gold-line transcription tool)
- `src/setu/eval/` — `metrics.py`, the single source of truth for CER/chrF++/BLEU
- `src/setu/demo/` — live demo interface
- `data/s1/` — rendered old-Kannada line images + old text, for training the recogniser
- `data/s2/` — rendered image + old text + **modern text** pairs (from KannadaLit4NLP); backbone for joint training and all reported end-to-end numbers
- `data/s3/` — old/modern text pairs with no images, rule-generated, for pre-training the modernizer
- `data/real/` — real HKHPL page photos, no transcriptions (demo + label-free measurements, see CLAUDE.md Part 6)
- `data/gold/` — real image + old-text transcriptions, best effort, never blocking
- `data/splits/manifest.jsonl` — append-only record of gold-line test/adaptation split assignment (by hash, see CLAUDE.md)
- `data/splits/s2_test_verse_ids.txt` — frozen S2 test verse IDs, committed before rendering
- `demo_pages/` — curated real pages for the live demo only, never used for reported numbers
- `test_split/` — frozen synthetic test set
- `runs/` — one folder per experiment run (config, results, git hash)
