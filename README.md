# SETU

Reading old Kannada palm-leaf manuscripts and rewriting them in modern Kannada.

See [SETU_Roadmap_v2.md](SETU_Roadmap_v2.md) for the plan and [CLAUDE.md](CLAUDE.md)
for the project rules that Claude Code (and everyone else) must follow when
working in this repository.

## Layout

- `src/setu/data/` — dataset loading, damage simulator, transcription tool
- `src/setu/models/` — recogniser (Track A/B) and modernization model
- `src/setu/bridge/` — the soft bridge (hand-written, not AI-generated — see CLAUDE.md)
- `src/setu/eval/` — `metrics.py`, the single source of truth for CER/chrF++/BLEU
- `src/setu/demo/` — live demo interface
- `data/gold/` — hand-transcribed manuscript lines
- `data/splits/manifest.jsonl` — append-only record of test/adaptation split assignment (by hash, see CLAUDE.md)
- `data/synthetic/` — rendered + damaged synthetic line images
- `data/text_pairs/` — old/modern Kannada text pairs (real + rule-based synthetic)
- `demo_pages/` — curated pages for the live demo only, never used for reported numbers
- `test_split/` — frozen test set
- `runs/` — one folder per experiment run (config, results, git hash)
