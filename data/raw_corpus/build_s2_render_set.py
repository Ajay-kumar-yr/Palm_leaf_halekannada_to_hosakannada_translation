"""One-off script (scratch, not src/setu/ pipeline code): subsamples the
full S2 candidate pool (9,386 verses with interpretations) down to the
roadmap's target rendered size (2-4k; targeting ~3,500) for actual
rendering. The full pool was needed to freeze a robust test split (rule 1
of CLAUDE.md: freeze before rendering), but CLAUDE.md's data table caps
S2 itself at 2-4k -- rendering all 9,386 would both overshoot that and
take ~4x longer than necessary.

ALL frozen test verse IDs are included (never dropped) -- the remaining
budget is filled from the train pool, seeded for reproducibility. Writes:
- s2_corpus_render.txt: old_text<TAB>modern_text, for generate.py --paired,
  in the exact row order that becomes line_000000, line_000001, ...
- s2_corpus_render_meta.jsonl: verse_id + is_test per row, SAME order, so
  rendered lines can be traced back to the frozen split membership.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S2_CORPUS = ROOT / "data/raw_corpus/s2_corpus.txt"  # verse_id\told_text\tmodern_text
TEST_IDS_PATH = ROOT / "data/splits/s2_test_verse_ids.txt"
OUT_RENDER = ROOT / "data/raw_corpus/s2_corpus_render.txt"
OUT_META = ROOT / "data/raw_corpus/s2_corpus_render_meta.jsonl"

TARGET_TOTAL = 3500
SEED = 0

test_ids = {int(line) for line in TEST_IDS_PATH.open(encoding="utf-8") if line.strip()}

records = []
for line in S2_CORPUS.open(encoding="utf-8"):
    verse_id_s, old_text, modern_text = line.rstrip("\n").split("\t")
    records.append({"verse_id": int(verse_id_s), "old_text": old_text, "modern_text": modern_text})

test_records = [r for r in records if r["verse_id"] in test_ids]
train_pool = [r for r in records if r["verse_id"] not in test_ids]
assert len(test_records) == len(test_ids), f"{len(test_records)} vs {len(test_ids)} -- a frozen test ID is missing from the corpus"

n_train_wanted = max(TARGET_TOTAL - len(test_records), 0)
rng = np.random.default_rng(SEED)
train_sample_idx = rng.choice(len(train_pool), size=min(n_train_wanted, len(train_pool)), replace=False)
train_sample = [train_pool[i] for i in sorted(train_sample_idx)]

selected = test_records + train_sample  # test rows first, then train -- order otherwise irrelevant
print(f"Test: {len(test_records)} (all frozen test IDs). Train sample: {len(train_sample)}. Total: {len(selected)}")

with OUT_RENDER.open("w", encoding="utf-8") as f_render, OUT_META.open("w", encoding="utf-8") as f_meta:
    for r in selected:
        f_render.write(f"{r['old_text']}\t{r['modern_text']}\n")
        f_meta.write(json.dumps({"verse_id": r["verse_id"], "is_test": r["verse_id"] in test_ids}, ensure_ascii=False) + "\n")

print(f"Wrote {OUT_RENDER} and {OUT_META}")
