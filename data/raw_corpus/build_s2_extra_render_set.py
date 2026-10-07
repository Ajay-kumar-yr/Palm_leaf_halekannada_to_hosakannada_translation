"""One-off script (scratch, not src/setu/ pipeline code): builds a SECOND
S2 render set from the in-band verses that were never rendered.

Why: the reported B0/B3/B4 comparison put both learned systems ~6 chrF++
BELOW simply copying the input. Inspecting the generations showed the
modernizer had learned the target genre's surface form (it emits the
corpus's "ಅರ್ಥ:"/"ಭಾವಾರ್ಥ:" commentary markers correctly) but not the
content mapping, filling the rest with high-frequency filler. 651
training examples cannot teach old->modern Kannada. Meanwhile
B0_oracle beat B0_recognised by only 0.30 chrF++, so the recogniser is
nowhere near the bottleneck -- the modernizer's data is.

build_s2_render_set.py subsampled 3,500 of the 9,386 S2 candidates long
before anyone knew that ~66% of S2's targets are free-form commentary
rather than modernizations. Of the 3,261 verses that ARE in band, only
1,186 happen to have been rendered. This renders the other 2,075, which
takes the modernizer's in-band training pool from ~721 to ~2,400 lines.

Rule 1 is untouched. The test split was frozen by verse ID in
data/splits/s2_test_verse_ids.txt BEFORE any rendering, so a newly
rendered verse simply lands on whichever side its frozen hash already
assigned it to. Nothing is reassigned and nothing is rewritten -- this is
exactly the growth case CLAUDE.md rule 2 anticipates ("if the S2
candidate pool grows later ... new IDs get assigned by the same hash rule
and appended -- existing test IDs here are never reassigned").

Writes, in matching row order (the positional join downstream depends on
it, same contract as build_s2_render_set.py):
  s2_extra_corpus_render.txt        old_text<TAB>modern_text, for --paired
  s2_extra_corpus_render_meta.jsonl verse_id + is_test per row

Usage:
    python data/raw_corpus/build_s2_extra_render_set.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S2_CORPUS = ROOT / "data/raw_corpus/s2_corpus.txt"
ALREADY_META = ROOT / "data/raw_corpus/s2_corpus_render_meta.jsonl"
INBAND_PATH = ROOT / "data/splits/s2_inband_verse_ids.txt"
TEST_IDS_PATH = ROOT / "data/splits/s2_test_verse_ids.txt"
OUT_RENDER = ROOT / "data/raw_corpus/s2_extra_corpus_render.txt"
OUT_META = ROOT / "data/raw_corpus/s2_extra_corpus_render_meta.jsonl"


def main() -> None:
    inband = {int(x) for x in INBAND_PATH.read_text(encoding="utf-8").split()}
    frozen_test = {int(x) for x in TEST_IDS_PATH.read_text(encoding="utf-8").split()}
    with ALREADY_META.open(encoding="utf-8") as f:
        already = {json.loads(l)["verse_id"] for l in f}

    rows = []
    for line in S2_CORPUS.open(encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        verse_id, old_text, modern_text = int(parts[0]), parts[1], parts[2]
        if verse_id not in inband or verse_id in already:
            continue
        if not old_text or not modern_text:
            continue
        rows.append((verse_id, old_text, modern_text))

    # Deterministic order: by verse_id. No sampling -- we want all of them.
    rows.sort(key=lambda r: r[0])

    with OUT_RENDER.open("w", encoding="utf-8") as fr, OUT_META.open("w", encoding="utf-8") as fm:
        for verse_id, old_text, modern_text in rows:
            fr.write(f"{old_text}\t{modern_text}\n")
            fm.write(json.dumps({"verse_id": verse_id, "is_test": verse_id in frozen_test},
                                ensure_ascii=False) + "\n")

    n_test = sum(1 for r in rows if r[0] in frozen_test)
    print(f"in-band verses total:        {len(inband)}")
    print(f"already rendered (data/s2):  {len(already & inband)}")
    print(f"NEW to render:               {len(rows)}  ({n_test} frozen test, {len(rows)-n_test} train)")
    print(f"  -> {OUT_RENDER}")
    print(f"  -> {OUT_META}")


if __name__ == "__main__":
    main()
