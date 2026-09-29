"""One-off script (scratch, not src/setu/ pipeline code): turns the
downloaded KannadaLit4NLP_master.jsonl into the two flat text corpora
generate.py expects, and freezes the S2 test-verse-ID split per CLAUDE.md
rules 1-2 BEFORE any S2 rendering happens.

S1 corpus: every verse's raw text (old-Kannada only), one per line.
S2 corpus: verse_id<TAB>old_text<TAB>modern_text for the 9,597 verses that
have at least one scholarly interpretation (modern_text = the SHORTEST
interpretation for that verse, picked to stay closest to a line-level
modernization target rather than a long free-form commentary -- still an
unedited, real scholarly interpretation per CLAUDE.md rule 9, just the
shortest one on offer when there's a choice).

Split freeze: verse_id -> train/test via sha256(f"s2:{verse_id}"), first
byte < threshold => test (~15%). Deterministic, independent of set growth
order, recorded in an append-only data/splits/manifest.jsonl entry.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path("D:/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/src")))
from setu.data.wx import decode as wx_decode  # noqa: E402
from setu.data.wx import encode as wx_encode  # noqa: E402


def wx_round_trips(text: str) -> bool:
    """encode() succeeding isn't enough: a bare consonant (virama-terminated)
    immediately followed by an independent vowel letter is legal Unicode but
    is indistinguishable, in WX symbols, from consonant+matra -- decode()
    always reconstructs the matra form. This is a known, very rare (~1 in
    25k lines in this corpus) limitation of the WX scheme's symbol set, not
    something to fix by extending the vocabulary here; lines that hit it are
    dropped instead."""
    try:
        symbols = wx_encode(text)
    except ValueError:
        return False
    return wx_decode(symbols) == text

# KannadaLit4NLP is a modern critical-edition transcription: it carries
# Western-typeset punctuation (commas, periods, quotes, brackets) that a
# hand-inscribed old-Kannada palm leaf would never have, plus an ASCII '|'
# used informally in place of the real danda/double-danda verse markers.
# Normalizing here -- rather than extending setu.data.wx's symbol table --
# keeps the WX vocabulary exactly as CLAUDE.md defines it (extending it is
# an explicit "ask before" item); this is corpus cleaning, not a vocabulary
# change.
_ZW_CHARS = "\u200b\u200c\u200d\ufeff"
_STRIP_PUNCT = re.compile(
    r"[,.;:?!\-\u2013\u2014\[\]()`\"'\u2018\u2019\u201c\u201d\u00a0\u02bb\u02bc]|[a-zA-Z0-9]"
)


def normalize_for_wx(text: str) -> str:
    text = text.replace("||", "॥").replace("|", "।")
    for zw in _ZW_CHARS:
        text = text.replace(zw, "")
    text = _STRIP_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

ROOT = Path("D:/major_proj/Palm_leaf_halekannada_to_hosakannada_translation")
MASTER = ROOT / "data/raw_corpus/extracted/KannadaLit4NLP/KannadaLit4NLP_master.jsonl"
S1_CORPUS = ROOT / "data/raw_corpus/s1_corpus.txt"
S2_CORPUS = ROOT / "data/raw_corpus/s2_corpus.txt"  # id<TAB>old<TAB>modern, id stripped before generate.py sees it
S2_CORPUS_FOR_GENERATE = ROOT / "data/raw_corpus/s2_corpus_for_generate.txt"  # old<TAB>modern only, --paired input
TEST_IDS_PATH = ROOT / "data/splits/s2_test_verse_ids.txt"
SPLITS_MANIFEST = ROOT / "data/splits/manifest.jsonl"

TEST_FRACTION_THRESHOLD = 38  # out of 256 -> ~14.8% test

# A KannadaLit4NLP "verse" is a content unit, not a physical manuscript-line
# unit -- some Vachanas run to 8,500+ characters, which generate.py renders
# as a single, ever-longer image line (measured up to ~36px/char once page
# warp/skew are applied), and the rendering pipeline itself runs out of
# memory well before that (observed: numpy tried to allocate 556MiB for one
# line's warp array). 700 chars (~p95 of S1's length distribution, keeping
# ~95% of lines) keeps worst-case rendered width in a safely renderable
# range and also reads more like a plausible single manuscript line than a
# multi-sentence paragraph.
MAX_OLD_TEXT_CHARS = 700

records = [json.loads(line) for line in MASTER.open(encoding="utf-8")]
print(f"Loaded {len(records)} verses from KannadaLit4NLP_master.jsonl")

# --- S1: every verse's old-Kannada text, one per line ---
s1_lines = []
n_s1_dropped_wx = 0
for r in records:
    text = normalize_for_wx(r["verse"].strip().replace("\n", " ").replace("\t", " "))
    if not text or len(text) > MAX_OLD_TEXT_CHARS:
        continue
    if not wx_round_trips(text):
        n_s1_dropped_wx += 1
        continue
    s1_lines.append(text)
S1_CORPUS.write_text("\n".join(s1_lines) + "\n", encoding="utf-8")
print(f"S1 corpus: {len(s1_lines)} lines -> {S1_CORPUS} ({n_s1_dropped_wx} dropped: still fail WX encode after normalization)")

# --- S2: verses with >=1 interpretation, modern_text = shortest interpretation ---
s2_records = []
n_s2_dropped_wx = 0
for r in records:
    interps = [r[k] for k in r if k.startswith("interpretation")]
    if not interps:
        continue
    old_text = normalize_for_wx(r["verse"].strip().replace("\n", " ").replace("\t", " "))
    # modern_text is the modernizer's target: real modern Kannada, which DOES use
    # ordinary punctuation -- only whitespace is cleaned, nothing is stripped.
    modern_text = re.sub(
        r"\s+", " ", min(interps, key=lambda s: len(s.split())).strip().replace("\n", " ").replace("\t", " ")
    ).strip()
    if not old_text or not modern_text or len(old_text) > MAX_OLD_TEXT_CHARS:
        continue
    if not wx_round_trips(old_text):  # modern_text is the modernizer's target -- no WX requirement on it
        n_s2_dropped_wx += 1
        continue
    s2_records.append({"verse_id": r["id"], "old_text": old_text, "modern_text": modern_text})
print(f"S2 candidates (verses with >=1 interpretation): {len(s2_records)} ({n_s2_dropped_wx} dropped: old_text still fails WX encode after normalization)")

# --- Freeze the test split BEFORE writing any S2 corpus / rendering (rules 1-2) ---
test_ids = []
for rec in s2_records:
    digest = hashlib.sha256(f"s2:{rec['verse_id']}".encode("utf-8")).digest()
    if digest[0] < TEST_FRACTION_THRESHOLD:
        test_ids.append(rec["verse_id"])
test_id_set = set(test_ids)
test_ids_sorted = sorted(test_ids)

TEST_IDS_PATH.write_text("\n".join(str(i) for i in test_ids_sorted) + "\n", encoding="utf-8")
print(f"Frozen S2 test split: {len(test_ids_sorted)}/{len(s2_records)} verse IDs ({len(test_ids_sorted)/len(s2_records):.1%}) -> {TEST_IDS_PATH}")

git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
manifest_entry = {
    "event": "s2_test_split_freeze",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit,
    "method": "sha256(f's2:{verse_id}')[0] < 38  (out of 256, ~14.8% target)",
    "corpus_source": "KannadaLit4NLP master.jsonl, Mendeley Data DOI 10.17632/nvjydxpxjr.3, sha256 013245506e09a9f51c9102faa8f0db42410db888500605dd5c233df0dbeae1b1",
    "n_s2_candidates": len(s2_records),
    "n_test": len(test_ids_sorted),
    "test_fraction_actual": len(test_ids_sorted) / len(s2_records),
    "note": "Frozen before any S2 image rendering. Per CLAUDE.md rule 1/2: never rewrite this split; "
            "if the S2 candidate pool grows later (e.g. more KannadaLit4NLP verses gain interpretations "
            "in a future dataset version), new IDs get assigned by the same hash rule and appended -- "
            "existing test IDs here are never reassigned.",
}
with SPLITS_MANIFEST.open("a", encoding="utf-8") as f:
    f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
print(f"Appended freeze record to {SPLITS_MANIFEST}")

# --- Now write the S2 corpus for generate.py (old<TAB>modern, --paired) ---
# generate.py's manifest doesn't carry verse_id, so we also keep an id-carrying
# sidecar (S2_CORPUS) for traceability between verse_id and the rendered line_NNNNNN.
with S2_CORPUS.open("w", encoding="utf-8") as f_full, S2_CORPUS_FOR_GENERATE.open("w", encoding="utf-8") as f_gen:
    for rec in s2_records:
        f_full.write(f"{rec['verse_id']}\t{rec['old_text']}\t{rec['modern_text']}\n")
        f_gen.write(f"{rec['old_text']}\t{rec['modern_text']}\n")
print(f"S2 corpus (with verse_id): {S2_CORPUS}")
print(f"S2 corpus (for generate.py --paired): {S2_CORPUS_FOR_GENERATE}")

print("\nDone. S1 rows == S2 rows order is NOT test-split-filtered -- generate.py renders everything; "
      "the frozen test IDs mark which S2 lines are held out for scoring, not which get rendered.")
