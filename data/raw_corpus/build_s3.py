"""One-off script (scratch, not src/setu/ pipeline code): builds S3 --
"old/modern text pairs, no images, rule-generated" for pre-training the
modernizer (CLAUDE.md data table, target 100-300k pairs).

Source of real modern-Kannada text: every interpretationN field across all
24,746 KannadaLit4NLP records (not just the shortest one per verse, and not
restricted to the 9,517 S2 candidates -- S3's modern side has no
"real scholarly source" restriction beyond being real text, per CLAUDE.md
rule 9, which only constrains S2's modern side), split into sentence-like
units. Old side: setu.modernizer.reverse_spelling applied to each real
sentence -- never the other way around, and never scored against (S3 is
pretraining-only).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from setu.data.wx import decode as wx_decode  # noqa: E402
from setu.data.wx import encode as wx_encode  # noqa: E402
from setu.modernizer.reverse_spelling import ReverseSpellingConfig, old_style_text  # noqa: E402

MASTER = ROOT / "data/raw_corpus/extracted/KannadaLit4NLP/KannadaLit4NLP_master.jsonl"
S3_CORPUS = ROOT / "data/raw_corpus/s3_corpus_for_generate.txt"  # old<TAB>modern, WX-normalized old side
SEED = 0
MIN_WORDS, MAX_WORDS = 4, 35

_ZW_CHARS = "\u200b\u200c\u200d\ufeff"
_STRIP_PUNCT = re.compile(
    r"[,.;:?!\-\u2013\u2014\[\]()`\"'\u2018\u2019\u201c\u201d\u00a0\u02bb\u02bc]|[a-zA-Z0-9]"
)
_SENTENCE_SPLIT = re.compile(r"[।॥.!?\n]|\|\|?")


def normalize_for_wx(text: str) -> str:
    text = text.replace("||", "॥").replace("|", "।")
    for zw in _ZW_CHARS:
        text = text.replace(zw, "")
    text = _STRIP_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wx_round_trips(text: str) -> bool:
    try:
        symbols = wx_encode(text)
    except ValueError:
        return False
    return wx_decode(symbols) == text


records = [json.loads(line) for line in MASTER.open(encoding="utf-8")]

raw_sentences: set[str] = set()
for r in records:
    for key in r:
        if not key.startswith("interpretation"):
            continue
        for chunk in _SENTENCE_SPLIT.split(r[key]):
            chunk = chunk.strip()
            if not chunk:
                continue
            n_words = len(chunk.split())
            if MIN_WORDS <= n_words <= MAX_WORDS:
                raw_sentences.add(chunk)

print(f"Extracted {len(raw_sentences)} unique candidate sentences from interpretation fields")

rng = np.random.default_rng(SEED)
cfg = ReverseSpellingConfig(seed=SEED)

pairs = []
n_dropped_wx = 0
for modern_raw in sorted(raw_sentences):  # sorted for determinism given the seeded rng draws below
    modern_text = re.sub(r"\s+", " ", modern_raw).strip()
    old_candidate = normalize_for_wx(modern_raw)
    if not old_candidate:
        continue
    old_text = old_style_text(old_candidate, cfg, rng)
    if not wx_round_trips(old_text):
        n_dropped_wx += 1
        continue
    pairs.append((old_text, modern_text))

print(f"S3 pairs: {len(pairs)} ({n_dropped_wx} dropped: rule-generated old side failed WX round-trip)")

with S3_CORPUS.open("w", encoding="utf-8") as f:
    for old_text, modern_text in pairs:
        f.write(f"{old_text}\t{modern_text}\n")
print(f"Wrote {S3_CORPUS}")

git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
run_dir = ROOT / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_s3_corpus_build"
run_dir.mkdir(parents=True)
(run_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
config = {
    "task": "Build S3 (rule-generated old/modern pairs for modernizer pretraining)",
    "modern_text_source": "every interpretationN field across all 24746 KannadaLit4NLP records, "
                           "sentence-split, deduped, filtered to 4-35 words",
    "old_text_method": "setu.modernizer.reverse_spelling.old_style_text: deterministic anusvara-to-"
                        "nasal rule + probabilistic ra->rYa and la->zYa (intervocalic only), "
                        "rate 0.15 each, seed 0",
    "seed": SEED,
    "min_words": MIN_WORDS,
    "max_words": MAX_WORDS,
    "n_candidate_sentences": len(raw_sentences),
    "n_pairs_written": len(pairs),
    "n_dropped_wx_roundtrip": n_dropped_wx,
    "target_from_roadmap": "100-300k",
    "note": "Below the 100-300k roadmap target because it draws only from KannadaLit4NLP's own "
            "interpretation text (~2.4M tokens total), not yet a larger modern-Kannada corpus "
            "(e.g. Kannada Wikipedia). Sufficient to start modernizer pretraining; can be scaled "
            "up later with an additional modern-Kannada source if time permits.",
}
(run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
(run_dir / "results.json").write_text(
    json.dumps({"n_pairs": len(pairs), "n_candidate_sentences": len(raw_sentences)}, indent=2), encoding="utf-8"
)
print(f"Run recorded at {run_dir}")
