"""One-off script (scratch, not src/setu/ pipeline code): defines the
"in-band" evaluation subset of S2 -- the verses whose scholarly
interpretation is actually a line-level modernization rather than
free-form commentary.

Roadmap v4 §2.1 step 2 specifies keeping entries "where the interpretation
runs 0.7x-1.5x the length of the original -- those tend to be direct
rewrites rather than loose commentary." That filter was never implemented
in build_corpora.py (which only caps old_text at MAX_OLD_TEXT_CHARS and
picks the shortest interpretation per verse), so S2 as rendered pairs most
of its verses with commentary: measured over the 3,500 rendered lines the
modern/old length ratio has median 1.77x, p90 11.07x and max 32.6x, with
targets up to 5,284 characters that discuss neighbouring verses ("in the
previous five kaggas...") and address the reader directly ("O readers").

Why this matters: chrF++/BLEU against a 4,000-character commentary
measures whether the model wrote an essay, not whether it modernized the
line. It would also collapse B0 (copy the input unchanged), which CLAUDE.md
relies on as the reported floor, and the resulting noise is liable to swamp
the B3-vs-B4 difference that is the project's one novel contribution.

CLAUDE.md rule 1 forbids rewriting a frozen test split, and this script does
NOT: data/splits/s2_test_verse_ids.txt is left untouched and remains the
frozen split of record. This writes a SEPARATE, additional list marking
which verses pair with a genuine modernization, so end-to-end metrics can be
reported on that subset -- disclosed explicitly in the report -- while the
frozen split itself is never rewritten.

Usage:
    python data/raw_corpus/build_s2_inband_subset.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S2_CORPUS = ROOT / "data/raw_corpus/s2_corpus.txt"  # verse_id\told_text\tmodern_text
RENDER_META = ROOT / "data/raw_corpus/s2_corpus_render_meta.jsonl"  # verse_id + is_test per rendered row
TEST_IDS_PATH = ROOT / "data/splits/s2_test_verse_ids.txt"
OUT_INBAND = ROOT / "data/splits/s2_inband_verse_ids.txt"
SPLIT_MANIFEST = ROOT / "data/splits/manifest.jsonl"

# Roadmap v4 §2.1 step 2, verbatim: 0.7x-1.5x of the original's length.
RATIO_MIN = 0.7
RATIO_MAX = 1.5


def _git_commit_hash() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    inband_ids: list[int] = []
    n_total = 0
    for line in S2_CORPUS.open(encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        verse_id, old_text, modern_text = int(parts[0]), parts[1], parts[2]
        n_total += 1
        if not old_text:
            continue
        ratio = len(modern_text) / len(old_text)
        if RATIO_MIN <= ratio <= RATIO_MAX:
            inband_ids.append(verse_id)

    inband_set = set(inband_ids)
    OUT_INBAND.write_text("\n".join(str(i) for i in sorted(inband_ids)) + "\n", encoding="utf-8")

    # Report the intersection that actually matters operationally: of the
    # 3,500 verses that were rendered, how many are in band, split by frozen
    # test membership.
    frozen_test = {int(x) for x in TEST_IDS_PATH.read_text(encoding="utf-8").split()}
    rendered = [json.loads(l) for l in RENDER_META.open(encoding="utf-8")]
    r_test = [r for r in rendered if r["is_test"]]
    r_train = [r for r in rendered if not r["is_test"]]
    test_inband = sum(1 for r in r_test if r["verse_id"] in inband_set)
    train_inband = sum(1 for r in r_train if r["verse_id"] in inband_set)

    print(f"S2 candidates scanned: {n_total}")
    print(f"In band ({RATIO_MIN}x-{RATIO_MAX}x): {len(inband_ids)} ({100*len(inband_ids)/n_total:.1f}%)")
    print(f"  -> {OUT_INBAND}")
    print(f"\nOf the {len(rendered)} RENDERED lines:")
    print(f"  frozen test: {test_inband}/{len(r_test)} in band")
    print(f"  train:       {train_inband}/{len(r_train)} in band")
    print(f"\nFrozen test split itself is UNCHANGED ({len(frozen_test)} verse IDs) -- rule 1 intact.")

    record = {
        "event": "s2_inband_eval_subset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "criterion": f"{RATIO_MIN} <= len(modern_text)/len(old_text) <= {RATIO_MAX}",
        "source": "Roadmap v4 §2.1 step 2, never implemented in build_corpora.py",
        "n_candidates_scanned": n_total,
        "n_in_band": len(inband_ids),
        "rendered_test_in_band": test_inband,
        "rendered_test_total": len(r_test),
        "rendered_train_in_band": train_inband,
        "rendered_train_total": len(r_train),
        "note": "NOT a change to the frozen test split: s2_test_verse_ids.txt is untouched and "
                "remains the split of record. This is an additional, disclosed subset for "
                "end-to-end metric reporting, because ~66% of S2's modern-text targets are "
                "free-form commentary (median ratio 1.77x, max 32.6x) rather than line-level "
                "modernizations, and scoring chrF++/BLEU against commentary measures the wrong "
                "thing and would collapse the B0 floor. Recogniser-side numbers (CER) are "
                "unaffected and still use all rendered lines.",
    }
    with SPLIT_MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Appended provenance record to {SPLIT_MANIFEST}")


if __name__ == "__main__":
    main()
