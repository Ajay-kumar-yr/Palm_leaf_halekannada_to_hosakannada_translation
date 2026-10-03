"""Formats tagged_sample.jsonl into a plain-text review sheet and records
the run (CLAUDE.md rule 4). The actual tagging was done by hand, applying
prompt.md's fixed prompt (verbatim) to real S2 modern-Kannada text -- this
script only persists/formats the result and tallies stats, it does not
call any model itself.

Usage:
    python -m setu.pos.review
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from setu.runlog import finish_run, start_run

SAMPLE_PATH = Path(__file__).parent / "tagged_sample.jsonl"
REVIEW_OUT = Path(__file__).parent / "tagged_sample_review.tsv"


def main() -> None:
    records = [json.loads(line) for line in SAMPLE_PATH.open(encoding="utf-8")]
    tag_counts: Counter[str] = Counter()
    n_tokens = 0

    with REVIEW_OUT.open("w", encoding="utf-8") as f:
        f.write("sentence_id\ttoken\ttag\tcorrect(y/n)\tcorrected_tag(if n)\n")
        for sid, rec in enumerate(records):
            for token, tag in rec["tokens"]:
                f.write(f"{sid}\t{token}\t{tag}\t\t\n")
                tag_counts[tag] += 1
                n_tokens += 1

    run_dir = start_run(
        "pos_tagging",
        {
            "method": "fixed LLM prompt (src/setu/pos/prompt.md), applied by hand -- "
                      "CLAUDE.md: not a trained model, hand-checked on ~100 tags",
            "tagset": "Universal Dependencies (17 tags)",
            "text_source": "real S2 modern_text (KannadaLit4NLP scholarly interpretations), "
                            "a stand-in for the modernizer's actual output until a trained "
                            "CRNN + fine-tuned modernizer exist (Week 3)",
            "n_sentences": len(records),
        },
    )
    finish_run(run_dir, {"n_tokens_tagged": n_tokens, "tag_distribution": dict(tag_counts)})

    print(f"{n_tokens} tokens tagged across {len(records)} sentences.")
    print(f"Tag distribution: {dict(tag_counts)}")
    print(f"Review sheet written to {REVIEW_OUT} -- fill in correct(y/n) and corrected_tag by hand.")
    print(f"Run recorded at {run_dir}")


if __name__ == "__main__":
    main()
