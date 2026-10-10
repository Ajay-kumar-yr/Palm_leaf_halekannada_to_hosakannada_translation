"""A second, clean set of real lines for a human to transcribe.

The 32 gold lines (`gold_lines_corrected.jsonl`) have been used for
nearly everything: the true-CER measurement, the data-scaling curve's
held-out set, the glyph bank that the stitching experiment force-aligned
(RESULTS.md 3.4), and the sha/ka correction. They are entangled with the
experiments that were tuned against them, so a number measured on them
again is no longer independent.

These 15 are drawn from the **50-page sample set** instead -- the same
set behind the 6.4 domain-gap figures. It has no labels of any kind, was
never trained on, and `run_palmira_crops.py` builds it disjoint from
`demo_pages/` by page stem *and* photo signature, so a number measured
here does not touch CLAUDE.md rule 8.

**Selection is a seeded random draw, not a curated one** -- no filter on
Palmira's score, on crop size or on legibility. A draw that skipped the
hard lines would flatter every number measured against it. Lines the
transcriber cannot read are marked as such on the page, and that
fraction is itself worth reporting. The one constraint is at most two
lines per page, so 15 lines do not collapse onto two leaves.

What these buy, and nothing else can:

1. a real-line CER for the fine-tuned recogniser that is independent of
   every choice made while looking at the 32;
2. the untested question of whether U-Net binarized input helps the
   *fine-tuned* model (0.428 vs 0.447 is known only for the vision
   model, on the old gold lines).

Usage:
    python -m setu.eval.make_test_set
    python -m setu.demo.make_transcriber \
        --labels data/real_lines/sample/test15_lines.jsonl --split all --out page.html
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 0  # CLAUDE.md rule 6

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data/real_lines/sample/manifest.jsonl")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--per-page", type=int, default=2)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "data/real_lines/sample/test15_lines.jsonl")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8")]
    # Sort before shuffling: the manifest's order is whatever Palmira
    # emitted, so a draw from it is only reproducible once the order is.
    rows.sort(key=lambda r: r["crop"])
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    picked, per_page = [], Counter()
    for r in rows:
        if len(picked) >= args.n:
            break
        if per_page[r["page"]] >= args.per_page:
            continue
        img = args.manifest.parent / r["crop"]
        if not img.exists():
            raise SystemExit(f"missing crop {img}")
        per_page[r["page"]] += 1
        picked.append({
            "crop": r["crop"],
            "image": str(img.relative_to(REPO_ROOT)).replace("\\", "/"),
            "page": r["page"],
            "line_index": r["line_index"],
            "width": r["width"],
            "height": r["height"],
            "palmira_score": r["score"],
            "demo_split": "holdout",   # never trained on, by construction
            "set": "sample",           # reportable: not a demo page
        })
    if len(picked) < args.n:
        raise SystemExit(f"only {len(picked)} lines available under per-page cap {args.per_page}")

    picked.sort(key=lambda r: (r["page"], r["crop"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(picked)} lines over {len(per_page)} pages (seed {args.seed}, "
          f"max {args.per_page}/page, no quality filter)")
    for r in picked:
        print(f"  {r['crop']:22s} {r['width']:4d}x{r['height']:3d}  score {r['palmira_score']:.3f}")
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
