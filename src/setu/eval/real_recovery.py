"""Does the information B3 discards include the information needed?

"The branches differ" only shows the mechanism fires. This is the real
question, and the real-data analogue of the synthetic frame-level
result: where the consensus reading (what B3 commits to) is wrong, did
ANY of the samples read that character correctly -- i.e. was the right
answer still in the pile that B4 carries forward and argmax throws away?

Scored against the human transcriptions, over whichever demo lines have
one. Uses the RAW samples rather than the display list in demo.json,
which is capped at ten contested characters per line: measuring against
that cap understated the result as 9.6% when it is 34.2%.

Usage:
    python -m setu.eval.real_recovery --run runs/<ts>_demo_build_real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from setu.eval.metrics import align
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--gold", type=Path,
                default=Path("data/real_lines/gold/gold_lines_corrected.jsonl"))
args = ap.parse_args()

demo = json.load((args.run / "demo.json").open(encoding="utf-8"))
gold = {json.loads(l)["crop"]: json.loads(l)["text"]
        for l in args.gold.open(encoding="utf-8")}
per_line = []

tot_wrong = tot_any = 0
for e in demo:
    g = gold.get(e["crop"])
    if not g:
        continue
    b3 = e["b3_text"]
    samples = e.get("samples") or []
    # For each gold char, what did each sample put at the aligned spot?
    per_sample = []
    for s in samples:
        m = {}
        for gi, sj in align(g, s):
            if gi is not None and sj is not None:
                m[gi] = s[sj]
        per_sample.append(m)
    wrong = any_ok = 0
    for gi, hj in align(g, b3):
        if gi is None or hj is None or g[gi] == b3[hj]:
            continue
        wrong += 1
        if any(m.get(gi) == g[gi] for m in per_sample):
            any_ok += 1
    tot_wrong += wrong
    tot_any += any_ok
    per_line.append({"crop": e["crop"], "consensus_wrong": wrong, "some_sample_right": any_ok,
                     "n_samples": len(samples)})
    print(f"  {e['crop']:28s} consensus wrong {wrong:3d}, some sample right {any_ok:3d}  "
          f"({len(samples)} samples)")

print(f"\ncharacters the consensus got wrong        : {tot_wrong}")
print(f"...where at least one sample was correct  : {tot_any}  ({100*tot_any/max(tot_wrong,1):.1f}%)")
print(f"\nsynthetic, CTC top-5 (frame level)        : 80.6%")
