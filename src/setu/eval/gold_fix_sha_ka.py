"""Apply the ಶ -> ಕ correction the transcriber identified.

In this scribal hand the glyph for ಕ resembles ಶ, and the single blind
transcription used ಶ where the scribe wrote ಕ. The evidence: ಶ appears
49 times in the gold against ಕ's 23, which is backwards for Kannada,
and at 24 of the 39 aligned positions where the human wrote ಶ and the
machine disagreed, the machine read ಕ.

**The correction is machine-assisted, and that matters for scoring.**
Positions are located by where the machine read ಕ, so a CER measured
against the corrected gold is no longer independent of the system being
scored -- it can only flatter it. Three variants are therefore written
and all three reported:

    as_typed      the transcription exactly as entered
    machine_agreed  ಶ -> ಕ only where the machine read ಕ (what was asked
                  for; circular for scoring, defensible as a transcript)
    blanket       every ಶ -> ಕ, independent of the machine, which is an
                  upper bound on the correction's effect and is the
                  honest figure to quote if one number is needed

The original is never overwritten.

Usage:
    python -m setu.eval.gold_fix_sha_ka
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from setu.eval.metrics import agreement_cer, align
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
SHA, KA = "\u0cb6", "\u0c95"


def corrected(gold: str, machine: str) -> tuple[str, int]:
    """ಶ -> ಕ at aligned positions where the machine read ಕ."""
    out = list(gold)
    n = 0
    for gi, mj in align(gold, machine):
        if gi is None or mj is None:
            continue
        if gold[gi] == SHA and machine[mj] == KA:
            out[gi] = KA
            n += 1
    return "".join(out), n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-line", type=Path,
                   default=Path("runs/20261008T130753Z_gold_real_eval/per_line.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/real_lines/gold/gold_lines_corrected.jsonl"))
    args = p.parse_args()

    rows = [json.loads(l) for l in args.per_line.open(encoding="utf-8")]
    variants = {"as_typed": [], "machine_agreed": [], "blanket": []}
    out_rows, n_swapped = [], 0
    for r in rows:
        g, m = r["gold"], r["machine"]
        fixed, n = corrected(g, m)
        n_swapped += n
        blanket = g.replace(SHA, KA)
        variants["as_typed"].append(agreement_cer(list(g), list(m)))
        variants["machine_agreed"].append(agreement_cer(list(fixed), list(m)))
        variants["blanket"].append(agreement_cer(list(blanket), list(m.replace(SHA, KA))))
        out_rows.append({"crop": r["crop"], "page": r["page"], "text": fixed,
                         "text_as_typed": g, "n_sha_to_ka": n,
                         "source": "human transcription, ಶ->ಕ where the machine read ಕ"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out_rows) + "\n",
                        encoding="utf-8")

    res = {"n_lines": len(rows), "n_sha_to_ka_swapped": n_swapped,
           **{k: {"mean_cer": statistics.fmean(v), "median_cer": statistics.median(v)}
              for k, v in variants.items()},
           "caveat": "machine_agreed locates swaps by the machine's own output, so a CER "
                     "against it is not independent of the system scored; blanket is the "
                     "machine-independent figure.",
           "output": str(args.out)}
    run_dir = start_run("gold_fix_sha_ka", {"seed": SEED, "per_line": str(args.per_line),
                                            "sha": SHA, "ka": KA})
    finish_run(run_dir, res)
    print(f"{n_swapped} characters changed from {SHA} to {KA} across {len(rows)} lines\n")
    for k, v in variants.items():
        print(f"  machine-label CER, gold = {k:15s} {statistics.fmean(v):.4f}")
    print(f"\n-> {args.out}\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()
