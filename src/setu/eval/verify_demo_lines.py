"""Did argmax actually get it wrong, and did the bridge actually hold the truth?

The demo highlights where B3 and B4 differ, which is all it can honestly
claim on screen: the pipeline has no ground truth at demo time. These
lines do have it. They are frozen S2 test-split lines, never trained on,
and `data/s2/manifest.jsonl` carries the exact old-Kannada text for each.

So for every staged demo line this answers, per character:

    does B3's reading differ from the gold?            -> argmax is WRONG there
    is the gold symbol among the alternatives B4 kept? -> the bridge HELD it

which is the difference between "the branches disagree" and "argmax was
wrong and the information survived anyway".

**What this does NOT verify: the modernized text.** S2's `modern_text`
is scholarly commentary, not a line-level modernization -- for
`line_000859` it opens "ಸರಳಾನುವಾದ:" and runs 441 characters against a
209-character line, explaining that ಕೊಂಗಿತಿ stands for Maya. Scoring a
modernizer against it would measure nothing (STATUS.md 3.1: the title
and the training targets are different tasks). The verifiable claim is
at the reading level.

The demo replays entries produced from Gradio's re-encoded upload, while
this reads `data/s2/images/<id>.png` directly, so the script also checks
that both produce the same B3 reading and says so -- otherwise the
verdicts here would not describe what is on screen.

Usage:
    python -m setu.eval.verify_demo_lines
    python -m setu.eval.verify_demo_lines --lines line_000859 line_000282
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

SEED = 0  # CLAUDE.md rule 6

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# The staged demo order (setu.demo.stage_demo_inputs.ORDER).
DEFAULT_LINES = ["line_000859", "line_000282", "line_000508", "line_000494",
                 "line_000684", "line_000472", "line_000037", "line_000131"]


def load_gold(ids: set[str]) -> dict[str, str]:
    gold = {}
    for line in (REPO_ROOT / "data/s2/manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in ids:
            gold[r["id"]] = r["text"]
    return gold


def demo_readings() -> list[str]:
    """B3 strings from the cached entries the demo actually replays."""
    out = []
    for f in glob.glob(str(REPO_ROOT / "data/demo_cache/pipeline/*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("route") == "synthetic":
            for ln in d.get("lines", []):
                if ln.get("b3_text"):
                    out.append(ln["b3_text"])
    return out


def main() -> None:
    from setu.demo import pipeline as P
    from setu.eval.metrics import align
    from setu.runlog import finish_run, start_run

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", nargs="+", default=DEFAULT_LINES)
    ap.add_argument("--images-dir", type=Path, default=REPO_ROOT / "data/s2/images")
    args = ap.parse_args()

    gold = load_gold(set(args.lines))
    missing = [i for i in args.lines if i not in gold]
    if missing:
        raise SystemExit(f"no gold text for {missing}")
    on_screen = demo_readings()

    from setu.bridge.line_alternatives import analyse_line
    from setu.data import wx
    from setu.demo.build_demo_synthetic import score_line

    rows, tot_wrong, tot_held = [], 0, 0
    for line_id in args.lines:
        img = args.images_dir / f"{line_id}.png"
        # The demo's own forward pass, then the bridge's own analysis, so
        # the slots examined here are the slots the UI shows.
        lp, n_frames, _ = P.crnn_line(img, resize=False)
        symbols, slots = analyse_line(lp, n_frames, temperature=1.5, top_k=5,
                                      uncertain_below=0.9)
        b3 = P.run_synthetic(img, modernize_fn=None)["lines"][0]["b3_text"]

        # Align in WX SYMBOL space, not in the Kannada string. One
        # syllable is several code points, so a string index is not a
        # slot index: looking slots up by it finds nothing and reports
        # every recovery as a miss, which is exactly what it did.
        truth_syms = wx.encode(gold[line_id])
        held, subs, _where = score_line(symbols, slots, truth_syms)

        detail = []
        for ri, hj in align(truth_syms, symbols):
            if ri is None or hj is None:
                continue
            slot = slots[hj]
            if slot.chosen == truth_syms[ri]:
                continue
            alts = [(a.kannada, round(a.prob, 3)) for a in slot.alternatives
                    if a.symbol != "<blank>"][:4]
            detail.append({
                "slot": hj, "gold_symbol": truth_syms[ri],
                "argmax_symbol": slot.chosen,
                "argmax_confidence": round(slot.top1_prob, 3),
                "flagged_uncertain": bool(slot.uncertain),
                "bridge_held_it": bool(slot.uncertain and any(
                    a.symbol == truth_syms[ri] for a in slot.alternatives)),
                "alternatives": alts,
            })
        tot_wrong += subs
        tot_held += held
        matches = b3 in on_screen
        n_unc = sum(1 for s_ in slots if s_.uncertain)
        rows.append({"line_id": line_id, "argmax_wrong_symbols": subs,
                     "bridge_held": held, "matches_demo_cache": matches,
                     "n_flagged": n_unc, "n_slots": len(slots), "errors": detail})
        print(f"  {line_id}  argmax wrong: {subs:2d}  bridge held: {held:2d}  "
              f"flagged {n_unc:2d}/{len(slots):3d}  "
              f"({'same as demo' if matches else 'DIFFERS from demo cache'})")
        for d in detail:
            verdict = ("bridge HELD the gold" if d["bridge_held_it"] else
                       "bridge did NOT hold it" +
                       ("" if d["flagged_uncertain"] else " (slot not flagged)"))
            print(f"      slot {d['slot']}: argmax {d['argmax_symbol']!r} @ "
                  f"{d['argmax_confidence']} vs gold {d['gold_symbol']!r} -- {verdict}; "
                  f"carried {d['alternatives']}")

    print(f"\ncharacters argmax got wrong      : {tot_wrong}")
    print(f"...where the bridge held the gold : {tot_held}"
          f"  ({100 * tot_held / max(tot_wrong, 1):.1f}%)")
    agree = sum(r["matches_demo_cache"] for r in rows)
    print(f"readings identical to the demo cache: {agree}/{len(rows)}")

    run_dir = start_run("verify_demo_lines", {
        "seed": SEED, "lines": args.lines, "gold": "data/s2/manifest.jsonl",
        "images": str(args.images_dir),
        "claim": "per character: did argmax differ from the frozen-test gold, and was "
                 "the gold symbol among the alternatives the bridge carried",
        "not_verified": "the modernized text -- S2's modern_text is scholarly commentary, "
                        "not a line-level modernization (STATUS.md 3.1)",
    })
    finish_run(run_dir, {
        "n_lines": len(rows), "argmax_wrong_symbols": tot_wrong,
        "bridge_held_gold": tot_held,
        "held_rate": tot_held / max(tot_wrong, 1),
        "readings_matching_demo_cache": agree,
        "per_line": rows,
    })
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
