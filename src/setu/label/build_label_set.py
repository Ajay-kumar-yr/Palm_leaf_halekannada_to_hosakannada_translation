"""Turn raw vision-LLM transcriptions into a CRNN training set
(DEMO_PLAN.md, teacher-student). No human verification exists on this
project, so every filter here is automatic and every rejection is
counted — the counts are the quality evidence.

Filters, in order:

1. **errored / empty** — the labeller failed or said UNREADABLE.
2. **not WX-encodable** — the recogniser's alphabet has no symbol for
   something in the transcription. In practice this is Kannada digits
   (೦–೯), which appear as verse and folio numbers and as an old repha
   marker. Extending the vocabulary would need approval (CLAUDE.md "ask
   before ... changing the vocabulary") and would invalidate the
   existing checkpoint's 57-class output layer, so these lines are
   dropped, not accommodated.
3. **length implausible for the crop** — transcription length divided by
   the crop's aspect ratio (width/height) is tightly clustered for a
   given hand, because a line of a given physical length holds a
   predictable number of glyphs. The band is derived from the data by
   median and MAD, not hardcoded, and catches both hallucinated
   over-long output and truncated output.
4. **labeller disagreement** — where two or more label files cover the
   same crop, keep only crops whose transcriptions agree within
   `--max-disagreement` (symmetric `metrics.agreement_cer`). With one
   label file this filter is skipped and `n_cross_checked` is 0, which
   the output records so a single-labeller set can never be mistaken
   for a cross-checked one.

Output: `<set_dir>/train_labels.jsonl` (image path + text, the shape the
CRNN loader wants) plus a run folder with every count.

Usage:
    python -m setu.label.build_label_set --set-dir data/real_lines/train
    python -m setu.label.build_label_set --set-dir data/real_lines/train --group 1
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from setu.data import wx
from setu.eval.metrics import agreement_cer
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def encodable(text: str, digits: bool) -> bool:
    try:
        wx.encode(text, digits=digits)
        return True
    except ValueError:
        return False


def load_labels(set_dir: Path, patterns: list[str]) -> dict[str, dict[str, dict]]:
    """{label file stem: {crop: row}}, successful rows only."""
    out: dict[str, dict[str, dict]] = {}
    for pat in patterns:
        for path in sorted(set_dir.glob(pat)):
            rows = {}
            for line in path.open(encoding="utf-8"):
                r = json.loads(line)
                if "text" in r:
                    rows[r["crop"]] = r
            if rows:
                out[path.stem.replace("labels_", "")] = rows
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-dir", type=Path, default=REPO_ROOT / "data" / "real_lines" / "train")
    p.add_argument("--labels", nargs="+", default=["labels_*.jsonl"])
    p.add_argument("--group", default=None, help="Keep only this manuscript group (page stem prefix).")
    p.add_argument("--max-disagreement", type=float, default=0.35,
                   help="Max symmetric CER between two labellers for a crop to be kept.")
    p.add_argument("--length-mad", type=float, default=3.5,
                   help="Reject crops this many MADs from the median chars-per-aspect-unit.")
    p.add_argument("--digits", action="store_true",
                   help="Keep lines containing Kannada numerals, using wx.EXTENDED_VOCAB. "
                        "Without this they are rejected as unencodable -- 36%% of real lines.")
    p.add_argument("--min-chars", type=int, default=8)
    p.add_argument("--out-name", default="train_labels.jsonl")
    p.add_argument("--holdout-every", type=int, default=0,
                   help="Mark every Nth line of each page as held out (demo_split='holdout') "
                       "instead of training data. 4 keeps a quarter of every page unseen. "
                       "0 disables the split.")
    args = p.parse_args()

    manifest = {json.loads(l)["crop"]: json.loads(l)
                for l in (args.set_dir / "manifest.jsonl").open(encoding="utf-8")}
    label_sets = load_labels(args.set_dir, args.labels)
    if not label_sets:
        raise SystemExit(f"no label files matching {args.labels} in {args.set_dir}")
    print(f"label files: {', '.join(f'{k} ({len(v)})' for k, v in label_sets.items())}")

    # Primary labeller = the file with the most successful rows.
    primary_name = max(label_sets, key=lambda k: len(label_sets[k]))
    primary = label_sets[primary_name]
    others = {k: v for k, v in label_sets.items() if k != primary_name}
    print(f"primary: {primary_name}")

    counts = {"candidates": 0, "rejected_group": 0, "rejected_empty": 0, "rejected_short": 0,
              "rejected_unencodable": 0, "rejected_length": 0, "rejected_disagreement": 0}
    rows = []
    for crop, r in sorted(primary.items()):
        counts["candidates"] += 1
        m = manifest.get(crop)
        if m is None:
            continue
        if args.group and m["page"].split(".")[0] != args.group:
            counts["rejected_group"] += 1
            continue
        text = r["text"].strip()
        if not text:
            counts["rejected_empty"] += 1
            continue
        if len(text) < args.min_chars:
            counts["rejected_short"] += 1
            continue
        if not encodable(text, args.digits):
            counts["rejected_unencodable"] += 1
            continue
        rows.append({"crop": crop,
                     # Repo-root-relative, so label sets from different
                     # directories (demo pages + extra pages of the same
                     # hand) can be concatenated into one training set.
                     "image": str((args.set_dir / crop).resolve().relative_to(REPO_ROOT)
                                  ).replace("\\", "/"),
                     "page": m["page"], "text": text,
                     "aspect": m["width"] / m["height"] if m["height"] else 0.0,
                     "width": m["width"], "height": m["height"]})

    # Length plausibility, with the band derived from the surviving data.
    dens = [r["text"].__len__() / r["aspect"] for r in rows if r["aspect"]]
    if len(dens) >= 8:
        med = statistics.median(dens)
        mad = statistics.median([abs(d - med) for d in dens]) or 1e-6
        lo, hi = med - args.length_mad * mad, med + args.length_mad * mad
        kept = []
        for r in rows:
            d = len(r["text"]) / r["aspect"] if r["aspect"] else 0.0
            r["chars_per_aspect"] = round(d, 2)
            if lo <= d <= hi:
                kept.append(r)
            else:
                counts["rejected_length"] += 1
        rows = kept
        band = {"median": round(med, 2), "mad": round(mad, 3), "low": round(lo, 2), "high": round(hi, 2)}
    else:
        band = {"note": f"only {len(dens)} rows -- length filter skipped, needs >= 8"}

    # Cross-labeller agreement, where a second labeller covered the crop.
    n_cross, agreements = 0, []
    if others:
        kept = []
        for r in rows:
            scores = [agreement_cer(r["text"], o[r["crop"]]["text"])
                      for o in others.values() if r["crop"] in o and o[r["crop"]]["text"].strip()]
            if not scores:
                kept.append(r)
                continue
            n_cross += 1
            best = min(scores)
            r["disagreement"] = round(best, 3)
            agreements.append(best)
            if best <= args.max_disagreement:
                kept.append(r)
            else:
                counts["rejected_disagreement"] += 1
        rows = kept

    # Demo hold-out. The demo is writer-dependent by decision
    # (DEMO_PLAN.md 5b): the recogniser trains on lines from the very
    # pages it will be shown on. The least it must do is show lines it
    # never trained on, so every Nth line of each page -- by line_index,
    # not by hash, so the held-out lines are spread evenly down the page
    # rather than clumped -- is reserved for the demo.
    n_holdout = 0
    if args.holdout_every > 1:
        per_page: dict[str, int] = {}
        for r in sorted(rows, key=lambda r: (r["page"], r["crop"])):
            k = per_page.get(r["page"], 0)
            per_page[r["page"]] = k + 1
            r["demo_split"] = "holdout" if k % args.holdout_every == args.holdout_every - 1 else "train"
            n_holdout += int(r["demo_split"] == "holdout")
    else:
        for r in rows:
            r["demo_split"] = "train"

    out_path = args.set_dir / args.out_name
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results = {
        "n_kept": len(rows),
        "n_holdout_for_demo": n_holdout,
        "n_train": len(rows) - n_holdout,
        "holdout_every": args.holdout_every,
        "counts": counts,
        "length_band_chars_per_aspect": band,
        "primary_labeller": primary_name,
        "other_labellers": sorted(others),
        "n_cross_checked": n_cross,
        "mean_disagreement": round(statistics.fmean(agreements), 3) if agreements else None,
        "pages": sorted({r["page"] for r in rows}),
        "total_chars": sum(len(r["text"]) for r in rows),
        "output": str(out_path),
        "caveat": "machine labels, no human verification; cross-labeller agreement is the only "
                  "external check and is 0 crops when a single labeller was used",
    }
    run_dir = start_run("build_label_set", {
        "seed": SEED, "set_dir": str(args.set_dir), "labels": args.labels, "group": args.group,
        "max_disagreement": args.max_disagreement, "length_mad": args.length_mad,
        "min_chars": args.min_chars, "holdout_every": args.holdout_every,
        "digits": args.digits,
    })
    finish_run(run_dir, results)

    print(json.dumps({k: results[k] for k in
                      ("n_kept", "n_train", "n_holdout_for_demo", "counts",
                       "length_band_chars_per_aspect", "n_cross_checked",
                       "mean_disagreement", "total_chars")}, indent=2))
    print(f"-> {out_path}\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()
