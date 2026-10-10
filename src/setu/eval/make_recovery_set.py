"""The gold lines that may appear in a reported number.

CLAUDE.md rule 8: `demo_pages/` is for the live demo only and never
appears in any reported number. The 32 hand-transcribed gold lines
straddle that line --

    1.108, 1.12, 1.140          16 lines   train pages    reportable
    group1_1.128/.198/.26/.28   16 lines   demo pages     demo only

-- and the real-crop recovery figure first published in RESULTS.md row
13 (34.2%, run `20261008T152006Z_real_recovery`) was computed on six
lines from `group1_1.128` and `group1_1.198`, i.e. entirely on demo
pages. That number is withdrawn and recomputed on the 16 train-page
lines, which are disjoint from `demo_pages/` and from the 50-page
sample set behind the 6.4 figures (`run_palmira_crops.py` builds the
three sets disjoint by page stem and photo signature).

The 16 lines are also *more* than the 6 they replace, and they were
held out of every fine-tune, so `demo_split` is "holdout" for all of
them.

Images point at `gold_unet/`, the U-Net binarized crops: binarized
input reads slightly better for the vision model too (0.428 vs 0.447
CER, `setu.eval.ocr_input_check`), and it is what the 34.2% run used,
so the input is held constant.

Usage:
    python -m setu.eval.make_recovery_set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED = 0  # CLAUDE.md rule 6 -- selection is deterministic anyway

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# The four demo pages that carry gold lines. Named explicitly rather
# than inferred from a "group" prefix: the prefix is a manuscript
# group, not a set membership, and three of the four groups also appear
# in the sample set.
DEMO_PAGES = {"group1_1.128", "group1_1.198", "group1_1.26", "group1_1.28"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path,
                   default=REPO_ROOT / "data/real_lines/gold/gold_lines_corrected.jsonl")
    p.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data/real_lines/gold_unet")
    p.add_argument("--machine-labels", type=Path,
                   default=REPO_ROOT / "data/real_lines/train/train_labels.jsonl",
                   help="where `text` comes from. Everything downstream reads `text` as the "
                        "MACHINE label; putting the hand transcription there instead makes "
                        "`gold_real` score the gold against itself and report the labels as "
                        "97%% accurate (CER 0.028 instead of 0.370).")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "data/real_lines/gold/recovery_lines.jsonl")
    args = p.parse_args()

    # A relative --image-dir means "inside the repo", not "inside the
    # shell's cwd": the paths written into the manifest are repo-relative,
    # so the directory they are taken from has to be resolved that way too.
    image_dir = args.image_dir if args.image_dir.is_absolute() else REPO_ROOT / args.image_dir

    machine = {json.loads(l)["crop"]: json.loads(l).get("text", "")
               for l in args.machine_labels.open(encoding="utf-8")}

    rows = [json.loads(l) for l in args.gold.open(encoding="utf-8")]
    kept, dropped = [], []
    for r in rows:
        page = r["crop"].split("/")[0]
        if page in DEMO_PAGES:
            dropped.append(r["crop"])
            continue
        img = image_dir / r["crop"]
        if not img.exists():
            raise SystemExit(f"no crop for {r['crop']} at {img}")
        kept.append({
            "crop": r["crop"],
            "image": str(img.relative_to(REPO_ROOT)).replace("\\", "/"),
            "page": page,
            "text": machine.get(r["crop"], ""),   # the MACHINE label
            "gold_text": r["text"],               # the hand transcription
            "demo_split": "holdout",   # held out of every fine-tune
            "input": "binarized",
            "reportable": True,        # not a demo page -- CLAUDE.md rule 8
        })

    kept.sort(key=lambda r: (r["page"], r["crop"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pages = sorted({r["page"] for r in kept})
    print(f"{len(kept)} reportable gold lines over {len(pages)} train pages: {', '.join(pages)}")
    print(f"{len(dropped)} dropped as demo pages (rule 8): "
          f"{', '.join(sorted({c.split('/')[0] for c in dropped}))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
