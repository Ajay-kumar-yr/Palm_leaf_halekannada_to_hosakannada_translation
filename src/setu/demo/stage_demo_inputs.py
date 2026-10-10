"""Stage the images to upload during the demo, in one folder, in order.

Hunting for the right file while an examiner watches is how a demo dies.
This copies the images worth showing into `demo_inputs/`, in three
numbered folders matching the order to show them, with names that say
what each one does, plus a README naming the route each takes and
whether it replays from cache or spends API quota.

## The published examples do not reproduce on this machine

The six worked recoveries (`runs/20261008T100330Z_demo_build_synthetic`,
RESULTS.md 2.1) were computed from the transferred recogniser cache
`20261007T022632Z_cache_s2_recogniser`, which was built on the desktop.
Its frame counts do not match the S2 images on this laptop -- for
`line_000599` the cache has **492** frames where the local image gives
**468** (3747px / 8) -- so the desktop rendered S2 with a different
seed, and those exact renders are not here.

Consequence, measured rather than assumed: run the local
`line_000599` through the app and argmax reads **ಮುಗ್ಧೆ correctly**. The
documented ಮುಗ್ಹೆ → ಮುಗ್ಧೆ recovery cannot be shown live here. Worse,
`line_000322` reads the local render with **zero** uncertain characters,
so it would demonstrate nothing at all.

So the published examples stay where they belong -- in the published
viewer and the report, as measured results with their run folder -- and
what this folder stages is chosen by **what the live app actually does
on these files**, probed through its own HTTP API. The `LIVE` table
below records that probe; re-measure it after any change to the
recogniser, the renders or the bridge settings.

The dataset still matters: `data/s1/` and `data/s1_sample/` hold
different renders again, and `line_000599` from `s1_sample` reads with
one contested character at 0.87 instead of the three S2 gives.

Nothing here is written to; the frozen split is read only (CLAUDE.md
rule 1). Everything in `demo_inputs/` is a copy, and the folder is
regenerable, so it is gitignored.

Usage:
    python -m setu.demo.stage_demo_inputs
    python -m setu.demo.stage_demo_inputs --pages 4   # fewer real pages
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SEED = 0  # CLAUDE.md rule 6 -- selection is deterministic

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SYNTH_RUN = REPO_ROOT / "runs" / "20261009T121147Z_demo_build_synthetic" / "demo.json"
# Real pages with a warm pipeline cache, at the settings the demo runs
# at (5 reads, 4 lines, modernizer on). Both were pre-cached 2026-10-10;
# every line reads, both branches are modernized, and the branches
# differ on all four lines of each.
CACHED_PAGES = ("group1_1.128", "group2_2.15")
CACHED_READS = 5

# What the LIVE app does with each local render, probed through its own
# /gradio_api on 2026-10-09 at the synthetic route's fixed settings
# (temperature 1.5, top-5, uncertain below 0.9, no resize). These are
# this machine's renders, not the desktop cache the published examples
# came from -- see the module docstring.
#
#   contested  characters flagged / total
#   lowest     the most nearly-even slot, which is what argmax cannot
#              represent and therefore what the demo should point at
#   fix        the word argmax produced -> the true word, where argmax
#              actually got a character wrong. Only two of the eight do:
#              `build_demo_synthetic` counts substitutions against the
#              CTC-aligned WX reference, and most of those vanish once
#              WX is converted back to Kannada script. What an examiner
#              sees on screen is the Kannada, so that is what decides
#              which files lead.
#
# These are read off the CACHED pipeline entries, which is what the demo
# will actually replay -- not off a fresh run. They have to be, because
# the CRNN forward is **not bit-reproducible on this GPU**: the same file
# at the same settings gave 1/239 contested characters yesterday and
# 5/238 today, and `line_000131` went from 1/270 to 0/270, which flips
# it from a demonstration into a blank. Flagging thresholds at 0.9 sit
# close enough to the numerical noise for the count to move. Once a
# result is cached the demo is deterministic; re-cache and these change,
# so re-read them from the cache if you ever clear it.
LIVE = {
    # "err"  the one symbol argmax got wrong, verified against the frozen
    #        S2 gold by setu.eval.verify_demo_lines; "held" is whether the
    #        bridge still carried the gold symbol at that slot. It did on
    #        all eight -- but see the selection caveat in the README: these
    #        lines were CHOSEN for having a recoverable substitution.
    # "seen" whether the error survives into the printed Kannada. Only two
    #        do; the other six differ in WX only and render identically.
    "line_000859": {"contested": "1/239", "lowest": 0.48, "rivals": 4, "held": True,
                    "err": "ಗ್ @0.48 for gold ದ್ @0.253", "seen": True,
                    "fix": "ಬೀಗಿಯ → ಬೀದಿಯ",
                    "slug": "VISIBLE_argmax-0.48-beegiya_gold-beediya"},
    "line_000282": {"contested": "8/248", "lowest": 0.812, "rivals": 4, "held": True,
                    "err": "ದ್ @0.812 for gold ಧ್ @0.163", "seen": True,
                    "fix": "ಸಾದ್ಯವಿಲ್ಲಯ್ಯಾ → ಸಾಧ್ಯವಿಲ್ಲಯ್ಯಾ",
                    "slug": "VISIBLE_argmax-saadya_gold-saadhya"},
    "line_000037": {"contested": "15/235", "lowest": 0.507, "rivals": 4, "held": True,
                    "err": "ಎ @0.507 for gold ಅ @0.483", "seen": False,
                    "fix": None, "slug": "coin-flip-0.507_held_not-printed"},
    "line_000472": {"contested": "5/233", "lowest": 0.599, "rivals": 4, "held": True,
                    "err": "ಅ @0.599 for gold ಎ @0.384", "seen": False,
                    "fix": None, "slug": "argmax-0.599_held_not-printed"},
    "line_000508": {"contested": "2/257", "lowest": 0.65, "rivals": 4, "held": True,
                    "err": "ಎ @0.65 for gold ಅ @0.345", "seen": False,
                    "fix": None, "slug": "argmax-0.65_held_not-printed"},
    "line_000131": {"contested": "1/270", "lowest": 0.735, "rivals": 4, "held": True,
                    "err": "ಎ @0.735 for gold ಅ @0.147", "seen": False,
                    "fix": None, "slug": "argmax-0.735_held_not-printed"},
    "line_000494": {"contested": "5/256", "lowest": 0.739, "rivals": 4, "held": True,
                    "err": "ಎ @0.739 for gold ಅ @0.215", "seen": False,
                    "fix": None, "slug": "argmax-0.739_held_not-printed"},
    "line_000684": {"contested": "6/234", "lowest": 0.864, "rivals": 4, "held": True,
                    "err": "ಅ @0.864 for gold ಎ @0.134", "seen": False,
                    "fix": None, "slug": "argmax-0.864_held_not-printed"},
}
# Show order: the two recoveries first, then by how little noise sits
# around the contested slots, and the blank one last.
ORDER = ["line_000859", "line_000282", "line_000037", "line_000472",
         "line_000508", "line_000131", "line_000494", "line_000684"]


def stage_synthetic(out: Path) -> list[dict]:
    rows = json.loads(SYNTH_RUN.read_text(encoding="utf-8"))
    by_id = {r["crop"]: r for r in rows}
    missing = [i for i in ORDER if i not in by_id]
    if missing:
        raise SystemExit(f"{SYNTH_RUN} does not contain {missing}")

    staged = []
    for n, line_id in enumerate(ORDER, 1):
        r = by_id[line_id]
        src = REPO_ROOT / r["image"]
        if not src.exists():
            raise SystemExit(f"missing {src} -- S2 renders are required, not S1")
        live = LIVE[line_id]
        dst = out / f"{n:02d}_{line_id}__{live['slug']}.png"
        shutil.copy2(src, dst)
        staged.append({"file": dst.name, "line_id": line_id, "live": live,
                       "verse": r.get("label_text") or r.get("b3_text", "")})
    return staged


def stage_real_pages(out: Path, n_pages: int) -> list[dict]:
    pages = sorted((REPO_ROOT / "demo_pages").glob("*.jpg"))
    if not pages:
        raise SystemExit("demo_pages/ is empty")
    # The cached page first, then a spread across the manuscript groups,
    # so the fallback choices are not four photos of the same hand.
    def is_cached(stem: str) -> bool:
        return any(stem.endswith(c) or stem == c for c in CACHED_PAGES)

    cached = [p for p in pages if is_cached(p.stem)]
    rest, seen = [], {c.split("_")[0] for c in CACHED_PAGES}
    for p in pages:
        group = p.stem.split("_")[0]
        if not is_cached(p.stem) and group not in seen:
            rest.append(p)
            seen.add(group)
    rest += [p for p in pages if p not in cached and p not in rest]

    staged = []
    for n, src in enumerate((cached + rest)[:n_pages], 1):
        hit = is_cached(src.stem)
        tag = f"__CACHED-{CACHED_READS}-reads" if hit else "__spends-quota"
        dst = out / f"{n:02d}_{src.stem}{tag}{src.suffix}"
        shutil.copy2(src, dst)
        staged.append({"file": dst.name, "page": src.stem, "cached": hit})
    return staged


def stage_real_crops(out: Path) -> list[dict]:
    """Pre-cut lines from a cached page, which have hand transcriptions.

    Uploading one of these and overriding the route to `real` skips
    Palmira entirely -- useful when the worker is down, and it is the
    only way to show a line whose true text is known. They are demo-page
    lines, so CLAUDE.md rule 8 keeps them out of every reported number;
    the recovery figure is measured on train-page lines instead
    (RESULTS.md 6a).
    """
    gold = {}
    gf = REPO_ROOT / "data/real_lines/gold/gold_lines_corrected.jsonl"
    if gf.exists():
        gold = {json.loads(l)["crop"]: json.loads(l)["text"]
                for l in gf.open(encoding="utf-8")}

    staged = []
    for n, crop in enumerate([c for c in sorted(gold) if c.startswith(CACHED_PAGES[0])], 1):
        src = REPO_ROOT / "data/real_lines/demo" / crop
        if not src.exists():
            continue
        dst = out / f"{n:02d}_{crop.replace('/', '__')}"
        shutil.copy2(src, dst)
        staged.append({"file": dst.name, "crop": crop, "gold": gold[crop]})
    return staged


def write_readme(root: Path, synth: list[dict], pages: list[dict], crops: list[dict]) -> None:
    lines = [
        "# Demo inputs — what to upload, in order",
        "",
        "Generated by `python -m setu.demo.stage_demo_inputs`. Everything here",
        "is a copy; delete the folder and regenerate it freely.",
        "",
        "Start both processes first (`DEMO_RUNBOOK.md`), then open",
        "<http://127.0.0.1:7860>.",
        "",
        "| folder | set the Pipeline radio to | costs |",
        "|---|---|---|",
        "| `01_synthetic_soft_bridge/` | `synthetic` (or leave `auto`) | nothing — the CRNN is local |",
        "| `02_real_pages/` | `real` (or leave `auto`) | **API quota**, unless marked CACHED |",
        "| `03_real_line_crops/` | **`real` — you must override**, `auto` sends a crop to synthetic | **API quota** |",
        "",
        "## 01 — the soft bridge, on synthetic lines",
        "",
        "This is the project's own contribution and the only place the",
        "recogniser actually works (1.53% CER). Frozen S2 test split, true",
        "ground truth, never trained on.",
        "",
        "These were selected from a recogniser cache rebuilt on **this**",
        "machine's renders (`runs/20261009T120133Z_cache_s2_recogniser`), and",
        "every one was then replayed through `pipeline.run_synthetic` — the",
        "exact function the UI calls — and matched the cache's slot and",
        "confidence to three decimals. They reproduce live. The six examples",
        "published earlier do **not**: those came from the desktop's cache,",
        "whose renders are not on this laptop, and on the local render of",
        "`line_000599` argmax reads ಮುಗ್ಧೆ correctly, so the documented",
        "ಮುಗ್ಹೆ → ಮುಗ್ಧೆ moment never appears. Show that one from the published",
        "viewer, as a measured result with its run folder.",
        "",
        "| file | contested | argmax | the one symbol argmax got wrong | printed? |",
        "|---|---|---|---|---|",
    ]
    for s_ in synth:
        lv = s_["live"]
        low = "—" if lv["lowest"] is None else f"{lv['lowest']:.3f}"
        seen = "**yes**" if lv["seen"] else "no"
        lines += [f"| `{s_['file']}` | {lv['contested']} | {low} | {lv['err']} | {seen} |"]
    lines += [
        "",
        "**Every one of these eight lines has exactly one symbol argmax got",
        "wrong, and on all eight the bridge still carried the gold symbol.**",
        "Verified against the frozen S2 test-split gold by",
        "`setu.eval.verify_demo_lines`, which also confirms all eight readings",
        "are identical to the entries the demo replays.",
        "",
        "**Do not quote 8/8 as a result.** These lines were *selected* by",
        "`build_demo_synthetic` for having a recoverable substitution, so the",
        "100% is selection, not measurement. The honest unbiased figure is the",
        "frame-level one: 76.0% on this machine's renders, 80.6% on the",
        "desktop's (RESULTS.md 6b).",
        "",
        "**Only the first two show the error in the printed Kannada.** The",
        "other six differ in the WX romanisation only — argmax picks ಎ where",
        "the gold is ಅ and the rendered syllable comes out the same — so there",
        "is nothing on screen to point at. Show `01` and `02`; the rest are",
        "there to demonstrate the mechanism, not a visible fix.",
        "",
        "`01_line_000859` is the clearest: argmax is **0.48** confident, under",
        "half, and commits to **ಬೀಗಿಯ**, not a word; the bridge also carried ದ್",
        "at 0.253, which is the gold **ಬೀದಿಯ** (\"of the street\"). Its B4",
        "modern text gives ಬಾಗಿಲ (\"door\") — a real word where B3 keeps the",
        "non-word, though not the gold word either. Say that precisely.",
        "",
        "## 02 — real manuscript pages",
        "",
        "Palmira segmentation → line crops → U-Net binarization → vision-model",
        "ensemble → modernizer. Needs the Palmira worker up.",
        "",
    ]
    for p in pages:
        lines += [f"- **`{p['file']}`**" + ("  ← replays from cache, spends nothing"
                                            if p["cached"] else "")]
    lines += [
        "",
        f"Only the pages marked CACHED, at **{CACHED_READS} reads, 4 lines,",
        "modernizer on**, replay offline. Any other page, or these at different",
        "settings, calls the vision model live: a",
        "page at 5 reads × 4 lines costs 20 of the 20-per-key-per-day-per-model",
        "free tier. Pre-cache what you intend to show, then run the app with",
        "`--no-live` so it *cannot* call out.",
        "",
        "**Never say \"our system reads manuscripts\" here.** It does not: our",
        "CRNN scores 0.696 CER on real crops, so the reading is done by a",
        "vision model and the uncertainty comes from disagreement between",
        "repeated reads — same claim, different mechanism, and weaker (26.0%",
        "recovery against the bridge's 80.6%).",
        "",
        "## 03 — real line crops (skips Palmira)",
        "",
        "Override the Pipeline radio to `real`: left on `auto` a crop is",
        "routed to the synthetic CRNN, which cannot read it. Useful if the",
        "worker is down, and these are the only real lines whose true text is",
        "known, so you can show the hand transcription beside the machine's",
        "reading.",
        "",
    ]
    for c in crops:
        lines += [f"- **`{c['file']}`** — hand transcription: {c['gold'][:60]}…"]
    lines += [
        "",
        "These are demo-page lines. CLAUDE.md rule 8 keeps them out of every",
        "reported number — the recovery figure is measured on train-page lines",
        "instead (RESULTS.md 6a).",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "demo_inputs")
    p.add_argument("--pages", type=int, default=5, help="how many real pages to stage")
    p.add_argument("--clean", action="store_true", help="delete the folder first")
    args = p.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    dirs = {n: args.out / n for n in
            ("01_synthetic_soft_bridge", "02_real_pages", "03_real_line_crops")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    synth = stage_synthetic(dirs["01_synthetic_soft_bridge"])
    pages = stage_real_pages(dirs["02_real_pages"], args.pages)
    crops = stage_real_crops(dirs["03_real_line_crops"])
    write_readme(args.out, synth, pages, crops)

    print(f"{args.out}")
    print(f"  01_synthetic_soft_bridge : {len(synth)} lines (frozen S2 test split)")
    print(f"  02_real_pages            : {len(pages)} pages "
          f"({sum(p['cached'] for p in pages)} cached)")
    print(f"  03_real_line_crops       : {len(crops)} crops (hand-transcribed)")
    print("  README.md                : route, cost and script for each")


if __name__ == "__main__":
    main()
