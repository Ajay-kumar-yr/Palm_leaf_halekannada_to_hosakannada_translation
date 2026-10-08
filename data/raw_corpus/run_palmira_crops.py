"""Cut real HKHPL line crops with Palmira (STATUS.md 4.3 / TRANSFER_TO_LAPTOP.md
Part 7 step 1). Run inside the `palmira` conda env on WSL, from the Palmira
checkout directory (it needs `configs/`, `pretrained/`, `predictor.py` and
`defgrid/` relative to cwd) -- same isolation as run_palmira_survey.py:

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate palmira
    cd ~/Palmira
    python /mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation/data/raw_corpus/run_palmira_crops.py --time-steps 3
    python .../run_palmira_crops.py            # real run

Palmira is used exactly as downloaded (no fine-tuning) and receives the
ORIGINAL colour photo, never black-and-white (CLAUDE.md). Output is one
crop per detected "Character Line Segment":

    data/real_lines/<set>/<page_stem>/line_NN.png        mask-filled crop
    data/real_lines/<set>/<page_stem>/line_NN_bbox.png   raw bounding-box crop
    data/real_lines/<set>/manifest.jsonl                 one record per line

Two disjoint sets (CLAUDE.md rule 8: demo_pages never appears in any
reported number):

    demo    the 14 files in demo_pages/  -- live demo only
    sample  N pages drawn at random (seeded) from the 738 surveyed pages,
            EXCLUDING anything that shares a filename stem or a photo
            signature with a demo page. This is the set the label-free
            real-imagery measurements (CLAUDE.md Part 6) run on.

No quiet filtering: every detected line is written, with its box, score and
mask area recorded, so any size/score cut is made later and visibly.
Reading order is top-to-bottom by mask-centroid y (single-column pages;
facing-page pairs will interleave -- the recorded centroid_x lets a later
step split columns).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path("/mnt/d/major_proj/Palm_leaf_halekannada_to_hosakannada_translation")
REAL_DIR = REPO_ROOT / "data" / "real"
SURVEY_CSV = REPO_ROOT / "data" / "splits" / "hkhpl_page_survey.csv"
SURVEY_JSONL = REPO_ROOT / "data" / "raw_corpus" / "palmira_survey_results.jsonl"
DEMO_SELECTION = (
    REPO_ROOT / "runs" / "20260929T051027Z_page_survey_verdict_and_demo_selection" / "results.json"
)
DEMO_DIR = REPO_ROOT / "demo_pages"
OUT_ROOT = REPO_ROOT / "data" / "real_lines"
LINE_CLASS = "Character Line Segment"
SEED = 0

# Palmira's own dependencies (detectron2, defgrid, predictor) resolve from
# cwd; the survey script next to this file supplies build_cfg.
if not (Path.cwd() / "predictor.py").exists():
    sys.exit(f"cwd must be the Palmira checkout (no predictor.py in {Path.cwd()})")
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectron2.data.detection_utils import read_image  # noqa: E402

from predictor import VisualizationDemo  # noqa: E402
from run_palmira_survey import build_cfg  # noqa: E402


def page_stem(rel: str) -> str:
    """'dataset/.../train/4.64(1).jpg' -> '4.64' (drops the '(n)' copy suffix)."""
    return re.sub(r"\(\d+\)$", "", Path(rel).stem)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        raise RuntimeError(f"Could not determine git commit hash: {e}")


def survey_signatures() -> dict[str, tuple]:
    """rel path (forward slashes) -> (w, h, sharpness, contrast, brightness)."""
    out = {}
    for r in csv.DictReader(SURVEY_CSV.open(encoding="utf-8")):
        if r["width"]:
            out[r["path"].replace("\\", "/")] = (
                r["width"], r["height"], r["sharpness"], r["contrast"], r["brightness"],
            )
    return out


def build_demo_set() -> tuple[list[tuple[str, Path]], set[str], set[tuple]]:
    """Returns ([(page_id, image_path)], excluded filename stems, excluded
    photo signatures). The demo images are the files actually in demo_pages/."""
    sel = json.loads(DEMO_SELECTION.read_text(encoding="utf-8"))["demo_pages_selected"]
    sigs = survey_signatures()
    stems, signatures = set(), set()
    for d in sel:
        rel = d["source_path"].replace("\\", "/")
        stems.add(page_stem(rel))
        if rel in sigs:
            signatures.add(sigs[rel])
    demo = []
    for f in sorted(DEMO_DIR.glob("*.jpg")):
        demo.append((f.stem, f))
    if not demo:
        raise RuntimeError(f"no .jpg files in {DEMO_DIR}")
    return demo, stems, signatures


def eligible_pages(excl_stems: set[str], excl_sigs: set[tuple]) -> dict[str, tuple[str, int]]:
    """stem -> (rel path, surveyed line count), one entry per distinct page.

    The survey's "738 unique photos" are 568 distinct pages plus 170
    higher-resolution re-photos (dataset/hd_images) of the same pages --
    the signature dedup cannot see those because the resolution differs.
    Dedupe by page stem so no page is drawn twice; prefer the standard
    Dataset/ copy over its hd twin."""
    sigs = survey_signatures()
    by_page: dict[str, tuple[str, int]] = {}
    for line in SURVEY_JSONL.open(encoding="utf-8"):
        rec = json.loads(line)
        if "error" in rec:
            continue
        n_lines = rec.get("class_counts", {}).get(LINE_CLASS, 0)
        if n_lines < 1:
            continue
        rel = rec["path"]
        stem = page_stem(rel)
        if stem in excl_stems or sigs.get(rel) in excl_sigs:
            continue
        if stem not in by_page or ("hd_images" in by_page[stem][0] and "hd_images" not in rel):
            by_page[stem] = (rel, n_lines)
    return by_page


def page_id_for(rel: str) -> str:
    # Stems are unique after the dedupe, so the stem is the page id.
    return page_stem(rel) + ("_hd" if "hd_images" in rel else "")


def build_sample_set(n: int, excl_stems: set[str], excl_sigs: set[tuple]) -> list[tuple[str, Path]]:
    # Must stay byte-identical to the draw that produced the committed
    # sample run (20261008T055116Z): same pool, same order, same seed.
    eligible = sorted(rel for rel, _ in eligible_pages(excl_stems, excl_sigs).values())
    if len(eligible) < n:
        raise RuntimeError(f"only {len(eligible)} eligible pages, wanted {n}")
    rng = random.Random(SEED)
    picked = sorted(rng.sample(eligible, n))
    return [(page_id_for(rel), REAL_DIR / rel) for rel in picked]


def build_train_set(lines_per_group: int, excl_stems: set[str], excl_sigs: set[tuple],
                    sample: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Pages for the teacher-student labels (DEMO_PLAN.md). Excludes demo
    AND sample pages, so the fine-tuned recogniser never trains on a page
    a reported number or the demo is drawn from. Balanced across the four
    manuscript groups (stem prefix 1-4, which look like four different
    hands): pages are drawn per group until the surveyed line count
    reaches `lines_per_group`."""
    sample_stems = {page_stem(str(p)) for _, p in sample}
    pool = {s: v for s, v in eligible_pages(excl_stems, excl_sigs).items() if s not in sample_stems}
    rng = random.Random(SEED + 1)
    picked = []
    for group in sorted({s.split(".")[0] for s in pool}):
        stems = sorted(s for s in pool if s.split(".")[0] == group)
        rng.shuffle(stems)
        total = 0
        for s in stems:
            if total >= lines_per_group:
                break
            rel, n_lines = pool[s]
            picked.append(rel)
            total += n_lines
        print(f"  train group {group}: {total} surveyed lines")
    return [(page_id_for(rel), REAL_DIR / rel) for rel in sorted(picked)]


def mask_filled_crop(img_bgr: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Pixels outside the (slightly dilated) line mask are replaced by the
    median colour of the mask interior, so neighbouring lines that intrude
    into a slanted line's bounding box are blanked instead of read."""
    x0, y0, x1, y1 = box
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    crop = img_bgr[y0:y1, x0:x1].copy()
    k = max(3, (y1 - y0) // 10 | 1)
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    inside = crop[m.astype(bool)]
    fill = np.median(inside, axis=0).astype(np.uint8) if len(inside) else np.uint8([255, 255, 255])
    crop[~m.astype(bool)] = fill
    return crop


def process_page(demo_gpu, demo_cpu_holder, page_id: str, path: Path, out_dir: Path, f_manifest) -> dict:
    img = read_image(str(path), format="BGR")
    t0 = time.time()
    device = "cuda"
    try:
        pred, _ = demo_gpu.run_on_image(img)
    except RuntimeError as e:
        # Same OOM handling as run_palmira_survey.py: this torch build raises
        # a plain RuntimeError, so match the message.
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        if demo_cpu_holder[0] is None:
            print("  GPU OOM -- building CPU predictor (one-time cost)")
            demo_cpu_holder[0] = VisualizationDemo(build_cfg("cpu"))
        device = "cpu"
        pred, _ = demo_cpu_holder[0].run_on_image(img)
    elapsed = time.time() - t0

    inst = pred["instances"].to("cpu")
    names = demo_gpu.metadata.thing_classes
    line_idx = [i for i, c in enumerate(inst.pred_classes.tolist()) if names[c] == LINE_CLASS]

    records = []
    for i in line_idx:
        mask = inst.pred_masks[i].numpy()
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        records.append({
            "box": (x0, y0, x1, y1),
            "mask": mask,
            "cy": float(ys.mean()), "cx": float(xs.mean()),
            "area": int(mask.sum()),
            "score": float(inst.scores[i]),
        })
    records.sort(key=lambda r: r["cy"])

    page_dir = out_dir / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(records):
        x0, y0, x1, y1 = r["box"]
        rel_a, rel_b = f"{page_id}/line_{n:02d}.png", f"{page_id}/line_{n:02d}_bbox.png"
        cv2.imwrite(str(out_dir / rel_a), mask_filled_crop(img, r["mask"], r["box"]))
        cv2.imwrite(str(out_dir / rel_b), img[y0:y1, x0:x1])
        f_manifest.write(json.dumps({
            "page": page_id,
            "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "line_index": n,
            "crop": rel_a, "crop_bbox": rel_b,
            "box_xyxy": [x0, y0, x1, y1],
            "width": x1 - x0, "height": y1 - y0,
            "mask_area": r["area"], "score": round(r["score"], 4),
            "centroid_x": round(r["cx"], 1), "centroid_y": round(r["cy"], 1),
        }, ensure_ascii=False) + "\n")
    f_manifest.flush()
    return {"page": page_id, "device": device, "seconds": round(elapsed, 2),
            "n_lines": len(records), "image_shape": list(img.shape[:2])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=50)
    ap.add_argument("--train-lines-per-group", type=int, default=160)
    ap.add_argument("--sets", default="demo,sample",
                    help="Comma list of sets to cut: demo, sample, train. Sets whose "
                         "manifest already exists are refused, never overwritten.")
    ap.add_argument("--time-steps", type=int, default=0,
                    help="Measure N pages (demo set), project the run, write NOTHING, exit.")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = set(wanted) - {"demo", "sample", "train"}
    if unknown:
        sys.exit(f"unknown set(s): {sorted(unknown)}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    demo, excl_stems, excl_sigs = build_demo_set()
    sample = build_sample_set(args.n_sample, excl_stems, excl_sigs)
    overlap = {p for p, _ in demo} & {p for p, _ in sample}
    assert not overlap, f"demo/sample overlap: {overlap}"
    assert not (excl_stems & {page_stem(str(p)) for _, p in sample}), "sample contains a demo stem"
    assert len({page_stem(str(p)) for _, p in sample}) == len(sample), "sample contains a page twice"
    train = build_train_set(args.train_lines_per_group, excl_stems, excl_sigs, sample)
    train_stems = {page_stem(str(p)) for _, p in train}
    assert not (train_stems & excl_stems), "train contains a demo page"
    assert not (train_stems & {page_stem(str(p)) for _, p in sample}), "train contains a sample page"
    assert len(train_stems) == len(train), "train contains a page twice"
    print(f"demo pages: {len(demo)}   sample pages: {len(sample)}   train pages: {len(train)} "
          f"(excluded {len(excl_stems)} demo stems, {len(excl_sigs)} photo signatures)")
    all_sets = {"demo": demo, "sample": sample, "train": train}

    demo_gpu = VisualizationDemo(build_cfg("cuda"))
    cpu_holder = [None]

    if args.time_steps:
        # Dry run: no crops, no manifest, no run folder.
        class _Null:
            def write(self, *_): pass
            def flush(self): pass
        tmp = Path("/tmp/setu_crops_timing")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        stats = [process_page(demo_gpu, cpu_holder, pid, p, tmp, _Null()) for pid, p in demo[: args.time_steps]]
        per = (time.time() - t0) / len(stats)
        total = sum(len(all_sets[s]) for s in wanted)
        print(f"measured {len(stats)} pages: {per:.1f}s/page, peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"devices {sorted({s['device'] for s in stats})}, lines {[s['n_lines'] for s in stats]}")
        print(f"projected for {total} pages: ~{per * total / 60:.1f} min (wrote nothing to the repo)")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO_ROOT / "runs" / f"{ts}_palmira_line_crops"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"seed": SEED, "sets": wanted, "n_sample": args.n_sample,
              "train_lines_per_group": args.train_lines_per_group, "line_class": LINE_CLASS,
              "palmira_weights": "pretrained/Palmira_indiscapes.pth (as downloaded)",
              "input": "original colour photo (never black-and-white)",
              "score_thresh": 0.5, "mask_fill": "median colour of dilated mask interior",
              "sample_exclusion": "filename stem or photo signature shared with any demo page",
              "train_exclusion": "every demo and sample page",
              **{f"{s}_pages": [p for p, _ in all_sets[s]] for s in wanted}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")

    results = {}
    for set_name in wanted:
        pages = all_sets[set_name]
        out_dir = OUT_ROOT / set_name
        if (out_dir / "manifest.jsonl").exists():
            raise RuntimeError(f"{out_dir}/manifest.jsonl exists -- refusing to overwrite; move it aside first")
        out_dir.mkdir(parents=True, exist_ok=True)
        page_stats = []
        with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fm:
            for k, (pid, p) in enumerate(pages):
                s = process_page(demo_gpu, cpu_holder, pid, p, out_dir, fm)
                page_stats.append(s)
                print(f"  [{set_name} {k + 1}/{len(pages)}] {pid}: {s['n_lines']} lines "
                      f"({s['device']}, {s['seconds']}s)")
        lines = [s["n_lines"] for s in page_stats]
        results[set_name] = {"n_pages": len(pages), "n_lines": sum(lines),
                             "pages_with_zero_lines": [s["page"] for s in page_stats if s["n_lines"] == 0],
                             "cpu_fallback_pages": [s["page"] for s in page_stats if s["device"] == "cpu"],
                             "pages": page_stats}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Done. Run folder: {run_dir}")


if __name__ == "__main__":
    main()
