"""Heuristic first pass over the real HKHPL photos, feeding the Week 1
"survey every HKHPL page once" task (roadmap v4: mark good/acceptable/
broken -> demo page selection + the §6.4 segmentation success-rate
number).

This does NOT replace the human judgement call the roadmap assigns to
that survey -- it computes cheap, objective image-quality signals
(sharpness, contrast, exposure, resolution) and writes a heuristic
suggested bucket per page, so a human reviewer can sort/skim by that
column and confirm or correct roughly 1,259 photos in far less time than
opening each one cold. The output has an empty `human_verdict` column for
exactly that purpose, and CLAUDE.md's own accuracy language
("good/acceptable/broken") is kept as the heuristic's vocabulary so the
two are directly comparable.

Sharpness is the variance of a 3x3 Laplacian on a downsampled grayscale
copy (no scipy/opencv dependency -- ask-before-adding-a-dependency per
CLAUDE.md, and a hand-rolled 3x3 convolution is simple enough not to need
one). Images are downsampled to a max side of 512px before any heuristic
is computed, purely for speed -- 1,259 photos at up to 3264x2448 would
otherwise make this a multi-minute-per-metric pass.

Usage:
    python -m setu.data.survey --real-dir data/real --out data/splits/hkhpl_page_survey.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SIDE = 512
_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _list_photos(real_dir: Path) -> list[Path]:
    """Same real-photo selection as setu.render.textures: recursive, and
    excluding Ground_Truth_images (pre-binarized, not photos)."""
    return sorted(
        p
        for p in real_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _EXTS
        and not any("ground_truth" in part.lower() for part in p.parts)
    )


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian -- a standard cheap blur proxy (sharp
    edges produce high-magnitude second derivatives; a blurred photo has
    almost none)."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    lap = (
        k[0, 1] * np.roll(g, 1, axis=0)
        + k[1, 0] * np.roll(g, 1, axis=1)
        + k[1, 1] * g
        + k[1, 2] * np.roll(g, -1, axis=1)
        + k[2, 1] * np.roll(g, -1, axis=0)
    )
    # edges wrap with np.roll and would skew the variance; drop a 1px border
    return float(np.var(lap[1:-1, 1:-1]))


def compute_metrics(path: Path) -> dict:
    with Image.open(path) as im:
        width, height = im.size
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
        gray = np.array(im.convert("L"), dtype=np.uint8)

    sharpness = _laplacian_variance(gray)
    contrast = float(gray.std())
    brightness = float(gray.mean())
    return {
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "brightness": round(brightness, 2),
    }


def heuristic_bucket(m: dict) -> str:
    """Rough, tunable thresholds -- meant to rank candidates for human
    review, not to be trusted as ground truth. `broken` = likely unusable
    (near-blank, extreme exposure); `acceptable` = usable but marginal
    (soft focus or low contrast); `good` = everything else."""
    if m["contrast"] < 12 or m["brightness"] < 15 or m["brightness"] > 240:
        return "broken"
    if m["sharpness"] < 40 or m["contrast"] < 25:
        return "acceptable"
    return "good"


def run_survey(real_dir: Path, out_path: Path) -> None:
    photos = _list_photos(real_dir)
    if not photos:
        raise RuntimeError(f"No real photos found under {real_dir} (checked recursively).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"good": 0, "acceptable": 0, "broken": 0}
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["path", "width", "height", "sharpness", "contrast", "brightness",
             "heuristic_bucket", "human_verdict", "notes"]
        )
        for i, path in enumerate(photos):
            rel = path.relative_to(real_dir)
            try:
                m = compute_metrics(path)
            except Exception as e:  # a genuinely broken/corrupt file is exactly a real finding, not a crash
                writer.writerow([rel, "", "", "", "", "", "broken", "", f"unreadable: {e}"])
                counts["broken"] += 1
                continue
            bucket = heuristic_bucket(m)
            counts[bucket] += 1
            writer.writerow(
                [rel, m["width"], m["height"], m["sharpness"], m["contrast"],
                 m["brightness"], bucket, "", ""]
            )
            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(photos)} photos scanned")

    print(f"\nScanned {len(photos)} photos -> {out_path}")
    print(f"Heuristic buckets: {counts}")
    print(
        "This is a first pass, not the survey itself -- sort by "
        "heuristic_bucket/sharpness and fill in human_verdict for the "
        "roadmap's Week 1 survey and the §6.4 segmentation success-rate number."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=Path("data/splits/hkhpl_page_survey.csv"))
    args = parser.parse_args()
    run_survey(args.real_dir, args.out)


if __name__ == "__main__":
    main()
