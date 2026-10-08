"""Binarize labelled line crops with the Sajjan U-Net (setu.demo.binarize).

The glyph-stitching experiment failed on grayscale crops because every
glyph carried its own patch of leaf, so joins showed as brightness
seams. Binarizing removes the leaf: ink on uniform white, which stitches
without a seam. HKHPL ships its own binarized pages, but they cover only
3 of the 7 gold pages and are badly degraded on at least one (a large
black blob swallows a third of a line), so this uses the U-Net trained
on this dataset instead -- uniform quality over every crop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from setu.demo.binarize import binarize, load_binarizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-labels", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_binarizer(device=dev)
    rows = [json.loads(l) for l in args.labels.open(encoding="utf-8")]
    out = []
    for r in rows:
        src = Path(r["image"])
        dst = args.out_dir / r["crop"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        arr = binarize(Image.open(src), model, device=dev)
        Image.fromarray(arr).save(dst)
        out.append({**r, "image": dst.as_posix(), "binarized": "sajjan_unet"})
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    print(f"{len(out)} lines binarized -> {args.out_dir} (device={dev})")


if __name__ == "__main__":
    main()
