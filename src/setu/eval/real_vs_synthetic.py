"""CLAUDE.md Part 6, label-free real-imagery measurements #2 and #3:
the recogniser's top-1 confidence distribution on REAL vs SYNTHETIC
lines, and the flagging rate on each. No transcriptions of real HKHPL
lines exist, so neither measurement needs ground truth -- that is the
whole point ("a sharp drop quantifies the domain gap with zero
transcriptions needed").

Synthetic side: the S1 validation split, via train.py's own manifest
loading, hash-based split and collate -- the same held-out lines the
committed baseline `20261006T152242Z_measure_confidence_distribution`
measured (mean top-1 0.9606, 11.69% of frames below 0.9). This script
recomputes it rather than quoting it, so the two sides come from one
code path on one machine, and prints the committed numbers alongside as
a reproduction check.

Real side: the line crops Palmira cut from the 50-page sample set
(`run_palmira_crops.py`), which is disjoint from `demo_pages/` -- rule 8:
demo pages never appear in a reported number.

Four real conditions are measured, not one, so the result cannot be
dismissed as a preprocessing artifact:

    mask x as_is    mask-filled crop, native resolution
    mask x h64      mask-filled crop, resized to height 64 (the renderer's
                    nominal line_height, so glyph scale matches training)
    bbox x as_is    raw bounding-box crop, native resolution
    bbox x h64      raw bounding-box crop, resized to height 64

Per-frame statistics come from `soft_bridge.measure_confidence_distribution`
and per-line means from `flagging.mean_line_confidence` -- the same bridge
machinery used everywhere else, not reimplemented here.

Flagging rate is reported across several thresholds rather than one:
CLAUDE.md leaves the threshold to be tuned and `flagging.py` deliberately
requires it as an argument, so picking a single value here would be an
undisclosed choice.

Usage:
    python -m setu.eval.real_vs_synthetic --checkpoint runs/<...>/best_model.pt --time-steps 20
    python -m setu.eval.real_vs_synthetic --checkpoint runs/<...>/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# S1's wildly variable image sizes fragment the default allocator (STATUS.md
# 3.8); set before torch allocates anything.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from setu.bridge.flagging import mean_line_confidence  # noqa: E402
from setu.bridge.soft_bridge import measure_confidence_distribution  # noqa: E402
from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE, greedy_decode  # noqa: E402
from setu.recogniser.train import (  # noqa: E402
    AreaBucketBatchSampler,
    S1Dataset,
    _load_manifest,
    _split_train_val_indices,
    collate,
    scan_image_dims,
)
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
LINE_HEIGHT = 64  # setu.render.generate's nominal line_height
# The CRNN halves height five times (blocks 1-3 pool 2x2, blocks 4-5 pool
# 2x1), so an input shorter than 32px collapses to zero rows and the
# forward pass raises. S1's shortest rendered line is 78px, so training
# never met this; 12 of the 712 real crops are under 32px. They are padded
# (not rescaled) to 32 with the crop's median value, and counted in
# `n_padded_to_min_height` -- the alternative, dropping them, would quietly
# bias the real side toward the crops that happen to be tall.
MIN_HEIGHT = 32
THRESHOLDS = (0.75, 0.80, 0.85, 0.90)
# Committed synthetic baseline, for a reproduction check (not a target).
BASELINE_RUN = "20261006T152242Z_measure_confidence_distribution"


def load_real_crop(path: Path, scale: str) -> tuple[np.ndarray, bool]:
    """Grayscale uint8, ink-dark-on-light -- the same polarity the renderer
    writes to disk (generate.py applies no inversion), so no inversion here."""
    im = Image.open(path).convert("L")
    if scale == "h64":
        w = max(WIDTH_DOWNSAMPLE, round(im.width * LINE_HEIGHT / im.height))
        im = im.resize((w, LINE_HEIGHT), Image.LANCZOS)
    elif scale != "as_is":
        raise ValueError(f"unknown scale {scale!r}")
    a = np.array(im, dtype=np.uint8)
    if a.shape[0] < MIN_HEIGHT:
        pad = MIN_HEIGHT - a.shape[0]
        top = pad // 2
        a = np.pad(a, ((top, pad - top), (0, 0)), mode="constant",
                   constant_values=int(np.median(a)))
        return a, True
    return a, False


def accumulate(stats: dict, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> None:
    """Fold one batch into the running frame-level totals and per-line list."""
    input_lengths = input_lengths.to(log_probs.device)
    d = measure_confidence_distribution(log_probs, input_lengths)
    n = d["n_frames"]
    stats["n_frames"] += n
    stats["_sum_conf"] += d["mean_top1_confidence"] * n
    stats["_sum_lo90"] += d["fraction_below_0.9"] * n
    stats["_sum_lo99"] += d["fraction_below_0.99"] * n
    stats["line_conf"].extend(mean_line_confidence(log_probs, input_lengths).cpu().tolist())


def new_stats() -> dict:
    return {"n_frames": 0, "_sum_conf": 0.0, "_sum_lo90": 0.0, "_sum_lo99": 0.0, "line_conf": []}


def finalize(stats: dict) -> dict:
    n = max(stats["n_frames"], 1)
    lc = stats["line_conf"]
    out = {
        "n_lines": len(lc),
        "n_frames": stats["n_frames"],
        "mean_top1_confidence": stats["_sum_conf"] / n,
        "fraction_below_0.9": stats["_sum_lo90"] / n,
        "fraction_below_0.99": stats["_sum_lo99"] / n,
        "line_confidence": {
            "mean": statistics.fmean(lc) if lc else None,
            "median": statistics.median(lc) if lc else None,
            "min": min(lc) if lc else None,
            "max": max(lc) if lc else None,
        },
        "flagging_rate": {
            f"{t:.2f}": sum(1 for c in lc if c < t) / len(lc) if lc else None for t in THRESHOLDS
        },
    }
    return out


@torch.no_grad()
def measure_real(model, device, records, root: Path, field: str, scale: str, limit: int | None) -> tuple[dict, list]:
    stats = new_stats()
    per_line = []
    n_padded = 0
    for rec in records[: limit or len(records)]:
        gray, padded = load_real_crop(root / rec[field], scale)
        n_padded += int(padded)
        x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(device)
        log_probs = model(x)
        il = torch.tensor([max(1, gray.shape[1] // WIDTH_DOWNSAMPLE)], dtype=torch.long)
        if il[0] > log_probs.shape[0]:  # fail loudly rather than silently truncating
            raise RuntimeError(f"{rec[field]}: {int(il[0])} frames expected, model gave {log_probs.shape[0]}")
        accumulate(stats, log_probs, il)
        ids = greedy_decode(log_probs.cpu(), il)[0]
        per_line.append({
            "page": rec["page"], "crop": rec[field], "field": field, "scale": scale,
            "width": gray.shape[1], "height": gray.shape[0],
            "palmira_score": rec["score"],
            "mean_confidence": round(stats["line_conf"][-1], 4),
            "n_decoded_symbols": len(ids),
            "padded_to_min_height": padded,
        })
    out = finalize(stats)
    out["n_padded_to_min_height"] = n_padded
    return out, per_line


@torch.no_grad()
def measure_synthetic(model, device, args, limit: int | None) -> tuple[dict, list]:
    records = _load_manifest(args.s1_manifest)
    dims = scan_image_dims(records, args.s1_dir)
    keep = [i for i, (h, w) in enumerate(dims) if h * w <= args.max_single_image_pixels]
    records, dims = [records[i] for i in keep], [dims[i] for i in keep]
    _, val_idx = _split_train_val_indices(records, args.val_fraction)
    val_records, val_dims = [records[i] for i in val_idx], [dims[i] for i in val_idx]
    if limit:
        val_records, val_dims = val_records[:limit], val_dims[:limit]

    ds = S1Dataset(val_records, args.s1_dir, augment=False, seed=SEED)
    sampler = AreaBucketBatchSampler(
        list(range(len(val_records))), val_dims, args.max_pixels_per_batch, seed=SEED,
        max_batch_size=args.max_batch_size, drop_last=False,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=collate)

    stats = new_stats()
    for images, _t, input_lengths, _tl, _ls, _tx in loader:
        accumulate(stats, model(images.to(device)), input_lengths)
    return finalize(stats), []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/20261006T120718Z_crnn_train_s1/best_model.pt"))
    p.add_argument("--real-manifest", type=Path, default=Path("data/real_lines/sample/manifest.jsonl"))
    p.add_argument("--s1-manifest", type=Path, default=Path("data/s1/manifest.jsonl"))
    p.add_argument("--s1-dir", type=Path, default=Path("data/s1"))
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-pixels-per-batch", type=int, default=2_200_000)  # laptop 3050 value
    p.add_argument("--max-single-image-pixels", type=int, default=5_000_000)
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-synthetic", action="store_true",
                   help="Measure only the real side (synthetic needs data/s1, 12 GB).")
    p.add_argument("--time-steps", type=int, default=0,
                   help="Measure N real lines, project the run, write NOTHING, exit.")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_root = args.real_manifest.parent
    real_records = [json.loads(l) for l in args.real_manifest.open(encoding="utf-8")]
    print(f"real: {len(real_records)} line crops from {len({r['page'] for r in real_records})} pages"
          f"   device={device}")

    model = CRNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    conditions = [(f, s) for f in ("crop", "crop_bbox") for s in ("as_is", "h64")]

    if args.time_steps:
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        measure_real(model, device, real_records, real_root, "crop", "as_is", args.time_steps)
        per = (time.time() - t0) / args.time_steps
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"measured {args.time_steps} real lines: {per:.2f}s/line, peak GPU {peak:.2f} GiB")
        print(f"projected real side ({len(conditions)} conditions x {len(real_records)} lines): "
              f"~{per * len(real_records) * len(conditions) / 60:.1f} min")
        print("synthetic side adds an S1 header scan (~4 min on this machine) plus ~1055 val lines")
        print("wrote nothing")
        return

    results = {"checkpoint": str(args.checkpoint), "device": str(device), "real": {}}
    per_line_all = []
    for field, scale in conditions:
        t0 = time.time()
        r, pl = measure_real(model, device, real_records, real_root, field, scale, None)
        r["seconds"] = round(time.time() - t0, 1)
        results["real"][f"{field}:{scale}"] = r
        per_line_all.extend(pl)
        print(f"  real {field:9s} {scale:6s}  mean top-1 {r['mean_top1_confidence']:.4f}  "
              f"<0.9 {r['fraction_below_0.9']:.4f}  lines {r['n_lines']}  frames {r['n_frames']:,}")

    if not args.skip_synthetic:
        t0 = time.time()
        s, _ = measure_synthetic(model, device, args, None)
        s["seconds"] = round(time.time() - t0, 1)
        results["synthetic_s1_val"] = s
        print(f"  synthetic S1 val      mean top-1 {s['mean_top1_confidence']:.4f}  "
              f"<0.9 {s['fraction_below_0.9']:.4f}  lines {s['n_lines']}  frames {s['n_frames']:,}")
        baseline = json.loads((Path("runs") / BASELINE_RUN / "results.json").read_text())
        results["baseline_agreement_check"] = {
            "committed_run": BASELINE_RUN,
            "caveat": "NOT bit-exact reproduction unless this machine's data/s1 is the same "
                      "render the committed run used -- compare n_frames, which is sum(w//8) "
                      "over the split and so differs only if the images themselves differ",
            "committed_mean_top1": baseline["mean_top1_confidence"],
            "committed_fraction_below_0.9": baseline["fraction_below_0.9"],
            "delta_mean_top1": s["mean_top1_confidence"] - baseline["mean_top1_confidence"],
            "delta_fraction_below_0.9": s["fraction_below_0.9"] - baseline["fraction_below_0.9"],
        }
        primary = results["real"]["crop:as_is"]
        results["domain_gap"] = {
            "primary_real_condition": "crop:as_is",
            "mean_top1_real_minus_synthetic": primary["mean_top1_confidence"] - s["mean_top1_confidence"],
            "fraction_below_0.9_real_minus_synthetic": primary["fraction_below_0.9"] - s["fraction_below_0.9"],
            "flagging_rate_real_minus_synthetic": {
                k: primary["flagging_rate"][k] - s["flagging_rate"][k] for k in primary["flagging_rate"]
            },
        }

    run_dir = start_run("real_vs_synthetic_confidence", {
        "seed": SEED,
        "checkpoint": str(args.checkpoint),
        "real_manifest": str(args.real_manifest),
        "real_conditions": [f"{f}:{s}" for f, s in conditions],
        "thresholds": list(THRESHOLDS),
        "s1_manifest": str(args.s1_manifest),
        "val_fraction": args.val_fraction,
        "skip_synthetic": args.skip_synthetic,
        "note": "label-free: no transcriptions of real HKHPL lines exist, so this measures "
                "confidence and flagging only, never accuracy on real imagery",
    })
    with (run_dir / "per_line_real.jsonl").open("w", encoding="utf-8") as f:
        for row in per_line_all:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finish_run(run_dir, results)
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
