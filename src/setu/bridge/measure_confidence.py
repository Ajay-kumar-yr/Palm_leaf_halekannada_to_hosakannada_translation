"""CLAUDE.md Week 3: measure the fraction of frames with top-1 confidence
below 0.9 over the S1 validation split, using a real trained CRNN
checkpoint -- not Week 4 ("Measure the fraction of frames with top-1
confidence below 0.9 in Week 3, not Week 4; if that fraction is tiny,
raise the temperature"). Decides whether the soft bridge's temperature
needs raising above 1 before the blend is used anywhere for real: if CTC
is overconfident (as it typically is), too few sub-0.9 frames means the
blend would be numerically near-identical to argmax, erasing the
B3-vs-B4 comparison this whole project is built around.

Reuses setu.recogniser.train's manifest-loading, hash-based val split,
dataset, batching and collate logic directly rather than reimplementing
it, so this measures the exact same held-out validation set training
itself evaluated against.

Usage:
    python -m setu.bridge.measure_confidence --checkpoint runs/<...>/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from setu.bridge.soft_bridge import measure_confidence_distribution
from setu.recogniser.model import CRNN
from setu.recogniser.train import (
    AreaBucketBatchSampler,
    S1Dataset,
    _load_manifest,
    _split_train_val_indices,
    collate,
    scan_image_dims,
)
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/s1/manifest.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/s1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-pixels-per-batch", type=int, default=3_000_000)
    parser.add_argument("--max-single-image-pixels", type=int, default=5_000_000)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = _load_manifest(args.manifest)
    dims_all = scan_image_dims(records, args.data_dir)
    keep = [i for i, (h, w) in enumerate(dims_all) if h * w <= args.max_single_image_pixels]
    records = [records[i] for i in keep]
    dims_all = [dims_all[i] for i in keep]

    _, val_idx = _split_train_val_indices(records, args.val_fraction)
    val_records = [records[i] for i in val_idx]
    val_dims = [dims_all[i] for i in val_idx]
    print(f"Val split: {len(val_records)} lines (device={device})")

    model = CRNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    val_ds = S1Dataset(val_records, args.data_dir, augment=False, seed=SEED)
    val_sampler = AreaBucketBatchSampler(
        list(range(len(val_records))), val_dims, args.max_pixels_per_batch, seed=SEED,
        max_batch_size=args.max_batch_size, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate,
    )

    total_frames = 0
    weighted_mean_conf = 0.0
    weighted_below_09 = 0.0
    weighted_below_099 = 0.0

    with torch.no_grad():
        for images, _targets, input_lengths, _target_lengths, _label_seqs, _texts in val_loader:
            images = images.to(device)
            input_lengths = input_lengths.to(device)
            log_probs = model(images)
            stats = measure_confidence_distribution(log_probs, input_lengths)
            n = stats["n_frames"]
            total_frames += n
            weighted_mean_conf += stats["mean_top1_confidence"] * n
            weighted_below_09 += stats["fraction_below_0.9"] * n
            weighted_below_099 += stats["fraction_below_0.99"] * n

    overall = {
        "n_frames": total_frames,
        "mean_top1_confidence": weighted_mean_conf / total_frames,
        "fraction_below_0.9": weighted_below_09 / total_frames,
        "fraction_below_0.99": weighted_below_099 / total_frames,
    }

    run_dir = start_run(
        "measure_confidence_distribution",
        {
            "seed": SEED,
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "val_fraction": args.val_fraction,
            "max_single_image_pixels": args.max_single_image_pixels,
            "max_pixels_per_batch": args.max_pixels_per_batch,
            "n_val_records": len(val_records),
            "device": str(device),
        },
    )
    finish_run(run_dir, overall)

    print(f"n_frames: {overall['n_frames']:,}")
    print(f"mean_top1_confidence: {overall['mean_top1_confidence']:.4f}")
    print(f"fraction_below_0.9: {overall['fraction_below_0.9']:.4f}")
    print(f"fraction_below_0.99: {overall['fraction_below_0.99']:.4f}")
    print(f"Run folder: {run_dir}")
    if overall["fraction_below_0.9"] < 0.05:
        print(
            "fraction_below_0.9 is small -- per CLAUDE.md, the CTC model is overconfident "
            "and the soft bridge's temperature likely needs raising above 1 for the B3-vs-B4 "
            "comparison to show a real difference."
        )


if __name__ == "__main__":
    main()
