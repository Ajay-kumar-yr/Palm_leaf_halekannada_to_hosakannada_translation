"""Train the CRNN recogniser on S1 (CLAUDE.md Week 2: "train CRNN on S1").

Run only after `setu.recogniser.memorize_check` passes (CLAUDE.md rule 3);
this script does not re-check that itself, it's a separate, already-logged
run.

Usage:
    python -m setu.recogniser.train --manifest data/s1/manifest.jsonl --data-dir data/s1

Images on disk are ink-dark-on-light (setu.render.generate saves
compose_damage's output directly, no inversion) -- fed to the model as-is.
Note this differs from setu.recogniser.memorize_check, which feeds
render_clean_line's raw ink=255-bright-on-dark mask directly (no damage
compositing at all, since that check is only exercising the training loop
mechanics on clean text); the two conventions don't need to match, since
nothing requires the model to see the same polarity in both cases -- only
that a given image's polarity is what its label was matched against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Must be set before torch's CUDA allocator initializes (first `import torch`
# touching cuda). S1's images vary enormously in width/height, so batches hit
# a huge range of padded tensor shapes; PyTorch's default allocator caches a
# separate block per distinct size it has ever seen and never coalesces them,
# so GPU memory.used (as reported by nvidia-smi) ratchets upward through an
# epoch independent of --max-pixels-per-batch -- measured climbing from ~5GB
# to ~12GB/12GB (RTX 3060) with no new training-loop output in between, i.e.
# no single batch got bigger, the allocator's reserved pool just kept
# growing. expandable_segments lets the allocator grow/shrink existing
# segments instead of hoarding one per size, which is exactly this failure
# mode's documented fix. setdefault so an operator's own env setting wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from setu.data import wx
from setu.eval.metrics import cer
from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE, greedy_decode
from setu.render.augment import augment_line
from setu.runlog import finish_run, start_run


def _load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    for line in manifest_path.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        records.append(rec)
    return records


def _split_train_val_indices(records: list[dict], val_fraction: float) -> tuple[list[int], list[int]]:
    """Deterministic hash-of-id split -- same rationale as CLAUDE.md rule 2
    for S2/gold: stable regardless of manifest ordering or future growth.
    Returns indices (not records) so callers can slice parallel arrays
    (e.g. pre-scanned image dimensions) consistently with the split."""
    threshold = int(val_fraction * 256)
    train_idx, val_idx = [], []
    for i, rec in enumerate(records):
        digest = hashlib.sha256(f"s1:{rec['id']}".encode("utf-8")).digest()
        (val_idx if digest[0] < threshold else train_idx).append(i)
    return train_idx, val_idx


def scan_image_dims(records: list[dict], data_dir: Path) -> list[tuple[int, int]]:
    """(height, width) per record via a PIL header read (lazy -- .size
    reads only the image header, not the full pixel data, so this is
    much cheaper than opening every image fully). Needed because text
    length is NOT a reliable proxy for rendered pixel area: height varies
    independently of character count (damage.py's skew rotation uses
    PIL's expand=True, and the skew angle is random per line, so two
    lines of the same length can differ 2-3x in height). Measured: a
    single 700-char, heavily-skewed line was 924x14561px and used 3.3GB
    for a forward pass ALONE on this 4GB GPU -- char-length-based
    batching cannot see that coming, only actual pixel dimensions can."""
    dims = []
    for rec in records:
        rel_path = rec["image_path"].replace("\\", "/")
        with Image.open(data_dir / rel_path) as im:
            w, h = im.size
        dims.append((h, w))
    return dims


class AreaBucketBatchSampler(Sampler[list[int]]):
    """Bounds batch_size * (max_height_in_batch * max_width_in_batch) --
    actual padded pixel area, not a text-length proxy -- to a fixed
    budget calibrated against measured GPU memory (see scan_image_dims).
    Short/short lines get large batches, huge images get batches of 1;
    images whose OWN area already exceeds the budget are excluded
    entirely by the caller (no batch size can make them fit). Sorts by
    area so within-batch padding waste stays low; batch ORDER is
    shuffled each epoch so training isn't biased toward seeing
    short-then-long."""

    def __init__(
        self, indices: list[int], dims: list[tuple[int, int]], max_pixels_per_batch: int, seed: int,
        max_batch_size: int = 64, drop_last: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        order = sorted(indices, key=lambda i: dims[i][0] * dims[i][1])

        batches: list[list[int]] = []
        current: list[int] = []
        current_max_h, current_max_w = 0, 0
        for idx in order:
            h, w = dims[idx]
            prospective_h, prospective_w = max(current_max_h, h), max(current_max_w, w)
            prospective_area = prospective_h * prospective_w
            would_exceed_budget = current and (len(current) + 1) * prospective_area > max_pixels_per_batch
            would_exceed_size = len(current) >= max_batch_size
            if current and (would_exceed_budget or would_exceed_size):
                batches.append(current)
                current, current_max_h, current_max_w = [idx], h, w
            else:
                current.append(idx)
                current_max_h, current_max_w = prospective_h, prospective_w
        if current and not (drop_last and len(current) < 2):
            batches.append(current)
        self.batches = batches

    def __iter__(self):
        order = self.rng.permutation(len(self.batches))
        for i in order:
            yield self.batches[i]

    def __len__(self) -> int:
        return len(self.batches)


class S1Dataset(Dataset):
    def __init__(self, records: list[dict], data_dir: Path, augment: bool, seed: int):
        self.records = records
        self.data_dir = data_dir
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[int], str]:
        rec = self.records[idx]
        # normalize backslashes: a manifest written on Windows (rendering needs a
        # Windows font path) has "images\\line_000000.png", which Linux treats as
        # one literal filename rather than a path separator.
        rel_path = rec["image_path"].replace("\\", "/")
        img = np.array(Image.open(self.data_dir / rel_path).convert("L"), dtype=np.uint8)
        if self.augment:
            img = augment_line(img, self.rng)
        symbols = wx.encode(rec["text"])
        labels = [wx.SYMBOL_TO_INDEX[s] for s in symbols]
        return img, labels, rec["text"]


def collate(batch: list[tuple[np.ndarray, list[int], str]]):
    """Pads to the batch's max height AND width -- damage.py's skew rotation
    uses PIL's `expand=True` to avoid clipping corners, which grows a wide,
    short line strip's height well past LINE_HEIGHT (64 -> 128+ is common),
    so images in a batch do NOT all share the same height. The model's
    AdaptiveAvgPool2d collapses whatever height survives the conv blocks to
    1, so it tolerates this; only the padding here needs to account for it.

    Takes WIDTH_DOWNSAMPLE directly rather than a CRNN instance: passing a
    live nn.Module into DataLoader worker processes means pickling it on
    every worker spawn, and on Python 3.14/Linux (forkserver default) a
    bound-model closure fails to pickle at all -- output_length() only
    ever depended on the fixed downsample constant, not any learned
    parameter, so there's no need to carry the model here."""
    imgs, label_seqs, texts = zip(*batch)
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)
    padded = np.zeros((len(imgs), 1, max_h, max_w), dtype=np.float32)
    widths = []
    for i, img in enumerate(imgs):
        h, w = img.shape
        padded[i, 0, :h, :w] = img.astype(np.float32) / 255.0
        widths.append(w)

    input_lengths = torch.tensor([w // WIDTH_DOWNSAMPLE for w in widths], dtype=torch.long)
    target_lengths = torch.tensor([len(s) for s in label_seqs], dtype=torch.long)
    targets = torch.tensor([idx for seq in label_seqs for idx in seq], dtype=torch.long)
    return torch.from_numpy(padded), targets, input_lengths, target_lengths, label_seqs, texts


def evaluate(model: CRNN, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    total_cer, n = 0.0, 0
    with torch.no_grad():
        for i, (images, _targets, input_lengths, _target_lengths, label_seqs, _texts) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device)
            log_probs = model(images)
            decoded = greedy_decode(log_probs, input_lengths)
            for ref, hyp in zip(label_seqs, decoded):
                total_cer += cer(ref, hyp)
                n += 1
    model.train()
    return total_cer / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/s1/manifest.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/s1"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--max-pixels-per-batch", type=int, default=3_000_000,
        help="bounds batch_size * (max_height_in_batch * max_width_in_batch) -- actual padded "
             "pixel area. See AreaBucketBatchSampler. NOTE: a single-image (batch_size=1) "
             "profiling pass (varying h*w, see HANDOFF.md Step 5) suggested 6M was safe on an "
             "RTX 3060 12GB, extrapolating memory as linear in total padded pixels -- that "
             "extrapolation was wrong in practice: cuDNN's algorithm/workspace selection for "
             "real multi-image batch *shapes* is not simply a function of total element count, "
             "and 6M drove memory.used to the 12GB ceiling within one epoch (recovered from an "
             "OOM warning, then degraded to >150s/batch instead of crashing). 3M was verified "
             "end-to-end: a full epoch (21,272 train lines, RTX 3060 12GB, with "
             "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True also set, see top of file) "
             "peaked at ~5GB and completed in 2654s (~44min) with 0 batches skipped. "
             "Re-measure on any other GPU; don't just scale this number by VRAM ratio.",
    )
    parser.add_argument(
        "--max-single-image-pixels", type=int, default=5_000_000,
        help="images above this are dropped from training, not just batched differently. "
             "Re-profiled on an RTX 3060 12GB (previous defaults -- 2.2M/1.5M -- were tuned "
             "against a 4GB RTX 3050 and are far too conservative here). Method: single "
             "forward+backward+optimizer-step timing, torch.cuda.synchronize() around the "
             "timed block, one excluded warmup call first. Tall+wide images are "
             "disproportionately expensive, not just memory-heavy, at a given pixel AREA -- "
             "e.g. at height=924 (an extreme skew-rotation outlier), step time stayed under "
             "0.7s through 4.62M px but jumped to 1.6-6s by 7.39M px, a non-linear cliff tied "
             "to aspect ratio (CTC's conv blocks only pool width in blocks 1-3, so very tall "
             "images keep disproportionately large intermediate feature maps through blocks "
             "4-5), not pure OOM. 5M sits safely below that cliff and keeps 95.6%% of S1 "
             "(22,327/23,346 lines) vs. the old 1.5M cutoff's 75.2%%.",
    )
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=40, help="cap validation batches per epoch for speed")
    parser.add_argument("--max-lines-per-epoch", type=int, default=None, help="debug: cap train examples/epoch")
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="a best_model.pt checkpoint to load model weights from before training -- for "
             "running a long training budget in several shorter sessions (e.g. 4 epochs at a "
             "time toward a 12-epoch total). Mirrors setu.modernizer.pretrain's --resume-from: "
             "optimizer state is not saved/resumed, Adam restarts fresh each session (a minor "
             "imperfection, not worth checkpointing optimizer state at this scale).",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=1,
        help="epoch number to label the first epoch of this run as, for continuity in "
             "logs/history when resuming (e.g. 5 if epochs 1-4 already ran)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # CLAUDE.md rule 6
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = _load_manifest(args.manifest)
    n_before = len(records)
    print(f"Scanning image dimensions for {n_before} lines (one-time pass, header-only reads)...")
    dims_all = scan_image_dims(records, args.data_dir)
    keep = [i for i, (h, w) in enumerate(dims_all) if h * w <= args.max_single_image_pixels]
    n_dropped_huge = n_before - len(keep)
    records = [records[i] for i in keep]
    dims_all = [dims_all[i] for i in keep]

    train_idx, val_idx = _split_train_val_indices(records, args.val_fraction)
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    train_dims = [dims_all[i] for i in train_idx]
    val_dims = [dims_all[i] for i in val_idx]
    if args.max_lines_per_epoch is not None:
        train_records = train_records[: args.max_lines_per_epoch]
        train_dims = train_dims[: args.max_lines_per_epoch]
    print(
        f"S1: {n_before} lines total ({n_dropped_huge} dropped, over "
        f"{args.max_single_image_pixels:,} px) -> {len(train_records)} train, {len(val_records)} val (device={device})"
    )

    model = CRNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CRNN parameters: {n_params:,}")

    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"Resumed weights from {args.resume_from}")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = S1Dataset(train_records, args.data_dir, augment=True, seed=args.seed)
    val_ds = S1Dataset(val_records, args.data_dir, augment=False, seed=args.seed)

    train_sampler = AreaBucketBatchSampler(
        list(range(len(train_records))), train_dims, args.max_pixels_per_batch, seed=args.seed,
        max_batch_size=args.max_batch_size, drop_last=True,
    )
    val_sampler = AreaBucketBatchSampler(
        list(range(len(val_records))), val_dims, args.max_pixels_per_batch, seed=args.seed,
        max_batch_size=args.max_batch_size, drop_last=False,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate,
    )

    run_dir = start_run(
        "crnn_train_s1",
        {
            "seed": args.seed,
            "manifest": str(args.manifest),
            "max_single_image_pixels": args.max_single_image_pixels,
            "n_dropped_huge": n_dropped_huge,
            "n_train": len(train_records),
            "n_val": len(val_records),
            "val_fraction": args.val_fraction,
            "epochs": args.epochs,
            "max_pixels_per_batch": args.max_pixels_per_batch,
            "max_batch_size": args.max_batch_size,
            "lr": args.lr,
            "device": str(device),
            "n_params": n_params,
            "vocab_size": len(wx.VOCAB),
            "augmentation": "setu.render.augment.augment_line (CPU, per-example) -- GPU-batched "
                            "augmentation from the roadmap's speed guidance is not yet implemented",
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "start_epoch": args.start_epoch,
        },
    )

    history = []
    best_val_cer = float("inf")
    best_epoch = None
    end_epoch = args.start_epoch + args.epochs - 1

    for epoch in range(args.start_epoch, end_epoch + 1):
        epoch_start = time.time()
        model.train()
        total_loss, n_batches, n_skipped_bad_length = 0.0, 0, 0

        for images, targets, input_lengths, target_lengths, _label_seqs, _texts in train_loader:
            if bool((input_lengths < target_lengths).any()):
                n_skipped_bad_length += 1
                continue
            images = images.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        mean_train_loss = total_loss / max(n_batches, 1)
        val_cer = evaluate(model, val_loader, device, max_batches=args.val_batches)
        epoch_time = time.time() - epoch_start

        print(
            f"epoch {epoch:3d}/{end_epoch}  train_loss {mean_train_loss:.4f}  "
            f"val_CER {val_cer:.4f}  ({epoch_time:.1f}s, {n_skipped_bad_length} batches skipped: "
            f"CTC input shorter than target)"
        )
        history.append({"epoch": epoch, "train_loss": mean_train_loss, "val_cer": val_cer, "seconds": epoch_time})

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    finish_run(
        run_dir,
        {
            "history": history,
            "best_val_cer": best_val_cer,
            "best_epoch": best_epoch,
            "final_val_cer": history[-1]["val_cer"] if history else None,
        },
    )
    print(f"\nRun folder: {run_dir}")
    print(f"Best val CER {best_val_cer:.4f} at epoch {best_epoch} -> {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
