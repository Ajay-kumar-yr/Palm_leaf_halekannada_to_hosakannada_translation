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
import time
from pathlib import Path

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


def _split_train_val(records: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    """Deterministic hash-of-id split -- same rationale as CLAUDE.md rule 2
    for S2/gold: stable regardless of manifest ordering or future growth."""
    threshold = int(val_fraction * 256)
    train, val = [], []
    for rec in records:
        digest = hashlib.sha256(f"s1:{rec['id']}".encode("utf-8")).digest()
        (val if digest[0] < threshold else train).append(rec)
    return train, val


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Rendered line width varies enormously with verse length (measured:
    1,350px to 25,216px in this corpus) -- batching random lines together
    pads every image in a batch up to the widest one, which for an unlucky
    batch containing one of the longest verses tries to allocate tens of
    GB. Sorting by text length (a cheap, already-available proxy for
    rendered width -- both scale with character count for a fixed
    pixel-size range) before chunking into batches keeps each batch's
    images close in width, bounding padding waste to local variance
    instead of the whole dataset's spread. Batch ORDER is still shuffled
    each epoch so training isn't biased toward seeing short-then-long."""

    def __init__(self, records: list[dict], batch_size: int, seed: int, drop_last: bool):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        order = sorted(range(len(records)), key=lambda i: len(records[i]["text"]))
        self.batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        if drop_last and self.batches and len(self.batches[-1]) < batch_size:
            self.batches.pop()

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=40, help="cap validation batches per epoch for speed")
    parser.add_argument("--max-lines-per-epoch", type=int, default=None, help="debug: cap train examples/epoch")
    parser.add_argument(
        "--max-text-chars", type=int, default=1500,
        help="drop lines longer than this (p99 of S1 is ~1460 chars; the max is 8539 -- a single "
             "outlier verse rendered as one very wide line risks OOM even with length bucketing)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # CLAUDE.md rule 6
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = _load_manifest(args.manifest)
    n_before = len(records)
    records = [r for r in records if len(r["text"]) <= args.max_text_chars]
    n_dropped_long = n_before - len(records)
    train_records, val_records = _split_train_val(records, args.val_fraction)
    if args.max_lines_per_epoch is not None:
        train_records = train_records[: args.max_lines_per_epoch]
    print(
        f"S1: {n_before} lines total ({n_dropped_long} dropped, longer than "
        f"{args.max_text_chars} chars) -> {len(train_records)} train, {len(val_records)} val (device={device})"
    )

    model = CRNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CRNN parameters: {n_params:,}")

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = S1Dataset(train_records, args.data_dir, augment=True, seed=args.seed)
    val_ds = S1Dataset(val_records, args.data_dir, augment=False, seed=args.seed)

    train_sampler = LengthBucketBatchSampler(train_records, args.batch_size, seed=args.seed, drop_last=True)
    val_sampler = LengthBucketBatchSampler(val_records, args.batch_size, seed=args.seed, drop_last=False)
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
            "max_text_chars": args.max_text_chars,
            "n_dropped_long": n_dropped_long,
            "n_train": len(train_records),
            "n_val": len(val_records),
            "val_fraction": args.val_fraction,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "n_params": n_params,
            "vocab_size": len(wx.VOCAB),
            "augmentation": "setu.render.augment.augment_line (CPU, per-example) -- GPU-batched "
                            "augmentation from the roadmap's speed guidance is not yet implemented",
        },
    )

    history = []
    best_val_cer = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
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
            f"epoch {epoch:3d}/{args.epochs}  train_loss {mean_train_loss:.4f}  "
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
