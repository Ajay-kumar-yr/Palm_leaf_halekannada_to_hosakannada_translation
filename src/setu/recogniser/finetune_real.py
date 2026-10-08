"""Adapt the S1-trained CRNN to real palm-leaf handwriting
(DEMO_PLAN.md, teacher-student step 2).

The recogniser trained on S1 cannot read real manuscripts at all: mean
top-1 confidence 0.63 vs 0.96 on synthetic, output a string of repeated
ಅ (run `20261008T061052Z_real_vs_synthetic_confidence`). S1 taught it
font-rendered glyphs over leaf texture in which real handwriting only
ever appeared as *background*. This fine-tunes it on real crops carrying
machine labels from `setu.label.build_label_set`.

Two things keep the adaptation honest:

**S1 replay.** Training on a few hundred real lines alone would erase
the synthetic ability the reported 1.53% CER rests on. Each epoch mixes
`--replay-ratio` synthetic lines per real line, drawn fresh from the S1
*train* split (never its val split, which `measure_confidence` and the
reported CER use). Synthetic CER is evaluated every epoch and recorded
so any forgetting is visible rather than discovered later.

**Held-out real lines.** A hash-of-crop-id split (CLAUDE.md rule 2's
rationale) keeps `--val-fraction` of real lines out of training. Real
CER against those is CER **against machine labels, not human gold** —
the central caveat of this whole route, and it is written into the run
record so no later reader can mistake it.

Real crops are resized to height 64, the renderer's nominal
`line_height`, so glyph scale matches what the conv stack already
knows; the aspect ratio is preserved and width is free. A crop shorter
than 32px cannot survive five height-pooling stages, so those are padded
(counted, never dropped).

Per CLAUDE.md rule 3, run `--memorize-check` before any real run.

Usage:
    python -m setu.recogniser.finetune_real --memorize-check
    python -m setu.recogniser.finetune_real --time-steps 20
    python -m setu.recogniser.finetune_real --epochs 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # STATUS.md 3.8

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from setu.data import wx  # noqa: E402
from setu.eval.metrics import cer  # noqa: E402
from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE, expand_classifier, greedy_decode  # noqa: E402
from setu.recogniser.train import (  # noqa: E402
    AreaBucketBatchSampler,
    S1Dataset,
    _load_manifest,
    _split_train_val_indices,
    collate,
    scan_image_dims,
)
from setu.render.augment import augment_line  # noqa: E402
from setu.runlog import finish_run, start_run  # noqa: E402

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LINE_HEIGHT = 64
MIN_HEIGHT = 32  # five height-pooling stages: 2^5


# NOTE: the vocabulary choice must travel on the Dataset instance, never
# in a module global. DataLoader workers are separate processes that
# re-import this module, so a global set in main() is back to its default
# inside the worker -- which silently encoded with the wrong vocabulary
# until a Kannada numeral made it raise.
USE_DIGITS = False  # --digits; read ONLY in main/eval, never in a worker


class RealLineDataset(Dataset):
    """Labelled real crops, resized to LINE_HEIGHT. Same (image, labels,
    text) contract as S1Dataset so `collate` is reused unchanged."""

    def __init__(self, rows: list[dict], set_dir: Path, augment: bool, seed: int,
                 digits: bool = False):
        self.rows = rows
        self.set_dir = set_dir
        self.augment = augment
        self.digits = digits  # pickled to each worker; see the note above
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def dims(self) -> list[tuple[int, int]]:
        """(height, width) after the resize, for the batch sampler."""
        out = []
        for r in self.rows:
            w = max(WIDTH_DOWNSAMPLE, round(r["width"] * LINE_HEIGHT / r["height"]))
            out.append((LINE_HEIGHT, w))
        return out

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        # `image` is repo-root-relative and lets one training set mix
        # label files from several directories; `crop` is the older,
        # set_dir-relative form.
        path = REPO_ROOT / r["image"] if "image" in r else self.set_dir / r["crop"]
        im = Image.open(path).convert("L")
        w = max(WIDTH_DOWNSAMPLE, round(im.width * LINE_HEIGHT / im.height))
        img = np.array(im.resize((w, LINE_HEIGHT), Image.LANCZOS), dtype=np.uint8)
        if img.shape[0] < MIN_HEIGHT:  # cannot happen at LINE_HEIGHT=64, kept for safety
            pad = MIN_HEIGHT - img.shape[0]
            img = np.pad(img, ((pad // 2, pad - pad // 2), (0, 0)), constant_values=int(np.median(img)))
        if self.augment:
            img = augment_line(img, self.rng)
        _, sym_to_idx, _ = wx.tables(self.digits)
        labels = [sym_to_idx[s] for s in wx.encode(r["text"], digits=self.digits)]
        return img, labels, r["text"]


def split_real(rows: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    """Held-out real lines.

    If the label set carries `demo_split` (written by
    `build_label_set --holdout-every`), that decides: the demo is
    writer-dependent by decision (DEMO_PLAN.md 5b), so the lines it
    will display must be exactly the ones training never saw, chosen
    before training rather than by a hash that could drift as labels
    are added. Otherwise fall back to a hash of the crop id, which is
    stable as the label set grows."""
    if any("demo_split" in r for r in rows):
        train = [r for r in rows if r.get("demo_split") != "holdout"]
        val = [r for r in rows if r.get("demo_split") == "holdout"]
        return train, val
    threshold = int(val_fraction * 256)
    train, val = [], []
    for r in rows:
        digest = hashlib.sha256(f"real:{r['crop']}".encode("utf-8")).digest()
        (val if digest[0] < threshold else train).append(r)
    return train, val


def evaluate_real(model, loader, device) -> tuple[float, list[tuple[str, str]]]:
    model.eval()
    total, n, samples = 0.0, 0, []
    with torch.no_grad():
        for images, _t, input_lengths, _tl, label_seqs, texts in loader:
            log_probs = model(images.to(device))
            _, _, idx_to_sym = wx.tables(USE_DIGITS)
            for ref, hyp_ids, text in zip(label_seqs, greedy_decode(log_probs.cpu(), input_lengths), texts):
                total += cer(ref, hyp_ids)
                n += 1
                if len(samples) < 5:
                    samples.append((text, wx.decode([idx_to_sym[i] for i in hyp_ids])))
    model.train()
    return total / max(n, 1), samples


def load_rows(labels) -> list[dict]:
    paths = labels if isinstance(labels, (list, tuple)) else [labels]
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).open(encoding="utf-8")]
    return rows


def build_loaders(args, device):
    rows = load_rows(args.labels)
    if not rows:
        raise SystemExit(f"{args.labels} is empty -- run setu.label.build_label_set first")
    seen = set()
    rows = [r for r in rows if not (r.get("image", r["crop"]) in seen
                                    or seen.add(r.get("image", r["crop"])))]
    real_train, real_val = split_real(rows, args.val_fraction)
    set_dir = Path(args.labels[0]).parent
    how = "demo hold-out" if any("demo_split" in r for r in rows) else "hash split"
    print(f"real: {len(rows)} labelled lines -> {len(real_train)} train, {len(real_val)} val "
          f"({len({r['page'] for r in rows})} pages, {how})")

    train_ds = RealLineDataset(real_train, set_dir, augment=True, seed=SEED, digits=args.digits)
    val_ds = RealLineDataset(real_val, set_dir, augment=False, seed=SEED, digits=args.digits)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=AreaBucketBatchSampler(list(range(len(real_train))), train_ds.dims(),
                                             args.max_pixels_per_batch, seed=SEED,
                                             max_batch_size=args.max_batch_size, drop_last=False),
        num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(
        val_ds,
        batch_sampler=AreaBucketBatchSampler(list(range(len(real_val))), val_ds.dims(),
                                             args.max_pixels_per_batch, seed=SEED,
                                             max_batch_size=args.max_batch_size, drop_last=False),
        num_workers=args.num_workers, collate_fn=collate) if real_val else None
    return real_train, real_val, train_loader, val_loader


def build_replay(args, n_real: int):
    """S1 train-split lines for replay. Never the val split: that is what
    the reported synthetic CER and the confidence baseline measure."""
    records = _load_manifest(args.s1_manifest)
    dims = scan_image_dims(records, args.s1_dir)
    keep = [i for i, (h, w) in enumerate(dims) if h * w <= args.max_single_image_pixels]
    records, dims = [records[i] for i in keep], [dims[i] for i in keep]
    train_idx, val_idx = _split_train_val_indices(records, args.val_fraction_s1)
    rng = random.Random(SEED)
    pick = rng.sample(train_idx, min(int(n_real * args.replay_ratio), len(train_idx)))
    replay_records, replay_dims = [records[i] for i in pick], [dims[i] for i in pick]
    vpick = val_idx[: args.s1_val_lines]
    val_records, val_dims = [records[i] for i in vpick], [dims[i] for i in vpick]
    print(f"S1 replay: {len(replay_records)} train lines, {len(val_records)} val lines")

    def loader(recs, dms, augment, drop_last):
        ds = S1Dataset(recs, args.s1_dir, augment=augment, seed=SEED)
        return DataLoader(ds, batch_sampler=AreaBucketBatchSampler(
            list(range(len(recs))), dms, args.max_pixels_per_batch, seed=SEED,
            max_batch_size=args.max_batch_size, drop_last=drop_last),
            num_workers=args.num_workers, collate_fn=collate)

    return loader(replay_records, replay_dims, True, False), loader(val_records, val_dims, False, False)


def memorize_check(args, device) -> None:
    """CLAUDE.md rule 3: prove the model can drive 8 real examples to
    near-zero error before committing to a long run."""
    rows = load_rows(args.labels)[:8]
    ds = RealLineDataset(rows, Path(args.labels[0]).parent, augment=False, seed=SEED,
                         digits=args.digits)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    n_classes = len(wx.tables(USE_DIGITS)[0]) + 1
    model = CRNN(num_classes=n_classes).to(device)
    model.load_state_dict(expand_classifier(
        torch.load(args.resume_from, map_location=device), n_classes))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    t0 = time.time()
    for step in range(args.memorize_steps):
        for images, targets, il, tl, _ls, _tx in loader:
            if bool((il < tl).any()):
                raise SystemExit("a real line has more label symbols than CTC frames -- "
                                 "the crop is too narrow for its transcription; fix the label set")
            opt.zero_grad()
            loss = ctc(model(images.to(device)), targets, il, tl)
            loss.backward()
            opt.step()
        if (step + 1) % 20 == 0:
            c, _ = evaluate_real(model, loader, device)
            print(f"  step {step + 1:4d}  loss {loss.item():.4f}  CER {c:.4f}")
    final, samples = evaluate_real(model, loader, device)
    print(f"memorize-check: 8 real lines, CER {final:.4f} in {time.time() - t0:.0f}s")
    for ref, hyp in samples[:2]:
        print(f"   ref: {ref[:60]}\n   hyp: {hyp[:60]}")
    print("PASS -- safe to train" if final < 0.05 else
          "FAIL -- do NOT start the real run; something is broken (CLAUDE.md rule 3)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, nargs="+",
                   default=[Path("data/real_lines/train/train_labels.jsonl")],
                   help="One or more label sets; concatenated (each row carries its own "
                        "repo-root-relative image path).")
    p.add_argument("--resume-from", type=Path,
                   default=Path("runs/20261006T120718Z_crnn_train_s1/best_model.pt"))
    p.add_argument("--s1-manifest", type=Path, default=Path("data/s1/manifest.jsonl"))
    p.add_argument("--s1-dir", type=Path, default=Path("data/s1"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4, help="below train.py's 3e-4: this adapts, not retrains")
    p.add_argument("--val-fraction", type=float, default=0.15, help="held-out fraction of real lines")
    p.add_argument("--val-fraction-s1", type=float, default=0.05, help="must match train.py's S1 split")
    p.add_argument("--replay-ratio", type=float, default=2.0, help="S1 lines replayed per real line")
    p.add_argument("--s1-val-lines", type=int, default=120, help="S1 val lines for the forgetting check")
    p.add_argument("--max-pixels-per-batch", type=int, default=2_200_000)  # 3050 value
    p.add_argument("--max-single-image-pixels", type=int, default=1_500_000)
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-replay", action="store_true", help="train on real lines only (expect forgetting)")
    p.add_argument("--digits", action="store_true",
                   help="Use wx.EXTENDED_VOCAB (adds Kannada numerals, 10 extra CTC classes). "
                        "The checkpoint's classifier is widened, keeping every existing class.")
    p.add_argument("--memorize-check", action="store_true")
    p.add_argument("--memorize-steps", type=int, default=100)
    p.add_argument("--time-steps", type=int, default=0,
                   help="Measure N steps, project the run, write NOTHING, exit.")
    args = p.parse_args()

    global USE_DIGITS
    USE_DIGITS = args.digits
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if args.memorize_check:
        memorize_check(args, device)
        return

    real_train, real_val, real_loader, real_val_loader = build_loaders(args, device)
    replay_loader, s1_val_loader = (None, None) if args.no_replay else build_replay(args, len(real_train))

    n_classes = len(wx.tables(USE_DIGITS)[0]) + 1
    model = CRNN(num_classes=n_classes).to(device)
    model.load_state_dict(expand_classifier(
        torch.load(args.resume_from, map_location=device), n_classes))
    model.train()
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(measure_only: int = 0) -> tuple[float, int, int]:
        """One pass over real lines interleaved with replay lines."""
        batches = [("real", b) for b in real_loader]
        if replay_loader is not None:
            batches += [("s1", b) for b in replay_loader]
        random.shuffle(batches)
        if measure_only:
            batches = batches[:measure_only]
        total, n, skipped = 0.0, 0, 0
        for _src, (images, targets, il, tl, _ls, _tx) in batches:
            if bool((il < tl).any()):
                skipped += 1
                continue
            opt.zero_grad()
            loss = ctc(model(images.to(device)), targets, il, tl)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        return total / max(n, 1), n, skipped

    if args.time_steps:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _, n, _ = run_epoch(measure_only=args.time_steps)
        per = (time.time() - t0) / max(n, 1)
        n_batches = len(real_loader) + (len(replay_loader) if replay_loader else 0)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"measured {n} steps: {per:.2f}s/step, peak GPU {peak:.2f} GiB")
        print(f"projected: {n_batches} steps/epoch x {args.epochs} epochs ~= "
              f"{per * n_batches * args.epochs / 60:.1f} min (+ per-epoch eval)")
        print("wrote nothing")
        return

    rows = load_rows(args.labels)
    run_dir = start_run("crnn_finetune_real", {
        "seed": SEED, "labels": [str(p) for p in args.labels], "resume_from": str(args.resume_from),
        "n_real_train": len(real_train), "n_real_val": len(real_val),
        "epochs": args.epochs, "lr": args.lr, "replay_ratio": None if args.no_replay else args.replay_ratio,
        "val_fraction": args.val_fraction, "line_height": LINE_HEIGHT,
        "vocabulary": "EXTENDED (with Kannada digits)" if USE_DIGITS else "base",
        "n_classes": len(wx.tables(USE_DIGITS)[0]) + 1,
        "split": "demo hold-out (demo_split field)" if any("demo_split" in r for r in rows)
                 else "hash of crop id",
        "max_pixels_per_batch": args.max_pixels_per_batch, "device": str(device),
        "label_provenance": "vision-LLM transcriptions filtered by setu.label.build_label_set -- "
                            "NO human verification; real CER below is against machine labels",
    })

    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, n_steps, skipped = run_epoch()
        real_cer, samples = evaluate_real(model, real_val_loader, device) if real_val_loader else (float("nan"), [])
        s1_cer = evaluate_real(model, s1_val_loader, device)[0] if s1_val_loader else None
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "real_val_cer": round(real_cer, 4),
                        "s1_val_cer": round(s1_cer, 4) if s1_cer is not None else None,
                        "steps": n_steps, "skipped": skipped, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:3d}/{args.epochs}  loss {loss:.4f}  real_val_CER {real_cer:.4f}  "
              f"S1_val_CER {s1_cer if s1_cer is None else f'{s1_cer:.4f}'}  "
              f"{history[-1]['seconds']:.0f}s")
        if real_cer < best:
            best = real_cer
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            history[-1]["saved"] = True
    for ref, hyp in samples[:3]:
        print(f"  ref: {ref[:70]}\n  hyp: {hyp[:70]}")

    finish_run(run_dir, {
        "best_real_val_cer": best, "final_s1_val_cer": history[-1]["s1_val_cer"],
        "history": history, "n_real_train": len(real_train), "n_real_val": len(real_val),
        "samples": [{"ref": r, "hyp": h} for r, h in samples],
        "caveat": "real_val_cer is measured against machine labels, not human ground truth",
    })
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
