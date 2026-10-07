"""Joint unlocked training: recogniser + soft bridge + modernizer trained
end to end (CLAUDE.md Week 3: "joint unlocked training").

This is B4-only, and not by choice -- argmax is not differentiable, so no
gradient can reach the recogniser through B3's committed string. That is
precisely the roadmap's claim ("argmax isn't smooth, which is why it
blocks learning"). B4's blend is differentiable all the way back to the
CRNN, so it can do something B3 structurally cannot.

Reporting caveat worth carrying into the report: because B3 cannot be
jointly trained at all, "B3 fine-tuned vs B4 fine-tuned + jointly trained"
differs in TWO ways and is not the matched interface comparison. The
matched comparison is the pair of fine-tunes (both from the same S3 warm
start, same data, same procedure). Joint training is an ADDITIONAL
capability claim and should be reported as such, not folded into the
head-to-head.

No cache here, unlike the fine-tunes: the recogniser's weights change
every step, so cached log-probs are stale by definition. Every step pays a
full CRNN forward AND backward over real images, which is why this is the
memory-heaviest thing in the project -- CRNN peaked 5.0-6.3GB alone during
S1 training and the modernizer fine-tune peaked 4.7-5.4GB alone; naively
summed that exceeds what this 12GB card has free. Batches are therefore
bucketed on image pixel area with a deliberately conservative budget.

CLAUDE.md: "recogniser unlocked, ~1/10th normal learning rate, a few
hundred steps on S2. Only lock the recogniser (weakening the claim) if
training is genuinely unstable (NaN loss / falling accuracy), and only
after first trying a lower learning rate and a shorter phase." The 1/10th
applies to the recogniser: default 3e-5 against the CRNN's own 3e-4.

Recogniser drift is measured, not assumed: CER over the held-out slice is
computed before and after, because nothing in the modernizer's loss asks
the CRNN to keep reading correctly, and a recogniser that silently
degrades into a feature extractor would invalidate every downstream claim.

Usage:
    python -m setu.bridge.joint_train --crnn <ckpt> --modernizer <ckpt> --temperature 1.5
    ... --time-steps 10     # measure per-step cost and peak memory, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # see recogniser/train.py

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from setu.bridge.soft_bridge import SoftBridgeConfig, apply_soft_bridge
from setu.data import wx
from setu.eval.metrics import cer
from setu.modernizer.model import Modernizer
from setu.modernizer.pretrain import char_accuracy
from setu.modernizer.vocab import CharVocab
from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE, greedy_decode
from setu.recogniser.train import AreaBucketBatchSampler
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
VOCAB_PATH = Path("data/modernizer_vocab.json")
INBAND_PATH = Path("data/splits/s2_inband_verse_ids.txt")


def _load_rows(manifest: Path, meta: Path, inband: set[int]) -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    with meta.open(encoding="utf-8") as f:
        metas = [json.loads(l) for l in f]
    if len(recs) != len(metas):
        raise RuntimeError(
            f"manifest has {len(recs)} rows but meta has {len(metas)} -- the positional join "
            f"that carries frozen-test membership is broken; refusing to train."
        )
    rows = []
    for rec, m in zip(recs, metas):
        if rec.get("skipped") or m["is_test"] or m["verse_id"] not in inband:
            continue
        rows.append({**rec, "verse_id": m["verse_id"]})
    return rows


def _split_train_val(rows: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    """Identical hash to finetune_s2 so the held-out slice is the same lines."""
    threshold = int(val_fraction * 256)
    train, val = [], []
    for r in rows:
        digest = hashlib.sha256(f"s2ft:{r['id']}".encode("utf-8")).digest()
        (val if digest[0] < threshold else train).append(r)
    return train, val


class JointDataset(Dataset):
    def __init__(self, rows: list[dict], data_dir: Path, vocab: CharVocab):
        self.rows = rows
        self.data_dir = data_dir
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        rel = r["image_path"].replace("\\", "/")  # Windows-written manifest
        img = np.array(Image.open(self.data_dir / rel).convert("L"), dtype=np.uint8)
        tgt = self.vocab.encode(r["modern_text"])
        truth = [wx.SYMBOL_TO_INDEX[s] for s in wx.encode(r["text"])]
        return img, tgt, truth


class Collate:
    """Picklable (not a closure) -- forkserver needs DataLoader args to pickle."""

    def __init__(self, vocab: CharVocab):
        self.vocab = vocab

    def __call__(self, batch):
        imgs, tgts, truths = zip(*batch)
        b = len(batch)
        vocab = self.vocab
        max_h = max(i.shape[0] for i in imgs)
        max_w = max(i.shape[1] for i in imgs)
        padded = np.zeros((b, 1, max_h, max_w), dtype=np.float32)
        widths = []
        for i, im in enumerate(imgs):
            h, w = im.shape
            padded[i, 0, :h, :w] = im.astype(np.float32) / 255.0
            widths.append(w)
        input_lengths = torch.tensor([w // WIDTH_DOWNSAMPLE for w in widths], dtype=torch.long)

        max_tgt = max(len(t) for t in tgts) + 1
        tgt_in = torch.full((max_tgt, b), vocab.pad_id, dtype=torch.long)
        tgt_out = torch.full((max_tgt, b), vocab.pad_id, dtype=torch.long)
        tgt_mask = torch.ones(b, max_tgt, dtype=torch.bool)
        for i, t in enumerate(tgts):
            tin = [vocab.bos_id] + list(t)
            tout = list(t) + [vocab.eos_id]
            tgt_in[: len(tin), i] = torch.tensor(tin, dtype=torch.long)
            tgt_out[: len(tout), i] = torch.tensor(tout, dtype=torch.long)
            tgt_mask[i, : len(tin)] = False

        return torch.from_numpy(padded), input_lengths, tgt_in, tgt_out, tgt_mask, list(truths)


def _forward(crnn, modernizer, batch, bridge_cfg, ce_loss, device):
    images, input_lengths, tgt_in, tgt_out, tgt_mask, _truths = batch
    images = images.to(device)
    input_lengths = input_lengths.to(device)
    tgt_in, tgt_out, tgt_mask = tgt_in.to(device), tgt_out.to(device), tgt_mask.to(device)

    log_probs = crnn(images)  # (T, B, C) -- gradients flow back through here
    blended, mem_mask = apply_soft_bridge(
        log_probs, input_lengths, modernizer.src_embed.weight, bridge_cfg
    )
    memory = modernizer.encode_embeds(blended, mem_mask)
    logits = modernizer.decode(tgt_in, memory, tgt_mask, mem_mask)
    loss = ce_loss(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    return loss, logits, tgt_out, log_probs, input_lengths


@torch.no_grad()
def evaluate(crnn, modernizer, loader, bridge_cfg, ce_loss, device) -> dict:
    """Modernizer quality AND recogniser CER -- the second is the drift
    check: nothing in the loss asks the CRNN to keep reading correctly."""
    crnn.eval()
    modernizer.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    total_cer, n_lines = 0.0, 0
    for batch in loader:
        loss, logits, tgt_out, log_probs, input_lengths = _forward(
            crnn, modernizer, batch, bridge_cfg, ce_loss, device
        )
        total_loss += loss.item()
        total_acc += char_accuracy(logits, tgt_out, modernizer.pad_id)
        n += 1
        for ref, hyp in zip(batch[5], greedy_decode(log_probs.cpu(), input_lengths.cpu())):
            total_cer += cer(ref, hyp)
            n_lines += 1
    crnn.train()
    modernizer.train()
    return {
        "val_loss": total_loss / max(n, 1),
        "val_char_acc": total_acc / max(n, 1),
        "recogniser_cer": total_cer / max(n_lines, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/s2/manifest.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/s2"))
    parser.add_argument("--meta", type=Path, default=Path("data/raw_corpus/s2_corpus_render_meta.jsonl"))
    parser.add_argument("--crnn", type=Path, required=True, help="trained CRNN checkpoint")
    parser.add_argument("--modernizer", type=Path, required=True, help="B4 fine-tuned modernizer checkpoint")
    parser.add_argument("--temperature", type=float, required=True, help="must match the B4 fine-tune's")
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300, help="CLAUDE.md: 'a few hundred steps'")
    parser.add_argument(
        "--lr-recogniser", type=float, default=3e-5,
        help="CLAUDE.md: '~1/10th normal learning rate'; the CRNN trained at 3e-4",
    )
    parser.add_argument("--lr-modernizer", type=float, default=1e-4, help="its own fine-tune rate")
    parser.add_argument("--lock-recogniser", action="store_true",
                        help="CLAUDE.md: only after instability, and it WEAKENS the claim")
    parser.add_argument(
        "--max-pixels-per-batch", type=int, default=3_000_000,
        help="matches the recogniser's own proven budget. An initial 1.2M guess -- picked "
             "fearing that holding CRNN AND modernizer activations for backward together would "
             "approach the card's limit -- measured at only 2.27GB peak but produced ~1.4 "
             "lines/batch, so 300 steps covered just 0.67 of a pass over the data. 3M measures "
             "3.78GB peak (card has ~11.1GB usable), 329ms/step, 256 batches, so the same 300 "
             "steps covers ~1.2 passes. Re-measure with --time-steps on any other GPU.",
    )
    parser.add_argument("--max-single-image-pixels", type=int, default=5_000_000,
                        help="training-set only; the frozen test split is never filtered")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--eval-every", type=int, default=50, help="steps between validations")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=None,
                        help="measure per-step cost and peak memory, write nothing, exit")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocab.load(VOCAB_PATH)
    inband = {int(x) for x in INBAND_PATH.read_text(encoding="utf-8").split()}
    rows = _load_rows(args.manifest, args.meta, inband)

    dims = []
    for r in rows:
        with Image.open(args.data_dir / r["image_path"].replace("\\", "/")) as im:
            w, h = im.size
        dims.append((h, w))
    keep = [i for i, (h, w) in enumerate(dims) if h * w <= args.max_single_image_pixels]
    n_dropped = len(rows) - len(keep)
    rows = [rows[i] for i in keep]
    dims = [dims[i] for i in keep]

    train_rows, val_rows = _split_train_val(rows, args.val_fraction)
    tr_idx = {r["id"] for r in train_rows}
    train_dims = [d for r, d in zip(rows, dims) if r["id"] in tr_idx]
    val_dims = [d for r, d in zip(rows, dims) if r["id"] not in tr_idx]
    print(
        f"S2 in-band train pool: {len(rows)} lines ({n_dropped} dropped over "
        f"{args.max_single_image_pixels:,}px) -> {len(train_rows)} train, {len(val_rows)} val "
        f"(device={device})"
    )

    crnn = CRNN().to(device)
    crnn.load_state_dict(torch.load(args.crnn, map_location=device))
    modernizer = Modernizer(len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id).to(device)
    modernizer.load_state_dict(torch.load(args.modernizer, map_location=device))
    crnn.train()
    modernizer.train()
    print(f"CRNN       <- {args.crnn}")
    print(f"modernizer <- {args.modernizer}")

    if args.lock_recogniser:
        for p in crnn.parameters():
            p.requires_grad = False
        print("RECOGNISER LOCKED -- this weakens the claim (CLAUDE.md); only valid after instability")

    bridge_cfg = SoftBridgeConfig(temperature=args.temperature, pool_size=args.pool_size)
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    groups = [{"params": modernizer.parameters(), "lr": args.lr_modernizer}]
    if not args.lock_recogniser:
        groups.append({"params": crnn.parameters(), "lr": args.lr_recogniser})
    optimizer = torch.optim.Adam(groups)

    collate = Collate(vocab)
    loaders = {}
    for name, subset, sdims in [("train", train_rows, train_dims), ("val", val_rows, val_dims)]:
        sampler = AreaBucketBatchSampler(
            list(range(len(subset))), sdims, args.max_pixels_per_batch, seed=SEED,
            max_batch_size=args.max_batch_size, drop_last=False,
        )
        loaders[name] = DataLoader(
            JointDataset(subset, args.data_dir, vocab), batch_sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate,
        )
        print(f"  {name}: {len(subset)} lines in {len(sampler)} batches")

    if args.time_steps is not None:
        it = iter(loaders["train"])
        times = []
        for i in range(args.time_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loaders["train"])
                batch = next(it)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            optimizer.zero_grad()
            loss, _, _, _, _ = _forward(crnn, modernizer, batch, bridge_cfg, ce_loss, device)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i > 0:
                times.append(time.time() - t0)
        a = np.array(times)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        print(f"\nper-step: mean={a.mean()*1000:.0f}ms median={np.median(a)*1000:.0f}ms "
              f"max={a.max()*1000:.0f}ms (n={len(a)}, warmup dropped)")
        print(f"projected {args.steps} steps: ~{a.mean()*args.steps/60:.1f} min (+ validation)")
        print(f"peak GPU memory: {peak:.2f} GB   (card has ~11.1GB usable)")
        print("\n(--time-steps: nothing written, no run folder created)")
        return

    run_dir = start_run(
        "joint_unlocked_train",
        {
            "seed": SEED, "crnn": str(args.crnn), "modernizer": str(args.modernizer),
            "temperature": args.temperature, "pool_size": args.pool_size,
            "steps": args.steps, "lr_recogniser": args.lr_recogniser,
            "lr_modernizer": args.lr_modernizer, "recogniser_locked": args.lock_recogniser,
            "max_pixels_per_batch": args.max_pixels_per_batch,
            "max_single_image_pixels": args.max_single_image_pixels,
            "max_batch_size": args.max_batch_size,
            "n_train": len(train_rows), "n_val": len(val_rows), "n_dropped_oversize": n_dropped,
            "device": str(device),
            "note": "B4 only: argmax is non-differentiable so B3 cannot be jointly trained. "
                    "Report as an additional capability, not part of the matched B3-vs-B4 "
                    "interface comparison.",
        },
    )

    before = evaluate(crnn, modernizer, loaders["val"], bridge_cfg, ce_loss, device)
    print(f"\nBEFORE: val_loss {before['val_loss']:.4f}  val_char_acc {before['val_char_acc']:.4f}  "
          f"recogniser_CER {before['recogniser_cer']:.4f}")

    history = []
    step = 0
    it = iter(loaders["train"])
    running = []
    t_start = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loaders["train"])
            batch = next(it)
        optimizer.zero_grad()
        loss, _, _, _, _ = _forward(crnn, modernizer, batch, bridge_cfg, ce_loss, device)
        if not torch.isfinite(loss):
            # CLAUDE.md rule 7: fail loudly, do not quietly carry on.
            finish_run(run_dir, {"history": history, "aborted": "non-finite loss", "step": step})
            raise RuntimeError(
                f"Non-finite loss at step {step}. CLAUDE.md: try a lower learning rate and a "
                f"shorter phase BEFORE resorting to --lock-recogniser, which weakens the claim."
            )
        loss.backward()
        optimizer.step()
        running.append(loss.item())
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(crnn, modernizer, loaders["val"], bridge_cfg, ce_loss, device)
            mean_train = float(np.mean(running))
            running = []
            print(f"step {step:4d}/{args.steps}  train_loss {mean_train:.4f}  "
                  f"val_loss {ev['val_loss']:.4f}  val_char_acc {ev['val_char_acc']:.4f}  "
                  f"recogniser_CER {ev['recogniser_cer']:.4f}")
            history.append({"step": step, "train_loss": mean_train, **ev})

    after = history[-1] if history else before
    torch.save(crnn.state_dict(), run_dir / "crnn_joint.pt")
    torch.save(modernizer.state_dict(), run_dir / "modernizer_joint.pt")

    cer_delta = after["recogniser_cer"] - before["recogniser_cer"]
    finish_run(
        run_dir,
        {
            "history": history, "before": before, "after": after,
            "recogniser_cer_before": before["recogniser_cer"],
            "recogniser_cer_after": after["recogniser_cer"],
            "recogniser_cer_delta": cer_delta,
            "val_loss_delta": after["val_loss"] - before["val_loss"],
            "seconds": time.time() - t_start,
        },
    )
    print(f"\nAFTER:  val_loss {after['val_loss']:.4f}  val_char_acc {after['val_char_acc']:.4f}  "
          f"recogniser_CER {after['recogniser_cer']:.4f}")
    print(f"recogniser CER drift: {cer_delta:+.4f} "
          f"({'DEGRADED' if cer_delta > 0.01 else 'stable'})")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
