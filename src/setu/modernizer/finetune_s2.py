"""Fine-tune the modernizer on S2's real recogniser output -- one branch
per invocation (CLAUDE.md Week 3: "fine-tune both modernizers on their
respective noisy inputs").

This is deliberately ONE script with a --branch flag rather than two
scripts. CLAUDE.md's whole comparison rests on "same recogniser, same
data, same modernizer training procedure -- only the interface differs",
and rule 10 requires both branches to train on the recogniser's actual
noisy output. A single file makes that true by construction: both
branches share this optimizer, learning rate, batching, split, seed,
early-stopping rule and warm start. Two files would be free to drift
apart, and any drift would silently become part of the measured B3-vs-B4
difference.

  B3 (conventional): cached argmax indices -- the recogniser's committed
      CTC-collapsed string -- through Modernizer.encode. The uncertainty
      is already gone by the time the modernizer sees anything.
  B4 (ours):         cached per-frame log-probs through the soft bridge
      (temperature -> softmax -> top-5 incl. blank -> renormalise ->
      confidence-blend) into Modernizer.encode_embeds. Nothing commits to
      a single symbol before the modernizer.

Batches are formed by TARGET length, identical for both branches, so each
branch sees the same examples grouped the same way; bucketing by source
length would have given them different batch compositions (B3's source is
characters, B4's is frames) and so different gradient noise, confounding
the thing being measured. In-band S2 holds the ratio near 1:1, so target
length tracks source length closely anyway.

Trains on the in-band, non-test portion of S2 (721 lines), split 90/10 by
hash of line id for early stopping. The 465 in-band frozen-test lines are
never touched here. See data/raw_corpus/build_s2_inband_subset.py for why
the in-band restriction exists.

Usage:
    python -m setu.modernizer.finetune_s2 --cache-dir runs/<...>_cache_s2_recogniser --branch b3
    python -m setu.modernizer.finetune_s2 --cache-dir runs/<...>_cache_s2_recogniser --branch b4 --temperature 1.0
    ... --time-steps 20     # measure per-step cost and project, write nothing
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
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from setu.bridge.soft_bridge import SoftBridgeConfig, apply_soft_bridge
from setu.modernizer.model import Modernizer
from setu.modernizer.pretrain import char_accuracy
from setu.modernizer.vocab import CharVocab
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
VOCAB_PATH = Path("data/modernizer_vocab.json")
INBAND_PATH = Path("data/splits/s2_inband_verse_ids.txt")


def _load_cache_lines(cache_dir: Path, inband: set[int]) -> list[dict]:
    with (cache_dir / "lines.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return [r for r in rows if not r["is_test"] and r["verse_id"] in inband]


def _split_train_val(rows: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    """Hash-of-id split -- same rationale as CLAUDE.md rule 2 and
    recogniser/train.py: stable regardless of ordering or future growth."""
    threshold = int(val_fraction * 256)
    train, val = [], []
    for r in rows:
        digest = hashlib.sha256(f"s2ft:{r['id']}".encode("utf-8")).digest()
        (val if digest[0] < threshold else train).append(r)
    return train, val


class TargetLengthBatchSampler(Sampler[list[int]]):
    """Bounds len(batch) * max_target_len_in_batch, so memory stays bounded
    across a 4x spread in target length. Ordering is by target length (low
    padding waste); batch order is shuffled each epoch so training is not
    biased short-to-long. Same construction for both branches."""

    def __init__(self, tgt_lens: list[int], max_tokens: int, max_batch_size: int, seed: int):
        self.rng = np.random.default_rng(seed)
        order = sorted(range(len(tgt_lens)), key=lambda i: tgt_lens[i])
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in order:
            prospective_max = max(current_max, tgt_lens[idx])
            if current and (
                (len(current) + 1) * prospective_max > max_tokens
                or len(current) >= max_batch_size
            ):
                batches.append(current)
                current, current_max = [idx], tgt_lens[idx]
            else:
                current.append(idx)
                current_max = prospective_max
        if current:
            batches.append(current)
        self.batches = batches

    def __iter__(self):
        for i in self.rng.permutation(len(self.batches)):
            yield self.batches[i]

    def __len__(self) -> int:
        return len(self.batches)


class S2FinetuneDataset(Dataset):
    """Yields (src, tgt_ids) where src depends on the branch: argmax index
    list for B3, (n_frames, C) log-prob block for B4. The memmap is opened
    lazily per worker -- a np.memmap handle does not survive being pickled
    into a DataLoader worker."""

    def __init__(self, rows: list[dict], vocab: CharVocab, branch: str, logprobs_path: Path):
        self.rows = rows
        self.vocab = vocab
        self.branch = branch
        self.logprobs_path = logprobs_path
        self._lp = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        tgt = self.vocab.encode(r["modern_text"])
        if self.branch == "b3":
            return np.asarray(r["argmax_indices"], dtype=np.int64), tgt
        if self._lp is None:
            self._lp = np.load(self.logprobs_path, mmap_mode="r")
        # np.array (not asarray) to force a copy: the memmap is opened
        # read-only, and asarray would hand back a read-only view, which
        # torch.from_numpy warns about as a non-writable tensor.
        block = np.array(
            self._lp[r["offset"] : r["offset"] + r["n_frames"]], dtype=np.float32
        )
        return block, tgt


class Collate:
    """Picklable callable, not a closure -- see pretrain.Collate for why
    (forkserver requires DataLoader args to pickle)."""

    def __init__(self, vocab: CharVocab, branch: str):
        self.vocab = vocab
        self.branch = branch

    def __call__(self, batch):
        srcs, tgts = zip(*batch)
        b = len(batch)
        vocab = self.vocab
        max_tgt = max(len(t) for t in tgts) + 1  # + BOS/EOS shift

        tgt_in_ids = torch.full((max_tgt, b), vocab.pad_id, dtype=torch.long)
        tgt_out_ids = torch.full((max_tgt, b), vocab.pad_id, dtype=torch.long)
        tgt_pad_mask = torch.ones(b, max_tgt, dtype=torch.bool)
        for i, t in enumerate(tgts):
            tin = [vocab.bos_id] + list(t)
            tout = list(t) + [vocab.eos_id]
            tgt_in_ids[: len(tin), i] = torch.tensor(tin, dtype=torch.long)
            tgt_out_ids[: len(tout), i] = torch.tensor(tout, dtype=torch.long)
            tgt_pad_mask[i, : len(tin)] = False

        src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
        max_src = int(src_lens.max())

        if self.branch == "b3":
            # index 0 is the shared blank/pad: src_embed has padding_idx=0.
            src = torch.zeros(max_src, b, dtype=torch.long)
            src_pad_mask = torch.ones(b, max_src, dtype=torch.bool)
            for i, s in enumerate(srcs):
                src[: len(s), i] = torch.from_numpy(s)
                src_pad_mask[i, : len(s)] = False
            return src, src_pad_mask, src_lens, tgt_in_ids, tgt_out_ids, tgt_pad_mask

        # B4: (T, B, C) log-probs. The soft bridge derives its own padding
        # mask from src_lens, so none is built here.
        c = srcs[0].shape[1]
        src = torch.zeros(max_src, b, c, dtype=torch.float32)
        for i, s in enumerate(srcs):
            src[: len(s), i, :] = torch.from_numpy(s)
        return src, None, src_lens, tgt_in_ids, tgt_out_ids, tgt_pad_mask


def _encode(model: Modernizer, branch: str, src, src_pad_mask, src_lens, bridge_cfg, device):
    """The ONLY place the two branches differ."""
    if branch == "b3":
        return model.encode(src, src_pad_mask), src_pad_mask
    blended, key_padding_mask = apply_soft_bridge(
        src, src_lens.to(device), model.src_embed.weight, bridge_cfg
    )
    return model.encode_embeds(blended, key_padding_mask), key_padding_mask


def _run_batch(model, branch, batch, bridge_cfg, ce_loss, device):
    src, src_pad_mask, src_lens, tgt_in_ids, tgt_out_ids, tgt_pad_mask = batch
    src = src.to(device)
    if src_pad_mask is not None:
        src_pad_mask = src_pad_mask.to(device)
    tgt_in_ids, tgt_out_ids, tgt_pad_mask = (
        tgt_in_ids.to(device), tgt_out_ids.to(device), tgt_pad_mask.to(device)
    )
    memory, mem_mask = _encode(model, branch, src, src_pad_mask, src_lens, bridge_cfg, device)
    logits = model.decode(tgt_in_ids, memory, tgt_pad_mask, mem_mask)
    loss = ce_loss(logits.reshape(-1, logits.shape[-1]), tgt_out_ids.reshape(-1))
    return loss, logits, tgt_out_ids


def evaluate(model, branch, loader, bridge_cfg, ce_loss, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, logits, tgt_out_ids = _run_batch(model, branch, batch, bridge_cfg, ce_loss, device)
            total_loss += loss.item()
            total_acc += char_accuracy(logits, tgt_out_ids, model.pad_id)
            n += 1
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="a runs/<...>_cache_s2_recogniser folder")
    parser.add_argument("--branch", choices=["b3", "b4"], required=True)
    parser.add_argument(
        "--resume-from", type=Path,
        default=Path("runs/20260930T052116Z_modernizer_pretrain_s3/best_model.pt"),
        help="S3-pretrained warm start. Both branches MUST start from the same one.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="B4 only, required for it. soft_bridge deliberately gives this no default: CTC is "
             "overconfident, and leaving it at 1 can make the blend numerically indistinguishable "
             "from argmax, erasing the B3-vs-B4 difference. Measured on the real checkpoint: "
             "11.69%% of frames sit below 0.9 top-1 confidence (runs/*_measure_confidence_*).",
    )
    parser.add_argument(
        "--pool-size", type=int, default=2,
        help="B4 only. MUST be >=2, not optional as CLAUDE.md implies: one in-band frozen-test "
             "line is 2,151 frames, over the modernizer's MAX_LEN=2048, at pool_size=1.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8, help="early stop after N epochs with no val_loss gain")
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="below pretrain's 3e-4: this adapts an already-converged checkpoint on 649 examples, "
             "where the pretrain rate tends to wash out what S3 taught",
    )
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--time-steps", type=int, default=None,
        help="measure per-step cost over N steps, project the full run, write nothing, exit",
    )
    args = parser.parse_args()

    if args.branch == "b4" and args.temperature is None:
        parser.error("--temperature is required for --branch b4 (see --help)")
    if args.branch == "b4" and args.pool_size < 2:
        parser.error("--pool-size must be >= 2 (see --help)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocab.load(VOCAB_PATH)
    inband = {int(x) for x in INBAND_PATH.read_text(encoding="utf-8").split()}
    rows = _load_cache_lines(args.cache_dir, inband)
    train_rows, val_rows = _split_train_val(rows, args.val_fraction)
    print(
        f"branch={args.branch}  in-band S2 train lines: {len(rows)} -> "
        f"{len(train_rows)} train, {len(val_rows)} val (device={device})"
    )

    model = Modernizer(len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id).to(device)
    model.load_state_dict(torch.load(args.resume_from, map_location=device))
    print(f"Warm start from {args.resume_from}")

    bridge_cfg = (
        SoftBridgeConfig(temperature=args.temperature, pool_size=args.pool_size)
        if args.branch == "b4"
        else None
    )
    if bridge_cfg is not None:
        print(f"soft bridge: {bridge_cfg}")

    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logprobs_path = args.cache_dir / "logprobs.npy"
    collate = Collate(vocab, args.branch)
    loaders = {}
    for name, subset in [("train", train_rows), ("val", val_rows)]:
        tgt_lens = [len(vocab.encode(r["modern_text"])) for r in subset]
        sampler = TargetLengthBatchSampler(
            tgt_lens, args.max_tokens_per_batch, args.max_batch_size, seed=SEED
        )
        loaders[name] = DataLoader(
            S2FinetuneDataset(subset, vocab, args.branch, logprobs_path),
            batch_sampler=sampler, num_workers=args.num_workers, collate_fn=collate,
        )
        print(f"  {name}: {len(subset)} lines in {len(sampler)} batches")

    # --- measure-first mode (HANDOFF.md Step 5's lesson: do not estimate
    # from arithmetic when a measurement is cheap) ---
    if args.time_steps is not None:
        model.train()
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
            loss, _, _ = _run_batch(model, args.branch, batch, bridge_cfg, ce_loss, device)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            if i > 0:  # drop step 0: one-time cuDNN/alloc warmup
                times.append(dt)
        a = np.array(times)
        per_epoch = a.mean() * len(loaders["train"])
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        print(
            f"\nper-step: mean={a.mean()*1000:.0f}ms median={np.median(a)*1000:.0f}ms "
            f"max={a.max()*1000:.0f}ms  (n={len(a)}, warmup step dropped)"
        )
        print(f"batches/epoch: {len(loaders['train'])}  -> ~{per_epoch:.1f}s/epoch train")
        print(f"projected {args.epochs} epochs: ~{per_epoch*args.epochs/60:.1f} min (+ validation)")
        print(f"peak GPU memory: {peak:.2f} GB")
        print("\n(--time-steps: nothing written, no run folder created)")
        return

    # Temperature in the b4 run name: a sweep produces several b4 runs that
    # are otherwise distinguishable only by reading each config.json.
    run_name = (
        f"modernizer_finetune_s2_b4_T{args.temperature:g}"
        if args.branch == "b4"
        else "modernizer_finetune_s2_b3"
    )
    run_dir = start_run(
        run_name,
        {
            "seed": SEED,
            "branch": args.branch,
            "cache_dir": str(args.cache_dir),
            "resume_from": str(args.resume_from),
            "temperature": args.temperature,
            "pool_size": args.pool_size if args.branch == "b4" else None,
            "blank_drop_threshold": bridge_cfg.blank_drop_threshold if bridge_cfg else None,
            "top_k": bridge_cfg.top_k if bridge_cfg else None,
            "n_inband_train_total": len(rows),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "val_fraction": args.val_fraction,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "max_tokens_per_batch": args.max_tokens_per_batch,
            "max_batch_size": args.max_batch_size,
            "device": str(device),
            "vocab_size": len(vocab),
            "note": "in-band S2 only (data/splits/s2_inband_verse_ids.txt); the 465 in-band "
                    "frozen-test lines are not used here",
        },
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss, logits, tgt_out_ids = _run_batch(
                model, args.branch, batch, bridge_cfg, ce_loss, device
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += char_accuracy(logits, tgt_out_ids, vocab.pad_id)
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        train_acc = total_acc / max(n_batches, 1)
        val_loss, val_acc = evaluate(model, args.branch, loaders["val"], bridge_cfg, ce_loss, device)
        epoch_time = time.time() - epoch_start

        print(
            f"epoch {epoch:3d}/{args.epochs}  train_loss {train_loss:.4f} "
            f"train_char_acc {train_acc:.4f}  val_loss {val_loss:.4f} "
            f"val_char_acc {val_acc:.4f}  ({epoch_time:.1f}s)"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_char_acc": train_acc,
             "val_loss": val_loss, "val_char_acc": val_acc, "seconds": epoch_time}
        )

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_since_best = val_loss, epoch, 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(
                    f"Early stop: no val_loss improvement for {args.patience} epochs "
                    f"(best {best_val_loss:.4f} at epoch {best_epoch})"
                )
                break

    finish_run(
        run_dir,
        {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "early_stopped": len(history) < args.epochs,
        },
    )
    print(f"\nBest val_loss {best_val_loss:.4f} at epoch {best_epoch} -> {run_dir / 'best_model.pt'}")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
