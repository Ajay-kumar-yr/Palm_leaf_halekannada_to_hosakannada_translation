"""Pre-train the modernizer on S3 (CLAUDE.md Week 2: "pre-train modernizer
on S3"; data table: S3 is rule-generated old/modern pairs, pretraining
only, never scored against -- see setu.modernizer.reverse_spelling).

Run only after setu.modernizer.memorize_check passes (CLAUDE.md rule 3);
this script does not re-check that itself.

Usage:
    python -m setu.modernizer.pretrain --corpus data/raw_corpus/s3_corpus_for_generate.txt
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from setu.data import wx
from setu.modernizer.model import Modernizer
from setu.modernizer.vocab import CharVocab, build_vocab
from setu.runlog import finish_run, start_run

VOCAB_PATH = Path("data/modernizer_vocab.json")


def _load_pairs(corpus_path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in corpus_path.open(encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue  # a truncated/malformed line -- skip rather than crash
        pairs.append((parts[0], parts[1]))
    return pairs


def _split_train_val(pairs: list[tuple[str, str]], val_fraction: float) -> tuple[list, list]:
    threshold = int(val_fraction * 256)
    train, val = [], []
    for i, pair in enumerate(pairs):
        digest = hashlib.sha256(f"s3:{i}".encode("utf-8")).digest()
        (val if digest[0] < threshold else train).append(pair)
    return train, val


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], vocab: CharVocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        old, modern = self.pairs[idx]
        src = [wx.SYMBOL_TO_INDEX[s] for s in wx.encode(old)]
        tgt = self.vocab.encode(modern)
        return src, tgt


class Collate:
    """A picklable callable (not a closure) -- Python 3.14 on Linux
    defaults multiprocessing to 'forkserver', which requires DataLoader
    worker arguments (including collate_fn) to be picklable; a local
    closure function is not."""

    def __init__(self, vocab: CharVocab):
        self.vocab = vocab

    def __call__(self, batch: list[tuple[list[int], list[int]]]):
        vocab = self.vocab
        srcs, tgts = zip(*batch)
        max_src = max(len(s) for s in srcs)
        max_tgt = max(len(t) for t in tgts) + 1  # + BOS/EOS shift

        src_ids = torch.zeros(max_src, len(batch), dtype=torch.long)
        src_pad_mask = torch.ones(len(batch), max_src, dtype=torch.bool)
        tgt_in_ids = torch.full((max_tgt, len(batch)), vocab.pad_id, dtype=torch.long)
        tgt_out_ids = torch.full((max_tgt, len(batch)), vocab.pad_id, dtype=torch.long)
        tgt_pad_mask = torch.ones(len(batch), max_tgt, dtype=torch.bool)

        for i, s in enumerate(srcs):
            src_ids[: len(s), i] = torch.tensor(s, dtype=torch.long)
            src_pad_mask[i, : len(s)] = False
        for i, t in enumerate(tgts):
            tin = [vocab.bos_id] + t
            tout = t + [vocab.eos_id]
            tgt_in_ids[: len(tin), i] = torch.tensor(tin, dtype=torch.long)
            tgt_out_ids[: len(tout), i] = torch.tensor(tout, dtype=torch.long)
            tgt_pad_mask[i, : len(tin)] = False

        return src_ids, src_pad_mask, tgt_in_ids, tgt_out_ids, tgt_pad_mask


def char_accuracy(logits: torch.Tensor, tgt_out_ids: torch.Tensor, pad_id: int) -> float:
    pred = logits.argmax(dim=-1)
    mask = tgt_out_ids != pad_id
    correct = ((pred == tgt_out_ids) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def evaluate(model: Modernizer, loader: DataLoader, device: torch.device, ce_loss, max_batches: int) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (src_ids, src_pad_mask, tgt_in_ids, tgt_out_ids, tgt_pad_mask) in enumerate(loader):
            if i >= max_batches:
                break
            src_ids, src_pad_mask = src_ids.to(device), src_pad_mask.to(device)
            tgt_in_ids, tgt_out_ids, tgt_pad_mask = (
                tgt_in_ids.to(device), tgt_out_ids.to(device), tgt_pad_mask.to(device)
            )
            logits = model(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask)
            loss = ce_loss(logits.reshape(-1, logits.shape[-1]), tgt_out_ids.reshape(-1))
            total_loss += loss.item()
            total_acc += char_accuracy(logits, tgt_out_ids, model.pad_id)
            n += 1
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/raw_corpus/s3_corpus_for_generate.txt"))
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # CLAUDE.md rule 6
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = _load_pairs(args.corpus)
    train_pairs, val_pairs = _split_train_val(pairs, args.val_fraction)
    print(f"S3: {len(pairs)} pairs -> {len(train_pairs)} train, {len(val_pairs)} val (device={device})")

    vocab = build_vocab([m for _, m in pairs])
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(VOCAB_PATH)
    print(f"Vocab: {len(vocab)} chars -> {VOCAB_PATH}")

    model = Modernizer(len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modernizer parameters: {n_params:,}")

    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    collate = Collate(vocab)
    train_loader = DataLoader(
        PairDataset(train_pairs, vocab), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, vocab), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    run_dir = start_run(
        "modernizer_pretrain_s3",
        {
            "seed": args.seed,
            "corpus": str(args.corpus),
            "n_train": len(train_pairs),
            "n_val": len(val_pairs),
            "vocab_size": len(vocab),
            "vocab_path": str(VOCAB_PATH),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "n_params": n_params,
        },
    )

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for src_ids, src_pad_mask, tgt_in_ids, tgt_out_ids, tgt_pad_mask in train_loader:
            src_ids, src_pad_mask = src_ids.to(device), src_pad_mask.to(device)
            tgt_in_ids, tgt_out_ids, tgt_pad_mask = (
                tgt_in_ids.to(device), tgt_out_ids.to(device), tgt_pad_mask.to(device)
            )
            optimizer.zero_grad()
            logits = model(src_ids, tgt_in_ids, src_pad_mask, tgt_pad_mask)
            loss = ce_loss(logits.reshape(-1, logits.shape[-1]), tgt_out_ids.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += char_accuracy(logits, tgt_out_ids, vocab.pad_id)
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        train_acc = total_acc / max(n_batches, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, ce_loss, args.val_batches)
        epoch_time = time.time() - epoch_start

        print(
            f"epoch {epoch:3d}/{args.epochs}  train_loss {train_loss:.4f} train_char_acc {train_acc:.4f}  "
            f"val_loss {val_loss:.4f} val_char_acc {val_acc:.4f}  ({epoch_time:.1f}s)"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_char_acc": train_acc,
             "val_loss": val_loss, "val_char_acc": val_acc, "seconds": epoch_time}
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    finish_run(run_dir, {"history": history, "best_val_loss": best_val_loss})
    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()
