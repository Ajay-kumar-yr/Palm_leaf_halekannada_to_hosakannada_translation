"""Cache the trained recogniser's output over S2 (CLAUDE.md Week 3: "cache
recogniser outputs over S2") -- the shared foundation for BOTH B3's and
B4's modernizer fine-tunes.

CLAUDE.md rule 10 requires both branches to be fine-tuned on the
recogniser's ACTUAL noisy output, not clean text, "otherwise the
comparison is unfair to whichever one meets noise for the first time at
test time". Caching one deterministic pass is what makes that fair in the
strong sense: B3 and B4 then read the *same bytes*, so the only thing
differing between them is the interface (argmax string vs. blended
confidence), not the data, not the recogniser, not the sampling.

Determinism: model.eval() (BatchNorm uses running statistics, so output
does not depend on batch composition), torch.no_grad(), and augment=False
-- S1 training applied random augmentation per epoch, but a cache must be
reproducible, and both branches must agree.

Batch size is 1 throughout, deliberately. S2 contains 92 lines over 5M px
(largest 21.2M px, 1232x17210) -- far past anything the GPU profiling in
HANDOFF.md Step 5 covered. Those lines CANNOT be dropped the way S1
training dropped its oversized tail: 1,358 of S2's 3,500 lines are frozen
test verses, and dropping even one silently corrupts every reported
end-to-end number. Batch-of-1 removes all padding waste and oversized
special-casing; a single line that still will not fit on the GPU falls
back to CPU and is recorded as such per line rather than skipped.

Writes into runs/<timestamp>_cache_s2_recogniser/:
  logprobs.npy  float32 (total_frames, NUM_CLASSES). Lines concatenated
                with NO padding, each at its own recorded offset. Stored
                as full distributions rather than top-5: the soft bridge's
                blank-drop test reads the FULL-vocabulary softmax
                (soft_bridge.apply_soft_bridge: `temp_probs[...,
                blank_index]`), which cannot be recovered from top-5 alone
                at any temperature other than 1 -- and CLAUDE.md requires
                temperature be tuned on validation data, not fixed at 1.
                Full float32 costs 0.41GB here, which is not worth trading
                for a silently degraded temperature sweep.
  lines.jsonl   one record per line, in manifest order.

Usage:
    python -m setu.bridge.cache_s2 --checkpoint runs/<...>/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # see recogniser/train.py

from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from setu.data import wx
from setu.eval.metrics import cer
from setu.recogniser.model import CRNN, NUM_CLASSES, WIDTH_DOWNSAMPLE, greedy_decode
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6


def _load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    for line in manifest_path.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        records.append(rec)
    return records


def _load_meta(meta_path: Path) -> list[dict]:
    return [json.loads(line) for line in meta_path.open(encoding="utf-8")]


class S2ImageDataset(Dataset):
    """Returns one un-augmented image at a time, plus its record index."""

    def __init__(self, records: list[dict], data_dir: Path):
        self.records = records
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rel = self.records[idx]["image_path"].replace("\\", "/")  # Windows-written manifest
        img = np.array(Image.open(self.data_dir / rel).convert("L"), dtype=np.uint8)
        return torch.from_numpy(img.astype(np.float32) / 255.0), idx


def _collate_single(batch):
    """batch_size=1, so no padding at all -- (1, 1, H, W) and the index."""
    (img, idx), = batch
    return img.unsqueeze(0).unsqueeze(0), idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/s2/manifest.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/s2"))
    parser.add_argument(
        "--meta", type=Path, default=Path("data/raw_corpus/s2_corpus_render_meta.jsonl"),
        help="verse_id + is_test per rendered row, SAME order as the manifest",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = _load_manifest(args.manifest)
    meta = _load_meta(args.meta)
    if len(meta) != len(records):
        raise RuntimeError(
            f"meta has {len(meta)} rows but manifest has {len(records)} -- the positional "
            f"join that maps rendered lines back to frozen-test membership is broken; "
            f"refusing to write a cache whose test split cannot be trusted."
        )

    # Frame count per line is exact and knowable up front: the CRNN's blocks
    # 1-3 each halve width and blocks 4-5 leave it alone, so T == W // 8.
    lengths = []
    for rec in records:
        rel = rec["image_path"].replace("\\", "/")
        with Image.open(args.data_dir / rel) as im:
            w, _h = im.size
        lengths.append(w // WIDTH_DOWNSAMPLE)
    total_frames = int(sum(lengths))

    n_test = sum(1 for m in meta if m["is_test"])
    print(
        f"S2: {len(records)} lines ({n_test} frozen test, {len(records) - n_test} train), "
        f"{total_frames:,} frames, device={device}"
    )

    model = CRNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    cpu_model = None  # built lazily, only if a line OOMs on the GPU

    run_dir = start_run(
        "cache_s2_recogniser",
        {
            "seed": SEED,
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "meta": str(args.meta),
            "n_lines": len(records),
            "n_test_lines": n_test,
            "total_frames": total_frames,
            "num_classes": NUM_CLASSES,
            "dtype": "float32",
            "augment": False,
            "batch_size": 1,
            "device": str(device),
            "stores": "full per-frame distributions (not top-5) -- blank-drop needs the "
                      "full-vocab softmax at tuned temperatures, see module docstring",
        },
    )

    logprobs_path = run_dir / "logprobs.npy"
    store = open_memmap(
        logprobs_path, mode="w+", dtype=np.float32, shape=(total_frames, NUM_CLASSES)
    )

    loader = DataLoader(
        S2ImageDataset(records, args.data_dir), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate_single,
    )

    out_lines: list[dict] = []
    offset = 0
    n_cpu_fallback = 0
    total_cer = 0.0

    with torch.no_grad():
        for images, idx in loader:
            rec = records[idx]
            expected_t = lengths[idx]
            input_lengths = torch.tensor([expected_t], dtype=torch.long)

            used_device = str(device)
            try:
                log_probs = model(images.to(device))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # The allocator raises OutOfMemoryError, but when the
                # *driver* cannot map the memory it comes through as a
                # plain RuntimeError ("CUDA driver error: out of memory"
                # from cuMemSetAccess). The fallback this module promises
                # has to cover both, or one oversized line kills the whole
                # pass -- which is what happened on the 4GB laptop GPU, at
                # the first line, where the 12GB desktop never hit it.
                # Anything that is not an out-of-memory condition must
                # still fail loudly (CLAUDE.md rule 7).
                if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                        and "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                if cpu_model is None:
                    cpu_model = CRNN()
                    cpu_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
                    cpu_model.eval()
                log_probs = cpu_model(images)
                used_device = "cpu"
                n_cpu_fallback += 1
                print(
                    f"  {rec['id']}: OOM on GPU ({images.shape[-2]}x{images.shape[-1]}px) "
                    f"-- computed on CPU instead (same weights, recorded per line)"
                )

            t = log_probs.shape[0]
            if t != expected_t:
                raise RuntimeError(
                    f"{rec['id']}: model produced {t} frames but width//{WIDTH_DOWNSAMPLE} "
                    f"predicted {expected_t} -- the offsets written into logprobs.npy would "
                    f"be wrong, so refusing to continue."
                )

            store[offset : offset + t] = log_probs[:, 0, :].float().cpu().numpy()

            decoded = greedy_decode(log_probs.cpu(), input_lengths)[0]
            argmax_wx = wx.to_string([wx.INDEX_TO_SYMBOL[i] for i in decoded])
            truth_indices = [wx.SYMBOL_TO_INDEX[s] for s in wx.encode(rec["text"])]
            total_cer += cer(truth_indices, decoded)

            out_lines.append(
                {
                    "id": rec["id"],
                    "verse_id": meta[idx]["verse_id"],
                    "is_test": meta[idx]["is_test"],
                    "text": rec["text"],
                    "modern_text": rec["modern_text"],
                    "argmax_indices": decoded,
                    "argmax_wx": argmax_wx,
                    "n_frames": t,
                    "offset": offset,
                    "device": used_device,
                }
            )
            offset += t

            if len(out_lines) % 500 == 0:
                print(f"  cached {len(out_lines)}/{len(records)} lines")

    store.flush()
    del store

    if offset != total_frames:
        raise RuntimeError(f"wrote {offset} frames but preallocated {total_frames}")

    with (run_dir / "lines.jsonl").open("w", encoding="utf-8") as f:
        for row in out_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean_cer = total_cer / len(out_lines)
    results = {
        "n_lines": len(out_lines),
        "n_test_lines": n_test,
        "total_frames": offset,
        "logprobs_bytes": logprobs_path.stat().st_size,
        "mean_cer_vs_s2_ground_truth": mean_cer,
        "n_cpu_fallback": n_cpu_fallback,
    }
    finish_run(run_dir, results)

    print(f"\nCached {len(out_lines)} lines, {offset:,} frames "
          f"({logprobs_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Recogniser CER on S2 (vs S2's own old-text ground truth): {mean_cer:.4f}")
    if n_cpu_fallback:
        print(f"{n_cpu_fallback} line(s) fell back to CPU -- see 'device' in lines.jsonl")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
