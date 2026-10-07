"""The reported comparison: B0 vs B3 vs B4 on the frozen test split
(CLAUDE.md Week 4: "run B0/B3/B4 on frozen test split").

  B0  copy the input unchanged -- the floor. CLAUDE.md keeps it because old
      and modern Kannada share most vocabulary, so a system that merely
      echoes its input already scores respectably, and BLEU in particular
      is misleading without it on the page. Two variants are reported:
        B0_recognised  the recogniser's own reading, decoded WX -> Kannada,
                       unmodernized. The pipeline-matched floor: identical
                       starting information to B3 and B4.
        B0_oracle      the ground-truth old text, unmodernized. Measures
                       pure old-vs-modern vocabulary overlap with no OCR
                       error, i.e. the floor B3/B4 would face given a
                       perfect recogniser.
  B3  recogniser -> argmax -> modernizer. The conventional pipeline.
  B4  recogniser -> soft bridge -> modernizer. Ours.

Metrics come only from setu.eval.metrics (CLAUDE.md rule 5): chrF++ is
primary, BLEU secondary and read against B0 rather than on its own.

Two different row sets, deliberately, and the report must keep them apart:
  CER is computed over ALL frozen test lines, since the recogniser side is
  untouched by the modern-text quality problem.
  chrF++/BLEU are computed over the IN-BAND frozen test lines only
  (data/splits/s2_inband_verse_ids.txt), because ~66% of S2 pairs its
  verses with free-form commentary rather than a line-level modernization,
  and scoring generation against commentary measures essay-writing and
  collapses the B0 floor. The frozen split itself is never rewritten --
  see data/raw_corpus/build_s2_inband_subset.py.

B3 and B4 read the same cached recogniser output and decode through the
same greedy_generate_from_memory, so the only difference between them is
the interface.

Usage:
    python -m setu.eval.run_comparison --cache-dir runs/<...>_cache_s2_recogniser \
        --b3 runs/<...>_b3/best_model.pt --b4 runs/<...>_b4_T1.5/best_model.pt --temperature 1.5
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import numpy as np
import torch

from setu.bridge.soft_bridge import SoftBridgeConfig, apply_soft_bridge
from setu.data import wx
from setu.eval.metrics import bleu, cer, chrf_plus_plus
from setu.modernizer.model import Modernizer
from setu.modernizer.vocab import CharVocab
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
VOCAB_PATH = Path("data/modernizer_vocab.json")
INBAND_PATH = Path("data/splits/s2_inband_verse_ids.txt")


def _wx_indices_to_kannada(indices: list[int]) -> str:
    """Recogniser output is WX symbol indices; targets are Kannada script."""
    return wx.decode([wx.INDEX_TO_SYMBOL[i] for i in indices])


def _load_modernizer(path: Path, vocab: CharVocab, device) -> Modernizer:
    m = Modernizer(len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def _generate_b3(model, rows, vocab, device, max_len, batch_size) -> list[str]:
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        srcs = [r["argmax_indices"] for r in chunk]
        max_src = max(max(len(s) for s in srcs), 1)
        src = torch.zeros(max_src, len(chunk), dtype=torch.long, device=device)
        mask = torch.ones(len(chunk), max_src, dtype=torch.bool, device=device)
        for j, s in enumerate(srcs):
            if s:
                src[: len(s), j] = torch.tensor(s, dtype=torch.long, device=device)
                mask[j, : len(s)] = False
            else:
                mask[j, 0] = False  # an empty reading still needs one real position
        for seq in model.greedy_generate(src, mask, max_len):
            out.append(vocab.decode(seq))
    return out


@torch.no_grad()
def _generate_b4(model, rows, logprobs, vocab, device, cfg, max_len, batch_size) -> list[str]:
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        blocks = [
            np.array(logprobs[r["offset"] : r["offset"] + r["n_frames"]], dtype=np.float32)
            for r in chunk
        ]
        max_t = max(b.shape[0] for b in blocks)
        src = torch.zeros(max_t, len(chunk), blocks[0].shape[1], dtype=torch.float32)
        for j, b in enumerate(blocks):
            src[: b.shape[0], j, :] = torch.from_numpy(b)
        src = src.to(device)
        lengths = torch.tensor([b.shape[0] for b in blocks], dtype=torch.long, device=device)
        blended, mem_mask = apply_soft_bridge(src, lengths, model.src_embed.weight, cfg)
        memory = model.encode_embeds(blended, mem_mask)
        for seq in model.greedy_generate_from_memory(memory, mem_mask, max_len):
            out.append(vocab.decode(seq))
    return out


def _score(name: str, hyps: list[str], refs: list[str]) -> dict:
    c = chrf_plus_plus(hyps, refs)
    b = bleu(hyps, refs)
    return {
        "system": name,
        "chrf++": c.score,
        "bleu": b.score,
        "mean_hyp_chars": float(np.mean([len(h) for h in hyps])),
        "mean_ref_chars": float(np.mean([len(r) for r in refs])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--b3", type=Path, default=None)
    parser.add_argument(
        "--b4", type=Path, default=None,
        help="omit to score B0/B3 only. Better than passing a B4 trained on different data "
             "just to fill the column -- that would look like the matched comparison and is not.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="required with --b4")
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--max-gen-len", type=int, default=1024,
                        help="in-band targets top out at 796 chars")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.b4 is not None and args.temperature is None:
        parser.error("--temperature is required with --b4 (must match its fine-tune)")
    if args.b3 is None and args.b4 is None:
        parser.error("pass at least one of --b3 / --b4 (B0 alone needs no model)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocab.load(VOCAB_PATH)
    inband = {int(x) for x in INBAND_PATH.read_text(encoding="utf-8").split()}
    with (args.cache_dir / "lines.jsonl").open(encoding="utf-8") as f:
        all_rows = [json.loads(l) for l in f]

    test_rows = [r for r in all_rows if r["is_test"]]
    inband_test = [r for r in test_rows if r["verse_id"] in inband]
    print(f"frozen test lines: {len(test_rows)}   in-band (scored end-to-end): {len(inband_test)}")

    # --- recogniser CER, over ALL frozen test lines ---
    cers = [
        cer([wx.SYMBOL_TO_INDEX[s] for s in wx.encode(r["text"])], r["argmax_indices"])
        for r in test_rows
    ]
    cer_all = float(np.mean(cers))
    cer_inband = float(np.mean([
        cer([wx.SYMBOL_TO_INDEX[s] for s in wx.encode(r["text"])], r["argmax_indices"])
        for r in inband_test
    ]))
    print(f"recogniser CER: all test {cer_all:.4f}   in-band test {cer_inband:.4f}")

    refs = [r["modern_text"] for r in inband_test]
    logprobs = np.load(args.cache_dir / "logprobs.npy", mmap_mode="r")

    b0_recognised = [_wx_indices_to_kannada(r["argmax_indices"]) for r in inband_test]
    b0_oracle = [r["text"] for r in inband_test]
    systems = [
        _score("B0_recognised", b0_recognised, refs),
        _score("B0_oracle", b0_oracle, refs),
    ]

    b3_hyps = b4_hyps = None
    if args.b3 is not None:
        print("generating B3 ...")
        b3_hyps = _generate_b3(
            _load_modernizer(args.b3, vocab, device), inband_test, vocab, device,
            args.max_gen_len, args.batch_size,
        )
        systems.append(_score("B3_argmax", b3_hyps, refs))
    if args.b4 is not None:
        print("generating B4 ...")
        cfg = SoftBridgeConfig(temperature=args.temperature, pool_size=args.pool_size)
        b4_hyps = _generate_b4(
            _load_modernizer(args.b4, vocab, device), inband_test, logprobs, vocab, device, cfg,
            args.max_gen_len, args.batch_size,
        )
        systems.append(_score("B4_soft_bridge", b4_hyps, refs))

    run_dir = start_run(
        "eval_b0_b3_b4",
        {
            "seed": SEED, "cache_dir": str(args.cache_dir), "b3": str(args.b3), "b4": str(args.b4),
            "temperature": args.temperature, "pool_size": args.pool_size,
            "max_gen_len": args.max_gen_len, "device": str(device),
            "n_frozen_test": len(test_rows), "n_inband_test_scored": len(inband_test),
            "note": "CER over all frozen test lines; chrF++/BLEU over in-band frozen test lines "
                    "only (disclosed) -- see data/raw_corpus/build_s2_inband_subset.py. The "
                    "frozen split itself is unmodified.",
        },
    )

    with (run_dir / "per_line.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(inband_test):
            rec = {
                "id": r["id"], "verse_id": r["verse_id"],
                "old_text": r["text"], "reference_modern": r["modern_text"],
                "b0_recognised": b0_recognised[i],
            }
            if b3_hyps is not None:
                rec["b3"] = b3_hyps[i]
            if b4_hyps is not None:
                rec["b4"] = b4_hyps[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    results = {
        "recogniser_cer_all_frozen_test": cer_all,
        "recogniser_cer_inband_test": cer_inband,
        "n_frozen_test": len(test_rows),
        "n_inband_test_scored": len(inband_test),
        "systems": systems,
    }
    finish_run(run_dir, results)

    print(f"\n{'system':18s} {'chrF++':>9s} {'BLEU':>8s} {'hyp chars':>10s} {'ref chars':>10s}")
    for s in systems:
        print(f"{s['system']:18s} {s['chrf++']:>9.2f} {s['bleu']:>8.2f} "
              f"{s['mean_hyp_chars']:>10.0f} {s['mean_ref_chars']:>10.0f}")
    by_name = {s["system"]: s for s in systems}
    b0 = by_name["B0_recognised"]
    for name in ("B3_argmax", "B4_soft_bridge"):
        if name in by_name:
            s = by_name[name]
            verdict = "BELOW the copy floor" if s["chrf++"] < b0["chrf++"] else "above the floor"
            print(f"{name} vs B0_recognised: chrF++ {s['chrf++'] - b0['chrf++']:+.2f}  "
                  f"BLEU {s['bleu'] - b0['bleu']:+.2f}   ({verdict})")
    if "B3_argmax" in by_name and "B4_soft_bridge" in by_name:
        b3s, b4s = by_name["B3_argmax"], by_name["B4_soft_bridge"]
        print(f"\nB4 - B3: chrF++ {b4s['chrf++'] - b3s['chrf++']:+.2f}  "
              f"BLEU {b4s['bleu'] - b3s['bleu']:+.2f}   <- the headline comparison")
    else:
        print("\n(only one branch scored -- the B3-vs-B4 headline needs both)")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
