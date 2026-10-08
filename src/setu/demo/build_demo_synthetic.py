"""The soft bridge shown on synthetic lines, where the recogniser works.

On real crops the CRNN cannot read at all (0.71 CER held out, even
adapted on the same pages — `20261008T094654Z_crnn_finetune_real`). On
the synthetic frozen test split it reads at **1.53% CER**, and that is
where the project's claim can actually be demonstrated rather than
asserted.

What makes this stronger than the real-crop demo, not merely a fallback:

- **true ground truth**, from the rendered corpus — not vision-model
  labels, so every CER here is a real number;
- lines the recogniser **never trained on** (the frozen test split,
  `CLAUDE.md` rule 1);
- genuine recoveries: lines selected because argmax got a character
  **wrong** while the bridge still held the right one.

Nothing is recomputed. It reads the committed per-frame cache
(`*_cache_s2_recogniser`), which already holds the full distribution for
every frame of all 3,500 S2 lines, so no GPU and no model load.

Selection ranks frozen-test lines by how much the bridge has to show:
lines where argmax is wrong at a few characters and the correct symbol
survives in the top-k. Lines argmax already gets right demonstrate
nothing, and lines it gets wholly wrong are unreadable either way.

Output matches `build_demo.py`'s `demo.json`, so `make_viewer` renders
it unchanged.

Usage:
    python -m setu.demo.build_demo_synthetic --cache-dir runs/<ts>_cache_s2_recogniser
    python -m setu.demo.build_demo_synthetic --cache-dir ... --no-llm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from setu.bridge.line_alternatives import (
    analyse_line,
    annotated_text,
    argmax_text,
    summarise,
    variant_readings,
)
from setu.data import wx
from setu.demo.modernize_llm import modernize
from setu.eval.metrics import align, cer
from setu.label.vlm_label import KeyRing, collect_keys, load_env
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def score_line(symbols: list[str], slots, truth_syms: list[str]) -> tuple[int, int, list[int]]:
    """(recoveries, substitutions, recovered slot indices).

    Counts only **substitutions** — a truth symbol aligned to a
    different predicted symbol — and asks whether the bridge still held
    the right one there. Insertions and deletions have no predicted
    character to carry an alternative, so they are not recoverable in
    this sense and are left out of both counts.

    Must align rather than compare position-by-position: one insertion
    shifts every later character, so a line with 3 real errors can show
    ~100 positional mismatches and would be ranked as a spectacular
    recovery case while actually being noise.
    """
    recoveries, subs, where = 0, 0, []
    for ri, hj in align(truth_syms, symbols):
        if ri is None or hj is None:
            continue
        true_sym = truth_syms[ri]
        slot = slots[hj]
        if slot.chosen == true_sym:
            continue
        subs += 1
        # Only count a recovery where the character is also FLAGGED
        # uncertain. The bridge keeps the top-k at every frame, but the
        # demo only displays alternatives for flagged characters, so
        # counting an unflagged one would claim a recovery the viewer
        # cannot see.
        if slot.uncertain and any(a.symbol == true_sym for a in slot.alternatives):
            recoveries += 1
            where.append(hj)
    return recoveries, subs, where


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/20261007T022632Z_cache_s2_recogniser"))
    p.add_argument("--images-dir", type=Path, default=Path("data/s2"))
    p.add_argument("--n-lines", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--uncertain-below", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-substitutions", type=int, default=6,
                   help="Skip lines with more wrong characters than this -- a demo line must "
                        "be mostly right for the recovery to be visible.")
    p.add_argument("--min-recoveries", type=int, default=1)
    p.add_argument("--max-cer", type=float, default=0.15,
                   help="Skip lines the recogniser largely failed on; they show nothing.")
    p.add_argument("--max-chars", type=int, default=220, help="skip lines too long to read on screen")
    p.add_argument("--llm-model", default="gemini-3.5-flash")
    p.add_argument("--no-llm", action="store_true")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    lines = [json.loads(l) for l in (args.cache_dir / "lines.jsonl").open(encoding="utf-8")]
    store = np.load(args.cache_dir / "logprobs.npy", mmap_mode="r")
    manifest = {r["id"]: r for r in
                (json.loads(l) for l in (args.images_dir / "manifest.jsonl").open(encoding="utf-8"))
                if not r.get("skipped")}
    test = [r for r in lines if r["is_test"] and len(r["text"]) <= args.max_chars]
    print(f"{len(test)} frozen-test lines within {args.max_chars} chars (of {len(lines)} cached)")

    scored = []
    n_unencodable = 0
    for r in test:
        try:
            truth_syms = wx.encode(r["text"])
        except ValueError:
            n_unencodable += 1
            continue
        lp = torch.from_numpy(np.asarray(store[r["offset"]: r["offset"] + r["n_frames"]]))
        symbols, slots = analyse_line(lp, r["n_frames"], temperature=args.temperature,
                                      top_k=args.top_k, uncertain_below=args.uncertain_below)
        rec, subs, where = score_line(symbols, slots, truth_syms)
        line_cer = cer([wx.SYMBOL_TO_INDEX[s] for s in truth_syms],
                       [wx.SYMBOL_TO_INDEX[s] for s in symbols])
        # A worked example has to be READABLE. Ranking by raw recovery
        # count selects the most broken lines -- one scored 22 recoveries
        # at 0.58 CER, which is illegible whichever branch reads it. Keep
        # lines the recogniser mostly got right, with a handful of
        # genuinely confusable characters, and rank those by accuracy.
        if rec and args.min_recoveries <= rec and subs <= args.max_substitutions                 and line_cer <= args.max_cer:
            scored.append((line_cer, -rec, r, symbols, slots, truth_syms, where, subs))

    # Most accurate line first; among equals, most recoveries.
    scored.sort(key=lambda t: (t[0], t[1]))
    chosen = scored[: args.n_lines]
    print(f"{len(scored)} lines where the bridge holds a symbol argmax got wrong; showing {len(chosen)}")
    if n_unencodable:
        print(f"({n_unencodable} lines skipped: text outside the base vocabulary)")
    if not chosen:
        raise SystemExit("no recovery lines found -- try a higher --temperature or --top-k")

    run_dir = start_run("demo_build_synthetic", {
        "seed": SEED, "cache_dir": str(args.cache_dir), "temperature": args.temperature,
        "uncertain_below": args.uncertain_below, "top_k": args.top_k,
        "max_substitutions": args.max_substitutions, "max_cer": args.max_cer,
        "n_lines": len(chosen), "llm_model": None if args.no_llm else args.llm_model,
        "checkpoint": json.loads((args.cache_dir / "config.json").read_text())["checkpoint"],
        "source": "synthetic S2 frozen test split -- TRUE ground truth, never trained on",
        "selection": "ranked by characters where argmax is wrong but the correct symbol "
                     "survives among the bridge's alternatives",
        "modernizer": "an LLM, NOT the project's from-scratch modernizer (a measured negative "
                      "result, STATUS.md 3.1)",
    })
    ring = None if args.no_llm else KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))

    out, n_calls = [], 0
    for line_cer, neg_rec, r, symbols, slots, truth_syms, where, subs in chosen:
        rec = -neg_rec
        b3_text = argmax_text(symbols)
        b4_block = annotated_text(symbols, slots)
        rivals = variant_readings(symbols, slots)
        img = manifest[r["id"]]["image_path"].replace("\\", "/")

        entry = {
            "crop": r["id"], "page": f"synthetic verse {r['verse_id']}",
            "image": f"{args.images_dir.as_posix()}/{img}",
            "trained": False,  # frozen test split
            "label_text": r["text"], "cer_vs_label": line_cer,
            "reference_modern": r.get("modern_text"),
            "b3_text": b3_text, "b4_block": b4_block,
            "rival_readings": [{"text": t, "prob": round(pr, 3)} for t, pr in rivals],
            "uncertainty": summarise(slots),
            "n_argmax_substitutions": subs, "n_recoverable": rec,
            "recovered_at": where,
            "uncertain_chars": [
                {"index": s.index, "chosen": s.chosen, "top1": round(s.top1_prob, 3),
                 "alternatives": [{"symbol": a.symbol, "kannada": a.kannada, "prob": round(a.prob, 3)}
                                  for a in s.alternatives if a.symbol != "<blank>"][:4]}
                for s in slots if s.uncertain
            ],
        }
        if not args.no_llm:
            for branch, text in (("b3", b3_text), ("b4", b4_block)):
                res = modernize(text, branch, args.llm_model, REPO_ROOT / "data" / "demo_cache", ring=ring)
                entry[f"{branch}_modern"] = res["modern"]
                n_calls += int(not res["cached"])
            entry["branches_differ"] = entry["b3_modern"] != entry["b4_modern"]
        out.append(entry)
        print(f"  {r['id']}  CER {line_cer:.3f}  {rec}/{subs} substitutions recoverable"
              + ("" if args.no_llm else f"  differ={entry['branches_differ']}"))

    (run_dir / "demo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {
        "n_lines": len(out),
        "n_heldout": len(out),
        "n_candidate_recovery_lines": len(scored),
        "mean_cer_vs_machine_label": sum(e["cer_vs_label"] for e in out) / len(out),
        "cer_is_true_ground_truth": True,
        "mean_fraction_uncertain": sum(e["uncertainty"]["fraction_uncertain"] for e in out) / len(out),
        "n_lines_with_uncertain_chars": sum(1 for e in out if e["uncertain_chars"]),
        "total_argmax_substitutions": sum(e["n_argmax_substitutions"] for e in out),
        "total_recoverable": sum(e["n_recoverable"] for e in out),
        "n_branches_differ": None if args.no_llm else sum(1 for e in out if e.get("branches_differ")),
        "n_llm_calls_made": n_calls,
    }
    finish_run(run_dir, results)
    print(json.dumps(results, indent=2))
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
