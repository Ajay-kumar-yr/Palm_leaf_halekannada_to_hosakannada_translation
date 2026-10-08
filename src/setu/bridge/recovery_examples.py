"""Worked recovery examples: the soft bridge's mechanism, shown at frame
level (roadmap v4 §4.5, "the highest-value hour in the project" -- and
STATUS.md §4.2 act 3, the one demo act that needs neither Palmira nor a
working modernizer).

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
Roadmap §4.5 asks for cases where the recogniser's top choice was wrong,
the correct symbol was in its top 5, and the bridge produced the RIGHT
final answer. That last clause needs a modernizer good enough for the
right answer to come out the other end, which (STATUS.md §2, §3.1, §3.9)
we do not have.

So this reports the weaker, defensible claim instead: **the information
survives the interface.** On frames where argmax is wrong, argmax assigns
the correct symbol weight exactly 0 -- by construction, it keeps one
symbol and discards the rest. The bridge carries that same correct symbol
forward with a measurable, non-zero weight. That is the mechanism, it is
the project's novel contribution, and it is true independently of what the
modernizer then does with it.

Do not let this drift into "the bridge fixed the error" under
questioning. It did not; it declined to throw the answer away.

HOW THE PER-FRAME TRUTH IS OBTAINED
-----------------------------------
CTC gives no frame-to-character alignment (soft_bridge.py's docstring:
"there is no clean top-5 per letter position without collapsing frames
first"). To know which symbol a frame SHOULD have carried, this runs
standard CTC forced alignment -- Viterbi over the blank-extended target
sequence -- against the ground-truth old text. That is a decoding of
known-correct labels, not a model decision, so it introduces no new
guesswork into the comparison.

Lines where alignment is impossible (fewer frames than the extended
target needs, T < 2L+1) are skipped and counted, never silently dropped.

Note top-k MEMBERSHIP is temperature-invariant -- temperature scales
logits monotonically, so it cannot reorder them. Temperature changes only
the WEIGHTS the bridge assigns. "Was the true symbol in the top 5" is
therefore a property of the recogniser alone; "how much of it survived" is
where the temperature dial acts.

Usage:
    python -m setu.bridge.recovery_examples \
        --cache-dir runs/<...>_cache_s2_recogniser --temperature 1.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from setu.data import wx
from setu.recogniser.model import WIDTH_DOWNSAMPLE
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
INBAND_PATH = Path("data/splits/s2_inband_verse_ids.txt")
BLANK = 0
NEG = np.float32(-1e30)


def forced_align(lp: np.ndarray, targets: list[int]) -> np.ndarray | None:
    """CTC forced alignment by Viterbi. lp: (T, C) log-probs, targets: the
    true symbol indices. Returns (T,) per-frame true label (BLANK where
    the alignment says "nothing here"), or None if T is too short to fit
    the blank-extended target."""
    n_targets = len(targets)
    if n_targets == 0:
        return None
    size = 2 * n_targets + 1
    n_frames = lp.shape[0]
    # Minimum frames a valid CTC path needs: one per symbol, plus a
    # mandatory blank between each pair of IDENTICAL adjacent symbols.
    # Not 2L+1 -- that is the longest path (a blank between every symbol),
    # not the shortest, and using it as the guard wrongly rejects ~13% of
    # perfectly alignable lines.
    repeats = sum(1 for i in range(n_targets - 1) if targets[i] == targets[i + 1])
    if n_frames < n_targets + repeats:
        return None

    ext = np.full(size, BLANK, dtype=np.int64)
    ext[1::2] = targets

    # A skip (s-2 -> s) is legal only onto a real symbol, and only when it
    # differs from the previous real symbol -- otherwise the blank between
    # a doubled letter would be jumped and the two would collapse into one.
    skip_ok = np.zeros(size, dtype=bool)
    skip_ok[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])

    dp = np.full(size, NEG, dtype=np.float32)
    dp[0] = lp[0, ext[0]]
    if size > 1:
        dp[1] = lp[0, ext[1]]
    back = np.zeros((n_frames, size), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2

    for t in range(1, n_frames):
        stay = dp
        from_prev = np.concatenate(([NEG], dp[:-1]))
        from_skip = np.concatenate(([NEG, NEG], np.where(skip_ok[2:], dp[:-2], NEG)))
        stacked = np.stack([stay, from_prev, from_skip])
        choice = stacked.argmax(axis=0)
        dp = stacked[choice, np.arange(size)] + lp[t, ext]
        back[t] = choice

    # A valid path ends on the final blank or the final real symbol.
    state = size - 1 if dp[size - 1] >= dp[size - 2] else size - 2
    path = np.zeros(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        path[t] = state
        if t > 0:
            # int() matters: back is int8, and numpy 2 promotes
            # `python_int - np.int8` to int8, overflowing for state > 127.
            state -= int(back[t, state])
    return ext[path]


def bridge_weights(frame_lp: np.ndarray, temperature: float, top_k: int):
    """Exactly what soft_bridge.apply_soft_bridge does to one frame:
    temperature -> softmax -> top-k -> renormalise. Returns (indices,
    renormalised weights), highest first."""
    scaled = frame_lp / temperature
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    idx = np.argpartition(-probs, top_k - 1)[:top_k]
    idx = idx[np.argsort(-probs[idx])]
    kept = probs[idx]
    return idx, kept / kept.sum()


def _sym(i: int) -> str:
    return wx.INDEX_TO_SYMBOL.get(int(i), f"<{i}>")


def scan_line(rec: dict, lp: np.ndarray, temperature: float, top_k: int) -> tuple[list[dict], dict]:
    """Returns (candidate examples, per-line counts)."""
    try:
        targets = [wx.SYMBOL_TO_INDEX[s] for s in wx.encode(rec["text"])]
    except (ValueError, KeyError):
        return [], {"unencodable": 1}

    truth = forced_align(lp, targets)
    if truth is None:
        return [], {"unalignable": 1}

    argmax = lp.argmax(axis=1)
    real = truth != BLANK
    wrong = real & (argmax != truth)

    counts = {
        "aligned_real_frames": int(real.sum()),
        "wrong_top1_frames": int(wrong.sum()),
        "recoverable_frames": 0,
    }

    out = []
    probs_full = None
    for t in np.flatnonzero(wrong):
        idx, weights = bridge_weights(lp[t], temperature, top_k)
        hit = np.flatnonzero(idx == truth[t])
        if hit.size == 0:
            continue  # true symbol not even in the top-k: the bridge cannot help either
        counts["recoverable_frames"] += 1
        rank = int(hit[0])
        if probs_full is None:
            probs_full = None  # only computed per frame below, kept local
        # top-1 confidence at temperature 1, i.e. what the recogniser itself believed
        raw = np.exp(lp[t] - lp[t].max())
        raw /= raw.sum()
        out.append({
            "line_id": rec["id"],
            "verse_id": rec["verse_id"],
            "frame": int(t),
            "n_frames": int(lp.shape[0]),
            "pixel_col_start": int(t) * WIDTH_DOWNSAMPLE,
            "pixel_col_end": (int(t) + 1) * WIDTH_DOWNSAMPLE,
            "true_symbol": _sym(truth[t]),
            "argmax_symbol": _sym(argmax[t]),
            "argmax_confidence_raw": float(raw[argmax[t]]),
            "true_symbol_confidence_raw": float(raw[truth[t]]),
            "true_symbol_rank": rank + 1,
            "argmax_weight_on_true_symbol": 0.0,  # by construction
            "bridge_weight_on_true_symbol": float(weights[rank]),
            "top_k": [
                {"symbol": _sym(i), "bridge_weight": float(w), "raw_confidence": float(raw[i])}
                for i, w in zip(idx, weights)
            ],
        })
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--temperature", type=float, required=True,
        help="same dial as the bridge; required, never defaulted (see soft_bridge.py)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=12,
        help="how many worked examples to write out, best-first (3-4 are needed for the report)",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="restrict to frozen test lines (default: scan every cached line, which is fine -- "
             "these are illustrations of a mechanism, not a reported metric)",
    )
    parser.add_argument(
        "--time-steps", type=int, default=None,
        help="measure cost over N lines, project the full run, write nothing, exit",
    )
    args = parser.parse_args()
    np.random.seed(SEED)

    rows = [json.loads(l) for l in (args.cache_dir / "lines.jsonl").open(encoding="utf-8")]
    if args.test_only:
        rows = [r for r in rows if r["is_test"]]
    logprobs = np.load(args.cache_dir / "logprobs.npy", mmap_mode="r")

    inband = set()
    if INBAND_PATH.exists():
        inband = {int(x) for x in INBAND_PATH.read_text(encoding="utf-8").split()}

    if args.time_steps:
        t0 = time.time()
        for r in rows[: args.time_steps]:
            lp = np.asarray(logprobs[r["offset"] : r["offset"] + r["n_frames"]], dtype=np.float32)
            scan_line(r, lp, args.temperature, args.top_k)
        per = (time.time() - t0) / max(args.time_steps, 1)
        print(f"{per * 1000:.0f} ms/line over {args.time_steps} lines")
        print(f"projected for {len(rows)} lines: {per * len(rows) / 60:.1f} min")
        print("(--time-steps: nothing written)")
        return

    run_dir = start_run("recovery_examples", {
        "seed": SEED,
        "cache_dir": str(args.cache_dir),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "test_only": args.test_only,
        "n_lines_scanned": len(rows),
        "alignment": "CTC forced alignment (Viterbi) against ground-truth old text",
        "claim": "information survives the interface; NOT 'the bridge fixed the error'",
    })

    totals = {"aligned_real_frames": 0, "wrong_top1_frames": 0, "recoverable_frames": 0,
              "unalignable": 0, "unencodable": 0}
    examples: list[dict] = []
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        lp = np.asarray(logprobs[r["offset"] : r["offset"] + r["n_frames"]], dtype=np.float32)
        got, counts = scan_line(r, lp, args.temperature, args.top_k)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        for g in got:
            g["is_test"] = bool(r["is_test"])
            g["is_inband"] = r["verse_id"] in inband
        examples.extend(got)
        if n % 500 == 0:
            print(f"  ...{n}/{len(rows)} lines, {len(examples)} candidates")
    elapsed = time.time() - t0

    # Best-first: the more of the correct symbol that survived the
    # interface, the more clearly the frame shows the mechanism.
    examples.sort(key=lambda e: -e["bridge_weight_on_true_symbol"])

    wrong = totals["wrong_top1_frames"]
    rec = totals["recoverable_frames"]
    mean_w = float(np.mean([e["bridge_weight_on_true_symbol"] for e in examples])) if examples else 0.0

    results = {
        **totals,
        "recoverable_fraction_of_wrong_frames": (rec / wrong) if wrong else None,
        "mean_bridge_weight_on_true_symbol": mean_w,
        "mean_argmax_weight_on_true_symbol": 0.0,
        "n_examples_written": min(args.limit, len(examples)),
        "elapsed_seconds": round(elapsed, 1),
        "note": (
            "argmax weight is 0.0 by construction, not by measurement: argmax keeps one "
            "symbol and discards the rest. These frames are where that discarding is "
            "demonstrably lossy. This does NOT show the bridge produced a correct final "
            "answer -- see the module docstring."
        ),
    }
    finish_run(run_dir, results)

    with (run_dir / "examples.jsonl").open("w", encoding="utf-8") as f:
        for e in examples[: args.limit]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    _write_report(run_dir, examples[: args.limit], results, args)

    print()
    print(f"aligned real (non-blank) frames : {totals['aligned_real_frames']:,}")
    print(f"frames where top-1 was wrong    : {wrong:,}")
    if wrong:
        print(f"  ...of those, true symbol still in top-{args.top_k}: "
              f"{rec:,} ({rec / wrong * 100:.1f}%)")
    print(f"mean bridge weight on the true symbol (those frames): {mean_w:.3f}  vs argmax 0.000")
    if totals["unalignable"]:
        print(f"lines skipped, too few frames to align: {totals['unalignable']:,}")
    print(f"\n{run_dir}")


def _write_report(run_dir: Path, examples: list[dict], results: dict, args) -> None:
    """Human-readable version, for the demo slides and the report."""
    lines = [
        "# Worked recovery examples — the soft bridge at frame level",
        "",
        f"Cache: `{args.cache_dir}` · temperature {args.temperature} · top-{args.top_k}",
        "",
        "## What this shows",
        "",
        "On these frames the recogniser's top choice was **wrong** and the correct",
        "symbol was still inside its top-5. Argmax keeps one symbol and discards the",
        "rest, so it assigns the correct symbol weight **0** — by construction, not by",
        "measurement. The soft bridge carries it forward with the weight shown.",
        "",
        "**This does not show that the bridge produced a correct final answer.** It",
        "shows the information survived the interface. That distinction matters under",
        "questioning.",
        "",
        "## Aggregate",
        "",
        f"- aligned real (non-blank) frames: **{results['aligned_real_frames']:,}**",
        f"- frames where top-1 was wrong: **{results['wrong_top1_frames']:,}**",
    ]
    frac = results["recoverable_fraction_of_wrong_frames"]
    if frac is not None:
        lines.append(
            f"- of those, correct symbol still in top-{args.top_k}: "
            f"**{results['recoverable_frames']:,} ({frac * 100:.1f}%)**"
        )
    lines += [
        f"- mean weight the bridge puts on the correct symbol there: "
        f"**{results['mean_bridge_weight_on_true_symbol']:.3f}**",
        "- mean weight argmax puts on it: **0.000**",
        "",
        "## Examples",
        "",
    ]
    for n, e in enumerate(examples, 1):
        lines += [
            f"### {n}. `{e['line_id']}` frame {e['frame']}/{e['n_frames']}"
            f"{'  (frozen test)' if e.get('is_test') else ''}",
            "",
            f"Image columns ≈ **{e['pixel_col_start']}–{e['pixel_col_end']}px** "
            f"(frame × {WIDTH_DOWNSAMPLE}, the CRNN's width downsample).",
            "",
            f"- true symbol: **`{e['true_symbol']}`** (rank {e['true_symbol_rank']}, "
            f"raw confidence {e['true_symbol_confidence_raw']:.3f})",
            f"- recogniser chose: **`{e['argmax_symbol']}`** "
            f"(raw confidence {e['argmax_confidence_raw']:.3f})",
            "",
            f"| rank | symbol | bridge weight | raw confidence |",
            f"|---|---|---|---|",
        ]
        for rank, c in enumerate(e["top_k"], 1):
            mark = " ← correct" if c["symbol"] == e["true_symbol"] else ""
            lines.append(
                f"| {rank} | `{c['symbol']}`{mark} | {c['bridge_weight']:.3f} "
                f"| {c['raw_confidence']:.3f} |"
            )
        lines += [
            "",
            f"**argmax carries `{e['true_symbol']}` forward with weight 0.000; "
            f"the bridge carries it with {e['bridge_weight_on_true_symbol']:.3f}.**",
            "",
        ]
    (run_dir / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
