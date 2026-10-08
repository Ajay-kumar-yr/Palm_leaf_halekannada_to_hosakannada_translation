"""Cut real glyphs out of hand-transcribed manuscript lines.

The data-scaling curve (DEMO_PLAN.md §5c) says the recogniser needs
orders of magnitude more labelled handwriting than anyone can transcribe
here. This attacks that directly: take the lines a human DID transcribe,
find where each character sits in the image, and cut it out. The result
is a bank of real glyphs in the scribe's own hand, every one perfectly
labelled — from which unlimited training lines can be stitched.

Finding the positions is the hard part, and it is done the only honest
way available: **CTC forced alignment** (Viterbi against the known text,
`setu.bridge.recovery_examples.forced_align`), using a recogniser
deliberately **overfit on these same lines**. That is circular for
measuring accuracy and would be indefensible there — but it is exactly
right here, because the model only has to reproduce text it has already
memorised in order to say *where* each character is. The alignment is
then checked, not trusted: a span that collapses to nothing, or swallows
a quarter of the line, is dropped and counted.

Each glyph is saved with a small horizontal margin, because CTC frames
are 8 image columns wide and a character's ink routinely bleeds a frame
either side of its peak.

Usage:
    python -m setu.render.glyphs --checkpoint runs/<overfit>/best_model.pt
    python -m setu.render.glyphs --checkpoint ... --contact-sheet sheet.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from setu.bridge.recovery_examples import forced_align  # noqa: E402
from setu.data import wx  # noqa: E402
from setu.recogniser.finetune_real import LINE_HEIGHT, MIN_HEIGHT  # noqa: E402
from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE  # noqa: E402
from setu.runlog import finish_run, start_run  # noqa: E402

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BLANK = 0


def load_line(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    w = max(WIDTH_DOWNSAMPLE, round(im.width * LINE_HEIGHT / im.height))
    a = np.array(im.resize((w, LINE_HEIGHT), Image.LANCZOS), dtype=np.uint8)
    if a.shape[0] < MIN_HEIGHT:
        pad = MIN_HEIGHT - a.shape[0]
        a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0)), constant_values=int(np.median(a)))
    return a


def spans_from_alignment(aligned: np.ndarray) -> list[tuple[int, int, int]]:
    """(symbol index, first frame, last frame) per aligned character run."""
    out, cur, start = [], None, 0
    for t, lab in enumerate(aligned.tolist()):
        if lab == cur:
            continue
        if cur is not None and cur != BLANK:
            out.append((cur, start, t - 1))
        cur, start = lab, t
    if cur is not None and cur != BLANK:
        out.append((cur, start, len(aligned) - 1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="a CRNN overfit on the gold lines (loss near zero)")
    p.add_argument("--labels", type=Path, default=Path("data/real_lines/gold/gold_train.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/glyph_bank"))
    p.add_argument("--margin-frames", type=int, default=1,
                   help="frames of context kept either side of a span (ignored with --midpoint)")
    p.add_argument("--midpoint", action="store_true",
                   help="Cut at the midpoint between neighbouring spans instead of a fixed "
                        "margin. CTC marks where a character PEAKS, not where it starts and "
                        "ends, so a fixed margin either clips the character (too narrow) or "
                        "drags in its neighbours (too wide). Splitting the gap between "
                        "consecutive peaks partitions the line instead of guessing a width.")
    p.add_argument("--max-span-fraction", type=float, default=0.25,
                   help="drop a glyph whose span covers more of the line than this")
    p.add_argument("--contact-sheet", type=Path, default=None)
    p.add_argument("--digits", action="store_true", default=True)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, sym_to_idx, idx_to_sym = wx.tables(args.digits)

    model = CRNN(num_classes=len(vocab) + 1).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.eval()

    rows = [json.loads(l) for l in args.labels.open(encoding="utf-8")]
    print(f"{len(rows)} transcribed lines, device={dev}")

    bank: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    per_line = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for r in rows:
        gray = load_line(REPO_ROOT / r["image"])
        x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(dev)
        with torch.no_grad():
            lp = model(x)[:, 0].cpu().numpy()
        n_frames = max(1, gray.shape[1] // WIDTH_DOWNSAMPLE)
        lp = lp[:n_frames]
        try:
            targets = [sym_to_idx[s] for s in wx.encode(r["text"], digits=args.digits)]
        except ValueError:
            counts["line_unencodable"] += 1
            continue
        aligned = forced_align(lp, targets)
        if aligned is None:
            counts["line_too_short_to_align"] += 1
            continue

        spans = spans_from_alignment(aligned)
        if len(spans) != len(targets):
            # Viterbi must emit every target exactly once; anything else
            # means the alignment is not trustworthy for this line.
            counts["line_span_count_mismatch"] += 1
            continue

        page_dir = args.out_dir / "glyphs"
        page_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for k, (cls, f0, f1) in enumerate(spans):
            sym = idx_to_sym[cls]
            if args.midpoint:
                prev_end = spans[k - 1][2] if k > 0 else -1
                next_start = spans[k + 1][1] if k + 1 < len(spans) else n_frames
                lo = (prev_end + f0 + 1) // 2 if k > 0 else max(0, f0 - args.margin_frames)
                hi = (f1 + next_start + 1) // 2 if k + 1 < len(spans) else f1 + 1 + args.margin_frames
                x0 = max(0, lo * WIDTH_DOWNSAMPLE)
                x1 = min(gray.shape[1], hi * WIDTH_DOWNSAMPLE)
            else:
                x0 = max(0, (f0 - args.margin_frames) * WIDTH_DOWNSAMPLE)
                x1 = min(gray.shape[1], (f1 + 1 + args.margin_frames) * WIDTH_DOWNSAMPLE)
            if x1 - x0 < 2:
                counts["glyph_empty_span"] += 1
                continue
            if (x1 - x0) / gray.shape[1] > args.max_span_fraction:
                counts["glyph_span_too_wide"] += 1
                continue
            crop = gray[:, x0:x1]
            safe = "".join(c if c.isalnum() else f"u{ord(c):04x}" for c in sym)
            name = f"{safe}__{r['crop'].replace('/', '__').replace('.png', '')}__{k:03d}.png"
            Image.fromarray(crop).save(page_dir / name)
            bank[sym].append({"file": f"glyphs/{name}", "width": int(x1 - x0),
                              "source_line": r["crop"], "index": k})
            counts["glyphs_kept"] += 1
            kept += 1
        per_line.append({"crop": r["crop"], "n_targets": len(targets), "n_kept": kept})

    with (args.out_dir / "bank.jsonl").open("w", encoding="utf-8") as f:
        for sym, items in sorted(bank.items()):
            f.write(json.dumps({"symbol": sym, "kannada": _render(sym), "n": len(items),
                                "glyphs": items}, ensure_ascii=False) + "\n")

    widths = [g["width"] for items in bank.values() for g in items]
    results = {
        "n_lines": len(rows), "n_lines_aligned": len(per_line),
        "n_glyphs": counts["glyphs_kept"],
        "n_distinct_symbols": len(bank),
        "vocabulary_coverage": round(len(bank) / len(vocab), 3),
        "symbols_with_1": sum(1 for v in bank.values() if len(v) == 1),
        "symbols_with_5_or_more": sum(1 for v in bank.values() if len(v) >= 5),
        "glyph_width_px": {"min": min(widths), "median": int(np.median(widths)),
                           "max": max(widths)} if widths else None,
        "rejections": {k: v for k, v in counts.items() if k != "glyphs_kept"},
        "alignment": "CTC forced alignment (Viterbi) using a recogniser overfit on these same "
                     "lines -- legitimate for locating characters it has memorised, NOT a "
                     "measurement of accuracy",
    }
    run_dir = start_run("glyph_bank", {
        "seed": SEED, "checkpoint": str(args.checkpoint), "labels": str(args.labels),
        "margin_frames": args.margin_frames, "midpoint": args.midpoint,
        "max_span_fraction": args.max_span_fraction,
        "out_dir": str(args.out_dir),
    })
    (run_dir / "per_line.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_line) + "\n", encoding="utf-8")
    finish_run(run_dir, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.contact_sheet:
        _contact_sheet(bank, args.out_dir, args.contact_sheet)
        print(f"contact sheet -> {args.contact_sheet}")
    print(f"Run folder: {run_dir}")


def _render(sym: str) -> str:
    try:
        return wx.decode([sym])
    except Exception:
        return sym


def _contact_sheet(bank: dict, out_dir: Path, path: Path, per_row: int = 12) -> None:
    """One row per symbol, up to `per_row` exemplars — the only way to
    judge whether the alignment actually cut characters or smeared."""
    syms = sorted(bank, key=lambda s: -len(bank[s]))[:28]
    cell_h = LINE_HEIGHT
    rows = []
    for sym in syms:
        imgs = [Image.open(out_dir / g["file"]) for g in bank[sym][:per_row]]
        w = sum(i.width + 4 for i in imgs)
        strip = Image.new("L", (max(w, 1), cell_h), 255)
        x = 0
        for im in imgs:
            strip.paste(im, (x, 0))
            x += im.width + 4
        rows.append((sym, strip))
    width = max(r[1].width for r in rows) + 60
    sheet = Image.new("L", (width, cell_h * len(rows) + 10), 255)
    for i, (sym, strip) in enumerate(rows):
        sheet.paste(strip, (60, i * cell_h))
    sheet.save(path)


if __name__ == "__main__":
    main()
