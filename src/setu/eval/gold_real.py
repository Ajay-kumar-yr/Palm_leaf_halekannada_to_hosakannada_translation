"""What the hand transcriptions are worth: true CER on real lines.

Until now every real-image number was *agreement with a vision model*,
which nothing had checked -- the weakest claim in the project. A Kannada
reader transcribed the 32 held-out lines blind (the machine reading was
never shown, so it could not anchor theirs), and this turns that into
three measurements:

1. **How good the machine labels actually were** -- gold against the
   Gemini transcription the recogniser was trained on. This decides
   whether the whole teacher-student route rested on sand.
2. **True CER for the fine-tuned recogniser** -- gold against its own
   output, replacing the machine-label CER reported so far.
3. **Whether real manuscripts carry the archaic letters** (ಱ, ೞ).
   STATUS.md 3.9 found none in the *corpus* and left open whether the
   real leaves have them. A human reading the leaves answers it.

Usage:
    python -m setu.eval.gold_real
    python -m setu.eval.gold_real --checkpoint runs/<...>/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import unicodedata
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from setu.eval.metrics import agreement_cer, cer  # noqa: E402
from setu.runlog import finish_run, start_run  # noqa: E402

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARCHAIC = {"ಱ": "ra-archaic", "ೞ": "zha-archaic"}


def normalise(s: str) -> str:
    """NFC, collapse whitespace, strip. Typed input carries stray leading
    newlines and double spaces that are not transcription differences;
    counting them as errors would add noise to every system alike."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def load_gold(d: Path) -> dict[str, str]:
    gold = {}
    for f in sorted(d.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        data = doc.get("data", doc)
        if data.get("illegible"):
            continue
        text = normalise(data.get("text", ""))
        if text:
            gold[data["crop"]] = text
    return gold


def score_crnn(args, shared, gold, meta, per_line):
    import numpy as np
    import torch
    from PIL import Image

    from setu.data import wx
    from setu.recogniser.finetune_real import LINE_HEIGHT, MIN_HEIGHT
    from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE, greedy_decode

    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, sym_to_idx, idx_to_sym = wx.tables(args.digits)
    model = CRNN(num_classes=len(vocab) + 1).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.eval()

    scores, skipped = [], 0
    for crop in shared:
        im = Image.open(REPO_ROOT / meta[crop]["image"]).convert("L")
        w = max(WIDTH_DOWNSAMPLE, round(im.width * LINE_HEIGHT / im.height))
        a = np.array(im.resize((w, LINE_HEIGHT), Image.LANCZOS), dtype=np.uint8)
        if a.shape[0] < MIN_HEIGHT:
            pad = MIN_HEIGHT - a.shape[0]
            a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0)),
                       constant_values=int(np.median(a)))
        x = torch.from_numpy(a.astype(np.float32) / 255.0)[None, None].to(dev)
        with torch.no_grad():
            lp = model(x)
        lengths = torch.tensor([max(1, a.shape[1] // WIDTH_DOWNSAMPLE)])
        ids = greedy_decode(lp.cpu(), lengths)[0]
        hyp = wx.decode([idx_to_sym[i] for i in ids])
        try:
            ref_idx = [sym_to_idx[s] for s in wx.encode(gold[crop], digits=args.digits)]
        except ValueError:
            skipped += 1  # the human read a character the vocabulary lacks
            continue
        c = cer(ref_idx, ids)
        scores.append(c)
        for r in per_line:
            if r["crop"] == crop:
                r["crnn_hyp"] = hyp
                r["crnn_vs_gold_cer"] = round(c, 4)
    return scores, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-dir", type=Path,
                   default=Path("data/real_lines/gold_raw/transcriptions"))
    p.add_argument("--labels", type=Path, nargs="+", default=[
        Path("data/real_lines/train/train_labels.jsonl"),
        Path("data/real_lines/demo/demo_labels.jsonl")])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="a fine-tuned CRNN to score against gold; omit to skip")
    p.add_argument("--no-digits", dest="digits", action="store_false", default=True)
    args = p.parse_args()

    gold = load_gold(args.gold_dir)
    if not gold:
        raise SystemExit(f"no usable transcriptions in {args.gold_dir}")

    rows = []
    for f in args.labels:
        rows += [json.loads(line) for line in f.open(encoding="utf-8")]
    machine = {r["crop"]: normalise(r["text"]) for r in rows}
    meta = {r["crop"]: r for r in rows}

    shared = sorted(set(gold) & set(machine))
    print(f"{len(gold)} hand transcriptions, {len(shared)} with a machine label to compare")

    per_line, scores = [], []
    for crop in shared:
        g, m = gold[crop], machine[crop]
        d = agreement_cer(list(g), list(m))
        scores.append(d)
        per_line.append({"crop": crop, "page": meta[crop]["page"], "gold": g, "machine": m,
                         "machine_vs_gold_cer": round(d, 4),
                         "gold_chars": len(g), "machine_chars": len(m)})

    print("\nmachine label vs hand transcription")
    print(f"  mean CER   {statistics.fmean(scores):.4f}")
    print(f"  median     {statistics.median(scores):.4f}")
    print(f"  best/worst {min(scores):.4f} / {max(scores):.4f}")
    print(f"  within 0.35 (the label filter's threshold): "
          f"{sum(1 for s in scores if s <= 0.35)}/{len(scores)}")

    g_arch = {c: sum(t.count(c) for t in gold.values()) for c in ARCHAIC}
    m_arch = {c: sum(t.count(c) for t in machine.values()) for c in ARCHAIC}
    g_lines = {c: sum(1 for t in gold.values() if c in t) for c in ARCHAIC}
    print("\narchaic letters (the open question in STATUS.md 3.9)")
    for c, name in ARCHAIC.items():
        print(f"  {c} ({name}): human found {g_arch[c]} in {g_lines[c]} line(s); "
              f"machine label had {m_arch[c]}")

    results = {
        "n_gold": len(gold), "n_compared": len(shared),
        "machine_vs_gold": {
            "mean_cer": statistics.fmean(scores), "median_cer": statistics.median(scores),
            "min": min(scores), "max": max(scores),
            "n_within_0.35": sum(1 for s in scores if s <= 0.35),
        },
        "archaic_letters": {
            name: {"in_gold": g_arch[c], "gold_lines": g_lines[c], "in_machine_label": m_arch[c]}
            for c, name in ARCHAIC.items()
        },
        "caveat": "gold is a single blind transcription by one reader, not double-annotated: "
                  "ground truth for this project's purposes, but it carries its own error rate.",
    }

    if args.checkpoint:
        crnn_scores, skipped = score_crnn(args, shared, gold, meta, per_line)
        if crnn_scores:
            print("\nfine-tuned CRNN vs hand transcription  (TRUE CER)")
            print(f"  mean {statistics.fmean(crnn_scores):.4f}   "
                  f"median {statistics.median(crnn_scores):.4f}   n={len(crnn_scores)}"
                  + (f"   ({skipped} skipped: gold outside the vocabulary)" if skipped else ""))
            results["crnn_vs_gold"] = {
                "checkpoint": str(args.checkpoint),
                "mean_cer": statistics.fmean(crnn_scores),
                "median_cer": statistics.median(crnn_scores),
                "n": len(crnn_scores), "n_skipped_unencodable": skipped,
            }

    run_dir = start_run("gold_real_eval", {
        "seed": SEED, "gold_dir": str(args.gold_dir),
        "labels": [str(x) for x in args.labels],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "normalisation": "NFC + collapsed whitespace",
        "provenance": "32 held-out lines transcribed by hand, blind to the machine reading",
    })
    (run_dir / "per_line.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_line) + "\n", encoding="utf-8")
    finish_run(run_dir, results)
    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()
