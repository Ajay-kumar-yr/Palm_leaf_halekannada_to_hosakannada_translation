"""Precompute everything the live demo shows (DEMO_PLAN.md day 3).

    real photo -> Palmira line crops -> fine-tuned CRNN -> per-frame
    distributions -> {B3 argmax | B4 soft bridge} -> LLM modernizer

Nothing here runs at showtime. Every CRNN forward and every LLM call is
done once, now, and written to `runs/<ts>_demo_build/demo.json`; the
viewer reads that file. A demo that calls an API live is one network
hiccup or one exhausted quota away from failing in front of an
examiner.

Scope, stated here because it is easy to overclaim (DEMO_PLAN.md 5b):
the recogniser was adapted on lines from these same pages. By default
this builds only lines marked `demo_split=holdout`, which training
never saw — but they share a hand, a leaf and an ink with lines it did
see. This demonstrates **adaptation to a known hand, not
generalisation**. `--include-trained` adds the rest of the page for a
full-page view; those lines are flagged `trained: true` and the viewer
must mark them.

Usage:
    python -m setu.demo.build_demo --checkpoint runs/<ts>_crnn_finetune_real/best_model.pt
    python -m setu.demo.build_demo --checkpoint ... --no-llm      # recogniser only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from setu.bridge.line_alternatives import (  # noqa: E402
    analyse_line,
    annotated_text,
    argmax_text,
    summarise,
    variant_readings,
)
from setu.demo.modernize_llm import modernize  # noqa: E402
from setu.eval.metrics import cer  # noqa: E402
from setu.data import wx  # noqa: E402
from setu.label.vlm_label import KeyRing, collect_keys, load_env  # noqa: E402
from setu.recogniser.model import CRNN, WIDTH_DOWNSAMPLE  # noqa: E402
from setu.recogniser.finetune_real import LINE_HEIGHT, MIN_HEIGHT  # noqa: E402
from setu.runlog import finish_run, start_run  # noqa: E402

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_crop(path: Path) -> np.ndarray:
    """Same preprocessing the fine-tune used: height 64, aspect kept."""
    im = Image.open(path).convert("L")
    w = max(WIDTH_DOWNSAMPLE, round(im.width * LINE_HEIGHT / im.height))
    a = np.array(im.resize((w, LINE_HEIGHT), Image.LANCZOS), dtype=np.uint8)
    if a.shape[0] < MIN_HEIGHT:
        pad = MIN_HEIGHT - a.shape[0]
        a = np.pad(a, ((pad // 2, pad - pad // 2), (0, 0)), constant_values=int(np.median(a)))
    return a


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path("data/real_lines/demo/train_labels.jsonl"))
    p.add_argument("--pages", nargs="*", default=None, help="Only these pages (default: all).")
    p.add_argument("--include-trained", action="store_true",
                   help="Also build lines the recogniser trained on, flagged trained=true.")
    p.add_argument("--temperature", type=float, default=1.5,
                   help="Soft-bridge temperature. Retune on real lines before trusting the default, "
                        "which was chosen on synthetic data by a metric later shown to move the "
                        "wrong way (STATUS.md 3.2).")
    p.add_argument("--uncertain-below", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--llm-model", default="gemini-3.5-flash")
    p.add_argument("--no-llm", action="store_true", help="Recogniser only; skip modernization.")
    p.add_argument("--max-lines", type=int, default=None)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in args.labels.open(encoding="utf-8")]
    if not args.include_trained:
        rows = [r for r in rows if r.get("demo_split") == "holdout"]
    if args.pages:
        rows = [r for r in rows if r["page"] in args.pages]
    rows.sort(key=lambda r: (r["page"], r["crop"]))
    if args.max_lines:
        rows = rows[: args.max_lines]
    if not rows:
        raise SystemExit("no lines selected -- check --labels, --pages, and the hold-out split")
    set_dir = args.labels.parent
    print(f"{len(rows)} lines from {len({r['page'] for r in rows})} page(s), device={device}")

    model = CRNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    run_dir = start_run("demo_build", {
        "seed": SEED, "checkpoint": str(args.checkpoint), "labels": str(args.labels),
        "temperature": args.temperature, "uncertain_below": args.uncertain_below,
        "top_k": args.top_k, "llm_model": None if args.no_llm else args.llm_model,
        "include_trained": args.include_trained, "n_lines": len(rows),
        "scope": "writer-dependent: the recogniser was adapted on other lines of these same "
                 "pages. Lines marked trained=false were held out from training; lines marked "
                 "trained=true were not. This shows adaptation to a known hand, not "
                 "generalisation to unseen manuscripts.",
        "modernizer": "an LLM, NOT the project's from-scratch modernizer, which is a measured "
                      "negative result (STATUS.md 3.1).",
    })
    cache_dir = REPO_ROOT / "data" / "demo_cache"
    ring = None if args.no_llm else KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))

    out, n_llm_calls, cers = [], 0, []
    for i, r in enumerate(rows):
        gray = load_crop(set_dir / r["crop"])
        x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(device)
        with torch.no_grad():
            log_probs = model(x)
        il = max(1, gray.shape[1] // WIDTH_DOWNSAMPLE)
        symbols, slots = analyse_line(
            log_probs.cpu(), il, temperature=args.temperature,
            top_k=args.top_k, uncertain_below=args.uncertain_below,
        )
        b3_text = argmax_text(symbols)
        b4_block = annotated_text(symbols, slots)
        rivals = variant_readings(symbols, slots)

        # CER against the machine label -- NOT human ground truth.
        label_syms = wx.encode(r["text"]) if r.get("text") else []
        line_cer = cer([wx.SYMBOL_TO_INDEX[s] for s in label_syms],
                       [wx.SYMBOL_TO_INDEX[s] for s in symbols]) if label_syms else None
        if line_cer is not None:
            cers.append(line_cer)

        entry = {
            "crop": r["crop"], "page": r["page"],
            "image": str((set_dir / r["crop"]).relative_to(REPO_ROOT)).replace("\\", "/"),
            "trained": r.get("demo_split") != "holdout",
            "label_text": r.get("text"), "cer_vs_label": line_cer,
            "b3_text": b3_text, "b4_block": b4_block,
            "rival_readings": [{"text": t, "prob": round(pr, 3)} for t, pr in rivals],
            "uncertainty": summarise(slots),
            "uncertain_chars": [
                {"index": s.index, "chosen": s.chosen, "top1": round(s.top1_prob, 3),
                 "alternatives": [{"symbol": a.symbol, "kannada": a.kannada, "prob": round(a.prob, 3)}
                                  for a in s.alternatives if a.symbol != "<blank>"][:4]}
                for s in slots if s.uncertain
            ],
        }

        if not args.no_llm:
            for branch, text in (("b3", b3_text), ("b4", b4_block)):
                res = modernize(text, branch, args.llm_model, cache_dir, ring=ring)
                entry[f"{branch}_modern"] = res["modern"]
                n_llm_calls += int(not res["cached"])
            entry["branches_differ"] = entry["b3_modern"] != entry["b4_modern"]

        out.append(entry)
        flag = "" if entry["trained"] else " [held-out]"
        print(f"  [{i + 1}/{len(rows)}] {r['crop']}{flag} "
              f"{entry['uncertainty']['n_uncertain']}/{entry['uncertainty']['n_chars']} uncertain"
              + ("" if args.no_llm else f"  differ={entry['branches_differ']}"))

    (run_dir / "demo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_differ = sum(1 for e in out if e.get("branches_differ"))
    results = {
        "n_lines": len(out),
        "n_heldout": sum(1 for e in out if not e["trained"]),
        "mean_cer_vs_machine_label": (sum(cers) / len(cers)) if cers else None,
        "mean_fraction_uncertain": sum(e["uncertainty"]["fraction_uncertain"] for e in out) / len(out),
        "n_lines_with_uncertain_chars": sum(1 for e in out if e["uncertain_chars"]),
        "n_branches_differ": None if args.no_llm else n_differ,
        "n_llm_calls_made": n_llm_calls,
        "caveats": [
            "CER is against machine labels, not human ground truth.",
            "Writer-dependent: adapted on other lines of these same pages.",
            "The modernizer is an LLM, not the project's own model.",
        ],
    }
    finish_run(run_dir, results)
    print(json.dumps(results, indent=2))
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
