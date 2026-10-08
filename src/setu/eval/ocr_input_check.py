"""Does the vision model read binarized crops better than grayscale?

Worth settling rather than assuming. Binarization helped the CRNN a
little (6% of the domain gap), but the CRNN was trained on textured
grayscale, so it had every reason to prefer what it knew. A vision LLM
has no such prior, and clean ink on white may genuinely be easier to
read than ink on a photographed palm leaf.

This is measurable because 32 held-out lines now carry human
transcriptions: read each line both ways and score both against the
human reading. Same model, same prompt, same temperature 0 -- only the
image differs.

Usage:
    python -m setu.eval.ocr_input_check --limit 32
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from setu.eval.metrics import agreement_cer
from setu.eval.gold_real import load_gold, normalise
from setu.label.vlm_label import KeyRing, collect_keys, load_env
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def read_once(ring, model, png, temperature=0.0):
    import base64
    import urllib.error

    from setu.label.vlm_label import PROMPT, _post, clean

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"text": PROMPT},
    ]}], "generationConfig": {"temperature": temperature}}
    for attempt in range(5):
        try:
            kname, key = ring.current()
        except RuntimeError:
            return None
        try:
            d = _post(url, {"x-goog-api-key": key}, body)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429:
                # The daily cap is named in the QuotaFailure details
                # (GenerateRequestsPerDayPerProjectPerModel-FreeTier), not
                # always in `message`, so check the whole body. Free tier
                # is 20 requests per key PER MODEL per day.
                if "PerDay" in msg or "per day" in msg.lower():
                    ring.retire(kname)
                else:
                    ring.advance()
                    time.sleep(min(20, 2 * 2 ** attempt))
                continue
            if e.code in (500, 502, 503):
                time.sleep(min(20, 2 * 2 ** attempt))
                continue
            raise
        except (TimeoutError, urllib.error.URLError):
            continue
        cands = d.get("candidates") or []
        ring.advance()
        time.sleep(1.5)
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", [])
        return clean("".join(p.get("text", "") for p in parts if not p.get("thought")))
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-dir", type=Path,
                   default=Path("data/real_lines/gold_raw/transcriptions"))
    p.add_argument("--gray-labels", type=Path, default=Path("data/real_lines/gold/gold_train.jsonl"))
    p.add_argument("--bin-labels", type=Path,
                   default=Path("data/real_lines/gold_unet/gold_train.jsonl"))
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--limit", type=int, default=32)
    args = p.parse_args()

    gold = load_gold(args.gold_dir)
    gray = {json.loads(l)["crop"]: json.loads(l) for l in args.gray_labels.open(encoding="utf-8")}
    binz = {json.loads(l)["crop"]: json.loads(l) for l in args.bin_labels.open(encoding="utf-8")}
    crops = sorted(set(gold) & set(gray) & set(binz))[: args.limit]
    print(f"{len(crops)} lines with a human transcription, read both ways\n")

    ring = KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))
    rows, scores = [], {"grayscale": [], "binarized": []}
    for i, crop in enumerate(crops):
        g = gold[crop]
        rec = {"crop": crop, "gold": g}
        for kind, src in (("grayscale", gray[crop]), ("binarized", binz[crop])):
            png = (REPO_ROOT / src["image"]).read_bytes()
            text = read_once(ring, args.model, png)
            if text is None:
                print("  all keys exhausted; stopping")
                crops = crops[:i]
                break
            text = normalise(text)
            d = agreement_cer(list(g), list(text))
            rec[kind] = text
            rec[f"{kind}_cer"] = round(d, 4)
            scores[kind].append(d)
        else:
            rows.append(rec)
            print(f"  [{i+1}/{len(crops)}] {crop}: grayscale {rec['grayscale_cer']:.3f}  "
                  f"binarized {rec['binarized_cer']:.3f}")
            continue
        break

    if not rows:
        raise SystemExit("nothing measured")
    res = {
        "n_lines": len(rows), "model": args.model,
        **{k: {"mean_cer": statistics.fmean(v), "median_cer": statistics.median(v)}
           for k, v in scores.items() if v},
        "binarized_minus_grayscale": statistics.fmean(scores["binarized"]) - statistics.fmean(scores["grayscale"]),
        "n_lines_binarized_better": sum(1 for r in rows if r["binarized_cer"] < r["grayscale_cer"]),
        "note": "CER against a single blind human transcription; lower is better",
    }
    run_dir = start_run("ocr_input_check", {
        "seed": SEED, "model": args.model, "temperature": 0,
        "gray_labels": str(args.gray_labels), "bin_labels": str(args.bin_labels),
        "binarizer": "sajjan_unet",
    })
    (run_dir / "per_line.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    finish_run(run_dir, res)
    print("\n" + json.dumps(res, indent=2))
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
