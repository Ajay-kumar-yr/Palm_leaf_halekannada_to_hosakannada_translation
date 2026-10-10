"""Build B — carrying uncertainty forward on REAL manuscript crops.

The project's claim is that collapsing a recogniser's output to one
string before modernization throws away information the modernizer could
have used. On synthetic lines that is demonstrated with our own CRNN's
per-frame CTC distributions (`build_demo_synthetic.py`). On real
manuscripts our CRNN reads nothing usable — 0.70 CER held out, and the
data-scaling curve says that is not fixable with the data obtainable
here (DEMO_PLAN.md §5c) — so the same principle is shown with a
different source of uncertainty.

**Read the crop N times at non-zero temperature.** Where the readings
agree, the model is confident. Where they disagree, the disagreement
*is* the uncertainty, and the sample frequencies are its probabilities.
Then:

    B3  the consensus reading alone — one string, uncertainty discarded
    B4  the same reading plus the alternatives and their frequencies

Same modernizer, same prompt, same temperature; only the input differs.
That is exactly the B3/B4 contrast, on real data.

**This is NOT the soft bridge, and the report must say so.** The bridge
operates on CTC frame distributions from our own recogniser; this is an
ensemble estimate from a vision model. The mechanism differs. The claim
is the same one, and the quantified result stays the frame-level
measurement on synthetic data (80.6% of wrong top-1 frames still held
the correct symbol in the top-5).

Consensus is built by alignment, not by position: readings differ in
length, so one inserted character would otherwise shift every later
position and manufacture disagreement everywhere.

Output matches `build_demo.py`'s `demo.json`, so `make_viewer` renders
it unchanged.

Usage:
    python -m setu.demo.build_demo_real --n-lines 10 --samples 5
    python -m setu.demo.build_demo_real --dry-run        # cost, no calls
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import statistics
import time
import urllib.error
from pathlib import Path

from setu.eval.metrics import agreement_cer, align
from setu.demo.modernize_llm import modernize
from setu.label.vlm_label import KeyRing, PROMPT, _post, clean, collect_keys, load_env
from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


READING_CACHE = REPO_ROOT / "data" / "demo_cache" / "readings"


def cache_path(crop: str, model: str, temperature: float) -> Path:
    """One file per crop per reading setting.

    Readings are the expensive thing in this project -- 20 requests per
    key per day per model -- and before this cache existed a crash on
    line 4 of 16 threw away the readings for lines 1-3 and the day's
    budget with them (a ConnectionResetError, which `sample_readings`
    did not catch). The prompt is part of the key: a reworded prompt is
    a different reading, and replaying the old one under the new prompt
    would be invisible.
    """
    tag = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]
    return (READING_CACHE / f"{model}_t{temperature}_{tag}" /
            f"{crop.replace('/', '__').replace('.png', '')}.json")


def load_cached(crop: str, model: str, temperature: float) -> list[dict]:
    f = cache_path(crop, model, temperature)
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))["readings"]


def save_cached(crop: str, model: str, temperature: float, readings: list[dict]) -> None:
    f = cache_path(crop, model, temperature)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"crop": crop, "model": model, "temperature": temperature,
                             "readings": readings}, ensure_ascii=False, indent=1),
                 encoding="utf-8")


def content_cache_path(png: bytes, model: str, temperature: float) -> Path:
    """Readings cache keyed by the IMAGE BYTES, not by a crop name.

    The crop-name cache below only helps `main()`. The demo UI reaches
    `sample_readings` through `app.make_reader`, which has no crop id --
    its crops are freshly cut by Palmira into `data/demo_work/<job>/`,
    and the job directory is new every run, so a path-keyed cache would
    never hit twice. Hashing the bytes is stable across jobs, across
    callers and across re-segmentation of the same page.

    This is what makes the demo pre-cache safe to interrupt: the
    pipeline cache only stores a page once every line of it has been
    read, so without this a crash on line 4 threw away the reads for
    lines 1-3, and those are 20 of a 100-call day.
    """
    tag = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]
    digest = hashlib.sha256(png).hexdigest()[:24]
    return READING_CACHE / f"by-content_{model}_t{temperature}_{tag}" / f"{digest}.json"


def sample_readings(ring: KeyRing, model: str, png: bytes, n: int, temperature: float,
                    gap: float = 1.5) -> list[dict]:
    """n independent readings of one crop. Each is its own request: the
    API has no n>1 for this shape, and separate calls are what makes the
    samples independent rather than one decode's beam.

    Rotates keys and backs off, like the labeller. Five rapid calls per
    crop otherwise trips the per-minute limit on the first image and the
    whole run dies before reading anything.
    """
    # Replay whatever has already been paid for on these exact bytes at
    # these exact settings, and only call out for the shortfall.
    cache_file = content_cache_path(png, model, temperature)
    out = []
    if cache_file.exists():
        try:
            out = json.loads(cache_file.read_text(encoding="utf-8"))["readings"]
        except (OSError, ValueError, KeyError):
            out = []   # a half-written cache file is not worth a crash
        if len(out) >= n:
            return out[:n]

    def _persist() -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"model": model, "temperature": temperature,
                                          "readings": out}, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}},
        {"text": PROMPT},
    ]}], "generationConfig": {"temperature": temperature}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for _ in range(n - len(out)):
        for attempt in range(5):
            try:
                kname, key = ring.current()
            except RuntimeError:
                _persist()  # every key spent; keep what we have paid for
                return out
            try:
                d = _post(url, {"x-goog-api-key": key}, body)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                if e.code == 429:
                    if "PerDay" in msg or "per day" in msg.lower():
                        ring.retire(kname)
                    else:
                        ring.advance()
                        time.sleep(min(20, 2 * 2 ** attempt))
                    continue
                if e.code in (500, 502, 503):
                    # 503 is per-routing, not per-key-quota: rotate rather
                    # than hammer one key that happens to be unlucky.
                    ring.advance()
                    time.sleep(min(20, 2 * 2 ** attempt))
                    continue
                raise
            except OSError:
                # Covers URLError and TimeoutError (both OSError) and the
                # bare ConnectionResetError that Google's edge throws under
                # load -- which was uncaught, so one reset killed a 16-line
                # run at line 4 and spent the readings it had already paid
                # for. Transient: back off and try the next key.
                ring.advance()
                time.sleep(min(20, 2 * 2 ** attempt))
                continue
            cands = d.get("candidates") or []
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                out.append({"raw": raw, "text": clean(raw), "usage": d.get("usageMetadata")})
                _persist()  # after every call, not at the end: a crash
                            # between here and the last line of a page
                            # would otherwise discard paid-for reads
            ring.advance()  # spread load across keys
            time.sleep(gap)
            break
    _persist()
    return out


def consensus(readings: list[str]) -> tuple[str, list[dict]]:
    """(base reading, per-character variants).

    The base is the medoid — the reading closest to all the others — not
    the first sample, so one outlier cannot become the spine everything
    else is measured against.

    Each base character gets the distribution of what the other readings
    put at that aligned position; `None` means a reading had nothing
    there (it deleted the character), which is itself a vote.
    """
    texts = [t for t in readings if t.strip()]
    if not texts:
        return "", []
    if len(texts) == 1:
        return texts[0], [{"index": i, "chosen": c, "votes": {c: 1}, "n": 1}
                          for i, c in enumerate(texts[0])]

    # Index-based, never identity: when several samples read the line
    # identically Python hands back the same string object, so `is not`
    # would discard every one of them and report a confident line as
    # maximally uncertain.
    dist = [sum(agreement_cer(a, b) for j, b in enumerate(texts) if j != i)
            for i, a in enumerate(texts)]
    bi_base = dist.index(min(dist))
    base = texts[bi_base]
    others = [t for j, t in enumerate(texts) if j != bi_base]

    votes: list[collections.Counter] = [collections.Counter({c: 1}) for c in base]
    for other in others:
        seen = set()
        for bi, oj in align(base, other):
            if bi is None:
                continue  # insertion in `other`; nothing in base to attribute it to
            votes[bi][other[oj] if oj is not None else ""] += 1
            seen.add(bi)
        for bi in range(len(base)):
            if bi not in seen:
                votes[bi][""] += 1

    n = len(texts)
    slots = [{"index": i, "chosen": base[i], "votes": dict(v), "n": n} for i, v in enumerate(votes)]
    return base, slots


def uncertain_slots(slots: list[dict], agree_below: float) -> list[dict]:
    """Characters the samples disagreed about, most contested first."""
    out = []
    for s in slots:
        top = s["votes"].get(s["chosen"], 0) / s["n"]
        if top < agree_below:
            alts = sorted(((c, k / s["n"]) for c, k in s["votes"].items()),
                          key=lambda t: t[1], reverse=True)
            out.append({**s, "agreement": round(top, 3),
                        "alternatives": [{"kannada": c or "(nothing)", "symbol": c,
                                          "prob": round(p, 3)} for c, p in alts]})
    out.sort(key=lambda s: s["agreement"])
    return out


def variant_lines(base: str, unc: list[dict], max_variants: int) -> list[tuple[str, float]]:
    """Whole-line readings the samples kept alive — the argmax reading
    first, then one per contested character, most probable alternative
    first. Whole lines because a Kannada reader judges a word, not a
    glyph."""
    out = [(base, 1.0)]
    swaps = []
    for s in unc:
        for a in s["alternatives"]:
            if a["symbol"] == s["chosen"]:
                continue
            swaps.append((a["prob"], s["index"], a["symbol"]))
    swaps.sort(reverse=True)
    for prob, idx, ch in swaps[: max_variants - 1]:
        out.append((base[:idx] + ch + base[idx + 1:], prob))
    return out


def annotated_block(base: str, variants: list[tuple[str, float]]) -> str:
    lines = [f"reading: {base}"]
    rivals = [(t, p) for t, p in variants[1:] if t != base]
    if rivals:
        lines.append("the recogniser was undecided here; it also read:")
        lines += [f"  {t}  ({p:.2f})" for t, p in rivals]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("data/real_lines/demo/demo_labels.jsonl"))
    p.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    p.add_argument("--n-lines", type=int, default=10)
    p.add_argument("--samples", type=int, default=5, help="readings per crop")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--agree-below", type=float, default=1.0,
                   help="flag a character when the samples' agreement is below this "
                        "(1.0 = any disagreement at all)")
    p.add_argument("--max-variants", type=int, default=5)
    p.add_argument("--binarized-labels", type=Path, default=None,
                   help="Read the U-Net binarized crops instead of the photo crops. Measured "
                        "on 10 gold lines: 0.428 CER against 0.447 grayscale, so a small but "
                        "consistent gain (setu.eval.ocr_input_check).")
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--llm-model", default="gemini-3.5-flash")
    p.add_argument("--no-llm", action="store_true", help="sample readings only; skip modernization")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore cached readings and call for every sample again")
    p.add_argument("--dry-run", action="store_true", help="print the call budget and exit")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.labels.open(encoding="utf-8")]
    if args.binarized_labels:
        binz = {json.loads(l)["crop"]: json.loads(l)["image"]
                for l in args.binarized_labels.open(encoding="utf-8")}
        rows = [{**r, "image": binz[r["crop"]], "input": "binarized"}
                for r in rows if r["crop"] in binz]
    if args.split != "all":
        rows = [r for r in rows if r.get("demo_split") == args.split]
    rows.sort(key=lambda r: (r["page"], r["crop"]))
    rows = rows[: args.n_lines]
    if not rows:
        raise SystemExit("no lines selected")

    # Price what is actually left to call, not the nominal total: with a
    # warm cache most of a rerun is free, and a budget that ignores that
    # is the reason a run gets postponed for quota it does not need.
    cached = 0 if args.no_cache else sum(
        min(len(load_cached(r["crop"], args.model, args.temperature)), args.samples)
        for r in rows)
    reads = len(rows) * args.samples - cached
    modern = 0 if args.no_llm else len(rows) * 2
    print(f"{len(rows)} lines x {args.samples} readings = {len(rows) * args.samples}; "
          f"{cached} already cached -> {reads} reading calls"
          f"{'' if args.no_llm else f', + {modern} modernizer calls'} = {reads + modern} total")
    if args.dry_run:
        print("dry run - nothing called")
        return

    ring = KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))
    if not ring.keys:
        raise SystemExit("no GEMINI_API_KEY in .env")

    run_dir = start_run("demo_build_real", {
        "seed": SEED, "labels": str(args.labels), "split": args.split,
        "n_lines": len(rows), "samples": args.samples, "temperature": args.temperature,
        "input": "U-Net binarized" if args.binarized_labels else "photo crop (grayscale)",
        "agree_below": args.agree_below, "model": args.model,
        "llm_model": None if args.no_llm else args.llm_model,
        "uncertainty_source": "ENSEMBLE over repeated vision-model readings at non-zero "
                              "temperature -- NOT the CTC soft bridge. The mechanism differs; "
                              "the claim (collapsing early discards usable information) is the "
                              "same. The quantified bridge result stays the frame-level "
                              "measurement on synthetic data.",
        "consensus": "medoid reading; per-character votes by Levenshtein alignment, not position",
    })

    out, n_calls = [], 0
    for i, r in enumerate(rows):
        png = (REPO_ROOT / r["image"]).read_bytes()
        samples = [] if args.no_cache else load_cached(r["crop"], args.model, args.temperature)
        reused = len(samples)
        if reused < args.samples:
            fresh = sample_readings(ring, args.model, png, args.samples - reused,
                                    args.temperature)
            n_calls += len(fresh)
            samples += fresh
            save_cached(r["crop"], args.model, args.temperature, samples)
        samples = samples[: args.samples]
        if reused:
            print(f"  [{i + 1}/{len(rows)}] {r['crop']}: {reused} reading(s) from cache")
        if len(samples) < args.samples:
            # Fewer samples is a smaller pile for B4 to carry, so a
            # short-read line understates recovery against the lines
            # beside it. Say so rather than average it in silently
            # (CLAUDE.md rule 7); the readings are cached, so a rerun
            # tomorrow tops it up for free.
            print(f"  [{i + 1}/{len(rows)}] {r['crop']}: WARNING only {len(samples)} of "
                  f"{args.samples} readings -- quota or network ran out")
        texts = [s["text"] for s in samples]
        base, slots = consensus(texts)
        if not base:
            print(f"  [{i + 1}/{len(rows)}] {r['crop']}: no readable samples, skipped")
            continue
        unc = uncertain_slots(slots, args.agree_below)
        variants = variant_lines(base, unc, args.max_variants)
        block = annotated_block(base, variants)

        pairwise = [agreement_cer(a, b) for ai, a in enumerate(texts)
                    for b in texts[ai + 1:]] or [0.0]
        entry = {
            "crop": r["crop"], "page": r["page"], "image": r["image"],
            "trained": r.get("demo_split") != "holdout",
            "label_text": r.get("text"), "cer_vs_label": None,
            "b3_text": base, "b4_block": block,
            "rival_readings": [{"text": t, "prob": round(pr, 3)} for t, pr in variants],
            "uncertainty": {
                "n_chars": len(slots), "n_uncertain": len(unc),
                "fraction_uncertain": len(unc) / len(slots) if slots else 0.0,
                "mean_top1": round(statistics.fmean(
                    [s["votes"].get(s["chosen"], 0) / s["n"] for s in slots]), 4) if slots else None,
                "mean_sample_disagreement": round(statistics.fmean(pairwise), 4),
            },
            "uncertain_chars": [
                {"index": s["index"], "chosen": s["chosen"], "top1": s["agreement"],
                 "alternatives": s["alternatives"][:4]} for s in unc[:10]
            ],
            # The display list above is capped for the viewer; analysis
            # needs every contested position, or a recovery measured
            # against it is silently understated (it read 9.6% instead of
            # the true 34.2% the first time).
            "uncertain_all": [
                {"index": s["index"], "chosen": s["chosen"], "top1": s["agreement"],
                 "alternatives": s["alternatives"]} for s in unc
            ],
            "samples": texts,
        }
        if not args.no_llm:
            for branch, text in (("b3", base), ("b4", block)):
                res = modernize(text, branch, args.llm_model, REPO_ROOT / "data" / "demo_cache", ring=ring)
                entry[f"{branch}_modern"] = res["modern"]
                n_calls += int(not res["cached"])
            entry["branches_differ"] = entry["b3_modern"] != entry["b4_modern"]
        out.append(entry)
        print(f"  [{i + 1}/{len(rows)}] {r['crop']}: {len(unc)}/{len(slots)} chars contested, "
              f"sample disagreement {entry['uncertainty']['mean_sample_disagreement']:.3f}"
              + ("" if args.no_llm else f", differ={entry['branches_differ']}"))

    if not out:
        raise SystemExit("nothing built")
    (run_dir / "demo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {
        "n_lines": len(out),
        "n_heldout": sum(1 for e in out if not e["trained"]),
        "mean_fraction_uncertain": statistics.fmean(
            e["uncertainty"]["fraction_uncertain"] for e in out),
        "mean_sample_disagreement": statistics.fmean(
            e["uncertainty"]["mean_sample_disagreement"] for e in out),
        "n_lines_with_uncertain_chars": sum(1 for e in out if e["uncertain_chars"]),
        "n_branches_differ": None if args.no_llm else sum(1 for e in out if e.get("branches_differ")),
        "n_api_calls": n_calls,
        "caveats": [
            "Uncertainty is an ENSEMBLE over repeated vision-model readings, not the CTC soft "
            "bridge. Same claim, different mechanism.",
            "The recogniser here is a vision LLM, not the project's CRNN, which cannot read real "
            "crops (0.70 CER; see the data-scaling curve).",
            "The modernizer is an LLM, not the project's own model.",
            # Written when only demo lines had no transcription. It is
            # false for a run over the gold lines, and a stale caveat in a
            # results file is worse than none -- it would have this run
            # claiming no ground truth exists for lines that have it.
            ("No ground truth: these lines have no human transcription, so no CER is reported."
             if not any(r.get("gold_text") for r in rows) else
             "These lines have hand transcriptions; CER against them is measured separately by "
             "setu.eval.real_recovery and setu.eval.gold_real, not here."),
        ],
    }
    finish_run(run_dir, results)
    print(json.dumps(results, indent=2))
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
