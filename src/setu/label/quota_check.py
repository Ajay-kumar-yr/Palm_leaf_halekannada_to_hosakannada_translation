"""Which keys still have quota today, for which model.

Every plan in DEMO_PLAN.md 6a is written against a budget of 20
requests per key per day *per model*, so the first question each
morning is how much of that budget actually exists. Guessing is worse
than asking: a run that starts on an exhausted key burns wall-clock and
reports a 429 as a failure.

The probe is one trivial text request per key -- it costs 1 of that
key's 20 for the model being probed, which is the price of knowing.
Nothing is written to a run folder: this measures the provider's
accounting, not the project's.

A 429 is read in full, not from the first 200 bytes (RESULTS.md 6: the
quota id sits further down the body, so daily exhaustion was once
misread as a rate limit).

Usage:
    python -m setu.label.quota_check --model gemini-3.5-flash
    python -m setu.label.quota_check --model gemini-3.5-flash,gemini-3.1-flash-lite
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from setu.label.vlm_label import REPO_ROOT, USER_AGENT, collect_keys, load_env

PER_KEY_PER_MODEL = 20  # free tier, GenerateRequestsPerDayPerProjectPerModel


def probe(key: str, model: str, timeout: int = 30) -> tuple[str, str]:
    """('live' | 'daily' | 'rate' | 'error', detail)."""
    body = {"contents": [{"parts": [{"text": "ok"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT,
                 "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return "live", ""
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")  # the WHOLE body
        if e.code == 429:
            kind = "daily" if "PerDay" in raw else "rate"
            retry = ""
            for line in raw.splitlines():
                if "retryDelay" in line:
                    retry = line.strip().strip(',')
                    break
            return kind, retry
        return "error", f"HTTP {e.code}: {raw[:200]}"
    except Exception as e:  # noqa: BLE001 -- network, DNS, timeout
        return "error", f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemini-3.5-flash",
                   help="comma-separated; each model has its own per-key daily quota")
    p.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    p.add_argument("--base", default="GEMINI_API_KEY")
    args = p.parse_args()

    keys = collect_keys(load_env(args.env), args.base)
    print(f"{len(keys)} distinct keys in {args.env.name}\n")

    for model in [m.strip() for m in args.model.split(",") if m.strip()]:
        print(f"== {model}")
        live = 0
        for name, value in keys:
            state, detail = probe(value, model)
            live += state == "live"
            mark = {"live": "OK  ", "daily": "SPENT", "rate": "RATE", "error": "ERR "}[state]
            print(f"  {mark} {name:20s} {detail}")
        spent_by_probe = len(keys)
        print(f"  -> {live} of {len(keys)} keys live; budget ~{live * PER_KEY_PER_MODEL}"
              f" requests today, {spent_by_probe} already spent by this probe\n")


if __name__ == "__main__":
    main()
