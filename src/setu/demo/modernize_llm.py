"""LLM modernizer for the demo's two branches (DEMO_PLAN.md).

The from-scratch modernizer is a measured negative result: it scores
6-7 chrF++ *below* simply copying its input, because its training
targets were ಭಾವಾರ್ಥ commentary rather than modernizations (STATUS.md
3.1). It cannot carry a demo, so the demo modernizes with an LLM
instead. **The report must say this plainly**: the modernizer shown is
not ours, and ours is documented as the negative result it is.

What the demo *does* still test is the project's actual claim — what
the recogniser hands forward:

    B3 (argmax)      one committed string. Uncertainty already gone.
    B4 (soft bridge) the same reading PLUS the rival readings the
                     recogniser was still holding, with probabilities.

Same model, same prompt, same temperature; only the input differs. That
is the comparison, and it is the one the project is about.

Both branches use temperature 0 so a difference between them is the
input's doing, not sampling noise. Every call is cached to disk by a
hash of (model, prompt, input): the demo must never depend on a network
or a quota at showtime, and a cached run is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
from pathlib import Path

from setu.label.vlm_label import KeyRing, collect_keys, load_env, _post

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SYSTEM = (
    "You rewrite old Kannada (halegannada) manuscript text into modern Kannada "
    "(hosagannada).\n"
    "- Keep the meaning and the word order as close to the original as you can. "
    "This is modernization, not translation and not commentary.\n"
    "- Output ONLY the modern Kannada text. No explanation, no English, no notes.\n"
    "- The text comes from OCR of a damaged palm leaf, so it may contain errors. "
    "Produce the most plausible reading."
)

B4_EXTRA = (
    "\n- The recogniser was NOT certain about some characters. Where you are given "
    "alternative readings of the whole line with their probabilities, choose whichever "
    "alternative makes a real Kannada word in context -- the most probable one is not "
    "always right. Then modernize the reading you chose."
)


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()[:32]


def modernize(text_block: str, branch: str, model: str, cache_dir: Path,
              ring: KeyRing | None = None, allow_network: bool = True) -> dict:
    """`text_block` is argmax text (B3) or the annotated block (B4).

    Returns {"modern": str, "cached": bool, ...}. With
    `allow_network=False` a cache miss raises rather than calling out --
    that is the mode the live demo runs in.
    """
    if branch not in ("b3", "b4"):
        raise ValueError(f"branch must be b3 or b4, got {branch!r}")
    prompt = (SYSTEM + (B4_EXTRA if branch == "b4" else "")
              + "\n\nOld Kannada:\n" + text_block + "\n\nModern Kannada:")

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(model, prompt)}.json"
    if path.exists():
        out = json.loads(path.read_text(encoding="utf-8"))
        out["cached"] = True
        return out

    if not allow_network:
        raise RuntimeError(
            f"cache miss for {branch} and allow_network=False -- the demo must run "
            f"from cache; pre-generate it with setu.demo.build_demo"
        )
    if ring is None:
        ring = KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), "GEMINI_API_KEY"))

    # Same quota handling as the labeller: a daily limit is per key, so
    # rotate rather than wait it out; a rate limit is worth a short
    # backoff. Without this a single 429 aborts the whole demo build.
    last_error = None
    for attempt in range(6):
        try:
            kname, key = ring.current()
        except RuntimeError as e:
            raise RuntimeError(f"all keys exhausted while modernizing: {e}") from e
        try:
            d = _post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"x-goog-api-key": key},
                {"contents": [{"parts": [{"text": prompt}]}],
                 "generationConfig": {"temperature": 0}},
            )
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            last_error = f"HTTP {e.code}: {body}"
            if e.code == 429:
                if "PerDay" in body or "per day" in body.lower():
                    ring.retire(kname)
                else:
                    ring.advance()
                    time.sleep(min(30, 5 * 2 ** attempt))
                continue
            if e.code in (500, 502, 503):
                time.sleep(min(30, 5 * 2 ** attempt))
                continue
            raise
        except (TimeoutError, urllib.error.URLError) as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
    else:
        raise RuntimeError(f"modernize failed after retries: {last_error}")

    cands = d.get("candidates") or []
    if not cands:
        raise RuntimeError(f"no candidates: {json.dumps(d)[:300]}")
    parts = cands[0].get("content", {}).get("parts", [])
    modern = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()

    out = {"branch": branch, "model": model, "prompt": prompt, "modern": modern,
           "usage": d.get("usageMetadata"), "key": kname, "cached": False}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
