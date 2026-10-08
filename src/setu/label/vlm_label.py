"""Machine labels for real line crops (DEMO_PLAN.md, teacher-student).

A vision LLM transcribes each real HKHPL line crop. Two independent model
families label the same crops; only lines where they agree become
training labels for the CRNN. There is no human verification on this
project, so cross-model agreement is the quality signal.

Providers (keys from `.env`, never from the command line):

    gemini      Google Gemini API (GEMINI_API_KEY)
    groq        Groq (GROQ_API_KEY)
    openrouter  OpenRouter, incl. its `:free` models (OPENROUTER_API_KEY)

Stdlib only (urllib), so no API client dependency is added. Groq sits
behind Cloudflare, which rejects Python's default User-Agent with error
1010, so every request sends its own.

Output is resumable and append-only, one JSON line per crop:

    <set_dir>/labels_<provider>_<model>.jsonl

A crop already labelled by that provider/model is skipped, so a run cut
short by a quota error resumes where it stopped. Errors are recorded,
never silently dropped. Each invocation also writes a run folder
(CLAUDE.md rule 4).

Usage:
    python -m setu.label.vlm_label --provider gemini --model gemini-3.5-flash --limit 30
    python -m setu.label.vlm_label --provider groq --model qwen/qwen3.8-27b --limit 30
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from setu.runlog import finish_run, start_run

SEED = 0  # CLAUDE.md rule 6
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
USER_AGENT = "setu-demo/0.1 (+palm-leaf research project)"

# Faithful transcription, NOT modernization -- the labels train the
# recogniser, which must read what is on the leaf. ಱ/ೞ are named
# explicitly because LLMs normalise them (STATUS.md 3.9).
PROMPT = (
    "This image is ONE line cropped from an old Kannada palm-leaf manuscript. "
    "Transcribe exactly the Kannada characters written on it, left to right.\n"
    "Rules:\n"
    "- Output ONLY the transcription: Kannada script, spaces between words, "
    "and danda (। ॥) if written. No translation, no Latin letters, no commentary.\n"
    "- Do NOT modernise or correct spelling. Keep archaic letters such as ಱ and ೞ "
    "exactly as written.\n"
    "- If part of the line is illegible, transcribe what you can read and skip the rest.\n"
    "- If nothing on the line is legible, output exactly: UNREADABLE"
)

# Batched variant: several crops in one request. The free tier throttles
# per request, not per image, so this is the difference between 163 calls
# and 17. Identity is guaranteed by construction -- we control the order
# the images go in -- provided the reply is parsed strictly (see
# parse_batch: a reply that does not account for every image is rejected
# whole rather than risk pairing a transcription with the wrong crop).
BATCH_PROMPT = (
    "You are given {n} separate images. Each is ONE line cropped from an old "
    "Kannada palm-leaf manuscript, in order.\n"
    "Transcribe each one exactly: the Kannada characters written on it, left to right.\n"
    "Rules:\n"
    "- Output exactly {n} lines, one per image, each starting with its number "
    "and a colon, like:\n"
    "1: <transcription of image 1>\n"
    "2: <transcription of image 2>\n"
    "- Nothing else: no blank lines between them, no commentary, no translation, "
    "no Latin letters. Kannada script, spaces between words, and danda (। ॥) if written.\n"
    "- Do NOT modernise or correct spelling. Keep archaic letters such as ಱ and ೞ "
    "exactly as written.\n"
    "- If an image is wholly illegible, write UNREADABLE as its transcription, "
    "but still output its numbered line."
)

# Kannada block, ZWJ/ZWNJ, danda/double danda, space.
_KEEP = re.compile(r"[^ಀ-೿‌‍।॥ ]")
_NUMBERED = re.compile(r"^\s*(\d+)\s*[:.)]\s*(.*)$")


def parse_batch(raw: str, n: int) -> list[str] | None:
    """Pull `n` numbered transcriptions out of a batch reply.

    Returns None unless the reply accounts for exactly images 1..n, once
    each. A partial or renumbered reply is rejected whole: silently
    shifting transcriptions onto the wrong crops would poison the
    training labels in a way nothing downstream could detect (CLAUDE.md
    rule 7 -- fail loudly)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    found: dict[int, str] = {}
    for line in raw.splitlines():
        m = _NUMBERED.match(line)
        if m:
            idx = int(m.group(1))
            if idx in found:  # a repeated number means the mapping is unreliable
                return None
            found[idx] = m.group(2)
    if set(found) != set(range(1, n + 1)):
        return None
    return [found[i] for i in range(1, n + 1)]


class KeyRing:
    """Rotates over the team's free-tier keys (DEMO_PLAN.md: 4 members, 4
    keys). Each key has its own daily quota, so a key that returns 429 is
    retired for the rest of the run and the next one takes over. Raises
    only when every key is exhausted, so a run never silently continues
    on a single key."""

    def __init__(self, keys: list[tuple[str, str]]):
        self.keys = keys            # [(env var name, value)]
        self.i = 0
        self.dead: set[str] = set()

    def current(self) -> tuple[str, str]:
        if len(self.dead) >= len(self.keys):
            raise RuntimeError("every key is out of daily quota")
        while self.keys[self.i][0] in self.dead:
            self.i = (self.i + 1) % len(self.keys)
        return self.keys[self.i]

    def retire(self, name: str) -> bool:
        """Mark a key exhausted. Returns True if another key is left."""
        self.dead.add(name)
        return len(self.dead) < len(self.keys)

    def advance(self) -> None:
        self.i = (self.i + 1) % len(self.keys)


def collect_keys(env: dict[str, str], base: str) -> list[tuple[str, str]]:
    """GEMINI_API_KEY plus GEMINI_API_KEY_2..9, in order, skipping blanks
    and duplicates (a pasted-twice key would otherwise look like extra
    quota that does not exist)."""
    out, seen = [], set()
    for name in [base] + [f"{base}_{i}" for i in range(2, 10)]:
        v = (env.get(name) or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append((name, v))
    return out


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"{path} not found -- create it with the API keys (see DEMO_PLAN.md 5)")
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def clean(text: str) -> str:
    """Drop reasoning blocks, then everything that is not Kannada script."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if text.strip() == "UNREADABLE":
        return ""
    return re.sub(r"\s+", " ", _KEEP.sub(" ", text)).strip()


def _post(url: str, headers: dict, body: dict, timeout: int = 240) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


GEMINI_THINKING: str | None = None  # set from --thinking; None = model default


def call_gemini(key: str, model: str, pngs: list[bytes]) -> tuple[str, dict]:
    parts: list[dict] = []
    if len(pngs) == 1:
        parts.append({"inline_data": {"mime_type": "image/png",
                                      "data": base64.b64encode(pngs[0]).decode()}})
        parts.append({"text": PROMPT})
    else:
        for i, png in enumerate(pngs, 1):
            parts.append({"text": f"Image {i}:"})
            parts.append({"inline_data": {"mime_type": "image/png",
                                          "data": base64.b64encode(png).decode()}})
        parts.append({"text": BATCH_PROMPT.format(n=len(pngs))})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0},
    }
    if GEMINI_THINKING:
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": GEMINI_THINKING}
    d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
              {"x-goog-api-key": key}, body)
    cands = d.get("candidates") or []
    if not cands:
        raise RuntimeError(f"no candidates: {json.dumps(d.get('promptFeedback', d))[:300]}")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return text, {"finish_reason": cands[0].get("finishReason"), "usage": d.get("usageMetadata")}


def _openai_style_content(pngs: list[bytes]) -> list[dict]:
    if len(pngs) == 1:
        content = [{"type": "text", "text": PROMPT}]
    else:
        content = [{"type": "text", "text": BATCH_PROMPT.format(n=len(pngs))}]
    for png in pngs:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    return content


def call_groq(key: str, model: str, pngs: list[bytes]) -> tuple[str, dict]:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": _openai_style_content(pngs)}],
    }
    d = _post("https://api.groq.com/openai/v1/chat/completions",
              {"Authorization": "Bearer " + key}, body)
    choice = d["choices"][0]
    return choice["message"].get("content") or "", {
        "finish_reason": choice.get("finish_reason"), "usage": d.get("usage")}


def call_openrouter(key: str, model: str, pngs: list[bytes]) -> tuple[str, dict]:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": _openai_style_content(pngs)}],
    }
    d = _post("https://openrouter.ai/api/v1/chat/completions",
              {"Authorization": "Bearer " + key}, body)
    if "choices" not in d:
        raise RuntimeError(f"no choices: {json.dumps(d)[:300]}")
    choice = d["choices"][0]
    return choice["message"].get("content") or "", {
        "finish_reason": choice.get("finish_reason"), "usage": d.get("usage")}


PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", call_gemini),
    "groq": ("GROQ_API_KEY", call_groq),
    "openrouter": ("OPENROUTER_API_KEY", call_openrouter),
}


def pick_crops(records: list[dict], limit: int | None, group: str | None = None,
               whole_pages: bool = False) -> list[dict]:
    """With --limit, a seeded draw stratified by manuscript group (stem
    prefix 1-4), so a 30-crop go/no-go sees every hand, not one page."""
    if group:
        records = [r for r in records if r["page"].split(".")[0] == group]
        if not records:
            raise SystemExit(f"no crops in manuscript group {group!r}")
    if not limit or limit >= len(records):
        return records
    rng = random.Random(SEED)
    if whole_pages:
        # Batching only pays off when a request holds consecutive lines of
        # one page, so a limited batched run takes whole pages in order
        # rather than crops scattered over many pages.
        by_page: dict[str, list[dict]] = {}
        for r in records:
            by_page.setdefault(r["page"], []).append(r)
        picked: list[dict] = []
        for page in sorted(by_page, key=lambda p: (p.split(".")[0], p)):
            if len(picked) >= limit:
                break
            picked += sorted(by_page[page], key=lambda r: r["line_index"])
        return picked[:limit]
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["page"].split(".")[0], []).append(r)
    per = -(-limit // len(groups))  # ceil
    picked = []
    for g in sorted(groups):
        picked += rng.sample(groups[g], min(per, len(groups[g])))
    return sorted(picked[:limit], key=lambda r: r["crop"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--set-dir", type=Path, default=REPO_ROOT / "data" / "real_lines" / "train")
    p.add_argument("--crop-field", default="crop", choices=["crop", "crop_bbox"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--group", default=None,
                   help="Only crops from this manuscript group (page stem prefix, e.g. 1).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Crops per request. The free tier throttles per request, not per "
                        "image, so 8-12 cuts a page to one or two calls. Batches never "
                        "span pages and a reply that does not account for every image is "
                        "rejected whole.")
    p.add_argument("--rpm", type=float, default=8.0,
                   help="Requests per minute ceiling -- stay under the free-tier limit.")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--out-suffix", default="",
                   help="Appended to the labels filename, to keep a second pass (different "
                        "thinking depth or batch size) separate for the agreement check.")
    p.add_argument("--thinking", default=None,
                   help="Gemini thinkingLevel (e.g. low). Output goes to a separate labels file.")
    args = p.parse_args()
    global GEMINI_THINKING
    GEMINI_THINKING = args.thinking
    random.seed(SEED)

    key_name, call = PROVIDERS[args.provider]
    ring = KeyRing(collect_keys(load_env(REPO_ROOT / ".env"), key_name))
    if not ring.keys:
        raise SystemExit(f"{key_name} is empty in .env")
    print(f"{len(ring.keys)} key(s): {', '.join(n for n, _ in ring.keys)}")

    records = [json.loads(l) for l in (args.set_dir / "manifest.jsonl").open(encoding="utf-8")]
    todo = pick_crops(records, args.limit, args.group, whole_pages=args.batch_size > 1)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model) + (f"_think-{args.thinking}" if args.thinking else "") + (f"_{args.out_suffix}" if args.out_suffix else "")
    out_path = args.set_dir / f"labels_{args.provider}_{safe_model}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            rec = json.loads(line)
            if "text" in rec:  # errored crops are retried on the next run
                done.add(rec["crop"])
    todo = [r for r in todo if r[args.crop_field] not in done]
    print(f"{args.provider}/{args.model}: {len(todo)} crops to label ({len(done)} already done)")

    run_dir = start_run(f"vlm_label_{args.provider}", {
        "seed": SEED, "provider": args.provider, "model": args.model,
        "set_dir": str(args.set_dir), "crop_field": args.crop_field, "limit": args.limit,
        "group": args.group,
        "rpm": args.rpm, "batch_size": args.batch_size, "thinking": args.thinking, "prompt": PROMPT if args.batch_size == 1 else BATCH_PROMPT, "temperature": 0, "output": str(out_path),
    })

    gap = 60.0 / args.rpm
    n_ok, n_err, n_empty = 0, 0, 0
    last = 0.0
    # Batches never span pages: one page's lines share a hand and a
    # layout, which is easier to read in one go, and a rejected batch
    # then costs one page's lines rather than an arbitrary slice.
    batches: list[list[dict]] = []
    for rec in todo:
        if (batches and len(batches[-1]) < args.batch_size
                and batches[-1][-1]["page"] == rec["page"]):
            batches[-1].append(rec)
        else:
            batches.append([rec])
    if args.batch_size > 1:
        print(f"{len(batches)} request(s), up to {args.batch_size} crops each")

    with out_path.open("a", encoding="utf-8") as f:
        for i, batch in enumerate(batches):
            pngs = [(args.set_dir / r[args.crop_field]).read_bytes() for r in batch]
            rows = [{"crop": r[args.crop_field], "page": r["page"], "provider": args.provider,
                     "model": args.model, "batch_size": len(batch)} for r in batch]
            row = rows[0]  # errors below are recorded on every row in the batch
            for attempt in range(args.max_retries + 1):
                wait = gap - (time.time() - last)
                if wait > 0:
                    time.sleep(wait)
                last = time.time()
                try:
                    kname, key = ring.current()
                except RuntimeError as e:
                    for r in rows:
                        r.update(error=str(e))
                    break
                try:
                    raw, meta = call(key, args.model, pngs)
                    secs = round(time.time() - last, 2)
                    if len(batch) == 1:
                        rows[0].update(raw=raw, text=clean(raw), seconds=secs, key=kname, **meta)
                        break
                    parsed = parse_batch(raw, len(batch))
                    if parsed is None:
                        # Never guess an alignment: redo this batch one crop
                        # at a time so each transcription is unambiguous.
                        if attempt < args.max_retries:
                            print(f"  batch reply did not account for all {len(batch)} images "
                                  f"-- retrying")
                            continue
                        for r in rows:
                            r.update(error="unparseable batch reply", raw=raw, key=kname)
                        break
                    for r, text in zip(rows, parsed):
                        r.update(raw=text, text=clean(text), seconds=secs, key=kname, **meta)
                    break
                except urllib.error.HTTPError as e:
                    msg = e.read()[:800].decode(errors="replace")
                    if e.code == 429:
                        # Daily quota is per key, so rotate rather than wait out
                        # a limit that will not reset for hours.
                        per_day = "PerDay" in msg or "per day" in msg.lower()
                        if per_day and ring.retire(kname):
                            print(f"  {kname} out of daily quota -- switching key")
                            continue
                        if not per_day and attempt < args.max_retries:
                            ring.advance()
                            backoff = min(60, 5 * 2 ** attempt)
                            print(f"  HTTP 429 (rate), next key, {backoff}s")
                            time.sleep(backoff)
                            continue
                    if e.code in (500, 502, 503) and attempt < args.max_retries:
                        backoff = min(60, 5 * 2 ** attempt)
                        print(f"  HTTP {e.code}, retry in {backoff}s")
                        time.sleep(backoff)
                        continue
                    for r in rows:
                        r.update(error=f"HTTP {e.code}: {msg}", key=kname)
                    break
                except (TimeoutError, urllib.error.URLError) as e:
                    if attempt < args.max_retries:
                        print(f"  {type(e).__name__}, retry")
                        continue
                    for r in rows:
                        r.update(error=f"{type(e).__name__}: {str(e)[:300]}")
                    break
                except Exception as e:
                    for r in rows:
                        r.update(error=f"{type(e).__name__}: {str(e)[:300]}")
                    break

            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            errored = [r for r in rows if "error" in r]
            n_err += len(errored)
            n_ok += len(rows) - len(errored)
            n_empty += sum(1 for r in rows if "error" not in r and not r["text"])
            label = f"[{i + 1}/{len(batches)}] {batch[0]['page']}"
            if errored:
                err = errored[0]["error"]
                print(f"  {label}: ERROR x{len(errored)} {err[:110]}")
                if "out of daily quota" in err:
                    print("  all keys exhausted -- stopping; rerun tomorrow or add keys (resumable)")
                    break
                if err.startswith(("HTTP 401", "HTTP 403", "HTTP 404")):
                    print("  auth/model error -- stopping rather than burning the rest of the list")
                    break
            else:
                print(f"  {label}: {len(rows)} lines, "
                      f"{[len(r['text']) for r in rows]} chars")

    finish_run(run_dir, {"n_attempted": n_ok + n_err, "n_ok": n_ok, "n_error": n_err,
                         "n_empty_or_unreadable": n_empty, "labels_file": str(out_path)})
    print(f"ok {n_ok}  errors {n_err}  empty {n_empty}   -> {out_path}")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
