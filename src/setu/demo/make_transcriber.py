"""Build a page for transcribing real line crops by hand.

Every accuracy number on real imagery so far is measured against
**machine labels** from a vision model, which is the weakest part of the
whole route: nothing has checked whether those labels are right. A few
dozen lines transcribed by a Kannada reader fix that, and they are worth
far more than their count suggests:

1. they give a **true CER** on real lines, instead of agreement with a
   model that may be confidently wrong;
2. they measure the machine labels themselves, turning "we had no human
   verification" into a number;
3. they are the held-out lines, so they cost nothing in training data.

The page is deliberately **blind**: it never shows the machine reading
beside the image. Seeing it first would anchor the transcription and
quietly turn the gold set into a review of the model's output.

Transcriptions are written to the artifact's shared store as they are
typed, so they can be read back directly rather than copied by hand.

Usage:
    python -m setu.demo.make_transcriber --out page.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

from setu.data.translit import js_tables

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CSS = """
/* Layout: a transcription worklist -- one row per line, image above the
   field that transcribes it, progress pinned to the top. */
:root {
  --ink: #2b2317; --ink-soft: #6b6052; --leaf: #f6f3ec; --leaf-edge: #e4ded1;
  --card: #fffdf8; --indigo: #3c4a7a; --done: #1f6b4f; --warn: #9a3d2e;
  --font-ui: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --font-kn: "Noto Serif Kannada", "Nirmala UI", serif;
  --font-mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #e8e2d6; --ink-soft: #9f978a; --leaf: #17150f; --leaf-edge: #322d23;
    --card: #211d16; --indigo: #9aa6d8; --done: #63c49b; --warn: #e08a76;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ink: #e8e2d6; --ink-soft: #9f978a; --leaf: #17150f; --leaf-edge: #322d23;
  --card: #211d16; --indigo: #9aa6d8; --done: #63c49b; --warn: #e08a76;
  color-scheme: dark;
}
body { background: var(--leaf); color: var(--ink); font-family: var(--font-ui); }
.wrap { max-width: 980px; margin: 0 auto; padding-inline: 20px; padding-block: 24px 80px; }
h1 { font-size: clamp(1.4rem, 1.1rem + 1.3vw, 1.9rem); font-weight: 600; margin: 0 0 6px;
     letter-spacing: -0.015em; }
.sub { color: var(--ink-soft); max-width: 60ch; line-height: 1.6; margin: 0 0 20px; }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5;
       background: var(--leaf); border-bottom: 1px solid var(--leaf-edge);
       padding: 10px 0 12px; margin-bottom: 20px;
       display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.count { font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 0.95rem; }
.track { flex: 1 1 160px; height: 6px; background: var(--leaf-edge); border-radius: 3px;
         overflow: hidden; min-width: 120px; }
.fill { height: 100%; background: var(--done); width: 0%; transition: width .25s ease; }
.state { font-size: 0.78rem; color: var(--ink-soft); }
.state[data-s="saving"] { color: var(--indigo); }
.state[data-s="error"] { color: var(--warn); }
.row { border: 1px solid var(--leaf-edge); border-radius: 8px; background: var(--card);
       margin-bottom: 18px; overflow: hidden; }
.row.done { border-color: var(--done); }
.row > header { display: flex; justify-content: space-between; gap: 10px; align-items: baseline;
                padding: 8px 14px; border-bottom: 1px solid var(--leaf-edge);
                font-family: var(--font-mono); font-size: 0.76rem; color: var(--ink-soft); }
.strip { padding: 12px 14px; background: var(--leaf); overflow-x: auto; }
.strip img { display: block; max-width: 100%; height: auto; min-width: 300px; border-radius: 3px; }
.field { padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
textarea { font-family: var(--font-kn); font-size: 1.05rem; line-height: 1.9; width: 100%;
           box-sizing: border-box; min-height: 3.2em; resize: vertical; padding: 9px 11px;
           border: 1px solid var(--leaf-edge); border-radius: 5px;
           background: var(--leaf); color: var(--ink); }
textarea:focus { outline: 2px solid var(--indigo); outline-offset: 1px; }
.roman { font-family: var(--font-mono); font-size: 0.95rem; width: 100%; box-sizing: border-box;
         padding: 9px 11px; border: 1px solid var(--leaf-edge); border-radius: 5px;
         background: var(--leaf); color: var(--ink); }
.roman:focus { outline: 2px solid var(--indigo); outline-offset: 1px; }
.preview { font-family: var(--font-kn); font-size: 1.1rem; line-height: 1.95; min-height: 1.95em;
           padding: 8px 11px; border-radius: 5px; background: var(--leaf);
           border: 1px dashed var(--leaf-edge); word-break: break-word; }
.preview:empty::before { content: "Kannada appears here as you type"; color: var(--ink-soft);
                         font-family: var(--font-ui); font-size: 0.82rem; }
.mode { display: flex; gap: 6px; }
.mode button { font: inherit; font-size: 0.74rem; padding: 3px 9px; cursor: pointer;
               border: 1px solid var(--leaf-edge); background: var(--card); color: var(--ink-soft);
               border-radius: 4px; }
.mode button[aria-pressed="true"] { background: var(--indigo); color: #fff; border-color: var(--indigo); }
.help { border: 1px solid var(--leaf-edge); border-radius: 8px; background: var(--card);
        padding: 14px 16px; margin-bottom: 22px; font-size: 0.84rem; line-height: 1.7; }
.help summary { cursor: pointer; font-weight: 600; color: var(--ink); }
.help table { border-collapse: collapse; margin-top: 10px; width: 100%; }
.help td { padding: 3px 10px 3px 0; vertical-align: top; }
.help code { font-family: var(--font-mono); font-size: 0.88em; color: var(--indigo); }
.help .kn { font-family: var(--font-kn); font-size: 1.05em; }
.who { color: var(--done); font-weight: 600; }
.tools { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
         font-size: 0.78rem; color: var(--ink-soft); }
label.chk { display: flex; gap: 5px; align-items: center; cursor: pointer; }
footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--leaf-edge);
         color: var(--ink-soft); font-size: 0.78rem; line-height: 1.7; }
.offline { background: var(--warn); color: #fff; padding: 10px 14px; border-radius: 6px;
           margin-bottom: 18px; font-size: 0.85rem; }
"""

JS = """
const KEY = c => c.replace(/[^A-Za-z0-9_.-]/g, "__");

// Same tables as setu.data.translit, emitted from it so there is one
// source of truth; tests/test_translit.py checks the two agree.
const T = __TRANSLIT_TABLES__;
const byLen = o => Object.keys(o).sort((a, b) => b.length - a.length);
const CONS_K = byLen(T.cons), VOW_K = byLen(T.matra), INDEP_K = byLen(T.indep);
const matchAt = (s, i, keys) => keys.find(k => s.startsWith(k, i)) || null;

function toKannada(text) {
  let out = "", i = 0;
  while (i < text.length) {
    if (text.startsWith("||", i)) { out += "॥"; i += 2; continue; }
    const ch = text[i];
    if (ch === "|") { out += "।"; i += 1; continue; }
    if (T.digits[ch]) { out += T.digits[ch]; i += 1; continue; }
    const cons = matchAt(text, i, CONS_K);
    if (cons) {
      i += cons.length; out += T.cons[cons];
      if (text[i] === "q") { out += T.virama; i += 1; continue; }
      const v = matchAt(text, i, VOW_K);
      if (v) { out += T.matra[v]; i += v.length; } else { out += T.virama; }
      continue;
    }
    const iv = matchAt(text, i, INDEP_K);
    if (iv) { out += T.indep[iv]; i += iv.length; continue; }
    if (T.signs[ch]) { out += T.signs[ch]; i += 1; continue; }
    out += ch; i += 1;
  }
  return out;
}

let db = null, me = null, myName = "you";
let ready = false;              // db resolved (or definitively absent)
const timers = {};              // per-crop debounce
const pending = new Set();      // crops edited before db was ready
let editing = null;             // crop this viewer has focused
const names = {};               // viewer id -> display name

const rowOf = crop => document.querySelector(`[data-crop="${CSS.escape(crop)}"]`);

function setState(s, msg) {
  const el = document.getElementById("state");
  el.dataset.s = s; el.textContent = msg;
}
function kannadaOf(row) {
  return row.dataset.mode === "direct"
    ? row.querySelector("textarea").value
    : row.querySelector(".preview").textContent;
}
function setMode(row, mode) {
  row.dataset.mode = mode;
  row.querySelector(".roman-wrap").hidden = mode !== "roman";
  row.querySelector("textarea").hidden = mode !== "direct";
  row.querySelectorAll(".mode button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
  try { localStorage.setItem("setu.mode", mode); } catch (e) { /* private window */ }
}
function refreshProgress() {
  const rows = [...document.querySelectorAll(".row")];
  let done = 0;
  rows.forEach(row => {
    const filled = !!(kannadaOf(row).trim() || row.querySelector(".ill").checked);
    row.classList.toggle("done", filled);
    if (filled) done += 1;
  });
  document.getElementById("count").textContent = done + " / " + rows.length;
  document.getElementById("fill").style.width = (100 * done / rows.length) + "%";
}

async function flush(crop) {
  // Writes are last-writer-wins and the store wants one write at a time
  // per document, so each crop has its own debounce and we await it.
  if (!db) { pending.add(crop); return; }
  const row = rowOf(crop);
  if (!row) return;
  const body = {
    crop,
    text: kannadaOf(row),
    illegible: row.querySelector(".ill").checked,
    by: me || "unknown",
    byName: myName,
    updatedAt: new Date().toISOString(),
  };
  setState("saving", "saving...");
  try {
    await db.collection("transcriptions").doc(KEY(crop)).set(body);
    setState("ok", "all work saved");
  } catch (e) {
    const code = (e && e.code) || "error";
    setState("error", code === "invalid_argument"
      ? "can't save - you may only have view access"
      : "not saved (" + code + ") - your typing is still here");
  }
}
function queueSave(crop) {
  setState("saving", "saving...");
  clearTimeout(timers[crop]);
  timers[crop] = setTimeout(() => flush(crop), 700);
}

function applyRemote(d) {
  // Never overwrite the line this viewer is typing in.
  if (!d || !d.crop || d.crop === editing) return;
  const row = rowOf(d.crop);
  if (!row) return;
  const ta = row.querySelector("textarea");
  const ill = row.querySelector(".ill");
  if ((d.text || "") !== kannadaOf(row)) {
    ta.value = d.text || "";
    if ((d.text || "").trim()) setMode(row, "direct");
  }
  ill.checked = !!d.illegible;
  ta.disabled = ill.checked;
  row.querySelector(".roman").disabled = ill.checked;
  const who = row.querySelector(".who");
  if (d.by && d.by !== me) {
    who.textContent = (names[d.by] || d.byName || "someone else") + " filled this";
    who.hidden = false;
  } else { who.hidden = true; }
}

function wire() {
  let saved = "roman";
  try { saved = localStorage.getItem("setu.mode") || "roman"; } catch (e) { /* ignore */ }
  document.querySelectorAll(".row").forEach(row => {
    const crop = row.dataset.crop;
    const ta = row.querySelector("textarea");
    const roman = row.querySelector(".roman");
    const prev = row.querySelector(".preview");
    const ill = row.querySelector(".ill");
    setMode(row, saved);
    [roman, ta].forEach(el => {
      el.addEventListener("focus", () => { editing = crop; });
      el.addEventListener("blur", () => { if (editing === crop) editing = null; });
    });
    roman.addEventListener("input", () => {
      prev.textContent = toKannada(roman.value);
      refreshProgress(); queueSave(crop);
    });
    ta.addEventListener("input", () => { refreshProgress(); queueSave(crop); });
    ill.addEventListener("change", () => {
      ta.disabled = ill.checked; roman.disabled = ill.checked;
      refreshProgress(); queueSave(crop);
    });
    row.querySelectorAll(".mode button").forEach(b =>
      b.addEventListener("click", () => setMode(row, b.dataset.mode)));
  });
  refreshProgress();
}

wire();
setState("saving", "connecting...");

(async () => {
  db = await window.claude?.use?.("db");
  ready = true;
  if (!db) {
    document.getElementById("offline").hidden = false;
    setState("error", "not saving - see the note above");
    return;
  }
  const user = await window.claude?.use?.("user");
  try { me = user ? await user.id() : null; } catch (e) { me = null; }
  try { myName = (user && (await user.me())?.name) || "you"; } catch (e) { /* no name scope */ }
  setState("ok", "connected");

  // Live: everyone's entries appear as they are typed, except in the
  // line this viewer is editing. Subscribe once.
  db.collection("transcriptions").onSnapshot(snap => {
    const ids = snap.docs.map(d => d.data()?.by).filter(b => b && b !== me);
    if (ids.length && window.claude?.use) {
      window.claude.use("user").then(u => u?.profiles?.([...new Set(ids)]).then(ps => {
        Object.entries(ps || {}).forEach(([id, p]) => { names[id] = p.name || ""; });
      })).catch(() => {});
    }
    snap.docs.forEach(doc => applyRemote(doc.data()));
    refreshProgress();
  }, err => {
    setState("error", "live sync stopped (" + err.code + ") - reload the page");
  });

  // Anything typed before the store was ready.
  pending.forEach(c => flush(c));
  pending.clear();
})();
"""


def crop_uri(path: Path, max_width: int) -> str:
    im = Image.open(path)
    if im.width > max_width:
        im = im.resize((max_width, max(1, round(im.height * max_width / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, nargs="+", default=[
        Path("data/real_lines/train/train_labels.jsonl"),
        Path("data/real_lines/demo/demo_labels.jsonl")])
    p.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    p.add_argument("--max-image-width", type=int, default=1500)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Manuscript Transcription",
                   help="each line set gets its own artifact, and its own shared store; "
                        "a distinct title is what keeps them apart")
    args = p.parse_args()

    rows = []
    for f in args.labels:
        rows += [json.loads(l) for l in f.open(encoding="utf-8")]
    if args.split != "all":
        rows = [r for r in rows if r.get("demo_split") == args.split]
    rows.sort(key=lambda r: (r["page"], r["crop"]))
    if not rows:
        raise SystemExit("no lines selected")

    cards = []
    for i, r in enumerate(rows, 1):
        uri = crop_uri(REPO_ROOT / r["image"], args.max_image_width)
        cards.append(f"""
<section class="row" data-crop="{r['crop']}">
  <header><span>{i} of {len(rows)} &middot; {r['page']}</span><span>{r['width']}&times;{r['height']}px</span></header>
  <div class="strip"><img src="{uri}" alt="manuscript line {i}" loading="lazy"></div>
  <div class="field">
    <div class="mode">
      <button type="button" data-mode="roman" aria-pressed="true">type in English letters</button>
      <button type="button" data-mode="direct" aria-pressed="false">type Kannada directly</button>
    </div>
    <div class="roman-wrap">
      <input class="roman" type="text" spellcheck="false" autocapitalize="off" autocorrect="off"
             placeholder="kannaDa  &rarr;  the Kannada appears below">
      <div class="preview" aria-live="polite"></div>
    </div>
    <textarea rows="2" spellcheck="false" hidden
              placeholder="Paste or type Kannada here"></textarea>
    <div class="tools">
      <label class="chk"><input type="checkbox" class="ill"> can't read this line</label>
      <span class="who" hidden></span>
      <span>Transcribe what is written, numerals included. Don't modernise the spelling.</span>
    </div>
  </div>
</section>""")

    html = f"""<title>{args.title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&family=Noto+Serif+Kannada:wght@400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<h1>{args.title}</h1>
<p class="sub">{len(rows)} palm-leaf lines, held out from training. These are the lines every
accuracy number is measured against &mdash; right now against a vision model's reading, which
nothing has checked. Your transcriptions become the ground truth.</p>

<div id="offline" class="offline" hidden>
Not connected to the shared store, so nothing typed here is being saved. Reopen the page from
your artifacts list; if it still says this, tell Claude before typing anything.
</div>

<p class="sub" style="margin-bottom:18px">Several people can work on this at once &mdash; lines
other people fill in appear here live, and the line you are typing in is never overwritten. To
let teammates in, open <strong>Share</strong> on this page and give them
<strong>Contributor</strong> or higher; Viewers can read it but cannot save.</p>

<details class="help" open>
<summary>Typing Kannada without a Kannada keyboard</summary>
<p>Type it the way it sounds, in English letters, and the Kannada appears underneath &mdash;
<code>kannaDa</code> gives <span class="kn">&#3221;&#3240;&#3277;&#3240;&#3233;</span>.
Capitals mark the long and retroflex letters. If you have a Kannada keyboard or want to paste,
switch any line to &ldquo;type Kannada directly&rdquo;.</p>
<table>
<tr><td><code>a aa/A i ii/I u uu/U</code></td><td class="kn">&#3205; &#3206; &#3207; &#3208; &#3209; &#3210;</td>
    <td><code>e E ai o O au</code></td><td class="kn">&#3214; &#3215; &#3216; &#3218; &#3219; &#3220;</td></tr>
<tr><td><code>k kh g gh</code></td><td class="kn">&#3221; &#3222; &#3223; &#3224;</td>
    <td><code>ch j jh ~n</code></td><td class="kn">&#3226; &#3228; &#3229; &#3230;</td></tr>
<tr><td><code>T Th D Dh N</code></td><td class="kn">&#3231; &#3232; &#3233; &#3234; &#3235;</td>
    <td><code>t th d dh n</code></td><td class="kn">&#3236; &#3237; &#3238; &#3239; &#3240;</td></tr>
<tr><td><code>p ph b bh m</code></td><td class="kn">&#3242; &#3243; &#3244; &#3245; &#3246;</td>
    <td><code>y r l L v</code></td><td class="kn">&#3247; &#3248; &#3250; &#3251; &#3253;</td></tr>
<tr><td><code>sh Sh s h</code></td><td class="kn">&#3254; &#3255; &#3256; &#3257;</td>
    <td><code>M (&#3202;) H (&#3203;)</code></td><td class="kn">&#3256;&#3202;&#3223; &#3238;&#3265;&#3203;</td></tr>
<tr><td><code>rY lY</code> &mdash; the archaic letters</td><td class="kn">&#3249; &#3294;</td>
    <td><code>| ||</code> and <code>0-9</code></td><td class="kn">&#2404; &#2405; &#3238;&#3240;</td></tr>
</table>
<p>Consonants join up on their own: <code>mugdhe</code> gives
<span class="kn">&#3246;&#3265;&#3223;&#3277;&#3238;&#3271;</span>,
<code>marYe</code> gives <span class="kn">&#3246;&#3249;&#3271;</span>.</p>
</details>

<div class="bar">
  <span class="count" id="count">0 / {len(rows)}</span>
  <span class="track"><span class="fill" id="fill"></span></span>
  <span class="state" id="state" data-s="ok">starting</span>
</div>

{"".join(cards)}

<footer>
Transcribe only what is on the leaf &mdash; no corrections, no modernisation. Leave a line blank
and tick &ldquo;can't read this line&rdquo; if it is illegible; a skipped line is more useful than
a guessed one. The machine's reading is deliberately not shown, so it cannot anchor yours.
Work saves as you type.
</footer>
</div>
<script>{JS}</script>"""

    html = html.replace("__TRANSLIT_TABLES__", js_tables())
    args.out.write_text(html, encoding="utf-8")
    print(f"{len(rows)} lines -> {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
